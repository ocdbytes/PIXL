"""Decoder-only transformer that draws a 32x32 image from a caption.

A left-padded caption is followed by 1,024 palette tokens in row-major order:

    <pad> ... <bos> a pine tree <eos> | p1 p2 ... p1023

Causal attention makes <eos> predict p1 and each pixel predict the next. Text and
pixels have separate embeddings; learned caption, row, and column positions give
them order and location. Training uses pixel cross-entropy and caption dropout;
generation uses a key-value cache and optional classifier-free guidance
(Ho & Salimans, 2022, https://arxiv.org/abs/2207.12598).

ClipPixl reads the caption through the frozen CLIP text encoder (ClipText)
instead of learned text embeddings; the pixel part is the same.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer, CLIPTextModel

from .tokenizer import BOS, EOS, PAD, Tokenizer

CLIP = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class Config:
    """Model shape.

    Attributes:
        text_vocab: Caption token ids (src/tokenizer.json).
        pixel_vocab: Pixel token ids (src/palette.json: transparent + 255 colours).
        caption: Caption slot length, <bos> and <eos> included.
        side: Image side in pixels.
        width: Size of every token's vector.
        layers: Transformer blocks.
        heads: Attention heads per block; width must divide by it.
        dropout: Dropout on each block's MLP output; 0 while the data is large for
            the model. Attention has none: on Apple GPUs it stops the fused
            attention kernel and costs ~5.7 GB more memory at batch 16.
    """

    text_vocab: int = 1024
    pixel_vocab: int = 256
    caption: int = 32
    side: int = 32
    # embedding size = 384
    # to make head size a power of 2
    # 384 / 6 (no. of heads) = 64
    width: int = 384
    layers: int = 8
    heads: int = 6
    dropout: float = 0.0

    @property
    def pixels(self) -> int:
        return self.side * self.side


class Block(nn.Module):
    """One transformer block: causal self-attention, then an MLP.

    Causal Attention = Masked Softmax (mask) & Dot Product Attention

    mask :
    0   -∞  -∞  -∞
    0    0  -∞  -∞
    0    0   0  -∞
    0    0   0   0

    Both parts normalise their input first and add their output to the residual
    stream ("pre-norm"), which keeps deep stacks stable to train.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.heads: int = config.heads
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.project = nn.Linear(config.width, config.width)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, cache: dict[str, Tensor] | None = None) -> Tensor:
        """Runs the block on new tokens.

        Args:
            x: (batch, length, width) vectors of the new tokens.
            cache: For generation: keys and values of earlier tokens, extended
                in place with the new ones. None in training.

        Returns:
            (batch, length, width) updated vectors.
        """
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).split(width, dim=2)
        # Divide attention among the heads
        q, k, v = (
            t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for t in (q, k, v)
        )
        if cache is not None:
            if cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # A single new token may see every cached one; a full sequence is causal.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=q.shape[2] == k.shape[2]
        )
        # Attention contributes to the original x
        # before passing attention to the project MLP it is reshaped
        x = x + self.project(attended.transpose(1, 2).reshape(batch, length, width))
        # another residual connection to x by add the main MLP result
        return x + self.mlp(self.mlp_norm(x))


class Pixl(nn.Module):
    """Caption-to-image transformer; see the module docstring for the layout."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        config = config or Config()
        self.config: Config = config
        # Token and Pixel Encodings
        self.text = nn.Embedding(config.text_vocab, config.width)
        self.pixel = nn.Embedding(config.pixel_vocab, config.width)

        # Positional Encodings for captions and pixels
        # This is a form of positional encoding
        self.caption_position = nn.Embedding(config.caption, config.width)
        self.row = nn.Embedding(config.side, config.width)
        self.column = nn.Embedding(config.side, config.width)

        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))

        # Output Layer params
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.pixel_vocab, bias=False)
        self.apply(self._init)
        # GPT-2 scaling: each residual branch starts small, so the stack starts
        # close to the identity however many layers it has.
        for module in self.blocks:
            block = cast(Block, module)
            for layer in (block.project, cast(nn.Linear, block.mlp[2])):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def embed_text(self, captions: Tensor) -> Tensor:
        """Turns (batch, caption) caption ids into vectors with their slot positions."""
        slots = torch.arange(self.config.caption, device=captions.device)
        return self.text(captions) + self.caption_position(slots)

    def embed_pixels(self, pixels: Tensor, start: int = 0) -> Tensor:
        """Turns pixel tokens into vectors with their row and column added.

        Args:
            pixels: (batch, n) pixel tokens.
            start: Index of the first of them in the image.

        Returns:
            (batch, n, width) vectors.
        """
        where = torch.arange(start, start + pixels.shape[1], device=pixels.device)
        return (
            self.pixel(pixels)
            + self.row(where // self.config.side)
            + self.column(where % self.config.side)
        )

    def run(self, x: Tensor, caches: list[dict[str, Tensor]] | None = None) -> Tensor:
        """Runs the blocks and returns pixel logits for every position."""
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None)
        return self.head(self.norm(x))

    def forward(self, captions: Tensor, pixels: Tensor) -> Tensor:
        """Predicts every pixel from the caption and the pixels before it.

        Args:
            captions: (batch, caption) caption ids.
            pixels: (batch, n) pixel tokens, n <= pixels; the last one is not
                needed as input, since nothing follows it.

        Returns:
            (batch, n, pixel_vocab) logits: entry i predicts pixel i.
        """
        x = torch.cat([self.embed_text(captions), self.embed_pixels(pixels[:, :-1])], 1)
        # The last caption position (<eos>) predicts pixel 0.
        return self.run(x)[:, self.config.caption - 1 :]

    def loss(self, captions: Tensor, pixels: Tensor) -> Tensor:
        """Mean cross-entropy over all pixel predictions."""
        logits = self(captions, pixels)
        return F.cross_entropy(logits.flatten(0, 1), pixels.flatten())

    def empty_captions(self, batch: int) -> Tensor:
        """The empty caption slot for each of batch rows: the unconditioned input for guidance."""
        return encode_captions([""] * batch, None)

    @torch.no_grad()
    def generate(
        self,
        captions: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        guidance: float = 1.0,
    ) -> Tensor:
        """Draws images one pixel at a time.

        Args:
            captions: (batch, caption) caption ids.
            temperature: Below 1 picks likely colours more often; above 1, less.
            top_k: If set, sample only from the k most likely tokens.
            guidance: Classifier-free guidance scale; 1 uses the caption as is,
                above 1 follows it more strongly at twice the cost per step.

        Returns:
            (batch, pixels) pixel tokens.
        """
        batch = captions.shape[0]
        if guidance != 1:
            captions = torch.cat(
                [captions, self.empty_captions(batch).to(captions.device)]
            )
        # Each block keeps the keys and values of earlier tokens, so every step
        # runs the network on the newest pixel only.
        caches: list[dict[str, Tensor]] = [{} for _ in self.blocks]
        logits = self.run(self.embed_text(captions), caches)[:, -1]
        pixels: list[Tensor] = []
        for i in range(self.config.pixels):
            logits = logits / temperature
            if guidance != 1:
                captioned, plain = logits[:batch], logits[batch:]
                logits = plain + guidance * (captioned - plain)
            if top_k is not None:
                cutoff = logits.topk(top_k).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            pixels.append(token)
            if guidance != 1:
                token = token.repeat(2, 1)
            if i + 1 < self.config.pixels:
                logits = self.run(self.embed_pixels(token, start=i), caches)[:, -1]
        return torch.cat(pixels, dim=1)


class ClipPixl(Pixl):
    """Pixl that reads captions as frozen CLIP text outputs instead of learned ids.

    Wherever Pixl takes (batch, caption) ids, this takes the (batch, caption,
    clip_width) slot from ClipText. One linear layer maps it to the model width;
    every other weight has Pixl's shape, so stage 1 weights load into it.
    """

    def __init__(self, config: Config | None = None, clip_width: int = 512) -> None:
        super().__init__(config)
        del self.text
        self.text_in = nn.Linear(clip_width, self.config.width)
        self._init(self.text_in)

    def embed_text(self, captions: Tensor) -> Tensor:
        """Turns (batch, caption, clip_width) CLIP outputs into vectors with their slot positions."""
        slots = torch.arange(self.config.caption, device=captions.device)
        return self.text_in(captions) + self.caption_position(slots)

    def empty_captions(self, batch: int) -> Tensor:
        """All-zero slots, which ClipText also gives empty and dropped captions."""
        return torch.zeros(batch, self.config.caption, self.text_in.in_features)


def encode_captions(
    texts: list[str],
    tokenizer: Tokenizer | None,
    drop: float = 0.0,
    length: int = Config.caption,
) -> Tensor:
    """Turns captions into the left-padded caption slot.

    Args:
        texts: Captions.
        tokenizer: Caption tokenizer; None is allowed when every text is empty.
        drop: Chance of replacing each caption with the empty caption, so the
            model also learns unconditioned images (for classifier-free guidance).
        length: Slot length; longer captions keep their first words.

    Returns:
        (len(texts), length) ids: <pad>..., <bos>, caption, <eos>.
    """
    rows: list[list[int]] = []
    for text in texts:
        if text and random.random() >= drop:
            if tokenizer is None:
                raise ValueError("tokenizer is required for non-empty captions")
            ids = tokenizer.encode(text)
        else:
            ids = []
        ids = [BOS, *ids[: length - 2], EOS]
        rows.append([PAD] * (length - len(ids)) + ids)
    return torch.tensor(rows)


class ClipText:
    """Frozen CLIP text encoder that fills the caption slot of a ClipPixl.

    Each caption becomes CLIP's per-token outputs from its last layer, padded
    with zeros to the slot length. Empty and dropped captions are all zeros.

    Attributes:
        device: Where CLIP runs and its outputs are put.
        tokenizer: CLIP's tokenizer.
        model: CLIP's text half, frozen in eval mode.
    """

    def __init__(self, device: str, name: str = CLIP) -> None:
        self.device = device
        self.tokenizer = cast(Any, AutoTokenizer.from_pretrained(name))
        self.model = (
            cast(Any, CLIPTextModel.from_pretrained(name))
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    @torch.no_grad()
    def __call__(
        self, texts: list[str], drop: float = 0.0, length: int = Config.caption
    ) -> Tensor:
        """Turns captions into the caption slot.

        Args:
            texts: Captions.
            drop: Chance of replacing each caption with the empty caption, as in
                encode_captions.
            length: Slot length in CLIP tokens; longer captions keep their first words.

        Returns:
            (len(texts), length, 512) float tensor on the device.
        """
        keep = [bool(text) and random.random() >= drop for text in texts]
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        mask = (
            inputs["attention_mask"] * torch.tensor(keep, device=self.device)[:, None]
        )
        return self.model(**inputs).last_hidden_state * mask[..., None]
