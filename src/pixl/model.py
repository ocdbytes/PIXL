"""Caption-to-image transformers for 32x32 palette-token images.

Pixl is the earlier decoder-only model. Its training input is a left-padded
caption followed by the first 1,023 palette tokens in row-major order:

    <pad> ... <bos> a pine tree <eos> | p0 p1 ... p1022

Causal attention makes <eos> predict p0 and each pixel predict the next, through
p1023. Text and pixels have separate embeddings; learned caption, row, and column
positions give them order and location. Training uses pixel cross-entropy and caption dropout;
generation uses a key-value cache and optional classifier-free guidance
(Ho & Salimans, 2022, https://arxiv.org/abs/2207.12598).

EncoderPixl is the model trained by pixl.train: a bidirectional caption encoder
and Pixl's causal pixel blocks reading it through cross-attention. Its decoder
starts with a learned start token instead of a caption prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .tokenizer import BOS, EOS, PAD, Tokenizer


@dataclass(frozen=True)
class Config:
    """Model shape.

    Attributes:
        text_vocab: Caption token ids (tokenizer.json).
        pixel_vocab: Pixel token ids (palette.json: transparent + 255 colours).
        caption: Caption slot length, <bos> and <eos> included.
        side: Image side in pixels.
        width: Size of every token's vector.
        layers: Transformer blocks.
        heads: Attention heads per block; width must be divisible by this number.
        dropout: Dropout on each block's MLP output; 0 while the data is large for
            the model. Attention has none: on Apple GPUs it stops the fused
            attention kernel and costs ~5.7 GB more memory at batch 16.
        recompute: In training, keep only each block's input, output and dropout
            mask, and recompute the rest in the backward pass (gradient
            checkpointing): about a third more compute for far less memory.
        cross_attention: Give every block an attention from its tokens to the
            caption's tokens (CrossAttention); EncoderPixl provides them.
        encoder_layers: Blocks of EncoderPixl's caption encoder; other models
            ignore it.
    """

    text_vocab: int = 1024
    pixel_vocab: int = 256
    caption: int = 32
    side: int = 32
    # Six heads of width 64 give an embedding width of 384.
    width: int = 384
    layers: int = 8
    heads: int = 6
    dropout: float = 0.0
    recompute: bool = False
    cross_attention: bool = False
    encoder_layers: int = 4

    @property
    def pixels(self) -> int:
        return self.side * self.side


class CrossAttention(nn.Module):
    """Attention from each token to every caption token, for the residual stream.

    The caption tokens (the memory) are not part of the sequence. The output
    layer starts at zero (Pixl.__init__), so each block starts as plain
    self-attention and MLP.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.heads: int = config.heads
        self.norm = nn.LayerNorm(config.width)
        self.query = nn.Linear(config.width, config.width)
        self.key_value = nn.Linear(config.width, 2 * config.width)
        self.project = nn.Linear(config.width, config.width)

    def forward(
        self, x: Tensor, memory: Tensor, cache: dict[str, Tensor] | None = None
    ) -> Tensor:
        """The attention result for each token of x.

        Args:
            x: (batch, length, width) vectors of the new tokens.
            memory: (batch, tokens, width) caption token vectors.
            cache: For generation: keeps the memory's keys and values, which
                every step reuses.

        Returns:
            (batch, length, width) update for the residual stream.
        """
        batch, length, width = x.shape
        if cache is not None and "memory_k" in cache:
            k, v = cache["memory_k"], cache["memory_v"]
        else:
            k, v = (
                t.view(batch, -1, self.heads, width // self.heads).transpose(1, 2)
                for t in self.key_value(memory).split(width, dim=2)
            )
            if cache is not None:
                cache["memory_k"], cache["memory_v"] = k, v
        q = self.query(self.norm(x))
        q = q.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.project(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    """One transformer block: causal self-attention, then an MLP.

    Causal self-attention adds this mask to the attention logits before softmax:
    0   -∞  -∞  -∞
    0    0  -∞  -∞
    0    0   0  -∞
    0    0   0   0

    Both parts normalise their input first and add their output to the residual
    stream ("pre-norm"), which keeps deep stacks stable to train. With
    Config.cross_attention, a CrossAttention to the caption runs between them.
    With causal=False every token attends to every other, as in EncoderPixl's
    caption encoder.
    """

    def __init__(self, config: Config, causal: bool = True) -> None:
        super().__init__()
        self.heads: int = config.heads
        self.causal: bool = causal
        self.recompute: bool = config.recompute
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.project = nn.Linear(config.width, config.width)
        self.mlp_norm = nn.LayerNorm(config.width)
        # Used by position, so keep this order: mlp[:-1] is the MLP (parts), mlp[-1] its
        # dropout (forward), and mlp[2] its output layer (Pixl.__init__).
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )
        self.cross = CrossAttention(config) if config.cross_attention else None

    def forward(
        self,
        x: Tensor,
        cache: dict[str, Tensor] | None = None,
        memory: Tensor | None = None,
    ) -> Tensor:
        """Runs the block on new tokens.

        Args:
            x: (batch, length, width) vectors of the new tokens.
            cache: For generation: keys and values of earlier tokens, extended
                in place with the new ones. None in training.
            memory: (batch, tokens, width) caption tokens for the cross-attention;
                None skips it.

        Returns:
            (batch, length, width) updated vectors.
        """
        if self.recompute and self.training and cache is None:
            # Keep dropout outside checkpoint replay: earlier Apple GPU runs
            # replayed its masks incorrectly and produced different gradients.
            # * checkpointing uses less memory while training but training takes longer than
            # the alternate without checkpojnt
            x, update = checkpoint(self.parts, x, None, memory, use_reentrant=False)
        else:
            x, update = self.parts(x, cache, memory)
        # Dropout is applied here: mlp[-1] is the Dropout, applied to the MLP update before
        # it is added to the residual stream.
        return x + self.mlp[-1](update)

    def parts(
        self,
        x: Tensor,
        cache: dict[str, Tensor] | None = None,
        memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """x after self/cross-attention updates, and the MLP's output before dropout."""
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).split(width, dim=2)
        # Divide attention among the heads
        q, k, v = (
            t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for t in (q, k, v)
        )
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # A single new token may see every cached one; a full sequence is causal.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal and q.shape[2] == k.shape[2]
        )
        # Rejoin the heads, project the attention output, and add the residual.
        x = x + self.project(attended.transpose(1, 2).reshape(batch, length, width))
        if self.cross is not None and memory is not None:
            x = x + self.cross(x, memory, cache)
        # mlp[:-1] is the MLP without its dropout, which forward applies.
        return x, self.mlp[:-1](self.mlp_norm(x))


class Pixl(nn.Module):
    """Caption-to-image transformer; see the module docstring for the layout."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        config = config or Config()
        self.config: Config = config
        # Separate caption-token and palette-token embeddings.
        self.text = nn.Embedding(config.text_vocab, config.width)
        self.pixel = nn.Embedding(config.pixel_vocab, config.width)

        # Caption slots and pixel rows/columns have separate learned positions.
        self.caption_position = nn.Embedding(config.caption, config.width)
        self.row = nn.Embedding(config.side, config.width)
        self.column = nn.Embedding(config.side, config.width)

        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))

        # Predict a palette-token distribution at each decoder position.
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.pixel_vocab, bias=False)
        self.apply(self._init)
        # GPT-2 scaling: each residual branch starts small, so the stack starts
        # close to the identity however many layers it has.
        for module in self.blocks:
            block = cast(Block, module)
            # mlp[2] is the MLP's output layer.
            for layer in (block.project, cast(nn.Linear, block.mlp[2])):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config.layers))
            if block.cross is not None:
                nn.init.zeros_(block.cross.project.weight)

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

    def caption_memory(self, captions: Tensor) -> Tensor | None:
        """Caption token vectors for the blocks' cross-attention; Pixl has none."""
        return None

    def run(
        self,
        x: Tensor,
        caches: list[dict[str, Tensor]] | None = None,
        memory: Tensor | None = None,
    ) -> Tensor:
        """Runs the blocks and returns pixel logits for every position.

        Args:
            x: (batch, length, width) token vectors.
            caches: For generation: one cache per block.
            memory: caption_memory's output, or None.
        """
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None, memory)
        return self.head(self.norm(x))

    def forward(self, captions: Tensor, pixels: Tensor) -> Tensor:
        """Predicts every pixel from the caption and the pixels before it.

        Args:
            captions: (batch, caption) caption ids.
            pixels: (batch, n) pixel tokens, n <= config.pixels; the last one is not
                needed as input, since nothing follows it.

        Returns:
            (batch, n, pixel_vocab) logits: entry i predicts pixel i.
        """
        text = self.embed_text(captions)
        x = torch.cat([text, self.embed_pixels(pixels[:, :-1])], 1)
        # The last text position (<eos>, or a start token) predicts pixel 0.
        return self.run(x, memory=self.caption_memory(captions))[:, text.shape[1] - 1 :]

    def loss(self, captions: Tensor, pixels: Tensor) -> Tensor:
        """(batch,) mean cross-entropy over each scene's pixel predictions."""
        logits = self(captions, pixels)
        return F.cross_entropy(logits.transpose(1, 2), pixels, reduction="none").mean(1)

    def empty_captions(self, batch: int) -> Tensor:
        """The empty caption slot for each of batch rows: the unconditioned input for guidance."""
        return encode_captions([""] * batch, None, length=self.config.caption)

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
        memory = self.caption_memory(captions)
        logits = self.run(self.embed_text(captions), caches, memory)[:, -1]
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
                x = self.embed_pixels(token, start=i)
                logits = self.run(x, caches, memory)[:, -1]
        return torch.cat(pixels, dim=1)


class EncoderPixl(Pixl):
    """Encoder-decoder Pixl, trained from scratch.

    The encoder reads the caption slot (Pixl's learned token and slot
    embeddings, padding included) with Config.encoder_layers bidirectional
    blocks, so every caption token sees the whole caption in order. The decoder
    is Pixl's pixel stack with Config.cross_attention: its sequence is a learned
    start token and the pixels, with no caption in front, and every block
    attends to the encoder's outputs:

        caption ids ──→ encoder ──→ memory
                                      │ cross-attention in every block
        <start> p0 ... p1022 ──→ decoder ──→ p0 p1 ... p1023
    """

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(replace(config or Config(), cross_attention=True))
        self.encoder = nn.ModuleList(
            Block(replace(self.config, cross_attention=False), causal=False)
            for _ in range(self.config.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(self.config.width)
        self.start = nn.Parameter(torch.zeros(self.config.width))
        self.encoder.apply(self._init)
        nn.init.normal_(self.start, std=0.02)

    def caption_memory(self, captions: Tensor) -> Tensor:
        """(batch, caption, width) encoder outputs for the decoder's cross-attention."""
        x = super().embed_text(captions)
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def embed_text(self, captions: Tensor) -> Tensor:
        """(batch, 1, width): the start token, which predicts pixel 0.

        The caption itself reaches the decoder only through caption_memory.
        """
        return self.start.expand(captions.shape[0], 1, -1)


def encode_captions(texts: list[str], tokenizer: Tokenizer | None, length: int) -> Tensor:
    """Turns captions into the left-padded caption slot.

    Args:
        texts: Captions; "" gives the empty caption.
        tokenizer: Caption tokenizer; None is allowed when every text is empty.
        length: Slot length, <bos> and <eos> included: the model's config.caption.

    Returns:
        (len(texts), length) ids: <pad>..., <bos>, caption, <eos>.

    Raises:
        ValueError: If a caption needs more than length slots, or a non-empty
            caption comes without a tokenizer.
    """
    rows: list[list[int]] = []
    for text in texts:
        if not text:
            ids = []
        elif tokenizer is None:
            raise ValueError("tokenizer is required for non-empty captions")
        else:
            ids = tokenizer.encode(text)
        if len(ids) + 2 > length:
            raise ValueError(f"caption needs {len(ids) + 2} slots, more than {length}: {text!r}")
        ids = [BOS, *ids, EOS]
        rows.append([PAD] * (length - len(ids)) + ids)
    return torch.tensor(rows)
