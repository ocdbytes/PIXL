"""Loads pixl-data from the Hugging Face Hub as captions and palette token grids.

Each row becomes {"caption": str, "tokens": SIZE x SIZE int64 tensor} using the
palette in src/palette.json (see src/preprocess.py). Photo targets and scenes
are already 32x32. Sprites are centred on a transparent canvas; the 51 sprites larger than
the canvas (Sunny Land backgrounds and big frames) are skipped. Tokens are
computed once per split and cached by the datasets library, so later loads are
fast; a new palette makes them be computed again.

    from src.data import loader

    for batch in loader("photos", "train", batch_size=64):
        batch["caption"]  # 64 strings
        batch["tokens"]   # (64, 32, 32) int64, ready for nn.Embedding
"""

from __future__ import annotations

import numpy as np
from datasets import Array2D, Dataset, Features, Value, load_dataset
from torch.utils.data import DataLoader

from .preprocess import REPO, TRANSPARENT, encode, load_palette

SIZE = 32
# Config -> (caption column, image column).
COLUMNS = {
    "photos": ("query", "target"),
    "sprites": ("label", "image"),
    "scenes": ("caption", "image"),
}


def on_canvas(tokens: np.ndarray) -> np.ndarray:
    """Centres a token grid on a transparent SIZE x SIZE canvas.

    Args:
        tokens: (height, width) tokens, each side at most SIZE.

    Returns:
        A (SIZE, SIZE) uint8 array.
    """
    canvas = np.full((SIZE, SIZE), TRANSPARENT, np.uint8)
    height, width = tokens.shape
    top, left = (SIZE - height) // 2, (SIZE - width) // 2
    canvas[top : top + height, left : left + width] = tokens
    return canvas


def load(config: str, split: str) -> Dataset:
    """Loads one split as captions and token grids.

    Args:
        config: "photos", "sprites" or "scenes".
        split: "train" or "test".

    Returns:
        A dataset in torch format with "caption" and "tokens" columns.
    """
    palette = load_palette()
    caption, image = COLUMNS[config]

    def to_tokens(batch: dict[str, list]) -> dict[str, list]:
        fits = [
            (text, picture)
            for text, picture in zip(batch[caption], batch[image])
            if max(picture.size) <= SIZE
        ]
        return {
            "caption": [text for text, _ in fits],
            "tokens": [on_canvas(encode(picture, palette)) for _, picture in fits],
        }

    data = load_dataset(REPO, config, split=split).select_columns([caption, image])
    data = data.map(
        to_tokens,
        batched=True,
        remove_columns=[caption, image],
        features=Features(
            {"caption": Value("string"), "tokens": Array2D((SIZE, SIZE), "uint8")}
        ),
    )
    return data.with_format("torch")


def loader(config: str, split: str, batch_size: int = 64) -> DataLoader:
    """Batches one split; the train split is shuffled every epoch.

    Args:
        config: "photos", "sprites" or "scenes".
        split: "train" or "test".
        batch_size: Rows per batch.

    Returns:
        A DataLoader of {"caption": list[str], "tokens": (batch, SIZE, SIZE) int64}.
    """
    return DataLoader(
        load(config, split), batch_size=batch_size, shuffle=split == "train"
    )
