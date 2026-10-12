"""Builds the shared colour palette and turns images into palette tokens and back.

Every image the model sees is a grid of tokens: token 0 is transparent and tokens
1-255 are the palette's colours, so each pixel fits in one byte. The palette holds
every colour the sprites use, so sprites are encoded exactly, and fills the other
slots with colours picked from photo targets by median cut. It is saved to
src/palette.json; a trained model only works with the palette it was trained on,
so rebuild it only before training from scratch.

Build the palette from the Hugging Face dataset (downloads the photos once):

    uv run python -m src.preprocess
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import load_dataset
from PIL import Image

REPO = "ocdbytes/pixl-data"
PALETTE = Path("src/palette.json")
TRANSPARENT = 0
# Colour tokens; with TRANSPARENT the vocabulary has 256 tokens.
COLOURS = 255


@dataclass
class Args(argparse.Namespace):
    photos: int = 5000


def opaque_pixels(image: Image.Image) -> np.ndarray:
    """Returns the colours of an image's opaque pixels.

    Args:
        image: Image in any mode.

    Returns:
        An (n, 3) uint8 array with one RGB row per pixel whose alpha is >= 128.
    """
    rgba = np.asarray(image.convert("RGBA")).reshape(-1, 4)
    return rgba[rgba[:, 3] >= 128, :3]


def build_palette(
    sprites: Iterable[Image.Image], photos: Iterable[Image.Image], size: int = COLOURS
) -> np.ndarray:
    """Builds the palette: every sprite colour, then photo colours up to size.

    Args:
        sprites: Sprites whose colours are all kept.
        photos: Photo targets the remaining colours are picked from by median cut.
        size: Most colours in the palette.

    Returns:
        A (n, 3) uint8 array of distinct colours, n <= size, sorted. Token i is
        colour i - 1.

    Raises:
        ValueError: If the sprites alone use more than size colours.
    """
    kept = np.unique(np.concatenate([opaque_pixels(s) for s in sprites]), axis=0)
    if len(kept) > size:
        raise ValueError(f"sprites use {len(kept)} colours, more than {size}")
    pixels = np.concatenate([opaque_pixels(p) for p in photos])
    free = size - len(kept)
    strip = Image.fromarray(pixels.reshape(1, -1, 3))
    picked = strip.quantize(free, method=Image.Quantize.MEDIANCUT, kmeans=3)
    picked_palette = picked.getpalette()
    if picked_palette is None:
        raise ValueError("quantization produced no palette")
    extra = np.array(picked_palette[: 3 * free], np.uint8).reshape(-1, 3)
    return np.unique(np.concatenate([kept, extra]), axis=0)


def encode(image: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Turns an image into palette tokens, without dithering.

    Args:
        image: Image in any mode.
        palette: Colours from load_palette.

    Returns:
        A (height, width) uint8 array: TRANSPARENT where alpha < 128, otherwise
        1 + the index of the nearest palette colour by RGB distance.
    """
    rgba = np.asarray(image.convert("RGBA"))
    # Images use few distinct colours, so only those are compared with the palette.
    colours, inverse = np.unique(
        rgba[..., :3].reshape(-1, 3), axis=0, return_inverse=True
    )
    # TODO :
    # Try Oklab distance if the colors generated are very off palette
    # It matters for non sprite images only
    distance = ((colours[:, None].astype(np.int32) - palette[None]) ** 2).sum(axis=-1)
    nearest = (distance.argmin(axis=1) + 1).astype(np.uint8)
    tokens = nearest[inverse.reshape(-1)].reshape(rgba.shape[:2])
    tokens[rgba[..., 3] < 128] = TRANSPARENT
    return tokens


def decode(tokens: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Turns palette tokens back into an image.

    Args:
        tokens: (height, width) tokens from encode.
        palette: The palette they were encoded with.

    Returns:
        An RGBA image; TRANSPARENT pixels have alpha 0.
    """
    rgba = np.zeros((*tokens.shape, 4), np.uint8)
    opaque = tokens != TRANSPARENT
    rgba[opaque, :3] = palette[tokens[opaque] - 1]
    rgba[opaque, 3] = 255
    return Image.fromarray(rgba)


def save_palette(palette: np.ndarray, path: Path = PALETTE) -> None:
    data = {"transparent": TRANSPARENT, "colours": palette.tolist()}
    _ = path.write_text(json.dumps(data), encoding="utf-8")


def load_palette(path: Path = PALETTE) -> np.ndarray:
    """Reads the palette; token i (1..n) is colours[i - 1]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["colours"], np.uint8)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument(
        "--photos",
        type=int,
        default=Args.photos,
        help="photo targets sampled for the extra colours",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    sprite_data = cast(Any, load_dataset(REPO, "sprites"))
    sprites = [
        row["image"]
        for split in sprite_data.values()
        for row in split.select_columns(["image"])
    ]
    # Train split only, so test photos do not shape the palette.
    photos = cast(Any, load_dataset(REPO, "photos", split="train")).select_columns(
        ["target"]
    )
    sample = [
        row["target"] for row in photos.shuffle(seed=0).select(range(args.photos))
    ]
    palette = build_palette(sprites, sample)
    save_palette(palette)
    print(f"saved {len(palette)} colours to {PALETTE}")

    # Check: every sprite pixel comes back exactly; photos lose a little colour.
    exact = 0
    for sprite in sprites:
        original = np.asarray(sprite.convert("RGBA"))
        back = np.asarray(decode(encode(sprite, palette), palette))
        opaque = original[..., 3] >= 128
        exact += np.array_equal(opaque, back[..., 3] > 0) and np.array_equal(
            original[opaque, :3], back[opaque, :3]
        )
    errors = [
        np.abs(
            np.asarray(p.convert("RGB"), np.int32)
            - np.asarray(decode(encode(p, palette), palette).convert("RGB"), np.int32)
        ).mean()
        for p in sample
    ]
    print(f"sprites encoded exactly: {exact}/{len(sprites)}")
    print(f"photo targets: mean error {np.mean(errors):.2f} per channel (0-255)")


if __name__ == "__main__":
    main()
