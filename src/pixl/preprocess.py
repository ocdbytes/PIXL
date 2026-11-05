"""Turns images into palette tokens and back, with the shared colour palette.

Every image the model sees is a grid of tokens: token 0 is transparent and tokens
1-255 are the palette's colours, so each pixel fits in one byte. The palette is
palette.json in the package, and it is fixed: trained models and cached tokens
depend on it. Sprite colours it lacks are mapped to their nearest entry.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

# Bundled with the package and read through its resources, so any working directory works.
PALETTE = resources.files("pixl").joinpath("palette.json")
TRANSPARENT = 0


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
    # TODO: Compare Oklab distance for sprite colours absent from the palette.
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


def load_palette(path: Traversable = PALETTE) -> np.ndarray:
    """Reads the palette; token i (1..n) is colours[i - 1]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["colours"], np.uint8)
