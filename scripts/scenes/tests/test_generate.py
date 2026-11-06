import random
import re
import unittest

import numpy as np
from PIL import Image

from scripts.scenes.generate import Sprite, make_scenes


def sprite(opaque: list[str]) -> Image.Image:
    """An RGBA sprite whose "#" cells are opaque."""
    alpha = np.array(
        [[255 if cell == "#" else 0 for cell in row] for row in opaque], np.uint8
    )
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


class SceneTest(unittest.TestCase):
    def test_four_objects_fill_the_quadrants_and_swapped_exchanges_two(self) -> None:
        # No pair of these labels is held out, so train scenes can use all four.
        objects = [
            Sprite(f"{label}.png", label, "pack", "train", sprite(["#"]))
            for label in ("tree", "barrel", "sword", "well")
        ]
        for row, _ in make_scenes(8, "train", objects, [], random.Random(0), four=True):
            phrases = [re.split(r", | and ", text) for text in (row["caption"], row["swapped"])]
            (labels, places), (swapped_labels, swapped_places) = (
                zip(*(phrase.split(" at the ") for phrase in group)) for group in phrases
            )
            self.assertEqual(sorted(places), ["bottom left", "bottom right", "top left", "top right"])
            self.assertEqual(swapped_labels, labels)
            self.assertEqual(swapped_places, (places[1], places[0], *places[2:]))


if __name__ == "__main__":
    _ = unittest.main()
