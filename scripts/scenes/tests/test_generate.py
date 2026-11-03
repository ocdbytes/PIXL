import random
import re
import unittest

import numpy as np
from PIL import Image

from scripts.scenes.generate import Sprite, make_scenes, whole


def sprite(opaque: list[str]) -> Image.Image:
    """An RGBA sprite whose "#" cells are opaque."""
    alpha = np.array(
        [[255 if cell == "#" else 0 for cell in row] for row in opaque], np.uint8
    )
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


class WholeTest(unittest.TestCase):
    def test_one_piece_and_diagonal_neighbours_count_as_whole(self) -> None:
        self.assertTrue(whole(sprite(["##..", "##..", "..#.", "...#"])))

    def test_two_equal_pieces_are_not_whole(self) -> None:
        self.assertFalse(whole(sprite(["##..", "##..", "....", "..##", "..##"])))


class SwappedTest(unittest.TestCase):
    def test_swapped_caption_reverses_the_positions(self) -> None:
        objects = [
            Sprite(f"{label}.png", label, "pack", "train", sprite(["#"]))
            for label in ("tree", "barrel", "house")
        ]
        kinds = set()
        mirrored = {"left": "right", "right": "left"}
        for row, _ in make_scenes(16, "train", objects, [], random.Random(0)):
            caption, swapped = row["caption"], row["swapped"]
            if pair := re.fullmatch(r"(an? \w+) (left of|above) (an? \w+)", caption):
                self.assertEqual(swapped, f"{pair[3]} {pair[2]} {pair[1]}")
            else:
                expected = re.sub(r"left|right", lambda m: mirrored[m[0]], caption)
                self.assertEqual(swapped, expected)
            kinds.add(bool(pair))
        self.assertEqual(kinds, {True, False})

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
