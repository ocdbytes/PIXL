import unittest

import numpy as np
from PIL import Image

from scripts.scenes.generate import whole


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


if __name__ == "__main__":
    _ = unittest.main()
