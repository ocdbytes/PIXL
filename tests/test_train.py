import random
import unittest

from pixl.train import PLACES, swap_places

CAPTION = (
    "a red soldier at the bottom right, a gray army base at the top right, "
    "a gray recon vehicle at the bottom left and a hand cursor at the top left on grass"
)


class SwapPlacesTest(unittest.TestCase):
    def test_exchanges_exactly_two_places(self) -> None:
        rng = random.Random(0)
        for _ in range(20):
            swapped = swap_places(CAPTION, rng)
            before, after = PLACES.findall(CAPTION), PLACES.findall(swapped)
            self.assertEqual(sorted(before), sorted(after))
            self.assertEqual(sum(a != b for a, b in zip(before, after)), 2)
            self.assertEqual(PLACES.split(swapped), PLACES.split(CAPTION))


if __name__ == "__main__":
    unittest.main()
