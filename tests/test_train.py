import random
import unittest
from itertools import pairwise
from typing import cast

import torch

from pixl.model import EncoderPixl
from pixl.train import PLACE_PHRASE, batch_loss, learning_rate, swap_places

CAPTION = (
    "a red soldier at the bottom right, a gray army base at the top right, "
    "a gray recon vehicle at the bottom left and a hand cursor at the top left on grass"
)
PIXELS = 16  # pixels per scene in the batch_loss tests


class SwapPlacesTest(unittest.TestCase):
    def test_exchanges_exactly_two_places(self) -> None:
        rng = random.Random(0)
        for _ in range(20):
            swapped = swap_places(CAPTION, rng)
            before, after = PLACE_PHRASE.findall(CAPTION), PLACE_PHRASE.findall(swapped)
            self.assertEqual(sorted(before), sorted(after))
            self.assertEqual(sum(a != b for a, b in zip(before, after)), 2)
            self.assertEqual(PLACE_PHRASE.split(swapped), PLACE_PHRASE.split(CAPTION))


class FixedLosses:
    """Stands in for the model: returns set per-pixel scene losses, the true captions' first.

    Attributes:
        rows: How many captions the last loss call scored.
    """

    def __init__(self, right: list[float], wrong: list[float]) -> None:
        self.losses = torch.tensor(right + wrong)
        self.rows = 0

    def loss(self, captions: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        self.rows = len(captions)
        return self.losses[: len(captions)]


def run(texts: list[str], right: list[float], wrong: list[float], margin: float) -> tuple[float, float, int]:
    """batch_loss on the stand-in model: the loss, the pixel loss, and the captions scored."""
    model = FixedLosses(right, wrong)
    pixels = torch.zeros(len(texts), PIXELS, dtype=torch.long)
    loss, pixel_loss = batch_loss(
        cast(EncoderPixl, model), lambda captions: torch.zeros(len(captions), 1), texts, pixels, margin,
        random.Random(0),
    )
    return loss.item(), pixel_loss.item(), model.rows


class BatchLossTest(unittest.TestCase):
    # Gaps are in nats per scene: (wrong - right) * PIXELS. The values are powers of two, so sums are exact.

    def test_margin_zero_is_exactly_the_pixel_loss(self) -> None:
        loss, pixel_loss, rows = run([CAPTION, CAPTION], [0.5, 0.25], [9.0, 9.0], margin=0)
        self.assertEqual(pixel_loss, 0.375)
        self.assertEqual(loss, pixel_loss)
        self.assertEqual(rows, 2)  # no swapped captions are scored

    def test_hinge_is_zero_once_the_gap_reaches_the_margin(self) -> None:
        for gap in (2.0, 4.0):  # at and beyond the margin of 2
            with self.subTest(gap=gap):
                step = gap / PIXELS
                loss, pixel_loss, _ = run([CAPTION, CAPTION], [0.5, 0.25], [0.5 + step, 0.25 + step], margin=2)
                self.assertEqual(loss, pixel_loss)

    def test_hinge_below_the_margin_adds_the_shortfall_per_scene(self) -> None:
        # Gap 1 against margin 2: each scene adds 1 nat, over 2 scenes of PIXELS pixels.
        loss, pixel_loss, rows = run([CAPTION, CAPTION], [0.5, 0.25], [0.5625, 0.3125], margin=2)
        self.assertEqual(loss, pixel_loss + 2 / (PIXELS * 2))
        self.assertEqual(rows, 4)

    def test_dropped_captions_add_nothing_to_the_hinge(self) -> None:
        # The middle caption is dropped: only the other two get a swapped copy, each 2 nats short of the margin,
        # and the hinge is still averaged over all 3 scenes. The dropped scene still counts in the pixel loss.
        loss, pixel_loss, rows = run([CAPTION, "", CAPTION], [0.5, 0.25, 0.75], [0.5, 0.75], margin=2)
        self.assertEqual(rows, 5)
        self.assertEqual(pixel_loss, 0.5)
        self.assertAlmostEqual(loss, 0.5 + 4 / (PIXELS * 3), places=6)


class LearningRateTest(unittest.TestCase):
    def test_warms_up_to_the_peak_then_decays_to_the_low_rate(self) -> None:
        steps, warmup, peak, low = 100, 10, 3e-4, 3e-5

        def rate(step: int) -> float:
            return learning_rate(step, steps, warmup, peak, low)

        self.assertAlmostEqual(rate(0), peak / warmup)
        self.assertAlmostEqual(rate(warmup - 1), peak)  # the last warm-up step
        self.assertAlmostEqual(rate(warmup), peak)  # the first decay step
        self.assertAlmostEqual(rate(steps), low)  # one past the last training step
        self.assertLess(rate(steps - 1) - low, 1e-7)  # the last training step ends a hair above it
        decay = [rate(step) for step in range(warmup, steps + 1)]
        self.assertTrue(all(a >= b for a, b in pairwise(decay)))


if __name__ == "__main__":
    unittest.main()
