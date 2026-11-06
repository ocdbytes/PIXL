import math
import random
import unittest
from itertools import pairwise
from typing import cast

import torch

from pixl.model import Config, EncoderPixl, encode_captions
from pixl.tokenizer import Tokenizer
from pixl.train import (
    PLACE_PHRASE,
    batch_loss,
    learning_rate,
    place_tokens,
    probes,
    swap_places,
)

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


def run(texts: list[str], right: list[float], wrong: list[float], margin: float) -> tuple[float, float, int, float]:
    """batch_loss on the stand-in model: the loss, the pixel loss, the captions scored, and the hinge's active share."""
    model = FixedLosses(right, wrong)
    pixels = torch.zeros(len(texts), PIXELS, dtype=torch.long)
    pixel_loss, margin_loss, active = batch_loss(
        cast(EncoderPixl, model), lambda captions: torch.zeros(len(captions), 1), texts, pixels, margin,
        random.Random(0),
    )
    return (pixel_loss + margin_loss).item(), pixel_loss.item(), model.rows, active.item()


class BatchLossTest(unittest.TestCase):
    # Gaps are in nats per scene: (wrong - right) * PIXELS. The values are powers of two, so sums are exact.

    def test_margin_zero_is_exactly_the_pixel_loss(self) -> None:
        loss, pixel_loss, rows, active = run([CAPTION, CAPTION], [0.5, 0.25], [9.0, 9.0], margin=0)
        self.assertEqual(pixel_loss, 0.375)
        self.assertEqual(loss, pixel_loss)
        self.assertEqual(rows, 2)  # no swapped captions are scored
        self.assertTrue(math.isnan(active))

    def test_hinge_is_zero_once_the_gap_reaches_the_margin(self) -> None:
        for gap in (2.0, 4.0):  # at and beyond the margin of 2
            with self.subTest(gap=gap):
                step = gap / PIXELS
                loss, pixel_loss, _, active = run([CAPTION, CAPTION], [0.5, 0.25], [0.5 + step, 0.25 + step], margin=2)
                self.assertEqual(loss, pixel_loss)
                self.assertEqual(active, 0.0)

    def test_hinge_below_the_margin_adds_the_shortfall_per_scene(self) -> None:
        # Gap 1 against margin 2: each scene adds 1 nat, over 2 scenes of PIXELS pixels.
        loss, pixel_loss, rows, active = run([CAPTION, CAPTION], [0.5, 0.25], [0.5625, 0.3125], margin=2)
        self.assertEqual(loss, pixel_loss + 2 / (PIXELS * 2))
        self.assertEqual(rows, 4)
        self.assertEqual(active, 1.0)

    def test_dropped_captions_add_nothing_to_the_hinge(self) -> None:
        # The middle caption is dropped: only the other two get a swapped copy, each 2 nats short of the margin,
        # and the hinge is still averaged over all 3 scenes. The dropped scene still counts in the pixel loss.
        loss, pixel_loss, rows, _ = run([CAPTION, "", CAPTION], [0.5, 0.25, 0.75], [0.5, 0.75], margin=2)
        self.assertEqual(rows, 5)
        self.assertEqual(pixel_loss, 0.5)
        self.assertAlmostEqual(loss, 0.5 + 4 / (PIXELS * 3), places=6)


class PlaceTokensTest(unittest.TestCase):
    def test_marks_only_the_place_words(self) -> None:
        tokenizer = Tokenizer.load()
        marked = torch.isin(torch.tensor(tokenizer.encode(CAPTION)), place_tokens(tokenizer))
        self.assertEqual(marked.sum().item(), 8)  # two words in each of the four place phrases


class ProbesTest(unittest.TestCase):
    def test_measures_without_changing_training(self) -> None:
        torch.manual_seed(0)
        config = Config(width=32, layers=2, heads=2, side=4, caption=48, encoder_layers=1, mask_padding=True)
        model = EncoderPixl(config).train()
        tokenizer = Tokenizer.load()
        pixels = torch.randint(0, config.pixel_vocab, (2, config.pixels))
        before = torch.get_rng_state()
        result = probes(
            model, lambda texts: encode_captions(texts, tokenizer, length=config.caption), [CAPTION, CAPTION], pixels,
            2.0, place_tokens(tokenizer),
        )
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(sorted(result), ["margin grad norm", "pixel grad norm", "place attention 0", "place attention 1"])
        self.assertTrue(all(0 < result[f"place attention {i}"] < 1 for i in range(config.layers)))


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
