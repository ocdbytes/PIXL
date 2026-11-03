import unittest
from dataclasses import replace

import torch

from pixl.model import Config, EncoderPixl

# Check the dropout-gradient regression on Apple's GPU when it is available.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# A small EncoderPixl shape.
SMALL = Config(width=32, layers=2, heads=2, side=4, caption=8, encoder_layers=2)


class RecomputeTest(unittest.TestCase):
    def test_recomputing_blocks_gives_the_same_gradients(self) -> None:
        gradients = []
        for recompute in (False, True):
            torch.manual_seed(0)
            model = EncoderPixl(replace(SMALL, dropout=0.5, recompute=recompute)).to(DEVICE)
            captions = torch.randint(3, SMALL.text_vocab, (3, SMALL.caption), device=DEVICE)
            pixels = torch.randint(0, SMALL.pixel_vocab, (3, SMALL.pixels), device=DEVICE)
            model.loss(captions, pixels).mean().backward()
            gradients.append([p.grad.cpu() for p in model.parameters() if p.grad is not None])
        for plain, recomputed in zip(*gradients):
            torch.testing.assert_close(plain, recomputed)


class EncoderPixlTest(unittest.TestCase):
    def test_pixel_by_pixel_matches_the_whole_image(self) -> None:
        torch.manual_seed(0)
        model = EncoderPixl(SMALL).eval()
        with torch.no_grad():
            # The cross-attention output starts at zero; give it weights, so it is tested too.
            for name, weight in model.named_parameters():
                if ".cross.project." in name:
                    weight.normal_(std=0.5)
        captions = torch.randint(3, SMALL.text_vocab, (2, SMALL.caption))
        captions[0, :2] = torch.tensor([10, 20])
        pixels = torch.randint(0, SMALL.pixel_vocab, (2, SMALL.pixels))
        with torch.no_grad():
            whole = model(captions, pixels)
            # As generate() runs: the start token, then one pixel per step from the cache.
            caches = [{} for _ in model.blocks]
            memory = model.caption_memory(captions)
            steps = [model.run(model.embed_text(captions), caches, memory)[:, -1]]
            for i in range(SMALL.pixels - 1):
                steps.append(model.run(model.embed_pixels(pixels[:, i : i + 1], start=i), caches, memory)[:, -1])
        torch.testing.assert_close(torch.stack(steps, 1), whole, atol=1e-5, rtol=1e-5)
        # Word order reaches the pixels through the encoder and cross-attention.
        swapped = captions.clone()
        swapped[0, [0, 1]] = captions[0, [1, 0]]
        self.assertFalse(torch.allclose(model(swapped, pixels)[0], whole[0]))
        # Guidance pairs each caption with an empty one of the model's caption length.
        self.assertEqual(model.generate(captions, guidance=3.0).shape, (2, SMALL.pixels))


if __name__ == "__main__":
    unittest.main()
