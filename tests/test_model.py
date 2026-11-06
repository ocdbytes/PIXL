import unittest
from dataclasses import replace

import torch

from pixl.model import Config, CrossAttention, EncoderPixl
from pixl.tokenizer import PAD

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


class CrossAttentionWeightsTest(unittest.TestCase):
    def test_weights_reproduce_forward(self) -> None:
        torch.manual_seed(0)
        cross = CrossAttention(SMALL)
        x, memory = torch.randn(2, 5, SMALL.width), torch.randn(2, SMALL.caption, SMALL.width)
        mask = torch.ones(2, 1, 1, SMALL.caption, dtype=torch.bool)
        mask[0, ..., :3] = False  # a left-padded caption
        weights = cross.weights(x, memory, mask)
        heads, size = SMALL.heads, SMALL.width // SMALL.heads
        v = cross.key_value(memory).split(SMALL.width, dim=2)[1].view(2, -1, heads, size).transpose(1, 2)
        attended = (weights @ v).transpose(1, 2).reshape(2, 5, SMALL.width)
        torch.testing.assert_close(cross.project(attended), cross(x, memory, None, mask))
        self.assertTrue((weights[0, ..., :3] == 0).all())


def weighted_cross_attention(model: EncoderPixl) -> EncoderPixl:
    """The model with weights in its cross-attention outputs, which start at zero, so captions reach the pixels."""
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if ".cross.project." in name:
                weight.normal_(std=0.5)
    return model


class EncoderPixlTest(unittest.TestCase):
    def test_pixel_by_pixel_matches_the_whole_image(self) -> None:
        for mask_padding in (False, True):
            with self.subTest(mask_padding=mask_padding):
                torch.manual_seed(0)
                model = weighted_cross_attention(EncoderPixl(replace(SMALL, mask_padding=mask_padding)).eval())
                captions = torch.randint(3, SMALL.text_vocab, (2, SMALL.caption))
                captions[0, :2] = torch.tensor([10, 20])
                captions[1, :3] = PAD  # left padding, as encode_captions writes it
                pixels = torch.randint(0, SMALL.pixel_vocab, (2, SMALL.pixels))
                with torch.no_grad():
                    whole = model(captions, pixels)
                    # As generate() runs: the start token, then one pixel per step from the cache.
                    caches = [{} for _ in model.blocks]
                    memory, mask = model.caption_memory(captions), model.caption_mask(captions)
                    steps = [model.run(model.embed_text(captions), caches, memory, mask)[:, -1]]
                    for i in range(SMALL.pixels - 1):
                        x = model.embed_pixels(pixels[:, i : i + 1], start=i)
                        steps.append(model.run(x, caches, memory, mask)[:, -1])
                torch.testing.assert_close(torch.stack(steps, 1), whole, atol=1e-5, rtol=1e-5)
                # Word order reaches the pixels through the encoder and cross-attention.
                swapped = captions.clone()
                swapped[0, [0, 1]] = captions[0, [1, 0]]
                self.assertFalse(torch.allclose(model(swapped, pixels)[0], whole[0]))
                # Guidance pairs each caption with an empty one of the model's caption length.
                self.assertEqual(model.generate(captions, guidance=3.0).shape, (2, SMALL.pixels))

    def test_masked_padding_does_not_reach_the_pixels(self) -> None:
        captions = torch.randint(3, SMALL.text_vocab, (1, SMALL.caption))
        captions[0, :3] = PAD
        pixels = torch.randint(0, SMALL.pixel_vocab, (1, SMALL.pixels))
        for mask_padding in (True, False):
            with self.subTest(mask_padding=mask_padding):
                torch.manual_seed(0)
                model = weighted_cross_attention(EncoderPixl(replace(SMALL, mask_padding=mask_padding)).eval())
                with torch.no_grad():
                    before = model(captions, pixels)
                    model.text.weight[PAD] += 1.0  # another <pad> embedding
                    after = model(captions, pixels)
                # Masked, <pad> changes nothing; unmasked, it changes the logits.
                self.assertEqual(torch.allclose(before, after), mask_padding)


if __name__ == "__main__":
    unittest.main()
