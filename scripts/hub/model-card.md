---
pipeline_tag: text-to-image
datasets:
  - ocdbytes/pixl-data
tags:
  - pixel-art
  - pytorch
license: cc0-1.0
---

![Pixl: 16 scenes drawn by the model](https://huggingface.co/ocdbytes/pixl/resolve/main/poster.png)

*16 scenes drawn by pixl from their captions, in one batch: seed 0, temperature 0.7, top-k 32, guidance 3.*

# pixl

> 🙏 **Thanks to [@harshitruwali](https://huggingface.co/harshitruwali) for the GPU compute used to train pixl.**

pixl draws a 32×32 pixel-art scene from a caption, one pixel at a time. It is a 26.7M-parameter encoder-decoder Transformer trained from scratch on one million generated scenes. Each scene has four sprites, one in each quadrant, and a caption that says which sprite goes where:

> a purple robed wizard at the top left, a closed treasure chest at the top right, a brown bat at the bottom left and a red potion at the bottom right on a dark dungeon floor

Code, training and the Docker image are at [github.com/ocdbytes/PIXL](https://github.com/ocdbytes/PIXL).

## Files

| File | Contents |
|---|---|
| `model.safetensors` | The weights in float32 (106.8 MB) |
| `model-bf16.safetensors` | The same weights in bf16 (53.4 MB) |
| `config.json` | The model's shape, for `pixl.model.Config` |
| `tokenizer.json`, `palette.json` | The caption tokenizer and colour palette the model was trained with; the `pixl` package bundles the same files |

## Model

| Part | Details |
|---|---|
| Caption tokens | Byte-level BPE with 1,024 tokens (3 special, 256 bytes, 765 merges), left-padded into a 96-token slot between `<bos>` and `<eos>` |
| Image tokens | 1,024 palette tokens per image in row-major order: 0 is transparent, 1–255 are the colours of a fixed palette |
| Encoder | 4 bidirectional pre-norm Transformer blocks over the caption slot, padding included, with learned token and slot embeddings |
| Decoder | 8 causal pre-norm blocks over a learned start token and the pixels. Each has self-attention, cross-attention to the encoder's output, and a GELU MLP. Pixels have learned token, row and column embeddings |
| Width | 384, with 6 heads of 64 and an MLP of 1,536 |
| Output | A distribution over the 256 palette tokens for each pixel |
| Parameters | 26.7M |
| Generation | Key-value cache, temperature, top-k, and classifier-free guidance ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)), which works because 10 % of training captions were left empty |

## Training data

The `scenes` config of [ocdbytes/pixl-data](https://huggingface.co/datasets/ocdbytes/pixl-data), CC0 1.0:

- **Train:** 1,000,000 scenes.
  - Each scene uses one sprite style: a background tile repeated in every 16 px slot, and four coloured objects with different labels, one centred in each quadrant.
  - The caption names each object's quadrant, in random order, then the ground.
- **Sprites:** 2,240 object labels and 40 background tiles from 10 sprite styles:
  - Kenney's Tiny Town, Tiny Dungeon, Tiny Farm, Tiny Battle, Pixel Shmup, Pixel Vehicle Pack and 1-Bit Pack (coloured).
  - ansimuz's Sunny Land and GrafxKid's Mini Pixel Pack 3.
  - The Dungeon Crawl Stone Soup 32×32 tiles, halved to 16 px.
- **Test:** 2,000 scenes, never trained on:
  - 1,000 *new sprites* scenes built from held-out sprite images.
  - 1,000 *new pair* scenes built from train sprites, each with at least one pair of labels that never appears together in a train scene. 5 % of label pairs are held out this way.

## Training

```bash
python -m pixl.train --run scenes-four-margin-2ep --epochs 2
```

| Setting | Value |
|---|---|
| Steps | 62,500 batches of 32 scenes: two passes over the train scenes |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices and embeddings |
| Learning rate | Linear warm-up to 3e-4 over 1,000 steps, then cosine decay to 3e-5 |
| Gradient clipping | 1.0 |
| Precision | bf16 autocast, with TF32 matrix multiplies |
| Caption dropout | 10 %, for guidance |
| Loss | Pixel cross-entropy, plus a hinge (`--margin 2`): each caption's loss must be at least 2 nats per scene below the loss of the same caption with two random objects' quadrants exchanged |
| Hardware | One 24 GB NVIDIA GPU, about 5 hours at 0.28–0.29 s per step |
| Seed | 0 |

The margin is what taught positions. Without it, the model learned which objects to draw from the caption but ignored where they go.

## Evaluation

These scores are from after the last step, over every evaluation scene:

- **Loss:** mean cross-entropy per pixel, in nats.
- **Position:** the share of scenes whose loss is lower under their caption than under the same caption with its first two objects' quadrants exchanged. Chance is 0.5.

| Scenes | Loss | Position |
|---|---:|---:|
| Train (500, seen in training) | 0.006 | 1.00 |
| New pair (1,000) | 0.005 | 1.00 |
| New sprites (1,000) | 1.706 | 0.86 |

The published files give the same results. They were scored again on the CPU in float32, on every test scene; the bf16 copy is loaded into float32 parameters:

| File | New pair loss | New pair position | New sprites loss | New sprites position |
|---|---:|---:|---:|---:|
| `model.safetensors` | 0.0053 | 1.000 | 1.7060 | 0.858 |
| `model-bf16.safetensors` | 0.0053 | 1.000 | 1.7054 | 0.859 |

New-pair position for the same data and settings:

| Run | New-pair position |
|---|---:|
| Pixel loss only, 1 pass | 0.50 |
| With the margin, 1 pass | 0.80 |
| With the margin, 2 passes | 1.00 |

## Limitations

- **Vocabulary:** the model knows only the object labels and ground names it was trained on. Other words still tokenize, into pieces, but mean nothing to it.
- **Layout:** always four objects in a 2×2 grid, in the training caption template. One or two objects, other positions and free-form text are outside what it learned.
- **Styles:** each training scene takes all its objects from one sprite style. Captions that mix styles are untested.
- **New shapes:** it draws sprites it has seen. On held-out sprite images the loss is 1.7 nats per pixel, against 0.005 for known sprites. Those held-out sprites can share labels with train sprites.
- **What was measured:** position compares captions against real scenes; it does not check where generated sprites land. Generated samples have only been judged by eye.
- **Size:** 32×32 pixels and a fixed palette of 255 colours.

## Use

Install the `pixl` package, which brings the model code, tokenizer and palette:

```bash
pip install git+https://github.com/ocdbytes/PIXL
```

`load_pretrained()` downloads the weights from this repo and checks that its tokenizer and palette match the package's. The training preview uses temperature 0.7, top-k 32 and guidance 3; `torch.manual_seed()` makes samples repeatable.

```python
import torch
from PIL import Image

from pixl.model import encode_captions
from pixl.preprocess import decode, load_palette
from pixl.tokenizer import Tokenizer
from pixl.train import load_pretrained

model = load_pretrained()  # load_pretrained(bf16=True) downloads the half-size copy
torch.manual_seed(0)
caption = ("a purple robed wizard at the top left, a closed treasure chest at the top right, "
           "a brown bat at the bottom left and a red potion at the bottom right on a dark dungeon floor")
ids = encode_captions([caption], Tokenizer.load(), length=model.config.caption)
pixels = model.generate(ids, temperature=0.7, top_k=32, guidance=3.0)
image = decode(pixels.view(32, 32).numpy(), load_palette())
image.resize((256, 256), Image.Resampling.NEAREST).save("scene.png")
```

## Licence

- **Code:** BSD 3-Clause.
- **Training data:** CC0 1.0.
- **Weights:** CC0 1.0.
