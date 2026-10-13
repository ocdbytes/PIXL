# Model and inputs

Pixl represents captions and 32×32 images as token sequences, then uses a decoder-only Transformer to generate image tokens from text.

## Image palette

[`preprocess.py`](preprocess.py) builds [`palette.json`](palette.json). Token `0` is transparent and tokens `1–255` are colours, so each pixel fits in one byte.

| Colours | Source | Result |
|---|---|---|
| 210 | Every sprite colour of the first 5,685 sprites | Those sprites encode pixel-exactly; later packs, such as Dungeon Crawl, snap to the nearest colour |
| 45 | Median cut over 5,000 train-split photo targets | Mean photo error is 6.7 per RGB channel |

`encode()` maps opaque pixels to their nearest palette colour without dithering; `decode()` converts token grids back to RGBA. Rebuilding the palette makes it incompatible with existing models and caches, so do it only before training from scratch:

```bash
uv run python -m src.preprocess
```

## Caption tokenizer

[`tokenizer.py`](tokenizer.py) implements byte-level byte-pair encoding (BPE), trained on train-split captions ([Sennrich et al. 2016](https://arxiv.org/abs/1508.07909)). Words start as UTF-8 bytes with a leading space. Training repeatedly merges the most frequent adjacent pair; common words become one token while unseen words fall back to smaller pieces or bytes.

```text
"a green pine tree left of a wooden barrel on grass"
  → " a" " green" " pine" " tree" " left" " of" " a" " wooden" " barrel" " on" " grass"
"basketball on grass" → " ba" "sk" "et" "ball" " on" " grass"
```

| IDs | Tokens |
|---|---|
| 0, 1, 2 | `<pad>`, `<bos>`, `<eos>` |
| 3–258 | 256 byte values |
| 259–1023 | 765 learned merges |

Every train and test caption round-trips exactly. Captions average 2.5 tokens for photos, 4.5 for sprites, and 12 for scenes, with a maximum of 28.

| Vocabulary | Single-token words | Share of word uses | Longest scene caption |
|---|---:|---:|---:|
| 512 | 79 of 1,083 | 66.5% | 42 tokens |
| **1,024** | 309 of 1,083 | 86.3% | 28 tokens |
| 2,048 | 830 of 1,083 | 99.3% | 20 tokens |

The default 1,024-token vocabulary keeps captions short without dedicating tokens to rare whole words. Retrain it only before training the model from scratch:

```bash
uv run python -m src.tokenizer  # optionally: --vocab-size 2048
```

## Data loader

[`data.py`](data.py) loads [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data). Photo targets and scenes are already 32×32; sprites are centred on a transparent canvas and oversized sprites are skipped. Encoded tokens are cached by the `datasets` library.

```python
from src.data import loader

for batch in loader("photos", "train", batch_size=64):
    batch["caption"]  # 64 strings
    batch["tokens"]   # (64, 32, 32) int64 palette tokens
```

| Config | Train | Test |
|---|---:|---:|
| `photos` | 82,651 | 1,639 |
| `sprites` | 10,151 | 1,135 |
| `scenes` | 100,000 | 2,000: half unseen sprites, half unseen object pairs |

The first photo load downloads about 2.4 GB into the Hugging Face cache.

## Model

[`model.py`](model.py) is a decoder-only Transformer, a “GPT for pixels” like [Image GPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf). It reads a fixed 32-token caption slot and generates 1,024 palette tokens in row-major order:

```text
input:    <pad> ... <bos> a pine tree <eos> | p1 p2 ... p1023
predicts:                                  p1 | p2 p3 ... p1024

caption embedding ─┐
                   ├─ positions ─→ 8 × [causal attention + MLP] ─→ 256 pixel logits
pixel embedding ───┘
```

| Part | Design |
|---|---|
| Sequence | Left-padded caption, then pixels; `<eos>` predicts the first pixel |
| Embeddings | Separate 1,024-token text and 256-token pixel vocabularies |
| Positions | Learned caption slots plus row and column pixel embeddings |
| Blocks | 8 pre-norm blocks, width 384, 6 attention heads, MLP width 1,536 |
| Loss | Cross-entropy over the 1,024 pixel predictions |
| Guidance | 10% caption dropout enables classifier-free guidance ([Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)) |

The model has 14.8M parameters. On the measured Apple-silicon GPU, a batch of 16 takes about 0.8 seconds and one photo epoch about 70 minutes; float16 and batches of 32 were not faster. Generation uses a key-value cache and takes about four seconds for eight images, with guidance doubling the work.

## Stage 1 training

Run [`notebooks/pretrain.ipynb`](../notebooks/pretrain.ipynb). It shows a live progress panel, evaluation logs, loss curves, reference images, generated samples, and a final guidance comparison. Evaluation runs every 500 steps and at each epoch end.

Three epochs take about 3.5 hours on the measured Apple-silicon GPU. Evaluations save `checkpoints/pretrain.pt` for resuming and update `checkpoints/pretrain-best.pt` when test loss improves. Set `QUICK = True` in the notebook for a 60-step smoke test.

The finished run reached a test loss of 1.353. Samples have photo-like colours and textures but no clear object shapes, which matches the 32×32 photo targets. The caption is used, but weakly: given a test photo, its true query has the lowest loss among it and 19 random queries 26% of the time (chance 5%).

## Stage 2 training

Run [`notebooks/clip-finetune.ipynb`](../notebooks/clip-finetune.ipynb). It trains `ClipPixl`, which reads captions through the frozen CLIP text encoder (`ClipText`, `openai/clip-vit-base-patch32`), and starts from `checkpoints/pretrain-best.pt`, keeping every stage 1 weight except the caption input. An earlier stage 2 run with stage 1's learned caption embeddings (`checkpoints/finetune-stage1-best.pt`) drew pixel-art style and backgrounds but barely followed captions: on drawn sprite pixels, the true caption lowered the loss by only about 0.02 nats out of 0.37 compared with another sprite's caption.

Training runs in phases of fixed step counts, each with its own data mix and its own warm-up and cosine learning rate. The default run, `clip-objects-then-scenes`, trains 5,000 steps on sprites only (objects), then 3,000 steps on 80% scenes and 20% sprites (positions). Sprite captions are cut to their main noun ("shield") in 80% of phase 1 samples and 20% of phase 2 samples; sprites are mirrored and shifted by up to 2 pixels, and dropout is 0.1. The notebook's Experiment section documents each setting.

The 8,000 steps take about 2.3 hours. Evaluations save `checkpoints/clip-objects-then-scenes.pt` for resuming and update `clip-objects-then-scenes-best.pt` when the mean of the sprite and scene test losses improves; the end of phase 1 also saves `clip-objects-then-scenes-phase1.pt`. Set `init="scratch"` in the experiment to start from random weights instead. The final table compares stage 1, the learned-caption run and every `checkpoints/clip-*-best.pt` run on:

| Measure | Question it answers |
|---|---|
| Test loss per source | How well each kind of image is modelled; scenes are split into unseen sprites and unseen object pairs |
| Sprite label, photo query | Does the true caption explain a test image better than 19 random other captions? (chance 5%) |
| Position | Does a scene's caption beat the same caption with the positions reversed? (chance 50%) |

The learned-caption run draws pixel-art style and backgrounds but barely follows captions: on drawn sprite pixels, the true caption lowers the loss by only about 0.02 nats out of 0.37 compared with another sprite's caption. [`notebooks/clip-finetune.ipynb`](../notebooks/clip-finetune.ipynb) repeats stage 2 with `ClipPixl`, which reads captions through the frozen CLIP text encoder (`ClipText`, `openai/clip-vit-base-patch32`) and keeps every stage 1 weight except the caption input. It runs 3,000 steps, saves `checkpoints/clip-finetune.pt` and `clip-finetune-best.pt`, and its final table compares it with stage 1 and the learned-caption run.
