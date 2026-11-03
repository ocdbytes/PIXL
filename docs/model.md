# Model and inputs

Pixl represents captions and 32×32 images as token sequences, then uses an encoder-decoder Transformer to generate image tokens from text. The code is the `pixl` package in [`src/pixl/`](../src/pixl); [development.md](development.md) covers how it is installed.

## Image palette

[`preprocess.py`](../src/pixl/preprocess.py) builds [`palette.json`](../src/pixl/palette.json). Token `0` is transparent and tokens `1–255` are colours, so each pixel fits in one byte.

| Colours | Source | Result |
|---|---|---|
| 210 | Every sprite colour of the first 5,685 sprites | Those sprites encode pixel-exactly; later packs, such as Dungeon Crawl, snap to the nearest colour |
| 45 | Median cut over 5,000 train-split photo targets | Mean photo error is 6.7 per RGB channel |

`encode()` maps opaque pixels to their nearest palette colour without dithering; `decode()` converts token grids back to RGBA. The bundled palette was built before the later sprite packs were added. The builder tries to preserve every source sprite colour and raises `ValueError` if those colours exceed its 255 opaque-colour slots. Rebuilding the palette makes it incompatible with existing models and caches, so do it only before training from scratch:

```bash
uv run python -m pixl.preprocess
```

It reads the sprites and the `photos` config from the Hugging Face dataset.

## Caption tokenizer

[`tokenizer.py`](../src/pixl/tokenizer.py) implements byte-level byte-pair encoding (BPE), trained on train-split captions ([Sennrich et al. 2016](https://arxiv.org/abs/1508.07909)). Words start as UTF-8 bytes with a leading space. Training repeatedly merges the most frequent adjacent pair; common words become one token while unseen words fall back to smaller pieces or bytes.

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

Decoding an encoded caption returns its lowercase text with whitespace normalised. The following measurements come from the earlier photo, sprite and one/two-object scene corpus: captions averaged 2.5 tokens for photos, 4.5 for sprites, and 12 for scenes, with a maximum of 28. They do not describe the current four-object scenes.

| Vocabulary | Single-token words | Share of word uses | Longest historical scene caption |
|---|---:|---:|---:|
| 512 | 79 of 1,083 | 66.5% | 42 tokens |
| **1,024** | 309 of 1,083 | 86.3% | 28 tokens |
| 2,048 | 830 of 1,083 | 99.3% | 20 tokens |

The bundled vocabulary has 1,024 tokens. In the current generated four-object snapshot, captions range from 30 to 80 tokens (median 41), excluding `<bos>` and `<eos>`. `pixl.train` uses a 96-token caption slot including those markers. Retrain the tokenizer only before training the model from scratch:

```bash
uv run python -m pixl.tokenizer  # optionally: --vocab-size 2048
```

## Model

[`model.py`](../src/pixl/model.py) holds `EncoderPixl`, an encoder-decoder trained from random weights. A caption encoder of `Config.encoder_layers` blocks (4) reads the left-padded caption slot through learned token and slot embeddings with bidirectional attention, so each caption token sees the whole caption in order. The pixel decoder is a "GPT for pixels" like [Image GPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf): 8 causal blocks that generate the 1,024 palette tokens in row-major order, each with a cross-attention to the encoder's outputs. A learned start token predicts the first pixel:

```text
caption ids ──→ encoder ──→ memory
                              │ cross-attention in every block
<start> p0 ... p1022 ──→ decoder ──→ p0 p1 ... p1023
```

| Part | Design |
|---|---|
| Encoder | 4 pre-norm bidirectional blocks over the caption slot (`Config.caption` tokens, padding included) |
| Decoder | 8 pre-norm causal blocks, each self-attention, cross-attention to the encoder, then MLP |
| Width | 384, 6 attention heads, MLP width 1,536 |
| Positions | Learned caption slots; row and column pixel embeddings |
| Loss | Cross-entropy over the 1,024 pixel predictions |
| Guidance | 10% caption dropout enables classifier-free guidance ([Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)) |

Each cross-attention's output layer starts at zero, so every decoder block starts as plain self-attention and MLP. With a 96-token caption slot the model has 26.7M parameters. Generation uses a key-value cache: the caption's keys and values are computed once, then each step runs the decoder on the newest pixel only. `Config.recompute` recomputes each block in the backward pass (gradient checkpointing) for far less memory at about a third more compute; batch 32 needs it on a 24 GB Mac.

`Pixl`, the base class, is the earlier decoder-only version, which read the caption as a prefix of the pixel sequence; `checkpoints/pretrain*.pt` and `finetune-stage1*.pt` hold its weights.

## Tests

[`tests/test_model.py`](../tests/test_model.py) checks that recomputing gives the same gradients, that cached generation matches the whole image at once, and that swapping two caption tokens changes the predictions:

```bash
uv run python -m unittest tests.test_model
```
