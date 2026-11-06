# Model

Pixl is an encoder-decoder Transformer for conditional generation of 32×32 palette-token images. [`EncoderPixl`](../src/pixl/model.py) encodes a caption once, then predicts the image autoregressively in row-major order.

```text
caption text ── BPE ──→ caption IDs ──→ bidirectional encoder ──→ caption memory
                                                                      │
image ── palette ──→ p0, p1, ... p1023 ──→ causal decoder ────────────┘
```

## Image representation

[`preprocess.py`](../src/pixl/preprocess.py) maps RGBA images to the fixed vocabulary in [`palette.json`](../src/pixl/palette.json):

- token `0` represents transparency;
- tokens `1–255` represent RGB colours;
- a 32×32 image becomes a sequence of 1,024 integer tokens.

Opaque colours absent from the palette are mapped to the nearest palette entry by squared RGB distance. No dithering is applied. `decode()` performs the inverse mapping from token grids to RGBA images.

The palette is part of the model format. Changing it invalidates trained weights and cached image tokens.

## Caption representation

[`tokenizer.py`](../src/pixl/tokenizer.py) implements byte-level byte-pair encoding (BPE). Text is lowercased, split into words and represented initially as UTF-8 bytes. Frequent adjacent tokens are merged during tokenizer training; unseen words remain representable through byte tokens.

| Token IDs | Meaning |
|---|---|
| `0`, `1`, `2` | `<pad>`, `<bos>`, `<eos>` |
| `3–258` | UTF-8 byte values |
| `259–1023` | Learned BPE merges |

The bundled vocabulary contains 1,024 tokens. Four-object captions in the current dataset require 30–80 tokens, excluding `<bos>` and `<eos>`, and training uses a 96-token caption slot. Captions are left-padded.

The tokenizer is coupled to the model weights and should only be rebuilt before training a new model:

```bash
uv run python -m pixl.tokenizer
# optional: --vocab-size 2048
```

## Architecture

The caption encoder applies bidirectional self-attention, allowing every non-padding caption token to use the full caption. The decoder uses causal self-attention and cross-attends to the encoder output in every block.

```text
caption IDs ──→ token + position embeddings ──→ 4 encoder blocks ──→ memory
                                                                        │
<start>, p0, ... p1022 ──→ 8 causal decoder blocks + cross-attention ──→ logits
                                                                        │
                                                               p0 ... p1023
```

| Component | Configuration |
|---|---|
| Caption encoder | 4 pre-norm bidirectional Transformer blocks |
| Pixel decoder | 8 pre-norm causal Transformer blocks |
| Hidden width | 384 |
| Attention | 6 heads of width 64 |
| MLP width | 1,536 |
| Image positions | Separate learned row and column embeddings |
| Caption positions | Learned slot embeddings |
| Pixel vocabulary | 256 tokens |
| Caption vocabulary | 1,024 tokens |
| Training caption length | 96 tokens |
| Parameters | 26.7 million |

A learned start token predicts the first pixel; each subsequent position receives the preceding pixel token. Cross-attention output projections start at zero, so the initial decoder behaves like an unconditional pixel decoder and learns caption conditioning during training.

`Config.mask_padding` excludes `<pad>` slots from encoder self-attention and decoder cross-attention. Current training enables it. The published model predates this option and was trained while attending to padding, so loading it retains `mask_padding=False`.

`Config.recompute` enables gradient checkpointing. It stores fewer intermediate activations and recomputes them during backpropagation, reducing memory use at the cost of additional compute.

## Training objective

For an image with pixel tokens $p_0, \ldots, p_{1023}$, the decoder models

$$
P(\text{image} \mid \text{caption})
= \prod_{i=0}^{1023} P(p_i \mid \text{caption}, p_{<i}).
$$

`model.loss()` applies cross-entropy to all 1,024 predictions and returns the mean loss per pixel for each scene. [`pixl.train`](training.md) adds caption dropout and the position-margin objective around this model loss.

## Generation

Generation encodes the caption once and samples pixels sequentially. A key-value cache avoids recomputing previous decoder states at every pixel. The default sampling settings are:

| Setting | Value |
|---|---:|
| Temperature | 0.7 |
| Top-k | 32 |
| Classifier-free guidance | 3.0 |

Classifier-free guidance compares conditional and empty-caption logits. Training leaves 10% of captions empty so the same model can provide both predictions.

## Tests

```bash
uv run python -m unittest tests.test_model
```

The model tests cover gradient checkpointing, cached generation, caption conditioning and padding masks.
