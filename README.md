# pixl

Pixl is a pixel-art generation model. It will learn to generate small, palette-controlled pixel art from text prompts.

The training data and its image tokens (the shared palette) are ready; the caption tokenizer, model and training code come next.

## Setup

```bash
uv sync
```

### Train

Nothing else is needed. [`src/data.py`](src/data.py) loads the public Hugging Face dataset [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data) on first use (about 2.4 GB into `~/.cache/huggingface`), and the palette is in the repo; see [Model input](#model-input).

### Change the dataset

Only needed to change the data itself: new queries, new sprite packs or new labels.

1. **Get the sprites.** The sprite packs are downloaded by hand into `downloaded-data/` (see [`sources.md`](sources.md)). To start from the current data instead, run `uv run python -m scripts.hub.sync pull`, which writes the Hub copy into `pixel-image-data/` and `downloaded-data/`.
2. **Rebuild:**

   ```bash
   uv run python -m src.setup_data --user-agent "PixlDataset/0.1 (your-email@example.com)"
   ```

   | Step | Script | What it does | Time |
   |---|---|---|---|
   | 1. Download | [`scripts/data`](scripts/data/README.md) | Photo pairs from Open Images, Wikimedia and Openverse for the queries in [`src/queries.json`](src/queries.json) | 1–2 days the first time; resumes if stopped |
   | 2. Label | [`scripts/labeler`](scripts/labeler/README.md) | Labels sprites that have no label yet (existing labels are kept) and gathers all sprites into `downloaded-data/labeled/` | Seconds when nothing is unlabelled |
   | 3. Refine | [`scripts/refine`](scripts/refine/README.md) | Drops photo pairs with nudity or graphic content, web photos that do not match their query, and CC BY photos that cannot be credited; writes `pixel-image-data/clean.csv` and review sheets | ~12 minutes the first time, then seconds |
   | 4. Scenes | [`scripts/scenes`](scripts/scenes/README.md) | Builds 52,000 captioned 32×32 scenes from the sprites (e.g. "a pine tree left of a barrel on grass") into `scene-data/`, with review sheets | ~15 seconds |

   The script stops at once if `downloaded-data/manifest.csv` is missing, and stops if a step fails. Add `--label-model florence` to label new sprites with Florence-2 instead of CLIP. Every step can also be run on its own; see its README.
3. **Review** the sheets in `pixel-image-data/review/` (photos dropped for nudity or graphic content) and `scene-data/review/` (sprite roles and sample scenes). Change the cut-offs or rules if needed.
4. **Upload** with `uv run python -m scripts.hub.sync push`. Every push is public at once.

## Model input

The model reads images as grids of palette tokens, loaded straight from the Hugging Face dataset; local data is only needed to build or change the dataset.

**Palette** (`src/preprocess.py`, saved in `src/palette.json`): token 0 is transparent and tokens 1–255 are colours, so the vocabulary is 256 tokens and each pixel fits in one byte.

| Colours | From | Effect |
|---|---|---|
| 210 | Every colour the sprites use | Sprites are encoded pixel-exact (5,685/5,685) |
| 45 | Median cut over 5,000 photo targets from the train split | Photo targets change by 6.7 per colour channel (0–255) on average |

`encode(image, palette)` maps each pixel to the nearest colour (no dithering) and pixels with alpha below 128 to token 0; `decode(tokens, palette)` turns tokens back into an RGBA image. A trained model only works with the palette it was trained on, so rebuild the palette only before training from scratch:

```bash
uv run python -m src.preprocess
```

**Loader** (`src/data.py`): each row becomes a caption and a 32×32 token grid. Photo targets are already 32×32; sprites are centred on a transparent 32×32 canvas, and the 51 sprites larger than that are skipped. Tokens are computed once per split and cached by the `datasets` library.

```python
from src.data import loader

for batch in loader("photos", "train", batch_size=64):  # or "sprites", "scenes"; "test"
    batch["caption"]  # 64 strings, e.g. "man riding horse"
    batch["tokens"]   # (64, 32, 32) int64 palette tokens (stored as uint8 in the cache)
```

| Config | train | test |
|---|---|---|
| `photos` | 82,651 | 1,639 |
| `sprites` | 5,104 | 530 |
| `scenes` | 50,000 | 2,000 (1,000 with unseen sprites, 1,000 with unseen object pairs) |

The first load of the photos downloads about 2.4 GB into `~/.cache/huggingface`.

## Project structure

```text
pixl/
├── src/
│   ├── setup_data.py  # Set up all training data: get, label, refine
│   ├── queries.json   # Queries for the photo download
│   ├── preprocess.py  # Build the shared palette; encode images as tokens
│   ├── palette.json   # The shared palette (256 tokens)
│   └── data.py        # Load the Hub dataset as captions + token grids
├── scripts/
│   ├── data/          # Download and build aligned photo/pixel-art pairs
│   ├── hub/           # Push and pull the data to the Hugging Face dataset repo
│   ├── labeler/       # Label and collect downloaded sprite assets
│   ├── refine/        # Drop unsuitable photo pairs (CLIP and licence checks)
│   └── scenes/        # Build captioned scenes from the sprites
├── pixel-image-data/  # Generated model-training dataset
├── scene-data/        # Generated scenes
├── downloaded-data/   # Downloaded sprite packs and generated labels
├── sources.md         # Approved and rejected data sources
├── LICENSE            # BSD 3-Clause license
└── pyproject.toml     # Python project metadata and dependencies
```

- [`scripts/data`](scripts/data/README.md) contains the automated data acquisition and dataset-building pipeline.
- [`scripts/hub`](scripts/hub/README.md) pushes the local data to the public dataset repo and pulls it back.
- [`scripts/labeler`](scripts/labeler/README.md) labels sprites with CLIP or Florence-2 and gathers them into a flat dataset.
- [`scripts/refine`](scripts/refine/README.md) drops photo pairs that should not be trained on or published and writes `clean.csv`.
- [`scripts/scenes`](scripts/scenes/README.md) builds captioned 32×32 scenes from the sprites.
- `pixel-image-data/` contains generated `input/` photos, `target/` pixel-art images, their manifest and `clean.csv`.
- `downloaded-data/` contains source sprite packs, archives, and generated labelled datasets.

The data directories are generated locally and excluded from Git.

## License

The project code is available under the [BSD 3-Clause License](LICENSE). Reuse and redistribution must retain the copyright and license notice crediting `ocdbytes`. Downloaded source images and datasets remain subject to their respective licenses.
