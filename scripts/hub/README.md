# Hub

`sync.py` keeps the training data in the Hugging Face dataset repo [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data), allowing a fresh setup without rebuilding the data locally.

## Contents

| Config | Local source | Images |
|---|---|---|
| `photos` | `pixel-image-data/clean.csv` | 256×256 input and 32×32 target |
| `sprites` | `downloaded-data/manifest.csv` | Sprite PNG |
| `scenes` | `scene-data/manifest.csv` | 32×32 scene PNG |

Manifest metadata is kept as text, and images retain their original file bytes. `queries.json` records the queries used to build the photos. [`card.md`](card.md) is the dataset card.

## Train and test splits

Each config has `train` and `test` splits:

| Config | Grouping | Test share |
|---|---|---|
| `photos` | Source-photo `sha256`, keeping crops together | 2% |
| `sprites` | Animation name or pack and filename, keeping frames and colour variants together | 10% |
| `scenes` | Existing manifest `split` value | Set by the scene generator |

Photos and sprites use a stable hash of their group key. This keeps related images in one split and ensures existing rows do not move when the command is rerun or new data is added. Code splitting local data should use `PART[name].split(row)`.

A pull combines both splits into `manifest.csv`, with train rows first.

## Run

Pulling needs no login. Pushing requires a Hugging Face token with write access:

```bash
hf auth login
```

From the project root:

```bash
uv run python -m scripts.hub.sync push
uv run python -m scripts.hub.sync pull
uv run python -m scripts.hub.sync card
uv run python -m scripts.hub.sync push --only scenes
```

`--only` also works with `pull`. Training can load the Hub dataset directly and does not require a pull.

## Important behavior

- **Pull never overwrites:** a config is skipped when its local `manifest.csv` exists.
- **Push replaces the selected Hub config:** run refine first so photos come from `clean.csv`.
- **Interrupted pulls are safe:** the manifest is written only after all images.
- Pulling photos writes the filtered `clean.csv` rows into `manifest.csv`.
- `dropped.csv`, `clip_embeddings.pt`, and `review/` remain local.
