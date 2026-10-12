# Hub

`sync.py` keeps the training data in the private Hugging Face dataset repo [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data), so a fresh setup takes minutes instead of 1–2 days, and the hand-checked sprite labels have a copy outside this machine.

## What the repo holds

| Config | Local source | Rows | Image columns |
|---|---|---|---|
| `photos` | `pixel-image-data/manifest.csv` | One per photo pair | `input` (256×256 JPEG), `target` (32×32 PNG) |
| `sprites` | `downloaded-data/manifest.csv` | One per sprite | `image` (PNG) |

Every manifest column is kept as text, including captions, labels, authors, licences and source URLs. Images are stored as their original file bytes in Parquet files, so a pull writes back exactly the files that were pushed. `queries.json` is uploaded as a record of the queries that built the photos.

The dataset card shown on the Hub is [`card.md`](card.md). `push` uploads it too; after editing only the card, run `sync card`. Its YAML is merged into the card's metadata block, which keeps the config and feature metadata that `push` writes there for the loader.

Parquet is used because the Hub allows at most 10,000 files per folder and recommends under 100,000 files per repo ([storage limits](https://huggingface.co/docs/hub/storage-limits)); `pixel-image-data/input/` alone has ~99k files.

## Run

Log in once with a token that has write access:

```bash
hf auth login
```

From the project root:

```bash
uv run python -m scripts.hub.sync push   # upload local data, replacing the repo's copy
uv run python -m scripts.hub.sync pull   # download into pixel-image-data/ and downloaded-data/
uv run python -m scripts.hub.sync card   # upload only the dataset card (card.md)
```

`pull` is step 1 of [`src/setup_data.py`](../../src/setup_data.py) by default.

- **Pull never overwrites:** a part whose `manifest.csv` already exists is skipped. Delete the manifest to pull that part again.
- **Interrupted pull:** the manifest is written after all images, so rerunning `pull` starts that part again.
- **After changing local data** (a new build, new sprite packs or new labels), run `push` so the repo matches.

`clean.csv` and `clip_scores.csv` are not uploaded; setup's refine step rebuilds them.
