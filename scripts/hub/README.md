# Hub

`sync.py` keeps the training data in the public Hugging Face dataset repo [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data), so a fresh setup takes minutes instead of 1–2 days, and the hand-checked sprite labels have a copy outside this machine.

## What the repo holds

| Config | Local source | Rows | Image columns |
|---|---|---|---|
| `photos` | `pixel-image-data/clean.csv` | One per photo pair that [refine](../refine/README.md) kept | `input` (256×256 JPEG), `target` (32×32 PNG) |
| `sprites` | `downloaded-data/manifest.csv` | One per sprite | `image` (PNG) |
| `scenes` | `scene-data/manifest.csv` | One per scene built by [scenes](../scenes/README.md) | `image` (32×32 PNG) |

Every manifest column is kept as text, including captions, labels, authors, licences and source URLs. Images are stored as their original file bytes in Parquet files, so a pull writes back exactly the files that were pushed. `queries.json` is uploaded as a record of the queries that built the photos.

### Train and test splits

Each config has a `train` and a `test` split. Splitting by row would put near-copies in both splits, so rows are split by group:

| Config | Split rule (`Part.split`) | Test share | train / test rows |
|---|---|---|---|
| `photos` | `sha256` of the source photo, so all crops of one photo stay together | 2% | 82,651 / 1,639 |
| `sprites` | `sprite_group`: animation frames without their number (`cherry-3` → `cherry`); other sprites by pack and file name, so the 4 colour variants of a 1-Bit Pack tile stay together | 10% | 5,144 / 541 |
| `scenes` | The manifest's `split` column, set by the generator from the sprites each scene uses | — | 50,000 / 2,000 |

For photos and sprites, `hashed_split(group, test_share)` puts a row in `test` when the first 8 hex digits of the SHA-256 of its group key, as a fraction of 16⁸, are below the test share. The result depends only on the key, so the split never changes between runs and new data never moves existing rows. Code that splits local data should call the same `PART[name].split(row)`.

All 73 Buch sprites are frames of one sprite sheet (`characters_0000` …), so they form one group and are all in `train`.

A pull writes both splits into one `manifest.csv`, train rows first, so its row order differs from the original manifest.

The dataset card shown on the Hub is [`card.md`](card.md). `push` uploads it too; after editing only the card, run `sync card`. Its YAML is merged into the card's metadata block, which keeps the config and feature metadata that `push` writes there for the loader.

Parquet is used because the Hub allows at most 10,000 files per folder and recommends under 100,000 files per repo ([storage limits](https://huggingface.co/docs/hub/storage-limits)); `pixel-image-data/input/` alone has ~99k files.

## Run

Pulling needs no login. To push, log in once with a token that has write access:

```bash
hf auth login
```

From the project root:

```bash
uv run python -m scripts.hub.sync push   # upload local data, replacing the repo's copy
uv run python -m scripts.hub.sync pull   # download into pixel-image-data/ and downloaded-data/
uv run python -m scripts.hub.sync card   # upload only the dataset card (card.md)
uv run python -m scripts.hub.sync push --only scenes   # upload one config (also works for pull)
```

Training does not need `pull`: [`src/data.py`](../../src/data.py) loads the dataset from the Hub directly. `pull` gives an editable local copy, for example to start [changing the dataset](../../README.md#change-the-dataset) on another machine.

- **Pull never overwrites:** a part whose `manifest.csv` already exists is skipped. Delete the manifest to pull that part again.
- **Interrupted pull:** the manifest is written after all images, so rerunning `pull` starts that part again.
- **Every push is public at once.** The repo history was squashed before it went public on 2026-10-12, so it holds only filtered data; push only `clean.csv` rows that refine has produced.
- **After changing local data** (a new build, new sprite packs or new labels), run refine and then `push` so the repo matches. `push` stops if `clean.csv` is missing.
- **A pull writes `clean.csv` rows as `manifest.csv`**, so pulled photo data holds only the kept pairs.

Only the rows of `clean.csv` are uploaded; refine's other outputs (`dropped.csv`, `clip_embeddings.pt`, `review/`) stay local.
