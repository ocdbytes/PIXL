# Refine

`clip_filter.py` filters photo pairs in `pixel-image-data/`, writing accepted rows to `clean.csv` and rejected rows to `dropped.csv`. It does not modify `manifest.csv` or image files.

Pairs are dropped for the first matching reason:

| Reason | Check |
|---|---|
| `nudity` | CLIP nudity margin is at least `--nudity` (default `0.02`) |
| `graphic` | CLIP graphic-content margin is at least `--graphic` (default `0.0395`) |
| `mismatch` | Query score is below `--min-score` (default `0.22`) or query rank is outside `--top-k` (default `50`) |
| `no author` | A CC BY photo has no author to credit |
| `email author` | The author field contains an email address |

Open Images rows skip the mismatch check because their labels come from human-drawn boxes.

## How the CLIP checks work

The script uses [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). CLIP maps photos and text to normalized 512-value vectors; their dot product gives a similarity score.

For mismatch detection, each photo is compared with `a photo of {query}` for every query in the manifest:

- **query score:** similarity to the row's own query;
- **rank:** number of other queries with a higher similarity.

For content detection, the photo's best similarity to nudity or graphic prompts is compared with its best similarity to neutral prompts. The difference is the content margin.

Photo embeddings are cached in `clip_embeddings.pt`. Later runs embed only new photos, while recalculating scores against the current queries and prompts.

## Run

This is step 3 of [`src/setup_data.py`](../../src/setup_data.py). From the project root:

```bash
uv run python -m scripts.refine.clip_filter
uv run python -m scripts.refine.clip_filter --sheets
uv run python -m scripts.refine.clip_filter --min-score 0.24 --top-k 20
uv run python -m scripts.refine.clip_filter --nudity 0.015
```

The first run takes about 12 minutes for 99k photos on an Apple-silicon GPU. Cached runs take about 6 seconds when there are no new photos.

## Outputs

| Path | Contents |
|---|---|
| `pixel-image-data/clean.csv` | Accepted manifest rows |
| `pixel-image-data/dropped.csv` | Rejected rows with reason and scores |
| `pixel-image-data/clip_embeddings.pt` | Cached float16 photo embeddings |
| `pixel-image-data/review/` | Optional nudity and graphic contact sheets from `--sheets` |

Delete `clip_embeddings.pt` when changing the CLIP model or image preprocessing so all photos are embedded again.
