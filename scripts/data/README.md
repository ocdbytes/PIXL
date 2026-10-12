# Data scripts

> [!WARNING]
> These scripts are entirely AI-generated. They are automated tools designed only to complete the data-download task: they work, but their approach may not be optimal.

The data scripts build aligned photo pairs in `pixel-image-data/`:

- `input`: a licensed photo, or a crop of one, at 256×256
- `target`: a synthetic 32×32, 24-color version of the input
- `manifest.csv`: caption (`query`), provenance, author, license and source URL

Three sources are used:

- **Open Images V7** for human-annotated objects and relationships. Each pair is a crop of an annotated box, captioned from the label, e.g. "dog" or "man riding horse".
- **Wikimedia Commons** for per-file public-domain, CC0 and CC BY images found by free-text search.
- **Openverse** for CC0, Public Domain Mark and CC BY images from its indexed providers.

## Build from the query file

`build.py` is step 1 of [`src/setup_data.py`](../../src/setup_data.py), which rebuilds the dataset when it changes; training loads the finished dataset from Hugging Face instead. To run only the download, call it directly. It reads [`src/queries.json`](../../src/queries.json), runs Open Images first, then the web sources:

```bash
uv run python -m scripts.data.build --user-agent "PixlDataset/0.1 (your-email@example.com)"
```

Add `--only open_images` or `--only web` to run one section, and `--config other.json` to use another query file. The script checks the whole file before downloading any images; an unknown Open Images name fails with suggestions.

Every run resumes. `per_query` is a total per source and query, so rerunning skips queries that already have enough pairs.

### `src/queries.json`

```jsonc
{
  "output": "pixel-image-data",
  "open_images": {
    "splits": ["validation", "test"],  // add "train" for ~10x more images (~3 GB of CSVs)
    "per_query": 150,
    "workers": 8,
    "min_box": 0.2,                    // smallest box side, as a fraction of the image
    "captions": {"Cattle": "cow"},     // caption overrides for class names
    "classes": {"animals": ["Dog", "Cat"]},
    "relationships": {"riding": [["Man", "ride", "Horse"], ["Man", "is", "Run"]]}
  },
  "web": {
    "sources": ["wikimedia", "openverse"],
    "per_query": 150,
    "workers": 4,
    "min_side": 512,
    "queries": {"retro": ["arcade cabinet", "cassette tape"]}
  }
}
```

The theme keys (`animals`, `retro`, …) only organise the file; repeated queries are dropped.

### How Open Images is labelled

Open Images has no free-text search, so its queries must use its own labels:

- **Classes**: the 600 boxable class names in [`oidv7-class-descriptions-boxable.csv`](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv), e.g. `Dog`, `Ball (Object)`, `Fixed-wing aircraft`. The caption is the lowercase name without the bracketed part, unless `captions` overrides it.
- **Relationships**: `[subject, relationship, object]` using 30 relationships such as `on`, `at`, `holds`, `wears`, `ride`, `plays`, `hang`, `contain`, `inside_of`. Captions read naturally: "man riding horse", "orange hanging from tree".
- **Attributes**: `[subject, "is", attribute]` with one of 15 attributes (Run, Jump, Sit, Smile, Wooden, Transparent, …), captioned as "running man" or "wooden table".

Relationship annotations are mostly about people: "Chair at Table", "Man wears Glasses" and "Man ride Horse" are common, while scenes such as "ball on grass" are not annotated. The validation and test splits hold about 300 relationships with 15 or more examples each; the train split has many more.

Crops grow to a square around the box so the whole object stays in view. Boxes marked as a group of objects, boxes smaller than `min_box`, and images whose rotation is not 0 or not given are skipped. Annotation CSVs are cached in `pixel-image-data/openimages-cache/`.

## One-off queries

`fetch.py` runs web queries from the command line without a query file:

```bash
uv run python -m scripts.data.fetch \
  --query "cat isolated" \
  --per-query 100 \
  --user-agent "PixlDataset/0.1 (your-email@example.com)"
```

Images download in parallel (`--workers`, default 4); searches stay serial. Use a real contact address in the user agent, inspect sample sheets before training, and retain `manifest.csv` whenever the data or model is distributed.

## Test

```bash
uv run python -m unittest discover -s scripts/data/tests -v
```

## Limits

These targets teach resizing and palette reduction, not all decisions made by pixel artists. Genuine pixel-art packs in [`sources.md`](../../sources.md) should be handled as a separate fine-tuning dataset; they are not converted into synthetic pairs by this script.
