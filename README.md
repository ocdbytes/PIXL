# pixl

Pixl is a pixel-art generation model. It will learn to generate small, palette-controlled pixel art from text prompts.

The project is currently preparing licensed training data. Model and training code will follow once the dataset baseline is validated.

## Setup

```bash
uv sync
```

### Training data

`src/setup_data.py` sets up all training data. By default it pulls the photo pairs and sprites from the private Hugging Face dataset repo [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data) (log in once with `hf auth login`):

```bash
uv run python -m src.setup_data
```

To rebuild the photo pairs from their original sources instead, use `--source build`. The sprite packs must then already be in `downloaded-data/` (downloaded by hand, see [`sources.md`](sources.md)); the script stops at once if their manifest is missing.

```bash
uv run python -m src.setup_data --source build --user-agent "PixlDataset/0.1 (your-email@example.com)"
```

| Step | Script | What it does | Time |
|---|---|---|---|
| 1. Get data (`hub`, default) | [`scripts/hub`](scripts/hub/README.md) | Pulls photo pairs and sprites from the dataset repo; skips data already on disk | Minutes |
| 1. Get data (`build`) | [`scripts/data`](scripts/data/README.md) | Downloads photo pairs from Open Images, Wikimedia and Openverse for the queries in [`src/queries.json`](src/queries.json) | 1–2 days; resumes if stopped |
| 2. Label | [`scripts/labeler`](scripts/labeler/README.md) | Labels sprites that have no label yet (existing labels are kept) and gathers all sprites into `downloaded-data/labeled/` | Seconds when nothing is unlabelled |
| 3. Refine | [`scripts/refine`](scripts/refine/README.md) | Drops web photo pairs whose photo does not match its query and writes `pixel-image-data/clean.csv` | ~5 minutes the first time |

Each step runs as its own process, and setup stops if one fails. Add `--label-model florence` to label new sprites with Florence-2 instead of CLIP. Every step can also be run on its own; see its README. After changing local data, upload it with `uv run python -m scripts.hub.sync push`.

## Project structure

```text
pixl/
├── src/
│   ├── setup_data.py  # Set up all training data: get, label, refine
│   └── queries.json   # Queries for the photo download
├── scripts/
│   ├── data/          # Download and build aligned photo/pixel-art pairs
│   ├── hub/           # Push and pull the data to the Hugging Face dataset repo
│   ├── labeler/       # Label and collect downloaded sprite assets
│   └── refine/        # Drop web pairs whose photo does not match its query
├── pixel-image-data/  # Generated model-training dataset
├── downloaded-data/   # Downloaded sprite packs and generated labels
├── sources.md         # Approved and rejected data sources
├── LICENSE            # BSD 3-Clause license
└── pyproject.toml     # Python project metadata and dependencies
```

- [`scripts/data`](scripts/data/README.md) contains the automated data acquisition and dataset-building pipeline.
- [`scripts/hub`](scripts/hub/README.md) pushes the local data to the private dataset repo and pulls it back.
- [`scripts/labeler`](scripts/labeler/README.md) labels sprites with CLIP or Florence-2 and gathers them into a flat dataset.
- [`scripts/refine`](scripts/refine/README.md) drops web pairs whose photo does not match its query and writes `clean.csv`.
- `pixel-image-data/` contains generated `input/` photos, `target/` pixel-art images, their manifest and `clean.csv`.
- `downloaded-data/` contains source sprite packs, archives, and generated labelled datasets.

The two data directories are generated locally and excluded from Git.

## License

The project code is available under the [BSD 3-Clause License](LICENSE). Reuse and redistribution must retain the copyright and license notice crediting `ocdbytes`. Downloaded source images and datasets remain subject to their respective licenses.
