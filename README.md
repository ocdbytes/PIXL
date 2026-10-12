# pixl

Pixl is a pixel-art generation model. It will learn to generate small, palette-controlled pixel art from text prompts.

The project is currently preparing licensed training data. Model and training code will follow once the dataset baseline is validated.

## Setup

```bash
uv sync
```

## Project structure

```text
pixl/
├── scripts/
│   ├── data/          # Download and build aligned photo/pixel-art pairs
│   └── labeler/       # Label and collect downloaded sprite assets
├── pixel-image-data/  # Generated model-training dataset
├── downloaded-data/   # Downloaded sprite packs and generated labels
├── sources.md         # Approved and rejected data sources
├── LICENSE            # BSD 3-Clause license
└── pyproject.toml     # Python project metadata and dependencies
```

- [`scripts/data`](scripts/data/README.md) contains the automated data acquisition and dataset-building pipeline.
- [`scripts/labeler`](scripts/labeler/README.md) labels sprites with CLIP or Florence-2 and gathers them into a flat dataset.
- `pixel-image-data/` contains generated `input/` photos, `target/` pixel-art images, and their manifest.
- `downloaded-data/` contains source sprite packs, archives, and generated labelled datasets.

The two data directories are generated locally and excluded from Git.

## License

The project code is available under the [BSD 3-Clause License](LICENSE). Reuse and redistribution must retain the copyright and license notice crediting `ocdbytes`. Downloaded source images and datasets remain subject to their respective licenses.
