# pixl

Pixl is a text-to-pixel-art model. It represents 32×32 images with a shared 256-token colour palette and generates them one pixel at a time with a decoder-only Transformer.

Stage 1 photo pretraining, the datasets, tokenizers, and model are ready. Later training stages are not yet implemented.

## Setup and train

```bash
uv sync
```

Open [`notebooks/pretrain.ipynb`](notebooks/pretrain.ipynb) in VS Code with the project's `.venv` kernel, or start Jupyter:

```bash
uv run --with jupyter jupyter lab
```

The notebook loads the public [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data) dataset automatically. The first photo load downloads about 2.4 GB. Three epochs take about 3.5 hours on the measured Apple-silicon GPU; checkpoints are written to `checkpoints/`. Set `QUICK = True` for a short smoke test.

See [`src/README.md`](src/README.md) for the palette, caption tokenizer, data loader, model architecture, and training details.

## Project map

| Path | Purpose |
|---|---|
| [`src/`](src/README.md) | Palette and caption tokenizers, data loader, model, and data setup |
| [`notebooks/pretrain.ipynb`](notebooks/pretrain.ipynb) | Stage 1 photo pretraining |
| [`misc/retro.py`](misc/retro.py) | Training displays |
| [`scripts/data/`](scripts/data/README.md) | Download and build aligned photo/pixel-art pairs |
| [`scripts/labeler/`](scripts/labeler/README.md) | Label and gather sprite assets |
| [`scripts/refine/`](scripts/refine/README.md) | Remove unsuitable or mismatched photo pairs |
| [`scripts/scenes/`](scripts/scenes/README.md) | Build captioned scenes from sprites |
| [`scripts/hub/`](scripts/hub/README.md) | Push and pull the Hugging Face dataset |
| [`sources.md`](sources.md) | Approved and rejected sprite sources |

Generated data lives in `pixel-image-data/`, `downloaded-data/`, and `scene-data/`. Model states live in `checkpoints/`. These directories are excluded from Git.

## Change the dataset

This is only needed for new photo queries, sprite packs, or labels. To start from the current Hub data:

```bash
uv run python -m scripts.hub.sync pull
```

Alternatively, place approved sprite packs in `downloaded-data/`, then run the complete pipeline:

```bash
uv run python -m src.setup_data --user-agent "PixlDataset/0.1 (your-email@example.com)"
```

The pipeline runs four steps:

| Step | Details | Typical time |
|---|---|---:|
| Download photo pairs | [`scripts/data`](scripts/data/README.md) | 1–2 days initially; resumable |
| Label sprites | [`scripts/labeler`](scripts/labeler/README.md) | Seconds when nothing is new |
| Filter photos | [`scripts/refine`](scripts/refine/README.md) | About 12 minutes initially, then seconds |
| Generate scenes | [`scripts/scenes`](scripts/scenes/README.md) | About 15 seconds |

`downloaded-data/manifest.csv` must exist before running the pipeline. Add `--label-model florence` to label new sprites with Florence-2. Each step can also run independently using the commands in its README.

Review `pixel-image-data/review/` and `scene-data/review/`, then upload:

```bash
uv run python -m scripts.hub.sync push
```

A push updates the public dataset immediately. See the [Hub README](scripts/hub/README.md) for split and overwrite behavior.

## License

The code is available under the [BSD 3-Clause License](LICENSE). Reuse must retain its copyright and license notice. Downloaded images and datasets remain subject to their original licences.
