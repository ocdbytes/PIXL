# pixl

Pixl is a text-to-pixel-art model. It draws a 32×32 image one pixel at a time from a caption, with an encoder-decoder Transformer trained from scratch. It learns from a million generated scenes, each with four sprites in the four quadrants of the image and a caption that says which sprite is where.

## Quick start

```bash
uv sync
OMP_NUM_THREADS=4 nice -n 19 uv run python -m pixl.train --quick --device cpu   # a light 40-step check
```

Full training belongs on a GPU server. From the repo, copy the code there (`SERVER` is `user@host`), then build and run the Docker image on the server:

```bash
rsync -av --relative --exclude __pycache__ pyproject.toml uv.lock README.md Dockerfile .dockerignore src SERVER:pixl/
cd pixl && docker build -t pixl .                                    # on the server
docker run --gpus all --rm -d --name pixl-train \
  -v "$PWD/checkpoints:/app/checkpoints" -v pixl-hf:/root/.cache/huggingface pixl
```

Copy `checkpoints/` back, then open [`notebooks/train.ipynb`](notebooks/train.ipynb) to see the run and [`notebooks/generate.ipynb`](notebooks/generate.ipynb) to draw your own captions.

## Docs

| Page | Covers |
|---|---|
| [Model](docs/model.md) | Colour palette, caption tokenizer, model architecture |
| [Data](docs/data.md) | The Hugging Face dataset, loading it, generating scenes, pushing and pulling |
| [Training](docs/training.md) | The training command, devices, notebooks, measures, checkpoints and resuming |
| [GPU server](docs/server.md) | The Docker image, running it, troubleshooting |
| [Development](docs/development.md) | Package layout, dependencies, checks and CI |

## Project map

| Path | Purpose |
|---|---|
| [`src/pixl/`](src/pixl) | The `pixl` package: tokenizers, data loader, model and training |
| [`notebooks/`](notebooks) | Train and generate |
| [`scripts/scenes/`](scripts/scenes) | Build the scenes from sprites |
| [`scripts/hub/`](scripts/hub) | Push and pull the Hugging Face dataset |
| [`docs/`](docs) | Technical documentation |
| [`sources.md`](sources.md) | Approved and rejected sprite sources |

## License

The code is available under the [BSD 3-Clause License](LICENSE). Reuse must retain its copyright and license notice. Downloaded images and datasets remain subject to their original licences.
