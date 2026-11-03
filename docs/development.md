# Development

## Layout

| Path | Contents |
|---|---|
| [`src/pixl/`](../src/pixl) | The `pixl` package: palette (`preprocess.py`), caption tokenizer (`tokenizer.py`), data loader (`data.py`), model (`model.py`) and training (`train.py`) |
| [`scripts/scenes/`](../scripts/scenes) | Scene generator ([data.md](data.md#scenes)) |
| [`scripts/hub/`](../scripts/hub) | Hugging Face sync and dataset card ([data.md](data.md#hugging-face-sync)) |
| [`notebooks/`](../notebooks) | Training and generation notebooks ([training.md](training.md#notebooks)) |
| [`tests/`](../tests) and `scripts/*/tests/` | Unit tests and the notebook import check |
| [`Dockerfile`](../Dockerfile) | Training image ([server.md](server.md)) |

Data lives in `downloaded-data/` (sprites) and `scene-data/` (scenes), and model states in `checkpoints/`. These directories are excluded from Git.

## The package

`uv sync` installs `src/pixl/` as the `pixl` package, built by `uv_build`, so `import pixl` works from any directory. [`palette.json`](../src/pixl/palette.json) and [`tokenizer.json`](../src/pixl/tokenizer.json) ship inside it and load through `importlib.resources`; rebuilding the palette or tokenizer writes over the package's own file. The scripts in `scripts/` run from the project root (`uv run python -m scripts.scenes.generate`) and import `pixl` like any other package.

## Dependencies

| Group | Packages | Installed by |
|---|---|---|
| main | `datasets`, `huggingface-hub`, `numpy`, `pillow`, `torch` | Everything, including the Docker image |
| `dev` | `ruff`, `basedpyright` | `uv sync` (a default group) |
| `notebook` | `ipykernel`, `ipywidgets`, `matplotlib`, `pandas` | `uv sync` (a default group) |

[`uv.lock`](../uv.lock) pins every version, and [`.python-version`](../.python-version) picks Python 3.11. The Docker image installs the main dependencies only (`--no-default-groups`).

## Checks

```bash
uv run ruff check
uv run basedpyright                                       # type checking
uv run python -m unittest discover -s . -p "test_*.py"   # unit tests and notebook import checks
```

[`tests/test_notebooks.py`](../tests/test_notebooks.py) imports every module and name the notebooks import, without running them. [GitHub Actions](../.github/workflows/checks.yml) runs linting, type checking, unit tests and notebook import checks on every push and pull request, with the locked dependencies (`uv sync --locked`).
