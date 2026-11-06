# Development

Pixl requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

This installs the package in `src/pixl/` together with the development, notebook and app dependencies locked in [`uv.lock`](../uv.lock). Dependency groups and package metadata are defined in [`pyproject.toml`](../pyproject.toml).

## Repository structure

- [`src/pixl/`](../src/pixl) contains the Python package.
- [`scripts/scenes/`](../scripts/scenes) contains scene generation and vocabulary tooling.
- [`scripts/hub/`](../scripts/hub) contains Hugging Face dataset tooling.
- [`notebooks/`](../notebooks) contains training and generation notebooks.
- [`tests/`](../tests) and `scripts/*/tests/` contain the test suite.
- [`docs/`](.) contains project documentation.

Generated data is stored in `downloaded-data/` and `scene-data/`. Training writes checkpoints to `checkpoints/`. These directories are not tracked by Git.

Run scripts as modules from the repository root:

```bash
uv run python -m scripts.scenes.generate --help
uv run python -m pixl.train --quick
```

## Checks

```bash
uv run ruff check
uv run basedpyright
uv run python -m unittest discover -s . -p "test_*.py"
```

The same checks run in [GitHub Actions](../.github/workflows/checks.yml).
