# Trains pixl on a GPU server: python -m pixl.train, with the scenes from Hugging Face.
#
#   docker build -t pixl .
#   docker run --gpus all --rm -v "$PWD/checkpoints:/app/checkpoints" -v pixl-hf:/root/.cache/huggingface pixl
#
# Arguments after the image name go to pixl.train, e.g. `pixl --quick` or `pixl --run my-run --batch 64`.
# PyPI's PyTorch brings its own CUDA libraries; the host needs the NVIDIA driver (580 or newer for
# CUDA 13) and the NVIDIA Container Toolkit for --gpus.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# The locked dependencies first, so a code change does not reinstall PyTorch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

# Then pixl itself, installed as a package: its palette and tokenizer load from the package.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1
# Checkpoints and sample images; the Hugging Face cache keeps the encoded scenes between runs.
VOLUME ["/app/checkpoints", "/root/.cache/huggingface"]
ENTRYPOINT ["python", "-m", "pixl.train"]
