# Remote training

The repository includes a Docker image for training on an NVIDIA server. It downloads the dataset from Hugging Face and writes checkpoints to a host directory, so neither data nor training state is stored inside the container.

## Requirements

The remote machine needs:

- Docker;
- an NVIDIA GPU and driver compatible with the CUDA version bundled by the locked PyTorch build;
- the NVIDIA Container Toolkit, which provides `docker run --gpus`.

The current PyTorch build uses CUDA 13 and requires NVIDIA driver 580 or newer. Check the host with:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## Copy and build

From the local repository, replace `user@host` with the server login:

```bash
rsync -av --relative \
  pyproject.toml uv.lock README.md Dockerfile .dockerignore src \
  user@host:pixl/
ssh user@host
```

On the server:

```bash
cd pixl
docker build -t pixl .
```

The image contains the locked runtime dependencies and the installed `pixl` package. It does not contain datasets or checkpoints.

## Run

First verify the image with the short training run:

```bash
docker run --gpus all --rm \
  -v pixl-hf:/root/.cache/huggingface \
  pixl --quick
```

Start the full run in the background:

```bash
mkdir -p checkpoints
docker run --gpus all --rm -d \
  --name pixl-train \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v pixl-hf:/root/.cache/huggingface \
  pixl
```

Arguments after the image name are passed to `python -m pixl.train`. For example, replace the final `pixl` with `pixl --run experiment-2 --epochs 2 --batch 64`.

The bind mount keeps checkpoints under `pixl/checkpoints/`. The `pixl-hf` volume keeps the downloaded and encoded dataset between containers.

## Monitor and resume

Follow the training log:

```bash
docker logs -f pixl-train
```

`Ctrl-C` stops following the log; it does not stop training. Stop the run with:

```bash
docker stop pixl-train
```

Because the container uses `--rm`, it is removed after stopping. Running the same command again resumes from `checkpoints/<run>/checkpoint.pt`. Use a new `--run` name when changing training settings.

Copy checkpoints back from the local machine:

```bash
rsync -a user@host:pixl/checkpoints/ checkpoints/
```

## Deploy code changes

From the local machine, stop the run, copy the changed files and rebuild:

```bash
ssh user@host 'docker stop pixl-train'
rsync -av --relative \
  pyproject.toml uv.lock README.md Dockerfile .dockerignore src \
  user@host:pixl/
ssh user@host 'cd pixl && docker build -t pixl .'
```

Restart with the previous training command to resume. A code change that alters the model configuration, data fingerprints or saved settings requires a new run.

## Troubleshooting

### Docker permission denied

Run Docker with `sudo`, or add the remote user to the `docker` group.

### `--gpus all` fails

Verify the NVIDIA driver and Container Toolkit installation with the CUDA test command under [Requirements](#requirements). This failure occurs before Pixl starts.

### The first run pauses at `Map`

The dataset is being converted to palette tokens and stored in the `pixl-hf` volume. This is CPU- and memory-intensive on the first run only. Monitor it with:

```bash
docker stats pixl-train
```

### Checkpoints are missing after the container stops

Confirm that the training command mounts the host checkpoint directory:

```text
-v "$PWD/checkpoints:/app/checkpoints"
```

Without that mount, `--rm` deletes the container together with its checkpoints.
