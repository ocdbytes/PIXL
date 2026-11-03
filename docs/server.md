# Training on a GPU server

The [`Dockerfile`](../Dockerfile) builds an image that runs `python -m pixl.train` ([training.md](training.md)). No data needs copying: the image downloads the scenes from the Hub.

## Requirements

The server needs Docker, the NVIDIA Container Toolkit (for `--gpus`), and an NVIDIA driver of version 580 or newer: PyPI's torch 2.14 brings its own CUDA 13 libraries ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)). `nvidia-smi` shows the driver version.

## The image

| Part | Choice |
|---|---|
| Base | `python:3.11-slim`, with uv 0.11.30 copied in |
| Dependencies | `uv sync --locked` without the `dev` and `notebook` groups, in their own layer, so a code change does not reinstall PyTorch |
| `pixl` | Installed as a package (not editable), so its palette and tokenizer load from the package |
| Volumes | `/app/checkpoints` for checkpoints and samples; `/root/.cache/huggingface` for the downloaded and encoded scenes |
| Entry point | `python -m pixl.train`; arguments after the image name go to it |

[`.dockerignore`](../.dockerignore) sends only `pyproject.toml`, `uv.lock`, `README.md` and `src/` to the build, since the data folders alone are gigabytes.

## Run

Copy the code to the server, either from GitHub or straight from your working copy (the image needs only these files). Replace `SERVER` with `user@host`; `ssh-copy-id SERVER` once saves typing the password on every copy.

```bash
git clone https://github.com/ocdbytes/PIXL.git pixl                                        # on the server, or
rsync -av --relative --exclude __pycache__ pyproject.toml uv.lock README.md Dockerfile .dockerignore src SERVER:pixl/   # from here
```

Then on the server:

```bash
cd pixl
docker build -t pixl .
docker run --gpus all --rm -v pixl-hf:/root/.cache/huggingface pixl --quick   # a short check
docker run --gpus all --rm -d --name pixl-train \
  -v "$PWD/checkpoints:/app/checkpoints" -v pixl-hf:/root/.cache/huggingface pixl
docker logs -f pixl-train                                  # one line per evaluation; Ctrl-C stops watching only
```

Arguments after the image name go to `pixl.train`, e.g. `pixl --run my-run --batch 64`. The `pixl-hf` volume keeps the encoded scenes between runs, and running the same command again resumes from `checkpoints/<run>/checkpoint.pt`. Copy the run back, then open it in [`notebooks/train.ipynb`](../notebooks/train.ipynb):

```bash
rsync -a SERVER:pixl/checkpoints/ checkpoints/
```

**After a code change:** copy the code again, rebuild (fast, since only the code layer changes), stop the running container with `docker stop pixl-train` (`--rm` removes it), and start it again. It resumes from its checkpoint; a run with changed settings needs `--run <new-name>`.

## Troubleshooting

- **`docker` says permission denied:** your user is not in the `docker` group; use `sudo docker`, or ask the server's admin to add you.
- **`--gpus all` fails** (for example "could not select device driver"): the NVIDIA Container Toolkit is missing or not configured for Docker.
- **The first run spends minutes on `Map`:** that is the one-time encoding of the million scenes into tokens. It runs on every core the container may use, with one worker process per core, each holding its own copy of the libraries (a few hundred MB); watch memory with `docker stats pixl-train`. Later runs read the result from the `pixl-hf` volume.
- **Fewer cores than expected:** the `docker run` commands above set no CPU limit. To check while the container runs:

  ```bash
  docker stats pixl-train                                              # CPU % (100% = one core)
  docker inspect pixl-train --format '{{.HostConfig.NanoCpus}} {{.HostConfig.CpusetCpus}} {{.HostConfig.CpuQuota}}'   # 0, empty, 0 = no limit
  docker exec pixl-train sh -c 'nproc; cat /sys/fs/cgroup/cpu.max'     # "max 100000" = no quota
  ```

  During training one busy core is normal: the model runs on the GPU, and one Python loop feeds it ([Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)).
