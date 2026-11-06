# Hub

`sync.py` pushes the local sprites and scenes to the Hugging Face dataset [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data), or pulls them back. [`card.md`](card.md) is the dataset card.

```bash
uv run python -m scripts.hub.sync pull                 # no login needed
uv run python -m scripts.hub.sync push --only scenes   # needs `hf auth login`
```

The dataset is public, so a push publishes at once: review the sprites and scene sheets first. A pull never overwrites an existing local manifest; remove or move it before restoring that config.
