# Labeler

`label.py` labels the sprites that `downloaded-data/manifest.csv` has no label for, then copies every sprite into one folder.

Models are downloaded to the Hugging Face cache on first run. Pick one with `--model`:

- **`clip`** (default): [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) zero-shot classification. Each sprite gets the closest of the broad categories in `CATEGORIES` (tree, house, road, person, digit, …), using the prompt `pixel art of {label}`. `score` is CLIP's softmax probability over those categories. Runs in about 90 seconds on a CPU.
- **`florence`**: [florence-community/Florence-2-base](https://huggingface.co/florence-community/Florence-2-base), the official transformers conversion of Microsoft's Florence-2-base. Each sprite gets a free-text `<CAPTION>`, such as "A pixel art image of a cactus." It has no score. Runs in about 12 minutes on an Apple-silicon GPU (MPS); on a CPU it takes about 4 seconds per sprite.

**Output:** `--output` (default `downloaded-data/labeled/`) holds every sprite under a flat file name, plus `labels.csv` with `file`, `label`, `label_source`, `score` and provenance columns.

## Run

This is step 2 of [`src/setup_data.py`](../../src/setup_data.py), which passes its `--label-model` here as `--model`. To run it alone, from the project root:

```bash
uv run python -m scripts.labeler.label                    # CLIP
uv run python -m scripts.labeler.label --model florence   # Florence-2
```

Delete the output folder before rerunning if sprites were removed from the manifest; the script overwrites files but does not remove old ones.

