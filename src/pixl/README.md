# pixl

The `pixl` package: a caption-to-pixel-art model and everything it needs.

| Module | Holds |
|---|---|
| [`preprocess.py`](preprocess.py) | The 256-token colour palette ([`palette.json`](palette.json)): images to tokens and back |
| [`tokenizer.py`](tokenizer.py) | The caption tokenizer ([`tokenizer.json`](tokenizer.json)) |
| [`data.py`](data.py) | Loads the scenes from the Hugging Face dataset as captions and tokens |
| [`model.py`](model.py) | `EncoderPixl`, the encoder-decoder model |
| [`train.py`](train.py) | `python -m pixl.train`, `load_model()` for trained checkpoints, and `load_pretrained()` for the published model |

Details are in [`docs/`](../../docs): [model](../../docs/model.md), [data](../../docs/data.md) and [training](../../docs/training.md).
