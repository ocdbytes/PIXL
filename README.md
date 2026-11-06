![16 scenes drawn by pixl](docs/poster.png)

*Seed 0, temperature 0.7, top-k 32 and guidance 3.*

Pixl is a small text-to-pixel-art model trained from scratch. It draws a 32×32 scene one pixel at a time from a caption such as “a wizard at the top left … on a dark dungeon floor”. Training uses one million generated scenes made from four sprites placed in the image's four quadrants.

> **GPU compute:** Thanks to [@harshitruwali](https://github.com/harshitruwali) for the GPU time used to train the published model.

## Generate an image

The weights are at [`ocdbytes/pixl`](https://huggingface.co/ocdbytes/pixl). Install the package directly from this repository:

```bash
pip install git+https://github.com/ocdbytes/PIXL
```

```python
import torch
from PIL import Image

from pixl.model import encode_captions
from pixl.preprocess import decode, load_palette
from pixl.tokenizer import Tokenizer
from pixl.train import load_pretrained

model = load_pretrained()  # pass bf16=True for the half-size weights
torch.manual_seed(0)
caption = ("a purple robed wizard at the top left, a closed treasure chest at the top right, "
           "a brown bat at the bottom left and a red potion at the bottom right on a dark dungeon floor")
ids = encode_captions([caption], Tokenizer.load(), length=model.config.caption)
pixels = model.generate(ids, temperature=0.7, top_k=32, guidance=3.0)
image = decode(pixels.view(32, 32).numpy(), load_palette())
image.resize((256, 256), Image.Resampling.NEAREST).save("scene.png")
```

The model expects the same shape of caption it saw during training: four objects from one sprite pack, each assigned to a quadrant, followed by the ground. See the [model card](https://huggingface.co/ocdbytes/pixl) for its vocabulary and limits, or use [`notebooks/generate.ipynb`](notebooks/generate.ipynb) to try prompts.

## App

![The pixl app](docs/app.png)

```bash
uv run python -m pixl.app
# open http://127.0.0.1:7860
```

Pick a preset or build a scene from the model's known sprite packs. **Randomize** chooses a new scene; **Rotate clockwise** is useful for checking whether the model follows positions. Advanced controls expose the seed, guidance and temperature.

| Flag | Effect |
| --- | --- |
| `--bf16` | Use the smaller half-precision weights |
| `--device auto` | Pick an available GPU, falling back to the CPU |
| `--share` | Create a temporary public Gradio link |

If you installed with `pip`, install `gradio` separately.

## Documentation

- [Model and inputs](docs/model.md): palette, tokenizer and architecture
- [Data](docs/data.md): datasets and scene generation
- [Training](docs/training.md): running, evaluating and resuming training
- [Remote training](docs/remote-training.md): Docker setup for an NVIDIA server
- [Development](docs/development.md): repository layout and checks
- [Sprite sources](docs/sources.md): accepted sprite packs and licences

## License

The code uses the [BSD 3-Clause License](LICENSE). Downloaded images and datasets keep their original licences.
