# Data

Pixl trains on [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data), a synthetic dataset built from the accepted [sprite sources](sources.md).

## Dataset

| Config | Split | Rows | Contents |
| --- | --- | ---: | --- |
| `scenes` | train | 1,000,000 | Four-object scenes used for training |
| `scenes` | test | 1,000 | `new sprites`: objects held out from training |
| `scenes` | test | 1,000 | `new pair`: training sprites with at least one held-out label pair |
| `sprites` | train | 10,361 | Labelled source sprites |
| `sprites` | test | 1,158 | Sprite groups held out from training |

Five percent of object-label pairs are deterministically reserved for `new pair`. No training scene contains a reserved pair, and every `new pair` scene contains at least one. Related sprite frames and colour variants remain in the same sprite split.

The current scene snapshot covers 10 visual styles and 2,240 object labels. Four-object captions contain 30–80 tokenizer tokens, with a median of 41, excluding `<bos>` and `<eos>`. Dataset counts and measurements can change when the scenes are regenerated.

The Hub dataset card, including column definitions and licences, is maintained in [`scripts/hub/card.md`](../scripts/hub/card.md).

## Loading

[`pixl.data.load()`](../src/pixl/data.py) returns captions and 32×32 palette-token tensors:

```python
from pixl.data import load

test = load("test", keep=("kind", "swapped"))[:]
test["caption"]  # 2,000 strings
test["tokens"]   # (2000, 32, 32) uint8 tensors
```

The conversion path is:

```text
Hugging Face `scenes` rows
        │
        │  caption, image, requested metadata
        ▼
`datasets.load_dataset()`
        │
        │  encode each image with pixl's fixed palette
        ▼
caption + 32×32 uint8 token grid + requested metadata
        │
        ▼
Hugging Face datasets cache → PyTorch-formatted Dataset
```

Images are converted to palette tokens because the model predicts integer colour IDs, not RGB values. The first load downloads and maps the selected split; later calls reuse the `datasets` cache while the input fingerprint and mapping code remain unchanged.

For splits of at least 100,000 rows, mapping uses the CPU cores available to the process. Smaller selections use one process because worker startup costs more than the conversion. Pass `num_proc` to override this behaviour. Row order is preserved.

## Scene generation

[`scripts/scenes/generate.py`](../scripts/scenes/generate.py) creates the synthetic `scenes` config from the local sprite manifest. Each 32×32 scene uses one sprite style, four distinct object labels and one object per quadrant. A compatible tile may fill the background. Captions name the objects and quadrants in random order, so caption order does not reveal spatial order.

The generator excludes unsuitable sprites and rejects placements with poor foreground/background contrast. Dungeon Crawl's 32 px sprites are downsampled to fit a quadrant. It also writes a `swapped` caption in which two listed objects exchange quadrants; training and evaluation use this to test whether the model reads position.

Train scenes use train-split objects. `new sprites` scenes use held-out objects, while `new pair` scenes use known objects in reserved pairings. Backgrounds come from the train split. Scene IDs prevent duplicates within each evaluation kind, and a fixed random seed makes generation reproducible for unchanged inputs and settings.

Selection rules, role classification and rejection criteria are implemented directly in [`generate.py`](../scripts/scenes/generate.py).

### Running the generator

From the project root, with the source sprites under `downloaded-data/`:

```bash
uv run python -m scripts.scenes.generate --sheets
```

A default run takes about ten minutes.

| Flag | Effect |
| --- | --- |
| `--train`, `--test` | Set the number of train and test scenes; the test rows are divided between `new sprites` and `new pair` |
| `--sheets` | Write sprite-role and sample-scene review sheets to `scene-data/review/` |
| `--no-four` | Generate one- and two-object scenes instead of four-object scenes |
| `--no-colour` | Allow monochrome objects and monochrome-only styles |

| Output | Contents |
| --- | --- |
| `scene-data/tokens.npy` | `(scenes, 32, 32)` uint8 palette tokens |
| `scene-data/manifest.csv` | Split, kind, captions, style, background, objects, slots and source paths |
| `scene-data/review/` | Optional contact sheets produced by `--sheets` |

Each run replaces `tokens.npy` and `manifest.csv`. Row `i` in the array corresponds to row `i` in the manifest.

## Dataset limitations

The dataset is designed for a narrow compositional-position experiment, not general text-to-image training.

- **Restricted layout:** default scenes always contain four objects in fixed quadrants. The data does not represent arbitrary coordinates, overlap, depth, object count or scale.
- **Templated language:** captions follow one grammatical form and use dataset labels rather than natural descriptions. Performance does not imply broad language understanding.
- **Limited visual domain:** images are 32×32 pixel art assembled from selected sprite packs. Results should not be extrapolated to photographs, illustrations or unseen art styles.
- **Palette quantisation:** every image is mapped to a fixed 256-token palette. Colours absent from the palette are replaced by the nearest available RGB colour.
- **Synthetic composition:** objects and backgrounds are combined algorithmically. Contrast filtering removes severe failures, but implausible combinations and visually busy scenes remain possible.
- **Narrow generalisation tests:** `new pair` measures held-out pairings of known labels, while `new sprites` changes the underlying sprite images. Neither split measures unrestricted compositional or semantic generalisation.
- **Source imbalance:** styles and labels occur at different frequencies, so aggregate loss can hide weak performance on rare labels or packs.
