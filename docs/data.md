# Data

Training reads the public Hugging Face dataset [`ocdbytes/pixl-data`](https://huggingface.co/datasets/ocdbytes/pixl-data). This page covers what it holds, how `pixl` loads it, how the scenes are made, and how the repo pushes and pulls it.

## The dataset

| Config | Train | Test | Contents |
|---|---:|---:|---|
| `scenes` | 1,000,000 | 2,000 | 32×32 scenes of four sprites, with captions naming each sprite's quadrant; [made here](#scenes) |
| `sprites` | 10,361 | 1,158 | Labelled sprites from approved packs ([`sources.md`](../sources.md)) |

These counts describe the current dataset snapshot; rebuilding and pushing a config can change them. [`scripts/hub/card.md`](../scripts/hub/card.md) is the dataset card on the Hub, with every column and licence.

## Loading

[`data.py`](../src/pixl/data.py) loads the scenes as captions and 32×32 token grids, cached by the `datasets` library. `keep` names other text columns to keep, such as a scene's `kind` and `swapped` caption. The package does not load the `sprites` config; the scene generator reads the local sprite files.

```python
from pixl.data import load

test = load("test", keep=("kind", "swapped"))[:]
test["caption"]  # 2,000 strings
test["tokens"]   # (2000, 32, 32) uint8 palette tokens
```

The first load downloads about 1 GB into the Hugging Face cache. It also encodes every image into tokens: splits of 100,000 rows or more (`PARALLEL_ROWS`) use every core the process may use, and smaller splits one process, since starting workers costs more than it saves. `num_proc` overrides this. The rows keep their order either way.

## Scenes

[`scripts/scenes/generate.py`](../scripts/scenes/generate.py) builds the synthetic scenes from the sprites in `downloaded-data/`. Each scene fills its four 16 px quadrants with four coloured sprites, and the caption names each sprite's quadrant, listing them in random order:

> "a pine tree at the bottom right, a red house at the top left, a well at the top right and a barrel at the bottom left on grass"

Only the caption tells which sprite goes where, so these scenes teach the model positions and keep the placement for evaluation.

### Splits

| Split | Kind | Default count | Purpose |
|---|---|---:|---|
| train | `train` | 1,000,000 | Train-split sprites, with no held-out label pair |
| test | `new sprites` | 1,000 | Sprites not used in training |
| test | `new pair` | 1,000 | Train-split sprites, with at least one held-out label pair |

Five percent of object-label pairs are deterministically held out: train scenes contain none of them, and every `new pair` scene contains at least one. Backgrounds always come from train-split sprites.

In the current generated snapshot, each of the 10 styles makes about 5–12% of the train scenes, every one of the 2,240 object labels appears in at least 153 train scenes (median 441), and captions run from 30 to 80 tokens (median 41), excluding `<bos>` and `<eos>`. Rebuilding with other sprites or settings changes these measurements.

### How a scene is made

Each scene uses one sprite style so scale and appearance remain consistent. The generator:

1. Selects four objects with different labels from one style; Pico-8 City is left out, since its 8 px slots make a 4×4 grid.
2. Puts them in the four quadrants in random order.
3. Adds a compatible background tile when available.
4. Rejects scenes where less than half of an object's visible pixels differ from the background.
5. Writes the caption, and `swapped`: the same caption with the first two listed objects' quadrants exchanged, for the position test.

Objects are not scaled, except the 32 px Dungeon Crawl sprites (`HALVED`), which are halved by keeping every other pixel; a halved object is skipped when its largest connected piece holds less than 80% of its pixels (`WHOLE`). A scene ID is derived from its style, background, objects, and positions in caption order. Repeated IDs are rejected within each kind; different caption orders can describe the same image. Fixed random seeds reproduce the scenes for unchanged sprite inputs and settings.

Sprites are eligible as objects when they are transparent, fit one slot, contain enough visible pixels, are not terrain fragments, and are not labelled "monochrome". Fully opaque terrain tiles can be backgrounds. Only CC0 sprites are used, so scenes stay CC0.

### Running the generator

From the project root (about 10 minutes, and `downloaded-data/` must hold the sprites):

```bash
uv run python -m scripts.scenes.generate --sheets
```

| Flag | Effect |
|---|---|
| `--train`, `--test` | Number of train and test scenes; `new sprites` gets `test // 2`, and `new pair` gets the remainder |
| `--sheets` | Also saves review sheets of the sprite roles and the first 60 scenes to `scene-data/review/` |
| `--no-four` | One or two objects per scene instead, with captions such as "a pine tree at the top left" or "a pine tree left of a red house"; Pico-8 City's 8 px slots are used too. The `scenes` config held 100,000 such scenes until November 2026 |
| `--no-colour` | Places monochrome objects too ("monochrome skeleton"); monochrome-only styles join |

| Output | Contents |
|---|---|
| `scene-data/tokens.npy` | A (scenes, 32, 32) uint8 array of palette tokens; row *i* is scene *i* of `manifest.csv`. A million scenes as PNG files would take about 4 GB |
| `scene-data/manifest.csv` | Split, kind, caption, swapped caption, style, background, objects, slots and sprite paths |
| `scene-data/review/` | Optional role and scene contact sheets from `--sheets` |

Every run replaces `tokens.npy` and `manifest.csv`; `--sheets` also replaces the review PNGs. Older image directories from the previous PNG-based workflow are not used by these scripts.

### Limits

Captions describe position but not counts, sizes, or colours beyond sprite labels. With the default settings every scene has four objects, so a model trained only on that dataset never sees a one- or two-object caption. `--no-four` produces those simpler scenes instead. Some busy backgrounds can still make objects difficult to see.

## Hugging Face sync

[`scripts/hub/sync.py`](../scripts/hub/sync.py) pushes the local sprites and scenes to the Hub, or pulls them back, allowing a fresh setup without rebuilding the data locally.

| Config | Local source | Images |
|---|---|---|
| `sprites` | `downloaded-data/manifest.csv` | Sprite PNG |
| `scenes` | `scene-data/manifest.csv` | 32×32 scene PNG, encoded from `scene-data/tokens.npy` |

Manifest metadata is kept as text, and sprite images retain their original file bytes. Scenes are kept locally as palette tokens (`tokens.npy`): a push encodes each one as a PNG in the shared palette, and a pull decodes the PNGs back into the same tokens when using that palette. Pushing data also updates the dataset card from [`card.md`](../scripts/hub/card.md) while preserving its config metadata.

### Splits

| Config | Grouping | Test share |
|---|---|---|
| `sprites` | Animation name or pack and filename, keeping frames and colour variants together | 10% |
| `scenes` | Existing manifest `split` value | Set by the scene generator |

Sprites use a stable hash of their group key. This keeps related images in one split and ensures existing rows do not move when the command is rerun or new data is added. Code splitting local data should use `PART[name].split(row)`.

A pull combines both splits into `manifest.csv` in the dataset loader's order. For scenes, row *i* of `tokens.npy` matches row *i* of that manifest.

### Running the sync

Pulling needs no login. Pushing requires a Hugging Face token with write access (`hf auth login`). From the project root:

```bash
uv run python -m scripts.hub.sync push
uv run python -m scripts.hub.sync pull
uv run python -m scripts.hub.sync card
uv run python -m scripts.hub.sync push --only scenes
```

`--only` also works with `pull`. Training loads the Hub dataset directly and does not require a pull.

- **Pull never overwrites:** a config is skipped when its local `manifest.csv` exists.
- **Push replaces the selected Hub config.** The default dataset is public, so review the sprites and scene sheets before pushing. A new `--repo` is created private; the script does not change an existing repo's visibility.
- **Interrupted pulls can be retried:** the manifest is written only after all sprite images or scene tokens. Files left by an incomplete pull may be rewritten on retry.
- `scene-data/review/` remains local.

## Changing the dataset

This is only needed for new sprite packs or labels. To start from the current Hub data:

```bash
uv run python -m scripts.hub.sync pull
```

Sprites live in `downloaded-data/`: the packs in `packs/` and one row per sprite, with its label, in `manifest.csv` ([`sources.md`](../sources.md) lists the approved packs). After adding packs and their labels, rebuild the scenes, review `scene-data/review/`, then upload:

```bash
uv run python -m scripts.scenes.generate --sheets   # about 10 minutes
uv run python -m scripts.hub.sync push
```

A push updates the public dataset immediately.
