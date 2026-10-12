# Scenes

`generate.py` builds synthetic 32×32 scenes from sprites. Captions are generated from object placement, for example:

> "a green pine tree left of a wooden barrel on grass"

These scenes teach the model spatial composition and retain placement metadata for evaluation.

## Dataset splits

| Split | Kind | Default count | Purpose |
|---|---|---:|---|
| train | `train` | 50,000 | Train-split sprites and known object pairs |
| test | `new sprites` | 1,000 | Sprites not used in training |
| test | `new pair` | 1,000 | Known sprites in object pairs excluded from training |

Five percent of object-label pairs are deterministically held out. Train scenes exclude these pairs, while `new pair` scenes use only them. Backgrounds always come from train-split sprites.

## Scene generation

Each scene uses one sprite style so scale and appearance remain consistent. The generator:

1. Selects one object, or two objects with different labels.
2. Places one object in a random slot, or two in the same row or column.
3. Adds a compatible background tile when available.
4. Rejects scenes where less than half of an object's visible pixels differ from the background.
5. Writes a caption such as `at the top left`, `left of`, or `above`.

Objects are never scaled. A scene ID is derived from its style, background, objects, and positions, preventing duplicates. Fixed random seeds make repeated runs deterministic.

Sprites are eligible as objects when they are transparent, fit one slot, contain enough visible pixels, and are not terrain fragments. Fully opaque terrain tiles can be backgrounds. Use `--sheets` to review sprite roles and generated scenes.

## Run

This is step 4 of [`src/setup_data.py`](../../src/setup_data.py). From the project root:

```bash
uv run python -m scripts.scenes.generate
uv run python -m scripts.scenes.generate --sheets
uv run python -m scripts.scenes.generate --train 100000 --test 4000
```

Generation requires `downloaded-data/` and takes about 15 seconds with the defaults.

## Outputs

| Path | Contents |
|---|---|
| `scene-data/image/<id>.png` | Generated 32×32 RGBA scene |
| `scene-data/manifest.csv` | Split, kind, caption, style, objects, positions, sprite paths, and image path |
| `scene-data/review/` | Optional role and scene contact sheets from `--sheets` |

Every run replaces `scene-data/image/` and `scene-data/manifest.csv`. Upload the result with:

```bash
uv run python -m scripts.hub.sync push --only scenes
```

## Limits

Captions describe position but not counts, sizes, or colours beyond sprite labels. Some busy backgrounds can still make objects difficult to see.
