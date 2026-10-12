# Scenes

`generate.py` builds synthetic 32×32 scenes from the sprites, with captions that are always right because they are written from the placement:

> "a green pine tree left of a wooden barrel on grass", "a yellow bus at the top right on pavement"

They teach the model to compose several objects, and their stored placement can score model output later (evaluation, RL rewards).

| Split | Kind | Scenes | Objects from |
|---|---|---|---|
| train | `train` | 50,000 | Train-split sprites, never in a held-out pair |
| test | `new sprites` | 1,000 | Test-split sprites only |
| test | `new pair` | 1,000 | Train-split sprites, in pairs that never appear together in train scenes |

## Architecture

### One style per scene

A scene never mixes styles, so pixel size and look stay consistent. A style is a pack, except the 1-Bit Pack, which has two: colored (its colored and colored-transparent tiles) and monochrome.

```
 32x32 canvas, 16 px slots (2x2)          Pico-8 City: 8 px slots (4x4)

 +--------+--------+
 |  pine  | barrel |   background tile in every slot,
 |  tree  |        |   objects centred in their slot,
 +--------+--------+   never scaled
 |        |        |
 |        |        |   caption: "a pine tree left of a wooden barrel on grass"
 +--------+--------+
```

### Sprite roles

`role()` decides how a sprite may be used. `--sheets` saves one sheet per style (`scene-data/review/roles-<style>.png`; yellow `BG` = background, white = object).

| Role | Rule | Train sprites |
|---|---|---|
| background | Fully opaque, exactly one slot in size, and its label is in `BACKGROUNDS` (grass, dirt, sand, water, pavement, road, floors, tilled soil…) | 90 |
| object | Has transparency, fits a slot, at least 5% opaque, not terrain, and its label has no `FRAGMENT` word | 2,055 (231 in test) |
| unused | Everything else: other full tiles, fragments ("roof edge", "water corner"), building parts ("door", "wall", "battlement") and sprites bigger than a slot | |

`BACKGROUNDS` maps each label to the caption text, e.g. "dirt ground with pebbles" → "pebbly dirt", "water center" → "water". Styles without background tiles (GrafxKid, cursors) use a transparent canvas. Buch's 32×32 characters are bigger than a slot, so the style is not used.

### Making a scene

1. Pick a style, then one object (half the time) or two objects with different labels.
2. Place them: one object in a random slot; two objects in the same row or the same column.
3. Pick a background tile from the style's train-split backgrounds.
4. Reject the scene if an object would be hard to see: at least half of its pixels must differ in colour from the background under them. This removes, for example, white monochrome sprites on white water.
5. Write the caption: "a X at the top left", "a X left of a Y" (same row) or "a X above a Y" (same column), then "on {background}" when there is one.

The scene id is a hash of the style, the background and each object's path and slot, so identical scenes are made once. Seeds are fixed (0 for train, 1 for test): a rerun writes the same scenes.

### No leaks between splits

- Train scenes use only train-split sprites as objects; `new sprites` scenes use only test-split sprites. Sprite splits come from [`scripts/hub`](../hub/README.md).
- 5% of object label pairs are held out (`held_out()`: the SHA-256 of the two sorted labels). Train scenes never contain them; `new pair` scenes contain only them. The key is the labels alone, because captions do not name the style.
- Backgrounds always come from train-split tiles: they are context, not what is tested.

I checked all three on the generated data: 0 train scenes use a test sprite, and 0 `new pair` label pairs appear in train scenes. All 52,000 scenes encode with the shared palette pixel-exact, because they only use sprite colours.

### Current data

| Measure | Value |
|---|---|
| Scenes per style | 1-Bit colored 8,064; Pico-8 City 7,659; Tiny Battle 6,893; Tiny Farm 6,175; Sunny Land 5,529; cursors 4,711; Tiny Town 4,551; 1-Bit monochrome 4,466; GrafxKid 3,952 |
| Captions | 17,953 "at the", 17,057 "left of", 16,990 "above"; 31,203 distinct |
| Without background | 8,663 |

Styles are picked evenly, but styles with few objects make more duplicate scenes, which are skipped, so large styles end up with more scenes.

## Run

Step 4 of [`src/setup_data.py`](../../src/setup_data.py). Alone, from the project root (needs `downloaded-data/`; about 15 seconds):

```bash
uv run python -m scripts.scenes.generate --sheets
uv run python -m scripts.scenes.generate --train 100000 --test 4000   # more scenes
```

### Files

| File | Contents |
|---|---|
| `scene-data/image/<id>.png` | 32×32 RGBA scene; transparent where there is no background |
| `scene-data/manifest.csv` | `id`, `split`, `kind`, `caption`, `style`, `background`, `objects` and `sprites` (labels and paths, `\|`-separated), `slots` (`row,column` per object), `background_path`, `path` |
| `scene-data/review/` | With `--sheets`: `roles-<style>.png` per style and `scenes.png` with 60 captioned scenes |

Every run replaces `scene-data/image/` and the manifest. Upload with `uv run python -m scripts.hub.sync push --only scenes`.

## Limits

- Captions describe position only: no counts ("two trees"), sizes or colours beyond the labels.
- Some 1-Bit monochrome backgrounds are busy, so objects on them pass the visibility check but are hard to see.
- Sunny Land's objects are mostly terrain pieces (platforms, logs, rocks); its characters are bigger than a slot.
