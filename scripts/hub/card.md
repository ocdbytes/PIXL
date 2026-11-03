---
pretty_name: pixl-data
license:
- cc0-1.0
- cc-by-2.0
- cc-by-2.5
- cc-by-3.0
- cc-by-4.0
task_categories:
- text-to-image
tags:
- pixel-art
- sprites
size_categories:
- 1M<n<10M
---

# pixl-data

Training data for **pixl**, a small text-to-pixel-art model trained from scratch.

| Config | Rows | What it holds |
|---|---|---|
| `photos` | ~84k | Photo pairs: a 256×256 photo (`input`), a 32×32, 24-colour version of it (`target`), and a caption (`query`) |
| `sprites` | ~11.5k | Game sprites and icons (`image`) with a short label (`label`) |
| `scenes` | 1.002M | 32×32 scenes of four sprites each (`image`), with a caption written from the placement (`caption`) |

## Load

```python
from datasets import load_dataset

photos = load_dataset("ocdbytes/pixl-data", "photos")  # {"train": ..., "test": ...}
sprites = load_dataset("ocdbytes/pixl-data", "sprites")
scenes = load_dataset("ocdbytes/pixl-data", "scenes")

train = load_dataset("ocdbytes/pixl-data", "photos", split="train")  # one split only
```

## Splits

| Config | train | test | Kept together in one split |
|---|---|---|---|
| `photos` | ~82.7k | ~1.6k (2% of source photos) | All crops of one source photo (same `sha256`) |
| `sprites` | ~10.4k | ~1.2k (10% of groups) | The colour variants of one tile, the frames of one animation, and the numbered and old/new versions of one Dungeon Crawl sprite |
| `scenes` | 1,000,000 | 2,000 | Set by the generator: test scenes use test sprites, or an object pair never seen together in train scenes |

Rows in each defined photo or sprite group stay in one split, reducing leakage from related crops, frames and variants. This grouping does not detect every possible near-copy. A group is in `test` when the first 8 hex digits of the SHA-256 of its key, as a fraction of 16⁸, are below 0.02 (photos) or 0.1 (sprites). The split is fixed: new data never moves existing rows.

## photos

| Source | Caption |
|---|---|
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | A box label or a relationship between two boxes drawn by people, e.g. "dog", "man riding horse", "wooden table". The photo is cropped around the box. |
| [Wikimedia Commons](https://commons.wikimedia.org/), [Openverse](https://openverse.org/) | The search query that found the photo, e.g. "arcade cabinet". |

Photos were filtered before upload. A photo was removed when:

- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) matched it more closely to prompts for nudity or graphic content (gore, dead bodies, injuries) than to neutral prompts;
- it came from a web search and CLIP found it does not show its query;
- it is CC BY but has no author to credit, or its author field is an email address.

Every photo was changed:

1. Cropped to a square.
2. Resized to 256×256: `input`.
3. Resized to 32×32 and reduced to 24 colours: `target`.

| Column | Meaning |
|---|---|
| `id` | Pair id |
| `provider` | `openimages`, `wikimedia` or `openverse:<provider>` |
| `source_id` | Id of the photo at its source |
| `query` | Caption |
| `title`, `author` | From the photo's source page |
| `license`, `license_url` | Licence of the original photo |
| `source_page`, `original_url` | Where the original photo is |
| `input_path`, `target_path` | File paths in the original folder layout |
| `sha256` | SHA-256 of the downloaded original |
| `input`, `target` | The two images |

## sprites

Sprites from CC0 packs by [Kenney](https://kenney.nl/assets) (1-Bit Pack, 1-Bit Platformer Pack, Board Game Icons, Cursor Pixel Pack, Game Icons, Monochrome Pirates, Pico-8 City, Pixel Shmup, Pixel Vehicle Pack, Tiny Battle, Tiny Dungeon, Tiny Farm, Tiny Town), [ansimuz](https://ansimuz.itch.io/sunny-land-pixel-game-art) (Sunny Land), [GrafxKid](https://grafxkid.itch.io/mini-pixel-pack-3) (Mini Pixel Pack 3), [Buch](https://opengameart.org/content/a-platformer-in-the-forest) (A Platformer in the Forest), the Dungeon Crawl Stone Soup artists ([Dungeon Crawl 32x32 tiles](https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental)), [Idylwild](https://opengameart.org/content/idylwilds-inventory) (Idylwild's Inventory), [Scrittl](https://opengameart.org/content/spaceship-set-32x32px) (Spaceship set 32x32px), [Jull](https://opengameart.org/node/50777) (Animated spaceships) and [wender](https://opengameart.org/node/176168) (Sports cars pixelart sprites). `label_source` says where each label comes from: checked by eye (`claude-visual`), a file name (`filename`) or a sprite-sheet atlas (`atlas`).

Icons by the [game-icons.net](https://game-icons.net) contributors (the row's `creator`, e.g. Lorc or Delapouite), available on https://game-icons.net under CC BY 3.0, rendered as white silhouettes 30×30 pixels in size on a 32×32 transparent canvas. Each row's `source_url` is the icon's page. Icons with abstract names, or about gore, drugs, self-harm or nudity, were left out. Kenney's Board Game Icons and Game Icons are converted the same way, keeping only icons of objects, not interface actions; the 1-Bit Platformer Pack keeps whole objects, not its ground and platform pieces.

Monochrome sprites (the 1-Bit Pack's white tiles, the 1-Bit Platformer Pack, the Cursor Pixel Pack, Monochrome Pirates and the icons) have labels starting with "monochrome", without the colour words of their coloured versions, so a caption tells colour and monochrome apart. The 1-Bit Pack's copies of objects on a dark square are left out; its ground tiles keep theirs.

| Column | Meaning |
|---|---|
| `path` | File path in the original pack layout |
| `pack`, `creator`, `source_url` | Which pack the sprite comes from |
| `license` | CC0 1.0, or CC BY 3.0 for game-icons.net icons |
| `width`, `height` | Size in pixels |
| `label`, `label_source` | Label and where it comes from |
| `image` | The sprite (PNG, with transparency) |

## scenes

Synthetic 32×32 scenes, each made from one sprite style: a background tile repeated in every 16 px slot and four coloured objects with different labels, one centred in each quadrant. Sprites are not scaled, except the 32×32 Dungeon Crawl tiles, which are halved to 16×16 by keeping every other pixel; halved objects that fall apart into pieces are left out. The caption names each object's quadrant, listing the objects in random order, so it is always right: "a pine tree at the bottom right, a red house at the top left, a well at the top right and a barrel at the bottom left on grass". Images are drawn in the shared 256-colour palette, so they encode back to palette tokens exactly. Until November 2026 this config held 100,000 scenes of one or two objects; they remain in the repo's history.

| Kind (`kind`) | Split | Scenes | Objects from |
|---|---|---|---|
| `train` | train | 1,000,000 | Train-split sprites, with no held-out label pair |
| `new sprites` | test | 1,000 | Test-split sprites only |
| `new pair` | test | 1,000 | Train-split sprites, with at least one label pair that never appears together in train scenes |

| Column | Meaning |
|---|---|
| `caption` | The scene's caption |
| `swapped` | The caption with the first two listed objects' quadrants exchanged, for testing whether a model reads positions |
| `kind` | See above |
| `style` | Sprite pack (or 1-Bit Pack colour family) the scene is made from |
| `background`, `background_path` | Background tile label and path in the `sprites` config; empty for a transparent scene |
| `objects`, `sprites`, `slots` | Each object's label, path and `row,column` slot, `\|`-separated |
| `image` | The scene (PNG, transparent where there is no background) |

## Licences

Photo and sprite rows carry their licence in `license` (and `license_url` for photos). Scene rows have no separate licence column; they use only CC0 sprites:

- **Sprites:** CC0 1.0, except the game-icons.net icons: CC BY 3.0 (credit the `creator`; see the sprites section).
- **Scenes:** CC0 1.0, as they are made only from CC0 sprites.
- **Photos:** CC0, public domain, or CC BY 1.0–4.0. All Open Images photos are listed as CC BY 2.0.
- **Open Images annotations** (the boxes and labels behind the crops and captions): CC BY 4.0, by Google.

To reuse a CC BY photo, credit it with the row's `author`, `title`, `source_page` and `license`, and say that it was changed (cropped, resized, colours reduced).

Google lists Open Images photos as CC BY 2.0 but [does not guarantee](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) each licence. Wikimedia and Openverse licences come from their providers' metadata. Check a row's source page before relying on its licence.

## Known issues

- Web captions are still somewhat noisy: in a sample of 30 web photos that passed the CLIP check, 4 did not show their query.
- The nudity and graphic-content filter is automatic and can miss photos. Please report any you find.
- Many photos show real people. Some show brand logos or photographers' watermarks.
