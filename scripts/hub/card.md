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
- 100K<n<1M
---

# pixl-data

Training data for **pixl**, a small text-to-pixel-art model trained from scratch.

| Config | Rows | What it holds |
|---|---|---|
| `photos` | ~84k | Photo pairs: a 256×256 photo (`input`), a 32×32, 24-colour version of it (`target`), and a caption (`query`) |
| `sprites` | ~5.7k | CC0 game sprites (`image`) with a short label (`label`) |

## Load

```python
from datasets import load_dataset

photos = load_dataset("ocdbytes/pixl-data", "photos", split="train")
sprites = load_dataset("ocdbytes/pixl-data", "sprites", split="train")
```

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

Sprites from CC0 packs by [Kenney](https://kenney.nl/assets) (1-Bit Pack, Cursor Pixel Pack, Pico-8 City, Tiny Battle, Tiny Farm, Tiny Town), [ansimuz](https://ansimuz.itch.io/sunny-land-pixel-game-art) (Sunny Land), [GrafxKid](https://grafxkid.itch.io/mini-pixel-pack-3) (Mini Pixel Pack 3) and [Buch](https://opengameart.org/content/a-platformer-in-the-forest) (A Platformer in the Forest). `label_source` says where each label comes from: checked by eye (`claude-visual`), a file name (`filename`) or a sprite-sheet atlas (`atlas`).

| Column | Meaning |
|---|---|
| `path` | File path in the original pack layout |
| `pack`, `creator`, `source_url` | Which pack the sprite comes from |
| `license` | CC0 1.0 |
| `width`, `height` | Size in pixels |
| `label`, `label_source` | Label and where it comes from |
| `image` | The sprite (PNG, with transparency) |

## Licences

Every row has its own licence in `license` (and `license_url` for photos):

- **Sprites:** CC0 1.0.
- **Photos:** CC0, public domain, or CC BY 1.0–4.0. All Open Images photos are listed as CC BY 2.0.
- **Open Images annotations** (the boxes and labels behind the crops and captions): CC BY 4.0, by Google.

To reuse a CC BY photo, credit it with the row's `author`, `title`, `source_page` and `license`, and say that it was changed (cropped, resized, colours reduced).

Google lists Open Images photos as CC BY 2.0 but [does not guarantee](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) each licence. Wikimedia and Openverse licences come from their providers' metadata. Check a row's source page before relying on its licence.

## Known issues

- Web captions are still somewhat noisy: in a sample of 30 web photos that passed the CLIP check, 4 did not show their query.
- The nudity and graphic-content filter is automatic and can miss photos. Please report any you find.
- Many photos show real people. Some show brand logos or photographers' watermarks.
