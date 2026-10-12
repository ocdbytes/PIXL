# Data sources

Only use assets whose individual source page provides a compatible license. “Free download” and public visibility are not licenses.

## Automated sources

The dataset service currently fetches from:

| Source | Accepted licenses | Use | API |
|---|---|---|---|
| [Wikimedia Commons](https://commons.wikimedia.org/) | Public domain, CC0, CC BY 1.0–4.0 | Ordinary source images | [MediaWiki API](https://commons.wikimedia.org/w/api.php) |
| [Openverse](https://openverse.org/) | CC0, Public Domain Mark, CC BY | Ordinary source images | [Openverse API](https://api.openverse.org/v1/images/) |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | Images listed as CC BY 2.0; annotations CC BY 4.0 | Captioned object and relationship crops | [Annotation CSVs](https://storage.googleapis.com/openimages/web/download_v7.html) |

Google states that Open Images [annotations are CC BY 4.0 and images are listed as CC BY 2.0](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), but makes no warranty about each image's licence and asks users to verify it. The manifest keeps every image's original Flickr URL, landing page and author so licences can be checked and attributed.

Openverse aggregates external providers. The manifest records both Openverse and the underlying provider so every result remains traceable. Search results still require visual review for incorrect licensing, trademarks, watermarks and unsuitable content.

## Curated pixel-art sources

These are suitable candidates for a later genuine pixel-art fine-tuning set. Download and record each pack separately; an artist may publish other work under different terms.

| Creator | Pack | Verified license |
|---|---|---|
| [Kenney](https://kenney.nl/assets) | Game assets on Kenney asset pages | CC0; confirmed by [Kenney support](https://kenney.nl/support) |
| [ansimuz](https://ansimuz.itch.io/sunny-land-pixel-game-art) | SunnyLand | CC0 1.0 |
| [0x72](https://0x72.itch.io/dungeontileset-ii) | Dungeon Tileset II | CC0 1.0 |
| [GrafxKid](https://grafxkid.itch.io/mini-pixel-pack-3) | Mini Pixel Pack 3 | CC0 1.0 |
| [Buch](https://opengameart.org/content/a-platformer-in-the-forest) | A Platformer in the Forest | CC0 |
| [surt](https://opengameart.org/content/another-nes-like-village) | Another NES-like Village | CC0 |
| [Screaming Brain Studios](https://screamingbrainstudios.itch.io/tiny-texture-pack) | Tiny Texture Pack | CC0 1.0; textures rather than sprites |

CC0 attribution is not required, but Pixl should retain creator and source information in its manifest.

