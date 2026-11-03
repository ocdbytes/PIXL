# Data sources

Only use assets whose individual source page provides a compatible license. “Free download” and public visibility are not licenses.

## Historical photo sources

The Hub's `photos` config was built from these sources. Its download and filtering scripts have been removed from the current tree; they remain in the Git history before November 2026. Training now uses generated sprite scenes, while the palette builder still samples the existing photo targets.

| Source | Accepted licenses | Use | API |
|---|---|---|---|
| [Wikimedia Commons](https://commons.wikimedia.org/) | Public domain, CC0, CC BY 1.0–4.0 | Ordinary source images | [MediaWiki API](https://commons.wikimedia.org/w/api.php) |
| [Openverse](https://openverse.org/) | CC0, Public Domain Mark, CC BY | Ordinary source images | [Openverse API](https://api.openverse.org/v1/images/) |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | Images listed as CC BY 2.0; annotations CC BY 4.0 | Captioned object and relationship crops | [Annotation CSVs](https://storage.googleapis.com/openimages/web/download_v7.html) |

Google states that Open Images [annotations are CC BY 4.0 and images are listed as CC BY 2.0](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), but makes no warranty about each image's licence and asks users to verify it. The manifest keeps every image's original Flickr URL, landing page and author so licences can be checked and attributed.

Openverse aggregates external providers. The manifest records both Openverse and the underlying provider so every result remains traceable. Search results still require visual review for incorrect licensing, trademarks, watermarks and unsuitable content.

## Curated pixel-art sources

These include the sprite packs used by the current scene generator and additional candidates. `downloaded-data/manifest.csv` records the assets actually present, with their labels, creators, licences and source URLs. The generator uses only eligible CC0 sprites; the sprite dataset can also hold the attributed CC BY icons listed below. Record each new pack separately; an artist may publish other work under different terms.

| Creator | Pack | Verified license |
|---|---|---|
| [Kenney](https://kenney.nl/assets) | Game assets on Kenney asset pages | CC0; confirmed by [Kenney support](https://kenney.nl/support). Board Game Icons and Game Icons: objects only, not interface actions. 1-Bit Platformer Pack: whole objects only. Not used: Micro Roguelike (8x8 tiles on dark squares) |
| [ansimuz](https://ansimuz.itch.io/sunny-land-pixel-game-art) | SunnyLand | CC0 1.0 |
| [0x72](https://0x72.itch.io/dungeontileset-ii) | Dungeon Tileset II | CC0 1.0 |
| [GrafxKid](https://grafxkid.itch.io/mini-pixel-pack-3) | Mini Pixel Pack 3 | CC0 1.0 |
| [Buch](https://opengameart.org/content/a-platformer-in-the-forest) | A Platformer in the Forest | CC0 |
| [surt](https://opengameart.org/content/another-nes-like-village) | Another NES-like Village | CC0 |
| [Screaming Brain Studios](https://screamingbrainstudios.itch.io/tiny-texture-pack) | Tiny Texture Pack | CC0 1.0; textures rather than sprites |
| Crawl Stone Soup artists, maintained by Chris Hamons ([OpenGameArt](https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental)) | Dungeon Crawl 32x32 tiles, Full (original + supplemental) | CC0 on the page and in the pack's `LICENSE.txt`; artist list in the pack. Not used, since the dataset is public: `player/base` (undressed character bodies), `water_nymph`, `harpy`, `human_slave` and `slave_freed` |
| [Scrittl](https://opengameart.org/content/spaceship-set-32x32px) | Spaceship set 32x32px | CC0 |
| [Jull](https://opengameart.org/node/50777) | Animated spaceships | CC0; most ships are 128x32 animation strips |
| [Idylwild](https://opengameart.org/content/idylwilds-inventory) | Idylwild's Inventory | CC0 |
| [wender](https://opengameart.org/node/176168) | Sports cars pixelart sprites | CC0 |
| PIXEL32, edited by [psycho_666](https://opengameart.org/node/15049) | Pixel Art Vehicles | CC0; sprite sheets, larger than 32x32 |
| [domsson](https://opengameart.org/content/two-pixelish-tree-sprites) | Two pixelish Tree sprites | CC0; 48x76, larger than 32x32 |
| Surt, uploaded by [vk](https://opengameart.org/content/trees-1) | trees | CC0; a 416x160 sheet |
| [game-icons.net](https://game-icons.net/about.html) contributors ([repository](https://github.com/game-icons/icons)) | Game Icons (SVG) | CC BY 3.0, with Viscious Speed and Zeromancer CC0 (`license.txt`); credit each author. Not used: abstract icons and names about gore, drugs, self-harm or nudity |

CC0 attribution is not required, but Pixl should retain creator and source information in its manifest.

