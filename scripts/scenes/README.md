# Scenes

`generate.py` builds the scenes pixl trains on: 32×32 images of four coloured sprites, one in each quadrant, with a caption that says which sprite is where. From the project root, with the sprites in `downloaded-data/` (about 10 minutes):

```bash
uv run python -m scripts.scenes.generate --sheets
uv run python -m scripts.hub.sync push --only scenes   # publishes to the public dataset
```

It writes `scene-data/tokens.npy` and `scene-data/manifest.csv`, and with `--sheets` review images in `scene-data/review/`. Splits, flags and how a scene is made are in [docs/data.md](../../docs/data.md#scenes).
