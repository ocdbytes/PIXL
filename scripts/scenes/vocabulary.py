"""Writes the captions pixl knows, per sprite pack, for the app's menus.

Reads the train scenes of scene-data/manifest.csv and writes
src/pixl/vocabulary.json: for each pack (a scene "style"), a display name, its
object labels with their articles, and its grounds, all exactly as the captions
write them. A pack drawn without a ground has the ground "".

    uv run python -m scripts.scenes.vocabulary
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

MANIFEST = Path("scene-data/manifest.csv")
OUTPUT = Path("src/pixl/vocabulary.json")
# One object phrase of a caption, e.g. "a red potion at the top left".
OBJECT = re.compile(r"(?:^|, | and )(an? .+?) at the (?:top|bottom) (?:left|right)")
# The text after the last place phrase: " on <ground>", or "" without a ground.
GROUND = re.compile(r" at the (?:top|bottom) (?:left|right)(?: on (.+))?$")
NAMES = {
    "ansimuz_sunny-land": "ansimuz · Sunny Land",
    "dcss_dungeon-crawl-32x32": "Dungeon Crawl 32x32 tiles",
    "grafxkid_mini-pixel-pack-3": "GrafxKid · Mini Pixel Pack 3",
    "kenney_1-bit-pack/colored": "Kenney · 1-Bit Pack (coloured)",
    "kenney_pixel-shmup": "Kenney · Pixel Shmup",
    "kenney_pixel-vehicle-pack": "Kenney · Pixel Vehicle Pack",
    "kenney_tiny-battle": "Kenney · Tiny Battle",
    "kenney_tiny-dungeon": "Kenney · Tiny Dungeon",
    "kenney_tiny-farm": "Kenney · Tiny Farm",
    "kenney_tiny-town": "Kenney · Tiny Town",
}


def main() -> None:
    objects: dict[str, set[str]] = {}
    grounds: dict[str, set[str]] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != "train":
                continue
            found = OBJECT.findall(row["caption"])
            ground = GROUND.search(row["caption"])
            if len(found) != 4 or ground is None:
                raise ValueError(f"not a four-object caption: {row['caption']!r}")
            objects.setdefault(row["style"], set()).update(found)
            grounds.setdefault(row["style"], set()).add(ground.group(1) or "")
    vocabulary = {
        style: {"name": NAMES[style], "objects": sorted(objects[style]), "grounds": sorted(grounds[style])}
        for style in sorted(objects, key=lambda style: NAMES[style].lower())
    }
    OUTPUT.write_text(json.dumps(vocabulary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for entry in vocabulary.values():
        print(f"{entry['name']}: {len(entry['objects'])} objects, grounds {entry['grounds']}")


if __name__ == "__main__":
    main()
