import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import DatasetDict

from scripts.hub import sync
from scripts.hub.sync import PART, pull, push, sprite_group

DCSS = "packs/dcss_dungeon-crawl-32x32/Dungeon Crawl Stone Soup Full/item/weapon"


def group(path: str, pack: str = "dcss_dungeon-crawl-32x32") -> str:
    return sprite_group({"path": path, "pack": pack})


class SpriteGroupTest(unittest.TestCase):
    def test_dungeon_crawl_variants_share_a_group(self) -> None:
        names = ("long_sword.png", "long_sword_5.png", "long_sword_new.png", "long_sword_2_old.png")
        self.assertEqual({group(f"{DCSS}/{name}") for name in names}, {f"{DCSS}/long_sword"})

    def test_existing_rules_are_unchanged(self) -> None:
        self.assertEqual(group("packs/a/Frames/cherry/cherry-3.png", "a"), "packs/a/Frames/cherry/cherry")
        self.assertEqual(group("packs/k/Tiles/colored/tile_0001.png", "k"), "k/tile_0001.png")


class TokensPartTest(unittest.TestCase):
    def test_push_then_pull_gives_back_the_tokens_and_rows(self) -> None:
        # Palette colours are all different, so any tokens survive the trip through PNG.
        grids = np.random.default_rng(0).integers(0, 256, (6, 32, 32), dtype=np.uint8)
        rows = [{"id": str(i), "split": "train" if i < 4 else "test", "caption": f"scene {i}"} for i in range(6)]
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            with Path(source, "manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.save(Path(source, "tokens.npy"), grids)
            pushed: list[DatasetDict] = []
            with (
                patch.object(DatasetDict, "push_to_hub", lambda data, *args, **kwargs: pushed.append(data)),
                patch.object(sync, "push_card"),
            ):
                push("repo", [replace(PART["scenes"], root=Path(source))])
            with patch.object(sync, "load_dataset", lambda repo, name: pushed[0]):
                pull("repo", [replace(PART["scenes"], root=Path(target))])
            np.testing.assert_array_equal(np.load(Path(target, "tokens.npy")), grids)
            with Path(target, "manifest.csv").open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), rows)


if __name__ == "__main__":
    _ = unittest.main()
