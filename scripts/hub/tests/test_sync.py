import unittest

from scripts.hub.sync import sprite_group

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


if __name__ == "__main__":
    _ = unittest.main()
