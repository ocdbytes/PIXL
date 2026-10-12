import unittest
from pathlib import Path
from typing import final

from scripts.data.build import flatten, load_config


@final
class BuildConfigTest(unittest.TestCase):
    def test_shipped_queries_file_is_valid(self) -> None:
        config = load_config(Path("scripts/data/queries.json"))
        self.assertIn("open_images", config)
        self.assertIn("web", config)

    def test_flatten_drops_repeats_in_order(self) -> None:
        self.assertEqual(flatten({"a": ["x", "y"], "b": ["y", "z"]}), ["x", "y", "z"])
