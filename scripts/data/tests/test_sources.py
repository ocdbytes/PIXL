import unittest
from typing import final

from scripts.data.sources import OpenverseSource


@final
class FakeClient:
    def json(self, url: str) -> object:
        _ = url
        return {
            "page_count": 1,
            "results": [
                {
                    "id": "abc",
                    "title": "Cat",
                    "creator": "Jane",
                    "license": "cc0",
                    "license_version": "1.0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "provider": "flickr",
                    "foreign_landing_url": "https://example.test/cat",
                    "url": "https://example.test/cat.jpg",
                    "thumbnail": "https://example.test/cat-thumb.jpg",
                    "width": 1024,
                    "height": 768,
                    "mature": False,
                }
            ],
        }


@final
class OpenverseSourceTest(unittest.TestCase):
    def test_maps_cc0_result(self) -> None:
        records = list(OpenverseSource(FakeClient()).search("cat"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].provider, "openverse:flickr")
        self.assertEqual(records[0].license, "CC0")
        self.assertEqual(records[0].download_url, "https://example.test/cat-thumb.jpg")
