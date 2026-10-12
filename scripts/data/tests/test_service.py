import csv
import hashlib
import io
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import final

from PIL import Image

from scripts.data.service import DatasetService, make_pair, plain_text, square_box
from scripts.data.sources import ImageRecord


@final
class FakeClient:
    def __init__(self, data: bytes) -> None:
        self.data: bytes = data

    def get(self, url: str) -> bytes:
        _ = url
        return self.data


@final
class FakeSource:
    name: str = "fake"

    def __init__(self, record: ImageRecord) -> None:
        self.record: ImageRecord = record

    def search(self, query: str) -> Iterator[ImageRecord]:
        _ = query
        yield self.record


@final
class DatasetServiceTest(unittest.TestCase):
    @staticmethod
    def image_data() -> bytes:
        source = Image.new("RGB", (800, 600), "red")
        data = io.BytesIO()
        source.save(data, format="PNG")
        return data.getvalue()

    def test_pair_dimensions_and_palette(self) -> None:
        normal, pixel = make_pair(self.image_data(), 256, 32, 24)
        colors = pixel.getcolors(maxcolors=256)
        self.assertEqual(normal.size, (256, 256))
        self.assertEqual(pixel.size, (32, 32))
        assert colors is not None
        self.assertLessEqual(len(colors), 24)

    def test_plain_text_removes_attribution_html(self) -> None:
        self.assertEqual(plain_text('<a href="example">Jane Doe</a>'), "Jane Doe")

    @staticmethod
    def record() -> ImageRecord:
        return ImageRecord(
            provider="fake",
            source_id="1",
            title="Test",
            author="<b>Jane Doe</b>",
            license="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            source_page="https://example.test/item",
            original_url="https://example.test/original.png",
            download_url="https://example.test/thumb.png",
            width=800,
            height=600,
        )

    def test_fetch_writes_pair_and_attribution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            service = DatasetService(FakeClient(self.image_data()), output)
            self.assertEqual(service.fetch([FakeSource(self.record())], ["cat"], 1), 1)
            self.assertEqual(len(list((output / "input").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output / "target").glob("*.png"))), 1)
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["author"], "Jane Doe")
            self.assertEqual(row["provider"], "fake")

    def test_fetch_records_files_left_by_an_interrupted_run(self) -> None:
        data = self.image_data()
        item_id = hashlib.sha256(data).hexdigest()[:16]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "input").mkdir()
            (output / "input" / f"{item_id}.jpg").write_bytes(b"partial")
            service = DatasetService(FakeClient(data), output)
            self.assertEqual(service.fetch([FakeSource(self.record())], ["cat"], 1), 1)
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(next(csv.DictReader(file))["id"], item_id)

    def test_square_box_keeps_the_object_inside_the_image(self) -> None:
        # A tall box at the left edge grows to a square and shifts right.
        self.assertEqual(
            square_box((0.0, 0.1, 0.1, 0.9), (1000, 500)), (0, 50, 400, 450)
        )

    def test_fetch_saves_each_crop_of_one_image(self) -> None:
        crops = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)]
        records = [
            ImageRecord(
                **{**self.record().__dict__, "source_id": f"img/{i}", "crop": crop}
            )
            for i, crop in enumerate(crops)
        ]

        @final
        class Source:
            name: str = "fake"

            def search(self, query: str) -> Iterator[ImageRecord]:
                _ = query
                yield from records

        with tempfile.TemporaryDirectory() as directory:
            service = DatasetService(FakeClient(self.image_data()), Path(directory))
            self.assertEqual(service.fetch([Source()], ["cat"], 2), 2)

    def test_rerun_stops_at_per_query_total(self) -> None:
        records = [
            ImageRecord(
                **{
                    **self.record().__dict__,
                    "source_id": f"img/{i}",
                    "crop": (0.0, 0.0, 0.5 + i / 10, 1.0),
                }
            )
            for i in range(3)
        ]

        @final
        class Source:
            name: str = "fake"

            def search(self, query: str) -> Iterator[ImageRecord]:
                _ = query
                yield from records

        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(self.image_data())
            self.assertEqual(
                DatasetService(client, Path(directory)).fetch([Source()], ["cat"], 2), 2
            )
            self.assertEqual(
                DatasetService(client, Path(directory)).fetch([Source()], ["cat"], 2), 0
            )
