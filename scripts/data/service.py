"""Dataset-building service shared by the command-line script and tests."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import final

from PIL import Image, ImageOps, UnidentifiedImageError

from .sources import BinaryClient, ImageRecord, ImageSource

FIELDS = [
    "id",
    "provider",
    "source_id",
    "query",
    "title",
    "author",
    "license",
    "license_url",
    "source_page",
    "original_url",
    "input_path",
    "target_path",
    "sha256",
]


def plain_text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def square_box(
    crop: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Turns a normalised box into a square pixel box around the same centre.

    The shorter side grows so the whole object stays in view instead of being
    cut by a centre crop; the square then shifts to stay inside the image.

    Args:
        crop: Normalised (xmin, ymin, xmax, ymax).
        size: Image (width, height) in pixels.

    Returns:
        Pixel box (left, top, right, bottom) for Image.crop.
    """
    width, height = size
    x0, y0, x1, y1 = (
        crop[0] * width,
        crop[1] * height,
        crop[2] * width,
        crop[3] * height,
    )
    side = min(max(x1 - x0, y1 - y0), width, height)
    left = min(max((x0 + x1 - side) / 2, 0), width - side)
    top = min(max((y0 + y1 - side) / 2, 0), height - side)
    return round(left), round(top), round(left + side), round(top + side)


def make_pair(
    data: bytes,
    input_size: int,
    pixel_size: int,
    colors: int,
    crop: tuple[float, float, float, float] | None = None,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if crop:
            image = image.crop(square_box(crop, image.size))
        normal = ImageOps.fit(image, (input_size, input_size), Image.Resampling.LANCZOS)
        pixel = ImageOps.fit(
            normal, (pixel_size, pixel_size), method=Image.Resampling.LANCZOS
        )
        pixel = pixel.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert(
            "RGB"
        )
    return normal, pixel


@final
class DatasetService:
    def __init__(
        self,
        client: BinaryClient,
        output: Path,
        input_size: int = 256,
        pixel_size: int = 32,
        colors: int = 24,
        min_side: int = 512,
        max_download_mb: int = 15,
        workers: int = 4,
    ) -> None:
        self.client: BinaryClient = client
        self.output: Path = output
        self.input_size: int = input_size
        self.pixel_size: int = pixel_size
        self.colors: int = colors
        self.min_side: int = min_side
        self.max_download_bytes: int = max_download_mb * 1024 * 1024
        self.workers: int = workers
        self.manifest: Path = output / "manifest.csv"
        (output / "input").mkdir(parents=True, exist_ok=True)
        (output / "target").mkdir(parents=True, exist_ok=True)
        # (provider, source_id), not URLs: one image can yield several crops.
        self.seen: set[tuple[str, str]] = set()
        # Ids, not files on disk: an interrupted run can leave files without rows.
        self.ids: set[str] = set()
        # Pairs already saved per (source, query), so per_query is a total.
        self.done: Counter[tuple[str, str]] = Counter()
        if self.manifest.exists():
            with self.manifest.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    self.seen.add((row["provider"], row["source_id"]))
                    self.ids.add(row["id"])
                    self.done[(row["provider"].split(":")[0], row["query"])] += 1

    def _eligible(self, record: ImageRecord) -> bool:
        # Crops are size-checked by their source; whole images by min_side.
        return (
            bool(record.original_url)
            and (record.provider, record.source_id) not in self.seen
            and (
                record.crop is not None
                or min(record.width, record.height) >= self.min_side
            )
        )

    def _download(
        self, record: ImageRecord
    ) -> tuple[bytes, Image.Image, Image.Image] | None:
        """Downloads one image and builds its pair; runs in a worker thread.

        Args:
            record: Image to download.

        Returns:
            (original bytes, normal image, pixel image), or None if it was skipped.
        """
        try:
            data = self.client.get(record.download_url)
            if len(data) > self.max_download_bytes:
                return None
            normal, pixel = make_pair(
                data, self.input_size, self.pixel_size, self.colors, record.crop
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            print(f"skip {record.title}: {error}")
            return None
        return data, normal, pixel

    def _save(
        self,
        record: ImageRecord,
        pair: tuple[bytes, Image.Image, Image.Image],
        query: str,
        writer: csv.DictWriter[str],
    ) -> bool:
        data, normal, pixel = pair
        digest = hashlib.sha256(data).hexdigest()
        if record.crop:
            digest_with_crop = f"{digest}{record.crop}".encode()
            item_id = hashlib.sha256(digest_with_crop).hexdigest()[:16]
        else:
            item_id = digest[:16]
        input_path = self.output / "input" / f"{item_id}.jpg"
        target_path = self.output / "target" / f"{item_id}.png"
        self.seen.add((record.provider, record.source_id))
        if item_id in self.ids:
            return False

        normal.save(input_path, quality=92, optimize=True)
        pixel.save(target_path, optimize=True)
        writer.writerow(
            {
                "id": item_id,
                "provider": record.provider,
                "source_id": record.source_id,
                "query": query,
                "title": record.title,
                "author": plain_text(record.author),
                "license": record.license,
                "license_url": record.license_url,
                "source_page": record.source_page,
                "original_url": record.original_url,
                "input_path": str(input_path.relative_to(self.output)),
                "target_path": str(target_path.relative_to(self.output)),
                "sha256": digest,
            }
        )
        self.ids.add(item_id)
        return True

    def fetch(
        self, sources: Sequence[ImageSource], queries: list[str], per_query: int
    ) -> int:
        total = 0
        with (
            self.manifest.open("a", newline="", encoding="utf-8") as file,
            ThreadPoolExecutor(self.workers) as pool,
        ):
            writer: csv.DictWriter[str] = csv.DictWriter(file, fieldnames=FIELDS)
            if file.tell() == 0:
                writer.writeheader()

            for source in sources:
                for query in queries:
                    saved = self.done[(source.name, query)]
                    # Searches stay serial; only downloads run in parallel.
                    records = islice(source.search(query), per_query * 20)
                    eligible = (r for r in records if self._eligible(r))
                    while saved < per_query:
                        batch = list(islice(eligible, self.workers))
                        if not batch:
                            break
                        for record, pair in zip(batch, pool.map(self._download, batch)):
                            if (
                                pair
                                and saved < per_query
                                and self._save(record, pair, query, writer)
                            ):
                                file.flush()
                                saved += 1
                                total += 1
                                print(f"[{source.name}:{query}] {saved}/{per_query}")
                    if saved < per_query:
                        print(
                            f"[{source.name}:{query}] only found {saved} eligible images"
                        )
        return total
