"""Open Images V7 source: captioned crops of annotated objects and relationships.

Open Images has no free-text search. Queries are exact class names ("Dog") and
(subject, relationship, object) triplets ("Man", "ride", "Horse"), which become
captions such as "dog" and "man riding horse". Each record crops its image to
the annotated box, or to the union of both boxes for a relationship.

Annotation CSVs are downloaded once into a cache folder. Image licences come
from the per-image CSV; Google lists them as CC BY 2.0 without warranty, so the
manifest keeps the original Flickr URL, landing page and author for checking.
"""

from __future__ import annotations

import csv
import difflib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, final

from .sources import ImageRecord

ANNOTATIONS = "https://storage.googleapis.com/openimages"
IMAGES = "https://open-images-dataset.s3.amazonaws.com"
CLASSES = "v7/oidv7-class-descriptions-boxable.csv"
ATTRIBUTES = "v6/oidv6-attributes-description.csv"
SPLITS = {
    "validation": (
        "2018_04/validation/validation-images-with-rotation.csv",
        "v5/validation-annotations-bbox.csv",
        "v6/oidv6-validation-annotations-vrd.csv",
    ),
    "test": (
        "2018_04/test/test-images-with-rotation.csv",
        "v5/test-annotations-bbox.csv",
        "v6/oidv6-test-annotations-vrd.csv",
    ),
    "train": (
        "2018_04/train/train-images-boxable-with-rotation.csv",
        "v6/oidv6-train-annotations-bbox.csv",
        "v6/oidv6-train-annotations-vrd.csv",
    ),
}
# Relationship labels as caption phrases, e.g. "man riding horse".
VERBS = {
    "at": "at",
    "on": "on",
    "under": "under",
    "inside_of": "inside",
    "contain": "containing",
    "hang": "hanging from",
    "holds": "holding",
    "wears": "wearing",
    "ride": "riding",
    "plays": "playing",
    "interacts_with": "with",
    "skateboard": "riding",
    "surf": "surfing on",
    "ski": "skiing on",
    "snowboard": "snowboarding on",
    "dance": "dancing with",
    "kiss": "kissing",
    "hug": "hugging",
    "holding_hands": "holding hands with",
    "highfive": "high-fiving",
    "handshake": "shaking hands with",
    "talk_on_phone": "talking on",
    "throw": "throwing",
    "catch": "catching",
    "kick": "kicking",
    "hits": "hitting",
    "drink": "drinking",
    "eat": "eating",
    "cut": "cutting",
    "read": "reading",
}
# Attributes used with the "is" relationship, as adjectives, e.g. "running man".
ADJECTIVES = {
    "Transparent": "transparent",
    "Plastic": "plastic",
    "(made of)Textile": "textile",
    "(made of)Leather": "leather",
    "Wooden": "wooden",
    "Stand": "standing",
    "Walk": "walking",
    "Run": "running",
    "Jump": "jumping",
    "Sit": "sitting",
    "Lay": "lying",
    "Smile": "smiling",
    "Cry": "crying",
    "Sing": "singing",
    "Talk": "talking",
}
# Rotation is documented as 0, 90, 180 or 270; images without a value are
# skipped too, because a box drawn on a rotated image would crop the wrong area.
UPRIGHT = {"0", "0.0"}

Box = tuple[float, float, float, float]


class FileClient(Protocol):
    def save(self, url: str, path: Path) -> None: ...


def license_name(url: str) -> str:
    """Maps a Creative Commons BY licence URL to a short name.

    Args:
        url: Licence URL, e.g. "https://creativecommons.org/licenses/by/2.0/".

    Returns:
        A name such as "CC BY 2.0", or "" for any other licence.
    """
    match = re.search(r"creativecommons\.org/licenses/by/(\d\.\d)", url)
    return f"CC BY {match.group(1)}" if match else ""


def _check(name: str, known: dict[str, str], kind: str) -> str:
    if name in known:
        return known[name]
    close = difflib.get_close_matches(name, list(known), n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise ValueError(f"unknown Open Images {kind} {name!r}{hint}")


@final
class OpenImagesSource:
    name: str = "openimages"

    def __init__(
        self,
        client: FileClient,
        cache: Path,
        splits: list[str],
        classes: list[str],
        relationships: list[tuple[str, str, str]],
        captions: dict[str, str],
        limit: int,
        min_box: float,
    ) -> None:
        """Validates the queries and indexes matching boxes from the annotations.

        Args:
            client: Downloads annotation CSVs that are not cached yet.
            cache: Folder for the annotation CSVs.
            splits: Open Images splits to read: validation, test or train.
            classes: Exact class names, e.g. "Dog".
            relationships: (subject, relationship, object or attribute) triplets.
            captions: Caption overrides for class names, e.g. Cattle -> cow.
            limit: Most candidate crops kept per query.
            min_box: Smallest crop width and height, as a fraction of the image.

        Raises:
            ValueError: If a split, class, relationship or attribute is unknown.
        """
        self.client: FileClient = client
        self.cache: Path = cache
        self.captions: dict[str, str] = captions
        self.limit: int = limit
        self.min_box: float = min_box
        for split in splits:
            _ = _check(split, {s: s for s in SPLITS}, "split")
        self.splits: list[str] = splits
        self.records: dict[str, list[ImageRecord]] = self._index(classes, relationships)

    @property
    def queries(self) -> list[str]:
        """Captions in config order; each is a query for search()."""
        return list(self.records)

    def search(self, query: str) -> Iterator[ImageRecord]:
        yield from self.records.get(query, [])

    def caption(self, name: str) -> str:
        """Turns a class name into caption text, e.g. "Orange (fruit)" -> "orange"."""
        return self.captions.get(name) or re.sub(r"\s*\(.*?\)", "", name).lower()

    def _file(self, path: str) -> Path:
        local = self.cache / Path(path).name
        if not local.exists():
            print(f"downloading Open Images {path}")
            self.cache.mkdir(parents=True, exist_ok=True)
            self.client.save(f"{ANNOTATIONS}/{path}", local)
        return local

    def _rows(self, path: str) -> Iterator[dict[str, str]]:
        with self._file(path).open(newline="", encoding="utf-8") as file:
            yield from csv.DictReader(file)

    def _names(self, path: str) -> dict[str, str]:
        """Reads a label description CSV as display name -> label id."""
        with self._file(path).open(newline="", encoding="utf-8") as file:
            return {name: mid for mid, name in csv.reader(file) if mid != "LabelName"}

    def _keep(self, box: Box) -> bool:
        return min(box[2] - box[0], box[3] - box[1]) >= self.min_box

    def _index(
        self, classes: list[str], relationships: list[tuple[str, str, str]]
    ) -> dict[str, list[ImageRecord]]:
        names = self._names(CLASSES)
        attributes = self._names(ATTRIBUTES)
        wanted: dict[str, str] = {
            _check(name, names, "class"): self.caption(name) for name in classes
        }
        triplets: dict[tuple[str, str, str], str] = {}
        for subject, verb, other in relationships:
            if verb == "is":
                key = (
                    _check(subject, names, "class"),
                    verb,
                    _check(other, attributes, "attribute"),
                )
                triplets[key] = f"{ADJECTIVES[other]} {self.caption(subject)}"
            else:
                _ = _check(verb, {v: v for v in VERBS}, "relationship")
                key = (
                    _check(subject, names, "class"),
                    verb,
                    _check(other, names, "class"),
                )
                triplets[key] = (
                    f"{self.caption(subject)} {VERBS[verb]} {self.caption(other)}"
                )

        # Captions in config order; one crop per image per caption.
        found: dict[str, list[tuple[str, str, Box]]] = {
            c: [] for c in [*wanted.values(), *triplets.values()]
        }
        used: set[tuple[str, str]] = set()

        def add(caption: str, split: str, image_id: str, box: Box) -> None:
            if (
                len(found[caption]) < self.limit
                and (caption, image_id) not in used
                and self._keep(box)
            ):
                used.add((caption, image_id))
                found[caption].append((split, image_id, box))

        for split in self.splits:
            _images, boxes, vrd = SPLITS[split]
            if wanted:
                for row in self._rows(boxes):
                    caption = wanted.get(row["LabelName"])
                    if caption and row["IsGroupOf"] != "1":
                        box = (
                            float(row["XMin"]),
                            float(row["YMin"]),
                            float(row["XMax"]),
                            float(row["YMax"]),
                        )
                        add(caption, split, row["ImageID"], box)
            if triplets:
                for row in self._rows(vrd):
                    key = (
                        row["LabelName1"],
                        row["RelationshipLabel"],
                        row["LabelName2"],
                    )
                    caption = triplets.get(key)
                    if caption:
                        box = (
                            min(float(row["XMin1"]), float(row["XMin2"])),
                            min(float(row["YMin1"]), float(row["YMin2"])),
                            max(float(row["XMax1"]), float(row["XMax2"])),
                            max(float(row["YMax1"]), float(row["YMax2"])),
                        )
                        add(caption, split, row["ImageID"], box)

        needed = {
            (split, image_id)
            for crops in found.values()
            for split, image_id, _ in crops
        }
        info: dict[tuple[str, str], dict[str, str]] = {}
        for split in self.splits:
            for row in self._rows(SPLITS[split][0]):
                key = (split, row["ImageID"])
                if (
                    key in needed
                    and row["Rotation"] in UPRIGHT
                    and license_name(row["License"])
                ):
                    info[key] = row

        records: dict[str, list[ImageRecord]] = {caption: [] for caption in found}
        for caption, crops in found.items():
            for split, image_id, box in crops:
                row = info.get((split, image_id))
                if not row:
                    continue
                records[caption].append(
                    ImageRecord(
                        provider=self.name,
                        source_id=f"{image_id}/{caption}",
                        title=row["Title"],
                        author=row["Author"],
                        license=license_name(row["License"]),
                        license_url=row["License"],
                        source_page=row["OriginalLandingURL"],
                        original_url=row["OriginalURL"],
                        download_url=f"{IMAGES}/{split}/{image_id}.jpg",
                        width=0,
                        height=0,
                        crop=box,
                    )
                )
        return records
