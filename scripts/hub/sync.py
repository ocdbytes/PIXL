"""Pushes the local training data to the public Hugging Face dataset repo, or pulls it back.

This script manages one config per local manifest:

- sprites: downloaded-data/manifest.csv rows, with their sprite image.
- scenes: scene-data/manifest.csv rows, with their scene image, made from
  scene-data/tokens.npy.

Each config has a train and a test split. Sprites use a stable hash of each
group key, keeping grouped colour variants and animation frames together.
Scenes keep the split assigned by the generator. A pull writes both splits back
into the folder's manifest.csv.

Sprite images retain their original file bytes. Scenes are kept locally as
palette tokens: a push encodes each as a PNG, and a pull turns the PNGs back into
the same tokens when using the same palette.
scripts/hub/card.md becomes the dataset card.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import PIL.Image
from datasets import Dataset, DatasetDict, Image, load_dataset
from huggingface_hub import DatasetCard

from pixl.preprocess import decode, encode, load_palette

REPO = "ocdbytes/pixl-data"
CARD = Path("scripts/hub/card.md")


Split = Callable[[dict[str, str]], str]


def hashed_split(group: Callable[[dict[str, str]], str], test_share: float) -> Split:
    """Makes a split function that keeps each group of rows in one split.

    Near-copies, such as a sprite's colour variants or the frames of one
    animation, get the same key, so they never land in both splits.

    Args:
        group: Returns a row's group key; rows with the same key share a split.
        test_share: Fraction of groups in the test split.

    Returns:
        A function that returns "test" for a row if the first 8 hex digits of
        the SHA-256 of its group key, as a fraction of 16**8, are below
        test_share, and "train" otherwise. It depends only on the key, so the
        split never changes and new rows never move old ones.
    """

    def split(row: dict[str, str]) -> str:
        digest = hashlib.sha256(group(row).encode()).hexdigest()
        return "test" if int(digest[:8], 16) / 16**8 < test_share else "train"

    return split


def sprite_group(row: dict[str, str]) -> str:
    """Groups the frames of one animation, and one tile's colour variants.

    Args:
        row: Sprite manifest row.

    Returns:
        The path without its frame number for animation frames, e.g.
        ".../Frames/cherry/cherry" for "cherry-3.png"; the path without its
        variant number and _new/_old for Dungeon Crawl, e.g. ".../long_sword"
        for "long_sword_5_new.png"; otherwise the pack and file name, shared by
        the 1-Bit Pack's colored and monochrome variants.
    """
    if "/Frames/" in row["path"]:
        return re.sub(r"[-_]?\d+\.png$", "", row["path"])
    if row["pack"].startswith("dcss_"):
        return re.sub(r"(_\d+)*(_new|_old)?(_\d+)*\.png$", "", row["path"])
    return f"{row['pack']}/{Path(row['path']).name}"


@dataclass(frozen=True)
class Part:
    """One local manifest and the image files it points to.

    Attributes:
        name: Config name in the dataset repo.
        root: Folder holding manifest.csv; manifest paths are relative to it.
        images: Image column in the repo -> manifest column with the file path.
        split: Returns the split ("train" or "test") of a manifest row.
        tokens: For a part kept as palette tokens: the .npy file in root whose
            row i holds the image of manifest row i, as the repo column "image".
    """

    name: str
    root: Path
    images: dict[str, str]
    split: Split
    tokens: str = ""

    @property
    def image_columns(self) -> list[str]:
        """Image columns in the repo: those read from files, then "image" for a tokens part."""
        return [*self.images, "image"] if self.tokens else list(self.images)


PARTS = (
    Part(
        "sprites",
        Path("downloaded-data"),
        {"image": "path"},
        split=hashed_split(sprite_group, test_share=0.1),
    ),
    # Scenes are built from train or test sprites, so the generator sets the split.
    Part(
        "scenes",
        Path("scene-data"),
        {},
        split=lambda row: row["split"],
        tokens="tokens.npy",
    ),
)
PART = {part.name: part for part in PARTS}


@dataclass
class Args(argparse.Namespace):
    command: str = ""
    repo: str = REPO
    only: list[str] | None = None


def push(repo: str, parts: list[Part]) -> None:
    """Uploads parts, replacing what the repo held for them before.

    Args:
        repo: Dataset repo id; created as private if it does not exist.
        parts: Parts to upload.

    Raises:
        SystemExit: If a part's manifest is missing.
    """
    for part in parts:
        manifest = part.root.joinpath("manifest.csv")
        if not manifest.exists():
            raise SystemExit(f"{manifest} not found: build or pull {part.name} first")
    for part in parts:
        with part.root.joinpath("manifest.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            rows = list(reader)
        grids = np.load(part.root.joinpath(part.tokens), mmap_mode="r") if part.tokens else None
        palette = load_palette()
        splits = DatasetDict()
        for name in ("train", "test"):
            chosen = [i for i, row in enumerate(rows) if part.split(row) == name]
            columns: dict[str, list[Any]] = {
                field: [rows[i][field] for i in chosen] for field in fields
            }
            for image, path_column in part.images.items():
                columns[image] = [
                    str(part.root.joinpath(rows[i][path_column])) for i in chosen
                ]
            if grids is not None:
                columns["image"] = [
                    {"bytes": png(np.asarray(grids[i]), palette), "path": None}
                    for i in chosen
                ]
            dataset = Dataset.from_dict(columns)
            for image in part.image_columns:
                dataset = dataset.cast_column(image, Image())
            splits[name] = dataset
        sizes = ", ".join(f"{name} {len(data)}" for name, data in splits.items())
        print(f"pushing {part.name} ({sizes}) to {repo}")
        _ = splits.push_to_hub(repo, config_name=part.name)
    push_card(repo)


def png(tokens: np.ndarray, palette: np.ndarray) -> bytes:
    """A palette token grid as PNG file bytes."""
    buffer = io.BytesIO()
    decode(tokens, palette).save(buffer, format="PNG")
    return buffer.getvalue()


def push_card(repo: str) -> None:
    """Replaces the dataset card with CARD, keeping the config metadata.

    push_to_hub writes the configs and features into the card's YAML block; the
    loader needs them, so CARD's YAML is merged into that block instead of
    replacing it.

    Args:
        repo: Dataset repo id.
    """
    card = DatasetCard.load(repo, repo_type="dataset")
    template = DatasetCard(CARD.read_text(encoding="utf-8"))
    for key, value in template.data.to_dict().items():
        setattr(card.data, key, value)
    card.text = template.text
    _ = card.push_to_hub(repo, repo_type="dataset")


def pull(repo: str, parts: list[Part]) -> None:
    """Writes parts' images and manifests back to their local folders.

    A part whose manifest already exists is skipped. Both splits go into one
    manifest in loader order. The manifest is written after all images/tokens:
    an interrupted pull leaves no manifest, and the next pull starts that part
    again, possibly rewriting files left by the incomplete pull.

    Args:
        repo: Dataset repo id.
        parts: Parts to download.
    """
    for part in parts:
        manifest = part.root.joinpath("manifest.csv")
        if manifest.exists():
            print(f"{manifest} exists; skipping {part.name} (delete it to pull again)")
            continue
        rows: list[dict[str, str]] = []
        fields: list[str] = []
        grids: list[np.ndarray] = []
        palette = load_palette()
        for name, dataset in load_dataset(repo, part.name).items():
            for image in part.image_columns:
                dataset = dataset.cast_column(image, Image(decode=False))
            fields = [c for c in dataset.column_names if c not in part.image_columns]
            print(f"writing {len(dataset)} {part.name} ({name}) to {part.root}")
            for raw_row in dataset:
                row = cast(dict[str, Any], raw_row)
                for image, path_column in part.images.items():
                    path = part.root.joinpath(row[path_column])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _ = path.write_bytes(row[image]["bytes"])
                if part.tokens:
                    picture = PIL.Image.open(io.BytesIO(row["image"]["bytes"]))
                    grids.append(encode(picture, palette))
                rows.append({field: row[field] for field in fields})
        if part.tokens:
            np.save(part.root.joinpath(part.tokens), np.stack(grids))
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument(
        "command", choices=("push", "pull", "card"), help="card: push only the card"
    )
    _ = parser.add_argument("--repo", default=Args.repo)
    _ = parser.add_argument(
        "--only",
        nargs="+",
        choices=list(PART),
        help="push or pull only these configs, e.g. --only scenes",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    parts = [PART[name] for name in args.only] if args.only else list(PARTS)
    if args.command == "push":
        push(args.repo, parts)
    elif args.command == "card":
        push_card(args.repo)
    else:
        pull(args.repo, parts)


if __name__ == "__main__":
    main()
