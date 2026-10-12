"""Pushes the local training data to the public Hugging Face dataset repo, or pulls it back.

The repo has one config per local manifest:

- photos: pixel-image-data/clean.csv rows (the manifest after scripts.refine
  dropped unsuitable photos), with their input and target images.
- sprites: downloaded-data/manifest.csv rows, with their sprite image.

Each config has a train and a test split. split() puts a row in test when the
hash of its group key falls below the part's test share, so the split never
changes between runs and near-copies (crops of one photo, colour variants of one
tile, frames of one animation) always land in the same split. A pull writes both
splits back into the folder's manifest.csv.

Images are stored as their original file bytes, so a pull writes back exactly the
files that were pushed. src/queries.json is uploaded too, as a record of the
queries that built the photos, and scripts/hub/card.md becomes the dataset card.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, Image, load_dataset
from huggingface_hub import DatasetCard, HfApi

REPO = "ocdbytes/pixl-data"
QUERIES = Path("src/queries.json")
CARD = Path("scripts/hub/card.md")


Split = Callable[[dict[str, str]], str]


def hashed_split(group: Callable[[dict[str, str]], str], test_share: float) -> Split:
    """Makes a split function that keeps each group of rows in one split.

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
        ".../Frames/cherry/cherry" for "cherry-3.png"; otherwise the pack and
        file name, shared by the 1-Bit Pack's colored and monochrome variants.
    """
    if "/Frames/" in row["path"]:
        return re.sub(r"[-_]?\d+\.png$", "", row["path"])
    return f"{row['pack']}/{Path(row['path']).name}"


@dataclass(frozen=True)
class Part:
    """One local manifest and the image files it points to.

    Attributes:
        name: Config name in the dataset repo.
        root: Folder holding manifest.csv; manifest paths are relative to it.
        images: Image column in the repo -> manifest column with the file path.
        split: Returns the split ("train" or "test") of a manifest row.
        upload: CSV in root whose rows are pushed; pulls write manifest.csv.
    """

    name: str
    root: Path
    images: dict[str, str]
    split: Split
    upload: str = "manifest.csv"


PARTS = (
    Part(
        "photos",
        Path("pixel-image-data"),
        {"input": "input_path", "target": "target_path"},
        # All crops of one source photo share a split.
        split=hashed_split(lambda row: row["sha256"], test_share=0.02),
        upload="clean.csv",
    ),
    Part(
        "sprites",
        Path("downloaded-data"),
        {"image": "path"},
        split=hashed_split(sprite_group, test_share=0.1),
    ),
    # Scenes are built from train or test sprites, so the generator sets the split.
    Part(
        "scenes", Path("scene-data"), {"image": "path"}, split=lambda row: row["split"]
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
        SystemExit: If a part's upload CSV is missing, e.g. refine has not run.
    """
    for part in parts:
        upload = part.root.joinpath(part.upload)
        if not upload.exists():
            raise SystemExit(f"{upload} not found: run src.setup_data to build it")
    for part in parts:
        with part.root.joinpath(part.upload).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            rows = list(reader)
        splits = DatasetDict()
        for name in ("train", "test"):
            chosen = [row for row in rows if part.split(row) == name]
            columns = {field: [row[field] for row in chosen] for field in fields}
            for image, path_column in part.images.items():
                columns[image] = [
                    str(part.root.joinpath(row[path_column])) for row in chosen
                ]
            dataset = Dataset.from_dict(columns)
            for image in part.images:
                dataset = dataset.cast_column(image, Image())
            splits[name] = dataset
        sizes = ", ".join(f"{name} {len(data)}" for name, data in splits.items())
        print(f"pushing {part.name} ({sizes}) to {repo}")
        _ = splits.push_to_hub(repo, config_name=part.name, private=True)
    _ = HfApi().upload_file(
        path_or_fileobj=str(QUERIES),
        path_in_repo="queries.json",
        repo_id=repo,
        repo_type="dataset",
    )
    push_card(repo)


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

    A part whose manifest already exists is skipped, so local data is never
    overwritten. Both splits go into one manifest, train rows first. The manifest
    is written last: an interrupted pull leaves no manifest, and the next pull
    starts that part again.

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
        for name, dataset in load_dataset(repo, part.name).items():
            for image in part.images:
                dataset = dataset.cast_column(image, Image(decode=False))
            fields = [c for c in dataset.column_names if c not in part.images]
            print(f"writing {len(dataset)} {part.name} ({name}) to {part.root}")
            for row in dataset:
                for image, path_column in part.images.items():
                    path = part.root.joinpath(row[path_column])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _ = path.write_bytes(row[image]["bytes"])
                rows.append({field: row[field] for field in fields})
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
