"""Pushes the local training data to the public Hugging Face dataset repo, or pulls it back.

The repo has one config per local manifest:

- photos: pixel-image-data/clean.csv rows (the manifest after scripts.refine
  dropped unsuitable photos), with their input and target images.
- sprites: downloaded-data/manifest.csv rows, with their sprite image.

A pull writes each config back as its folder's manifest.csv.

Images are stored as their original file bytes, so a pull writes back exactly the
files that were pushed. src/queries.json is uploaded too, as a record of the
queries that built the photos, and scripts/hub/card.md becomes the dataset card.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Image, load_dataset
from huggingface_hub import DatasetCard, HfApi

REPO = "ocdbytes/pixl-data"
QUERIES = Path("src/queries.json")
CARD = Path("scripts/hub/card.md")


@dataclass(frozen=True)
class Part:
    """One local manifest and the image files it points to.

    Attributes:
        name: Config name in the dataset repo.
        root: Folder holding manifest.csv; manifest paths are relative to it.
        images: Image column in the repo -> manifest column with the file path.
        upload: CSV in root whose rows are pushed; pulls write manifest.csv.
    """

    name: str
    root: Path
    images: dict[str, str]
    upload: str = "manifest.csv"


PARTS = (
    Part(
        "photos",
        Path("pixel-image-data"),
        {"input": "input_path", "target": "target_path"},
        upload="clean.csv",
    ),
    Part("sprites", Path("downloaded-data"), {"image": "path"}),
)


@dataclass
class Args(argparse.Namespace):
    command: str = ""
    repo: str = REPO


def push(repo: str) -> None:
    """Uploads every part, replacing what the repo held before.

    Args:
        repo: Dataset repo id; created as private if it does not exist.

    Raises:
        SystemExit: If a part's upload CSV is missing, e.g. refine has not run.
    """
    for part in PARTS:
        upload = part.root.joinpath(part.upload)
        if not upload.exists():
            raise SystemExit(
                f"{upload} not found: run scripts.refine.clip_filter first"
            )
    for part in PARTS:
        with part.root.joinpath(part.upload).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = {field: [] for field in reader.fieldnames or []}
            for row in reader:
                for field, values in columns.items():
                    values.append(row[field])
        for image, path_column in part.images.items():
            columns[image] = [str(part.root.joinpath(p)) for p in columns[path_column]]
        dataset = Dataset.from_dict(columns)
        for image in part.images:
            dataset = dataset.cast_column(image, Image())
        print(f"pushing {len(dataset)} {part.name} to {repo}")
        _ = dataset.push_to_hub(repo, config_name=part.name, private=True)
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


def pull(repo: str) -> None:
    """Writes every part's images and manifest back to their local folders.

    A part whose manifest already exists is skipped, so local data is never
    overwritten. The manifest is written last: an interrupted pull leaves no
    manifest, and the next pull starts that part again.

    Args:
        repo: Dataset repo id.
    """
    for part in PARTS:
        manifest = part.root.joinpath("manifest.csv")
        if manifest.exists():
            print(f"{manifest} exists; skipping {part.name} (delete it to pull again)")
            continue
        dataset = load_dataset(repo, part.name, split="train")
        for image in part.images:
            dataset = dataset.cast_column(image, Image(decode=False))
        print(f"writing {len(dataset)} {part.name} to {part.root}")
        for row in dataset:
            for image, path_column in part.images.items():
                path = part.root.joinpath(row[path_column])
                path.parent.mkdir(parents=True, exist_ok=True)
                _ = path.write_bytes(row[image]["bytes"])
        fields = [c for c in dataset.column_names if c not in part.images]
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dataset.remove_columns(list(part.images)))


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument(
        "command", choices=("push", "pull", "card"), help="card: push only the card"
    )
    _ = parser.add_argument("--repo", default=Args.repo)
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    if args.command == "push":
        push(args.repo)
    elif args.command == "card":
        push_card(args.repo)
    else:
        pull(args.repo)


if __name__ == "__main__":
    main()
