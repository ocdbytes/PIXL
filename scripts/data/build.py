"""Build the photo dataset from the queries in a JSON file.

Runs Open Images first (exact class and relationship queries, cropped to the
annotated boxes), then the web sources (free-text search). Both write pairs to
the same output folder and manifest, and every run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from .openimages import OpenImagesSource
from .service import DatasetService
from .sources import HttpClient, build_sources

# Theme name -> queries; themes only organise the file.
Groups = dict[str, list[str]]


class OpenImagesConfig(TypedDict):
    splits: list[str]
    per_query: int
    workers: int
    min_box: float
    captions: dict[str, str]
    classes: Groups
    relationships: dict[str, list[list[str]]]


class WebConfig(TypedDict):
    sources: list[str]
    per_query: int
    workers: int
    min_side: int
    queries: Groups


class Config(TypedDict, total=False):
    output: str
    open_images: OpenImagesConfig
    web: WebConfig


@dataclass
class Args(argparse.Namespace):
    config: Path = Path("scripts/data/queries.json")
    user_agent: str = ""
    only: str | None = None


def flatten(groups: Groups) -> list[str]:
    """Joins themed query lists, dropping repeats and keeping the first order."""
    return list(dict.fromkeys(query for group in groups.values() for query in group))


def load_config(path: Path) -> Config:
    """Reads and checks the query file before anything is downloaded.

    Args:
        path: JSON query file.

    Returns:
        The parsed config.

    Raises:
        ValueError: If a required setting is missing or out of range.
    """
    config = cast(Config, json.loads(path.read_text(encoding="utf-8")))
    if "output" not in config:
        raise ValueError(f"{path}: missing 'output'")
    sections = [config.get("open_images"), config.get("web")]
    for name, section in zip(("open_images", "web"), sections):
        if section is None:
            continue
        for key in ("per_query", "workers"):
            if not isinstance(section.get(key), int) or section[key] <= 0:
                raise ValueError(f"{path}: {name}.{key} must be a positive integer")
    if "open_images" in config:
        open_images = config["open_images"]
        if not 0 < open_images["min_box"] <= 1:
            raise ValueError(f"{path}: open_images.min_box must be in (0, 1]")
        for group in open_images["relationships"].values():
            for triplet in group:
                if len(triplet) != 3:
                    raise ValueError(f"{path}: relationship {triplet} needs 3 parts")
    if "web" in config:
        unknown = set(config["web"]["sources"]) - {"wikimedia", "openverse"}
        if unknown:
            raise ValueError(f"{path}: unknown web sources {sorted(unknown)}")
    return config


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--config", type=Path, default=Args.config)
    _ = parser.add_argument(
        "--user-agent",
        required=True,
        help='descriptive agent with contact, e.g. "PixlDataset/0.1 (you@example.com)"',
    )
    _ = parser.add_argument(
        "--only", choices=("open_images", "web"), help="run one section only"
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = Path(config["output"])
    client = HttpClient(args.user_agent)
    total = 0

    open_images = config.get("open_images")
    if open_images and args.only in (None, "open_images"):
        relationships = [
            (subject, verb, other)
            for group in open_images["relationships"].values()
            for subject, verb, other in group
        ]
        source = OpenImagesSource(
            client,
            cache=output / "openimages-cache",
            splits=open_images["splits"],
            classes=flatten(open_images["classes"]),
            relationships=list(dict.fromkeys(relationships)),
            captions=open_images["captions"],
            limit=open_images["per_query"] * 3,
            min_box=open_images["min_box"],
        )
        found = sum(len(records) for records in source.records.values())
        print(
            f"Open Images: {len(source.queries)} queries, {found} candidate crops,"
            f" up to {open_images['per_query']} pairs each"
        )
        service = DatasetService(client, output, workers=open_images["workers"])
        total += service.fetch([source], source.queries, open_images["per_query"])

    web = config.get("web")
    if web and args.only in (None, "web"):
        queries = flatten(web["queries"])
        print(
            f"Web: {len(queries)} queries x {len(web['sources'])} sources,"
            f" up to {web['per_query']} pairs each"
        )
        service = DatasetService(
            client, output, min_side=web["min_side"], workers=web["workers"]
        )
        total += service.fetch(
            build_sources(web["sources"], client), queries, web["per_query"]
        )

    print(f"Saved {total} new pairs to {output}")


if __name__ == "__main__":
    main()
