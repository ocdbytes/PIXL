"""Build licensed normal-image and synthetic pixel-art training pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .service import DatasetService
from .sources import HttpClient, build_sources


@dataclass
class Args(argparse.Namespace):
    source: list[str] | None = None
    query: list[str] = field(default_factory=list)
    per_query: int = 100
    output: Path = Path("pixel-image-data")
    input_size: int = 256
    pixel_size: int = 32
    colors: int = 24
    min_side: int = 512
    max_download_mb: int = 15
    delay: float = 0.1
    workers: int = 4
    user_agent: str = ""


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--source",
        action="append",
        choices=("wikimedia", "openverse"),
        help="repeat to select sources; defaults to both",
    )
    _ = parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="search term; repeat for balanced categories",
    )
    _ = parser.add_argument(
        "--per-query", type=int, default=100, help="images to save per source and query"
    )
    _ = parser.add_argument("--output", type=Path, default=Path("pixel-image-data"))
    _ = parser.add_argument("--input-size", type=int, default=256)
    _ = parser.add_argument("--pixel-size", type=int, default=32)
    _ = parser.add_argument("--colors", type=int, default=24)
    _ = parser.add_argument("--min-side", type=int, default=512)
    _ = parser.add_argument("--max-download-mb", type=int, default=15)
    _ = parser.add_argument("--delay", type=float, default=0.1)
    _ = parser.add_argument(
        "--workers", type=int, default=4, help="parallel image downloads"
    )
    _ = parser.add_argument(
        "--user-agent",
        required=True,
        help='descriptive agent with contact, e.g. "PixlDataset/0.1 (you@example.com)"',
    )
    args = parser.parse_args(namespace=Args())
    if (
        min(
            args.per_query,
            args.input_size,
            args.pixel_size,
            args.colors,
            args.min_side,
            args.workers,
        )
        <= 0
    ):
        parser.error("numeric size/count arguments must be positive")
    if args.pixel_size > args.input_size:
        parser.error("--pixel-size cannot exceed --input-size")
    if not 2 <= args.colors <= 256:
        parser.error("--colors must be between 2 and 256")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    client = HttpClient(args.user_agent, args.delay)
    source_names = args.source or ["wikimedia", "openverse"]
    service = DatasetService(
        client=client,
        output=args.output,
        input_size=args.input_size,
        pixel_size=args.pixel_size,
        colors=args.colors,
        min_side=args.min_side,
        max_download_mb=args.max_download_mb,
        workers=args.workers,
    )
    total = service.fetch(
        build_sources(source_names, client), args.query, args.per_query
    )
    print(f"Saved {total} pairs to {args.output}")


if __name__ == "__main__":
    main()
