"""Sets up the training data in three steps.

1. Get the photo pairs and sprites, from one of two sources (--source):
   - hub (default): pull them from the private Hugging Face dataset repo
     (scripts.hub.sync). Takes minutes; skips data that is already on disk.
   - build: download photo pairs from Open Images, Wikimedia and Openverse for
     the queries in src/queries.json (scripts.data.build). Takes 1-2 days and
     resumes if stopped. Sprites must already be in downloaded-data.
2. Label sprites that have no label yet, keeping existing labels, and gather
   every sprite into one folder (scripts.labeler.label).
3. Drop web photo pairs whose photo does not match its query and write
   clean.csv next to the photos (scripts.refine.clip_filter).

Each step runs as its own process, so one step's model is freed before the next.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

QUERIES = Path("src/queries.json")
# Built by hand from the downloaded sprite packs; the build source does not fetch them.
SPRITES = Path("downloaded-data/manifest.csv")


@dataclass
class Args(argparse.Namespace):
    source: str = "hub"
    user_agent: str = ""
    label_model: str = "clip"


def run(module: str, *args: str) -> None:
    """Runs one step as `python -m module args`.

    Args:
        module: Step module, e.g. "scripts.data.build".
        *args: Command-line arguments for the step.

    Raises:
        SystemExit: If the step fails, so later steps never see partial output.
    """
    print(f"\n== {module}", flush=True)
    if subprocess.run([sys.executable, "-m", module, *args]).returncode:
        raise SystemExit(f"setup stopped: {module} failed")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument(
        "--source",
        choices=("hub", "build"),
        default=Args.source,
        help="where step 1 gets the data",
    )
    _ = parser.add_argument(
        "--user-agent",
        help='needed with --source build: descriptive agent with contact, e.g. "PixlDataset/0.1 (you@example.com)"',
    )
    _ = parser.add_argument(
        "--label-model",
        choices=("clip", "florence"),
        default=Args.label_model,
        help="model for sprites that have no label yet",
    )
    args = parser.parse_args(namespace=Args())
    if args.source == "build" and not args.user_agent:
        parser.error("--source build needs --user-agent")
    return args


def main() -> None:
    args = parse_args()
    output = json.loads(QUERIES.read_text(encoding="utf-8"))["output"]

    if args.source == "hub":
        run("scripts.hub.sync", "pull")
    else:
        # Checked first: the build alone takes over a day.
        if not SPRITES.exists():
            raise SystemExit(f"{SPRITES} not found: add the sprite packs first")
        run(
            "scripts.data.build",
            "--config",
            str(QUERIES),
            "--user-agent",
            args.user_agent,
        )
    run("scripts.labeler.label", "--model", args.label_model)
    run("scripts.refine.clip_filter", "--data", output)
    print(
        f"\nDone: photo pairs in {output}/clean.csv, sprites in downloaded-data/labeled"
    )


if __name__ == "__main__":
    main()
