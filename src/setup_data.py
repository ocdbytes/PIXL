"""Rebuilds the training dataset from its sources, for changing the dataset.

Training does not need this: src/data.py loads the dataset from Hugging Face.
Run it after changing the queries, sprite packs or labels:

1. Download photo pairs from Open Images, Wikimedia and Openverse for the
   queries in src/queries.json (scripts.data.build). Takes 1-2 days the first
   time and resumes if stopped. Sprites must already be in downloaded-data.
2. Label sprites that have no label yet, keeping existing labels, and gather
   every sprite into one folder (scripts.labeler.label).
3. Drop photo pairs with nudity or graphic content, web photos that do not
   match their query, and CC BY photos that cannot be credited; write
   clean.csv and review sheets of the content drops (scripts.refine.clip_filter).
4. Build captioned 32x32 scenes from the sprites, with review sheets of the
   sprite roles and sample scenes (scripts.scenes.generate).

Each step runs as its own process, so one step's model is freed before the next.
Uploading stays a separate, manual step: every push is public at once, so check
the review sheets first, then run scripts.hub.sync push.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

QUERIES = Path("src/queries.json")
# Built by hand from the downloaded sprite packs; the download step does not fetch them.
SPRITES = Path("downloaded-data/manifest.csv")


@dataclass
class Args(argparse.Namespace):
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
        "--user-agent",
        required=True,
        help='descriptive agent with contact, e.g. "PixlDataset/0.1 (you@example.com)"',
    )
    _ = parser.add_argument(
        "--label-model",
        choices=("clip", "florence"),
        default=Args.label_model,
        help="model for sprites that have no label yet",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    # Checked first: the download alone takes over a day.
    if not SPRITES.exists():
        raise SystemExit(
            f"{SPRITES} not found: add the sprite packs, or get the current ones"
            " with `uv run python -m scripts.hub.sync pull`"
        )
    output = json.loads(QUERIES.read_text(encoding="utf-8"))["output"]

    run("scripts.data.build", "--config", str(QUERIES), "--user-agent", args.user_agent)
    run("scripts.labeler.label", "--model", args.label_model)
    run("scripts.refine.clip_filter", "--data", output, "--sheets")
    run("scripts.scenes.generate", "--sheets")
    print(
        f"\nDone. Check the review sheets in {output}/review/ and scene-data/review/,"
        " then upload with `uv run python -m scripts.hub.sync push`."
    )


if __name__ == "__main__":
    main()
