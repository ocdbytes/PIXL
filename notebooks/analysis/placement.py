"""Scores where a model draws the objects a caption names (review item R1).

For each evaluation caption, every checkpoint draws --samples images per guidance
value, sample s from seed s, at the model module's TEMPERATURE and TOP_K. Each 16x16
quadrant of a drawing is compared with the four quadrants of the caption's real scene.
The scene generator centres every object in its quadrant over the same background
tile, so a real quadrant is the exact picture of its object in any quadrant. A drawn
quadrant matches the real quadrant that agrees with it on the largest share of the
pixels where either differs from the background, when that share is at least --match:

- correct: it matches the object the caption puts there.
- swapped: it matches another of the caption's objects.
- other: it matches none of them: a wrong object, an empty quadrant or noise.

With the right four objects in random quadrants, chance is 0.25 correct per quadrant
and 1/24 for all four. The scored scenes are the first --per-kind of each kind: train
scenes from the train split, new sprites and new pairs from the test split.

Drawings are saved in --out/<run>/ and reused, so a rerun with another --match or more
checkpoints only draws what is missing. placement.csv in --out holds the table of the
runs given in the last invocation. Before scoring, the real scenes must score correct
and their quadrants moved one place must score swapped, which checks the background
tiles and the comparison. Without checkpoints the script runs only that check.

Drawing is heavy: run it on the GPU server.

    uv run python notebooks/analysis/placement.py \\
        checkpoints/scenes-four/checkpoint.pt checkpoints/scenes-four-margin-2ep/checkpoint.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image

from pixl.data import PLACES, REPO, SIZE, load
from pixl.model import TEMPERATURE, TOP_K, EncoderPixl, encode_captions
from pixl.preprocess import TRANSPARENT, encode, load_palette
from pixl.tokenizer import Tokenizer
from pixl.train import KINDS, load_model, pick_device

# Side of a quadrant, which holds one object.
SLOT = SIZE // 2


@dataclass
class Args(argparse.Namespace):
    checkpoints: list[Path] = field(default_factory=list)
    guidance: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])
    samples: int = 4
    per_kind: int = 250
    match: float = 0.5
    batch: int = 64
    out: Path = Path("runs/placement")
    device: str = "auto"


@dataclass
class Scenes:
    """The scored scenes.

    Attributes:
        tokens: (n, 32, 32) palette tokens of the real scenes.
        captions: Each scene's caption.
        kinds: Each scene's kind, one of KINDS.
        backgrounds: (n, 16, 16) tokens of each scene's background tile; TRANSPARENT without one.
    """

    tokens: np.ndarray
    captions: list[str]
    kinds: np.ndarray
    backgrounds: np.ndarray


def quadrants(grids: np.ndarray) -> np.ndarray:
    """Splits (..., 32, 32) grids into (..., 4, 16, 16) quadrants, in PLACES order."""
    *lead, height, width = grids.shape
    halves = grids.reshape(*lead, 2, height // 2, 2, width // 2).swapaxes(-3, -2)
    return halves.reshape(*lead, 4, height // 2, width // 2)


def background_tiles(paths: set[str]) -> dict[str, np.ndarray]:
    """(16, 16) palette tokens of each background tile, as the scene generator draws it.

    Args:
        paths: Tile paths in the sprites config; backgrounds are always train-split sprites.

    Returns:
        Path -> tokens.

    Raises:
        KeyError: If a path is missing from the sprites config.
    """
    palette = load_palette()
    sprites = load_dataset(REPO, "sprites", split="train").filter(lambda path: path in paths, input_columns="path")[:]
    tiles: dict[str, np.ndarray] = {}
    for path, image in zip(sprites["path"], sprites["image"], strict=True):
        image = image.convert("RGBA")
        if image.size != (SLOT, SLOT):
            # Tiles of the 32 px packs are halved, as scripts/scenes/generate.py does.
            image = image.resize((image.width // 2, image.height // 2), Image.Resampling.NEAREST)
        tiles[path] = encode(image, palette)
    if missing := paths - tiles.keys():
        raise KeyError(f"background tiles missing from the sprites config: {sorted(missing)}")
    return tiles


def eval_scenes(per_kind: int) -> Scenes:
    """The first per_kind scenes of each kind, with their background tiles."""
    keep = ("kind", "background_path")
    splits = [load(f"train[:{per_kind}]", keep=keep)[:], load("test", keep=keep)[:]]
    tokens = np.concatenate([split["tokens"].numpy() for split in splits])
    captions = [text for split in splits for text in split["caption"]]
    kinds = np.array([kind for split in splits for kind in split["kind"]])
    paths = [path for split in splits for path in split["background_path"]]
    rows = np.concatenate([np.flatnonzero(kinds == kind)[:per_kind] for kind in KINDS])
    tiles = background_tiles({paths[i] for i in rows if paths[i]})
    blank = np.full((SLOT, SLOT), TRANSPARENT, np.uint8)
    return Scenes(
        tokens=tokens[rows],
        captions=[captions[i] for i in rows],
        kinds=kinds[rows],
        backgrounds=np.stack([tiles[paths[i]] if paths[i] else blank for i in rows]),
    )


def classify(
    drawn: np.ndarray, templates: np.ndarray, backgrounds: np.ndarray, match: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tells which drawn quadrants hold the right object, and which another named one.

    Args:
        drawn: (samples, scenes, 4, 16, 16) quadrants of the drawings.
        templates: (scenes, 4, 16, 16) quadrants of the real scenes: what each caption
            puts in each quadrant.
        backgrounds: (scenes, 16, 16) background tile tokens.
        match: Least share of matching pixels, among those where the drawn quadrant
            or the template differs from the background, for the template to count.

    Returns:
        (samples, scenes, 4) masks of the correct and the swapped quadrants; the
        others match no template.
    """
    background = backgrounds[:, None]
    # Every drawn quadrant against every template: (samples, scenes, drawn, template, 16, 16).
    region = (drawn != background)[:, :, :, None] | (templates != background)[:, None]
    same = drawn[:, :, :, None] == templates[:, None]
    score = (same & region).sum((-2, -1)) / np.maximum(region.sum((-2, -1)), 1)
    best, matched = score.argmax(-1), score.max(-1) >= match
    right = best == np.arange(4)
    return matched & right, matched & ~right


def check(scenes: Scenes, match: float) -> None:
    """Stops unless the real scenes score correct and their quadrants moved one place score swapped."""
    templates = quadrants(scenes.tokens)
    correct, _ = classify(templates[None], templates, scenes.backgrounds, match)
    _, swapped = classify(np.roll(templates, 1, axis=1)[None], templates, scenes.backgrounds, match)
    print(f"check: real scenes {correct.mean():.4f} correct, quadrants moved one place {swapped.mean():.4f} swapped")
    # Two objects of one scene with identical pictures would tie; anything more is a bug.
    if min(correct.mean(), swapped.mean()) < 0.99:
        raise SystemExit("the comparison does not recognise the real scenes; check the background tiles")


def draw(model: EncoderPixl, captions: list[str], guidance: float, samples: int, batch: int) -> np.ndarray:
    """(samples, len(captions), 32, 32) drawings; sample s is drawn from seed s."""
    device = next(model.parameters()).device
    ids = encode_captions(captions, Tokenizer.load(), length=model.config.caption)
    side = model.config.side
    drawn = np.empty((samples, len(captions), side, side), np.uint8)
    for sample in range(samples):
        torch.manual_seed(sample)
        for start in range(0, len(captions), batch):
            pixels = model.generate(
                ids[start : start + batch].to(device), temperature=TEMPERATURE, top_k=TOP_K, guidance=guidance
            )
            drawn[sample, start : start + batch] = pixels.view(-1, side, side).cpu().numpy()
    return drawn


def summary(correct: np.ndarray, swapped: np.ndarray, kinds: np.ndarray) -> list[dict[str, float | int | str]]:
    """One table row per kind: shares of correct, swapped and other quadrants, all four correct, and each place."""
    rows: list[dict[str, float | int | str]] = []
    for kind in KINDS:
        right, moved = correct[:, kinds == kind], swapped[:, kinds == kind]
        rows.append({
            "kind": kind,
            "drawings": right.shape[0] * right.shape[1],
            "correct": right.mean(),
            "swapped": moved.mean(),
            "other": 1 - right.mean() - moved.mean(),
            "all four": right.all(-1).mean(),
            **{place: right[..., i].mean() for i, place in enumerate(PLACES)},
        })
    return rows


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("checkpoints", type=Path, nargs="*", help="checkpoint.pt files; the run is the folder name")
    _ = parser.add_argument("--guidance", type=float, nargs="+", default=Args().guidance)
    _ = parser.add_argument("--samples", type=int, default=Args.samples, help="drawings per caption")
    _ = parser.add_argument("--per-kind", type=int, default=Args.per_kind, help="scenes of each kind")
    _ = parser.add_argument("--match", type=float, default=Args.match, help="least share of matching pixels")
    _ = parser.add_argument("--batch", type=int, default=Args.batch, help="captions drawn at once")
    _ = parser.add_argument("--out", type=Path, default=Args.out)
    _ = parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=Args.device)
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    scenes = eval_scenes(args.per_kind)
    check(scenes, args.match)
    templates = quadrants(scenes.tokens)
    device = pick_device(args.device)
    table: list[dict[str, float | int | str]] = []
    for path in args.checkpoints:
        run = path.parent.name
        folder = args.out.joinpath(run)
        folder.mkdir(parents=True, exist_ok=True)
        model: EncoderPixl | None = None
        for guidance in args.guidance:
            saved = folder.joinpath(f"guidance-{guidance:g}_per-kind-{args.per_kind}_samples-{args.samples}.npy")
            if saved.exists():
                drawn = np.load(saved)
            else:
                if model is None:
                    model, _ = load_model(path, device)
                started = time.time()
                drawn = draw(model, scenes.captions, guidance, args.samples, args.batch)
                np.save(saved, drawn)
                print(f"{run} guidance {guidance:g}: {drawn.shape[0] * drawn.shape[1]:,} drawings "
                      f"in {time.time() - started:.0f} s", flush=True)
            correct, swapped = classify(quadrants(drawn), templates, scenes.backgrounds, args.match)
            table += [{"run": run, "guidance": guidance, **row} for row in summary(correct, swapped, scenes.kinds)]
    if table:
        frame = pd.DataFrame(table)
        frame.to_csv(args.out.joinpath("placement.csv"), index=False)
        print(frame.to_string(index=False, float_format="{:.3f}".format))
        print("chance with the right four objects in random quadrants: 0.250 correct per quadrant, 0.042 all four")


if __name__ == "__main__":
    main()
