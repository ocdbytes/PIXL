"""Builds synthetic 32x32 scenes from the sprites, with captions that are always right.

Each scene uses one sprite style (a pack, or one colour family of the 1-Bit Pack),
so pixel size and look stay consistent:

1. A background tile from BACKGROUNDS fills every slot. Styles without one use a
   transparent canvas.
2. Four objects with different labels fill the four 16 px slots (2x2). Sprites
   are not scaled, except those of the 32 px packs in HALVED, which are halved.
   With --no-four, one or two objects go into slots instead, also in the 8 px
   slots (4x4) of Pico-8 City; two objects share a row or a column.
3. The caption is written from the placement, e.g. "a pine tree at the bottom
   right, a red house at the top left, a well at the top right and a barrel at
   the bottom left on grass", the objects in random order; with --no-four, "a pine
   tree at the top left on grass" or "a pine tree left of a red house on grass".

Sprite roles come from role(); --sheets saves review sheets of them and of sample
scenes. Train scenes use only train-split sprites as objects. Test scenes use
test-split sprites ("new sprites"), or train sprites with at least one label pair
absent from train scenes ("new pair"). Backgrounds always use train-split sprites.

Only coloured objects are placed (--no-colour adds the monochrome ones). Writes
scene-data/manifest.csv and scene-data/tokens.npy, whose row i is the palette
tokens of manifest row i: a million scenes as PNG files would take ~4 GB. The seed
is fixed, so unchanged sprite inputs and settings reproduce the same scenes.

    uv run python -m scripts.scenes.generate --sheets
    uv run python -m scripts.scenes.generate --no-four --no-colour --train 100000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pixl.preprocess import encode, load_palette
from scripts.hub.sync import PART

SPRITES = Path("downloaded-data")
OUTPUT = Path("scene-data")
CANVAS = 32
# Full-tile labels that repeat cleanly, and how captions name them.
BACKGROUNDS = {
    "grass": "grass",
    "grass with tufts": "grass",
    "grass with flowers": "grass with flowers",
    "grass with pink flowers": "grass with pink flowers",
    "grass with shrubs": "grass with shrubs",
    "dense grass": "dense grass",
    "sparse grass": "sparse grass",
    "tall grass": "tall grass",
    "dry grass tufts": "dry grass",
    "dirt": "dirt",
    "dirt ground": "dirt",
    "dirt ground with pebbles": "pebbly dirt",
    "pebbles in dirt": "pebbly dirt",
    "dirt ground with stones": "stony dirt",
    "rocky dirt": "rocky dirt",
    "cobblestone ground": "cobblestones",
    "cave rock": "cave rock",
    "sand": "sand",
    "sand with pebbles": "pebbly sand",
    "sand with stones": "stony sand",
    "sand with shells": "sand with shells",
    "sand with pink shell": "sand with shells",
    "water": "water",
    "water center": "water",
    "water with sparkles": "water",
    "sea water": "water",
    "sea water with ripples": "water",
    "shallow water": "shallow water",
    "brown dirt ground": "dirt",
    "dark dungeon floor": "a dark dungeon floor",
    "sand floor with pebbles": "a sandy floor",
    "grass floor": "grass",
    "grass flowers blue floor": "grass with flowers",
    "grass flowers red floor": "grass with flowers",
    "grass flowers yellow floor": "grass with flowers",
    "dirt floor": "dirt",
    "pebble brown floor": "pebbly dirt",
    "sand floor": "sand",
    "shoals shallow water": "shallow water",
    "shoals deep water": "water",
    "marble floor": "a marble floor",
    "white marble floor": "a marble floor",
    "limestone floor": "a stone floor",
    "ice floor": "ice",
    "frozen floor": "ice",
    "lava floor": "lava",
    "pavement": "pavement",
    "cracked pavement": "cracked pavement",
    "road": "a road",
    "brown stone tile floor": "a stone tile floor",
    "stone brick floor": "a stone brick floor",
    "checkered floor": "a checkered floor",
    "herringbone brick floor": "a brick floor",
    "plowed field rows": "a plowed field",
    "tilled soil row": "tilled soil",
}
# Labels naming part of a larger picture, such as a tile edge or a piece of a
# building; such sprites are never placed as objects.
FRAGMENT = re.compile(
    r"\b(edge|corner|end|top|bottom|left|right|side|border|half|part|piece|segment"
    r"|middle|center|slope|ridge|gable|junction|crossing|crossroad|intersection"
    r"|bend|straight|vertical|horizontal|shore|shoreline|curve|marking"
    r"|roof|dormer|window|door|doorway|wall|facade|awning|chimney|battlement"
    r"|portcullis|arch|pavement"
    r"|north|south|east|west|northeast|northwest|southeast|southwest)\b"
)
# Share of object pairs kept out of train scenes for the "new pair" test.
HELD_OUT = 0.05
# Packs drawn at 32 px; their sprites are halved, keeping every other pixel, to fit 16 px slots.
HALVED = {"dcss_dungeon-crawl-32x32"}
# A halved object is only placed when its largest connected piece holds this share of its pixels.
WHOLE = 0.8
# The slots of a 2x2 grid, which four-object scenes fill.
QUADRANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


@dataclass
class Sprite:
    path: str
    label: str
    style: str
    split: str
    image: Image.Image


@dataclass
class Args(argparse.Namespace):
    train: int = 1_000_000
    test: int = 2_000
    sheets: bool = False
    colour: bool = True
    four: bool = True


def style_of(row: dict[str, str]) -> str:
    """Names a sprite's style; scenes never mix styles.

    Args:
        row: Sprite manifest row.

    Returns:
        The pack, or for the 1-Bit Pack the pack and colour family: its colored
        and colored-transparent tiles form one style, monochrome ones another.
    """
    if row["pack"] == "kenney_1-bit-pack":
        family = row["path"].split("/")[3].split("-")[0]
        return f"{row['pack']}/{family}"
    return row["pack"]


def slot_size(style: str) -> int:
    return 8 if style == "kenney_pico-8-city" else 16


def ground(label: str) -> str | None:
    """The caption name of a background label, or None when it is not a background.

    Monochrome tiles ("monochrome grass") are named like their coloured versions.
    """
    return BACKGROUNDS.get(label.removeprefix("monochrome "))


def role(image: Image.Image, label: str, slot: int) -> str:
    """Decides how a sprite may be used in scenes.

    Args:
        image: RGBA sprite.
        label: Its label.
        slot: Slot size of its style.

    Returns:
        "background" for an opaque one-slot tile labelled in BACKGROUNDS,
        "object" for a sprite with transparency that fits a slot, has at least
        5% opaque pixels, and is neither terrain nor a fragment, otherwise "".
    """
    opaque = np.asarray(image)[..., 3] >= 128
    if image.size == (slot, slot) and opaque.all():
        return "background" if ground(label) else ""
    fits = max(image.size) <= slot and 0.05 <= opaque.mean() and not opaque.all()
    if fits and not ground(label) and not FRAGMENT.search(label):
        return "object"
    return ""


def whole(image: Image.Image) -> bool:
    """Tells whether a sprite's opaque pixels mostly form one piece.

    Halving breaks thin outlines apart; such sprites are not placed.

    Args:
        image: RGBA sprite.

    Returns:
        True when the largest 8-connected piece holds at least WHOLE of the opaque pixels.
    """
    opaque = np.asarray(image)[..., 3] >= 128
    height, width = opaque.shape
    seen = np.zeros_like(opaque)
    largest = 0
    for start in zip(*np.nonzero(opaque)):
        if seen[start]:
            continue
        seen[start] = True
        stack, size = [start], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny in range(max(y - 1, 0), min(y + 2, height)):
                for nx in range(max(x - 1, 0), min(x + 2, width)):
                    if opaque[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        largest = max(largest, size)
    return largest >= WHOLE * opaque.sum()


def load_sprites() -> dict[str, list[Sprite]]:
    """Reads the local sprites and sorts them by role.

    Only CC0 sprites are used, so scenes stay CC0. Sprites of HALVED packs are halved
    first; halved objects that fall apart are dropped.

    Returns:
        "background" and "object" -> sprites with that role.
    """
    sprites = PART["sprites"]
    with SPRITES.joinpath("manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    roles: dict[str, list[Sprite]] = defaultdict(list)
    for row in rows:
        if not row["license"].startswith("CC0"):
            continue
        with Image.open(SPRITES.joinpath(row["path"])) as image:
            rgba = image.convert("RGBA")
        style = style_of(row)
        if style in HALVED:
            rgba = rgba.resize(
                (rgba.width // 2, rgba.height // 2), Image.Resampling.NEAREST
            )
        kind = role(rgba, row["label"], slot_size(style))
        if kind == "object" and style in HALVED and not whole(rgba):
            continue
        if kind:
            sprite = Sprite(row["path"], row["label"], style, sprites.split(row), rgba)
            roles[kind].append(sprite)
    return roles


def held_out(first: str, second: str) -> bool:
    """Tells whether a pair of object labels is kept out of train scenes.

    The key is the labels alone: captions do not name the style, so a pair held
    out in one style must not appear in another.
    """
    key = "|".join(sorted([first, second]))
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 16**8 < HELD_OUT


def article(label: str) -> str:
    return f"{'an' if label[0].lower() in 'aeiou' else 'a'} {label}"


def caption(
    placed: list[tuple[Sprite, tuple[int, int]]], ground: str, grid: int
) -> str:
    """Writes the caption of a scene from its placement.

    Args:
        placed: (object, (row, column)) per object: one, two, or four in the
            order the caption lists them.
        ground: Caption name of the background, or "" for none.
        grid: Slots per side.

    Returns:
        e.g. "a tree at the top left on grass", "a tree left of a house", or
        "a tree at the top left, a house at the top right, a well at the bottom
        left and a barrel at the bottom right".
    """
    if len(placed) == 2:
        (first, (row, _)), (second, (other, _)) = sorted(placed, key=lambda p: p[1])
        relation = "left of" if row == other else "above"
        text = f"{article(first.label)} {relation} {article(second.label)}"
    else:
        half = grid // 2
        phrases = [
            f"{article(sprite.label)} at the {'top' if row < half else 'bottom'} "
            f"{'left' if column < half else 'right'}"
            for sprite, (row, column) in placed
        ]
        text = phrases[0] if len(phrases) == 1 else f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    return f"{text} on {ground}" if ground else text


def visible(background: Sprite | None, sprite: Sprite, slot: int) -> bool:
    """Tells whether an object stands out from the background tile under it.

    Args:
        background: Background tile, or None for a transparent canvas.
        sprite: Object centred in a slot.
        slot: Slot size.

    Returns:
        True if at least half of the object's opaque pixels differ in colour
        from the background beneath them, e.g. False for a white sprite on the
        white water of the 1-Bit Pack's monochrome style.
    """
    if background is None:
        return True
    pixels = np.asarray(sprite.image)
    height, width = pixels.shape[:2]
    top, left = (slot - height) // 2, (slot - width) // 2
    under = np.asarray(background.image)[top : top + height, left : left + width, :3]
    differs = (pixels[..., :3] != under).any(axis=-1)
    return differs[pixels[..., 3] >= 128].mean() >= 0.5


def compose(
    background: Sprite | None, placed: list[tuple[Sprite, tuple[int, int]]], slot: int
) -> Image.Image:
    """Draws a scene: the background in every slot, objects centred in theirs."""
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    if background:
        for top in range(0, CANVAS, slot):
            for left in range(0, CANVAS, slot):
                canvas.paste(background.image, (left, top))
    for sprite, (row, column) in placed:
        width, height = sprite.image.size
        left = column * slot + (slot - width) // 2
        top = row * slot + (slot - height) // 2
        canvas.alpha_composite(sprite.image, (left, top))
    return canvas


def make_scenes(
    count: int,
    kind: str,
    objects: list[Sprite],
    backgrounds: list[Sprite],
    rng: random.Random,
    four: bool = False,
) -> Iterator[tuple[dict[str, str], Image.Image]]:
    """Makes scenes with distinct IDs within one kind, one at a time.

    IDs include object placement in caption order, so different caption orders
    can yield the same image.

    Args:
        count: Number of scenes.
        kind: "train", "new sprites" or "new pair". With four, a "new pair"
            scene holds at least one pair of labels kept out of train scenes,
            and a train scene none.
        objects: Sprites that may be placed.
        backgrounds: Background tiles, from train-split sprites.
        rng: Random source.
        four: Make four-object scenes: four different labels fill the four
            16 px slots, listed in random order. Styles with 8 px slots are
            left out.

    Yields:
        (manifest row, image) per scene. The row's "swapped" is the caption with
        the positions reversed: the first two listed objects exchange slots, or
        a single object is mirrored left to right.
    """
    by_style: dict[str, list[Sprite]] = defaultdict(list)
    for sprite in objects:
        by_style[sprite.style].append(sprite)
    grounds: dict[str, list[Sprite]] = defaultdict(list)
    for sprite in backgrounds:
        grounds[sprite.style].append(sprite)
    # A style needs two different labels to make pairs, or four 16 px ones for four-object scenes.
    styles = sorted(
        s
        for s, group in by_style.items()
        if len({x.label for x in group}) >= (4 if four else 2)
        and (not four or slot_size(s) == 16)
    )

    seen: set[str] = set()
    while len(seen) < count:
        style = rng.choice(styles)
        slot = slot_size(style)
        grid = CANVAS // slot
        pool = by_style[style]
        if four:
            chosen = rng.sample(pool, 4)
            if len({sprite.label for sprite in chosen}) < 4:
                continue
            pairs = [held_out(a.label, b.label) for a, b in combinations(chosen, 2)]
            if kind == "train" and any(pairs):
                continue
            if kind == "new pair" and not any(pairs):
                continue
            cells = rng.sample(QUADRANTS, 4)
        elif kind != "new pair" and rng.random() < 0.5:
            chosen = [rng.choice(pool)]
            cells = [(rng.randrange(grid), rng.randrange(grid))]
        else:
            chosen = rng.sample(pool, 2)
            pair_held_out = held_out(chosen[0].label, chosen[1].label)
            if chosen[0].label == chosen[1].label:
                continue
            if kind == "train" and pair_held_out:
                continue
            if kind == "new pair" and not pair_held_out:
                continue
            line, first, second = rng.randrange(grid), *rng.sample(range(grid), 2)
            cells = [(line, first), (line, second)]
            if rng.random() < 0.5:  # same column instead of same row
                cells = [(cell, row) for row, cell in cells]
        background = rng.choice(grounds[style]) if grounds[style] else None
        if not all(visible(background, sprite, slot) for sprite in chosen):
            continue
        placed = list(zip(chosen, cells))
        parts = [style, background.path if background else ""]
        parts += [f"{sprite.path}@{r},{c}" for sprite, (r, c) in placed]
        scene_id = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        if scene_id in seen:
            continue
        seen.add(scene_id)
        if len(placed) == 1:
            swapped = [(chosen[0], (cells[0][0], grid - 1 - cells[0][1]))]
        else:
            swapped = [(chosen[0], cells[1]), (chosen[1], cells[0]), *placed[2:]]
        name = ground(background.label) or "" if background else ""
        row = {
            "id": scene_id,
            "split": "train" if kind == "train" else "test",
            "kind": kind,
            "caption": caption(placed, name, grid),
            "swapped": caption(swapped, name, grid),
            "style": style,
            "background": background.label if background else "",
            "objects": "|".join(sprite.label for sprite, _ in placed),
            "slots": "|".join(f"{r},{c}" for _, (r, c) in placed),
            "sprites": "|".join(sprite.path for sprite, _ in placed),
            "background_path": background.path if background else "",
        }
        yield row, compose(background, placed, slot)


def save_sheets(
    roles: dict[str, list[Sprite]], scenes: list[tuple[dict[str, str], Image.Image]]
) -> None:
    """Saves review sheets: each style's backgrounds and objects, and sample scenes.

    Args:
        roles: Sprites by role, from load_sprites.
        scenes: Generated scenes; the first 60 are drawn with their captions.
    """
    folder = OUTPUT.joinpath("review")
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    styles = sorted({s.style for group in roles.values() for s in group})
    cell, columns = 96, 12
    for style in styles:
        items = [
            (kind, sprite)
            for kind in ("background", "object")
            for sprite in roles[kind]
            if sprite.style == style
        ]
        # One entry per label is enough to check a role.
        unique = list({(kind, s.label): (kind, s) for kind, s in items}.values())
        rows = -(-len(unique) // columns)
        sheet = Image.new("RGB", (columns * cell, rows * (cell + 12)), (128, 128, 128))
        draw = ImageDraw.Draw(sheet)
        for n, (kind, sprite) in enumerate(unique):
            x, y = n % columns * cell, n // columns * (cell + 12)
            big = sprite.image.resize(
                (sprite.image.width * 4, sprite.image.height * 4),
                Image.Resampling.NEAREST,
            )
            sheet.paste(
                big,
                (x + (cell - big.width) // 2, y + (cell - 8 - big.height) // 2),
                big,
            )
            draw.text(
                (x + 2, y + cell - 8),
                f"{'BG ' if kind == 'background' else ''}{sprite.label[:15]}",
                fill="white" if kind == "object" else "yellow",
            )
        sheet.save(folder.joinpath(f"roles-{style.replace('/', '-')}.png"))
    cell = 160
    sheet = Image.new("RGB", (6 * cell, 10 * (cell + 24)), (128, 128, 128))
    draw = ImageDraw.Draw(sheet)
    for n, (row, image) in enumerate(scenes[:60]):
        x, y = n % 6 * cell, n // 6 * (cell + 24)
        big = image.resize((cell - 4, cell - 4), Image.Resampling.NEAREST)
        sheet.paste(big, (x, y), big)
        text = row["caption"]
        draw.text((x + 2, y + cell - 2), text[:30], fill="white")
        draw.text((x + 2, y + cell + 9), text[30:60], fill="white")
    sheet.save(folder.joinpath("scenes.png"))
    print(f"saved review sheets to {folder}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument("--train", type=int, default=Args.train)
    _ = parser.add_argument(
        "--test", type=int, default=Args.test, help="half new sprites, half new pairs"
    )
    _ = parser.add_argument("--sheets", action="store_true", help="save review sheets")
    _ = parser.add_argument(
        "--colour",
        action=argparse.BooleanOptionalAction,
        default=Args.colour,
        help='place only coloured objects, not those labelled "monochrome"',
    )
    _ = parser.add_argument(
        "--four",
        action=argparse.BooleanOptionalAction,
        default=Args.four,
        help="four objects per scene, one in each 16 px slot of a 2x2 grid; "
        "--no-four places one or two",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    roles = load_sprites()
    objects = [
        s
        for s in roles["object"]
        if not (args.colour and s.label.startswith("monochrome "))
    ]
    train_objects = [s for s in objects if s.split == "train"]
    test_objects = [s for s in objects if s.split == "test"]
    backgrounds = [s for s in roles["background"] if s.split == "train"]
    print(
        f"roles: {len(backgrounds)} train backgrounds, {len(train_objects)} train objects, {len(test_objects)} test objects"
    )

    rng = random.Random(1)
    scenes = chain(
        make_scenes(
            args.train, "train", train_objects, backgrounds, random.Random(0), args.four
        ),
        make_scenes(
            args.test // 2, "new sprites", test_objects, backgrounds, rng, args.four
        ),
        make_scenes(
            args.test - args.test // 2, "new pair", train_objects, backgrounds, rng, args.four
        ),
    )

    OUTPUT.mkdir(exist_ok=True)
    palette = load_palette()
    tokens = np.empty((args.train + args.test, CANVAS, CANVAS), np.uint8)
    rows, samples = [], []
    for n, (row, image) in enumerate(scenes):
        tokens[n] = encode(image, palette)
        rows.append(row)
        if len(samples) < 60:
            samples.append((row, image))
        if (n + 1) % 100_000 == 0:
            print(f"{n + 1:,} scenes")
    # Row i of the manifest is tokens[i].
    np.save(OUTPUT.joinpath("tokens.npy"), tokens)
    with OUTPUT.joinpath("manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):,} scenes to {OUTPUT}")
    if args.sheets:
        save_sheets(roles, samples)


if __name__ == "__main__":
    main()
