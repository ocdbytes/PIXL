"""Label unlabelled sprites with CLIP or Florence-2 and gather every sprite into one folder."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Florence2ForConditionalGeneration,
)

CLIP = "openai/clip-vit-base-patch32"
# Official transformers conversion of microsoft/Florence-2-base; needs no Hub code.
FLORENCE = "florence-community/Florence-2-base"
MODELS = ("clip", "florence")
# Florence-2 generation is ~10x faster on a GPU; CLIP is fast enough on the CPU.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
PROMPT = "pixel art of {}"
# Mid-grey keeps both black and white sprites visible once transparency is flattened.
BACKGROUND = (128, 128, 128)
# Broad categories: CLIP matched the fine-grained manifest labels poorly on small tiles.
CATEGORIES = [
    # Nature and terrain.
    "tree",
    "bush",
    "grass",
    "flower",
    "crop plant",
    "water",
    "dirt ground",
    "rock",
    "sand",
    "cloud",
    "landscape background",
    # Buildings and structures.
    "house",
    "roof",
    "wall",
    "brick wall",
    "door",
    "window",
    "fence",
    "bridge",
    "ladder",
    "stairs",
    "road",
    "platform",
    "tower",
    "tent",
    # Vehicles.
    "car",
    "truck",
    "tank",
    "boat",
    "airplane",
    # Characters.
    "person",
    "soldier",
    "farm animal",
    "bird",
    "monster",
    # Items.
    "sword",
    "gun",
    "tool",
    "crate",
    "barrel",
    "treasure chest",
    "furniture",
    "sign",
    "street lamp",
    "coin",
    "gem",
    "key",
    "potion",
    "fruit or vegetable",
    "heart",
    "spikes",
    "explosion",
    # Interface.
    "mouse cursor arrow",
    "hand cursor",
    "hourglass",
    "magnifying glass",
    "crosshair",
    "direction arrow",
    "button",
    "digit",
    "letter",
    "symbol",
    "plain colored tile",
]
FIELDS = [
    "file",
    "label",
    "label_source",
    "score",
    "pack",
    "creator",
    "license",
    "source_url",
    "width",
    "height",
    "source_path",
]


@dataclass
class Args(argparse.Namespace):
    manifest: Path = Path("downloaded-data/manifest.csv")
    output: Path = Path("downloaded-data/labeled")
    model: str = "clip"


def prepare(image: Image.Image, size: int = 224) -> Image.Image:
    """Flattens transparency onto grey and scales up to a square without blurring.

    Args:
        image: Sprite in any mode.
        size: Side length the model expects.

    Returns:
        An RGB size x size image with the sprite centred.
    """
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, BACKGROUND)
    flat.paste(rgba, mask=rgba.getchannel("A"))
    return ImageOps.pad(
        flat, (size, size), method=Image.Resampling.NEAREST, color=BACKGROUND
    )


def flat_name(path: str) -> str:
    """Builds a flat file name from a manifest path.

    Args:
        path: Manifest path, e.g. "packs/kenney_tiny-town/Tiles/tile_0001.png".

    Returns:
        The file name, e.g. "kenney_tiny-town_Tiles_tile_0001.png".
    """
    return re.sub(r"[^\w.-]", "_", path.removeprefix("packs/"))


def load_images(paths: list[Path], size: int) -> list[Image.Image]:
    """Opens and prepares images for a model.

    Args:
        paths: Image files.
        size: Side length the model expects.

    Returns:
        Prepared images, in order.
    """
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(prepare(image, size))
    return images


def load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    """Loads CLIP, downloading it to the Hugging Face cache on first use.

    Returns:
        The model in eval mode and its processor.
    """
    return CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)


@torch.inference_mode()
def label_with_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    paths: list[Path],
    vocabulary: list[str],
    batch_size: int = 64,
) -> list[tuple[str, float]]:
    """Picks the vocabulary label whose text embedding best matches each image.

    Args:
        model: CLIP model from load_clip.
        processor: CLIP processor from load_clip.
        paths: Images to label.
        vocabulary: Candidate labels.
        batch_size: Images embedded per forward pass.

    Returns:
        (label, probability) per path, in order. The probability is a softmax
        over the whole vocabulary, so it stays low when several labels fit.
    """
    text = processor(
        text=[PROMPT.format(v) for v in vocabulary], padding=True, return_tensors="pt"
    )
    labels = model.text_projection(model.text_model(**text).pooler_output)
    labels = labels / labels.norm(dim=-1, keepdim=True)
    results: list[tuple[str, float]] = []
    for start in range(0, len(paths), batch_size):
        images = load_images(paths[start : start + batch_size], 224)
        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
        embeds = model.visual_projection(
            model.vision_model(pixel_values=pixels).pooler_output
        )
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * embeds @ labels.T
        best = logits.softmax(dim=-1).max(dim=-1)
        results += [
            (vocabulary[i], p)
            for p, i in zip(best.values.tolist(), best.indices.tolist())
        ]
        print(f"labelled {len(results)}/{len(paths)}")
    return results


def load_florence() -> tuple[Florence2ForConditionalGeneration, AutoProcessor]:
    """Loads Florence-2, downloading it to the Hugging Face cache on first use.

    Returns:
        The model in eval mode and its processor.
    """
    model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE).to(DEVICE)
    return model.eval(), AutoProcessor.from_pretrained(FLORENCE)


@torch.inference_mode()
def label_with_florence(
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    paths: list[Path],
    batch_size: int = 16,
) -> list[str]:
    """Writes a short Florence-2 caption for each image.

    Args:
        model: Florence-2 model from load_florence.
        processor: Florence-2 processor from load_florence.
        paths: Images to caption.
        batch_size: Images captioned per generate call.

    Returns:
        One caption per path, in order.
    """
    captions: list[str] = []
    for start in range(0, len(paths), batch_size):
        images = load_images(paths[start : start + batch_size], 768)
        inputs = processor(
            text=["<CAPTION>"] * len(images), images=images, return_tensors="pt"
        ).to(model.device)
        ids = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        captions += [
            c.strip() for c in processor.batch_decode(ids, skip_special_tokens=True)
        ]
        print(f"captioned {len(captions)}/{len(paths)}")
    return captions


def label_images(paths: list[Path], model_name: str) -> list[tuple[str, str, str]]:
    """Labels images with the selected model.

    Returns:
        (label, model identifier, score) per path, in order. Captioning models
        return an empty score.
    """
    if model_name == "clip":
        model, processor = load_clip()
        labels = label_with_clip(model, processor, paths, CATEGORIES)
        return [(label, CLIP, f"{score:.3f}") for label, score in labels]
    if model_name == "florence":
        model, processor = load_florence()
        return [
            (caption, FLORENCE, "")
            for caption in label_with_florence(model, processor, paths)
        ]
    raise ValueError(f"unsupported model: {model_name}")


def gather(rows: list[dict[str, str]], root: Path, output: Path) -> None:
    """Copies every sprite into one folder and writes labels.csv beside them.

    Args:
        rows: Manifest rows with label, label_source and optional score filled in.
        root: Folder the manifest paths are relative to.
        output: Destination folder.

    Raises:
        ValueError: If two sprites would get the same flat file name.
    """
    names = [flat_name(row["path"]) for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("two sprites flatten to the same file name")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name, row in zip(names, rows):
            _ = shutil.copyfile(root / row["path"], output / name)
            writer.writerow({**row, "file": name, "source_path": row["path"]})


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--manifest", type=Path, default=Args.manifest)
    _ = parser.add_argument("--output", type=Path, default=Args.output)
    _ = parser.add_argument("--model", choices=MODELS, default=Args.model)
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    todo = [row for row in rows if not row["label"]]
    paths = [args.manifest.parent / row["path"] for row in todo]
    # Do not load a model when the manifest is already fully labelled.
    results = label_images(paths, args.model) if todo else []
    for row, (label, source, score) in zip(todo, results):
        row.update(label=label, label_source=source, score=score)
    gather(rows, args.manifest.parent, args.output)
    print(f"Labelled {len(todo)} sprites; wrote {len(rows)} to {args.output}")


if __name__ == "__main__":
    main()
