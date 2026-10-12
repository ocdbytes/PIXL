"""Drops photo pairs that should not be trained on or published.

A pair is dropped for the first of these reasons that applies:

- nudity, graphic: the photo matches the CONTENT prompts better than the best
  NEUTRAL prompt by at least --nudity or --graphic. Checked for every photo.
- mismatch: a web photo (Wikimedia, Openverse) does not show its query. Its CLIP
  similarity to "a photo of {query}" is below --min-score, or its query is not
  among the --top-k best-matching of every query in the manifest. Open Images
  queries come from human-drawn boxes and are not checked.
- no author: a CC BY photo without an author, so it cannot be credited.
- email author: the author field is an email address.

CLIP photo embeddings are cached in clip_embeddings.pt, so later runs only embed
new photos, and scores are always computed against the current queries. Kept rows
are written to clean.csv and dropped rows, with their reason and scores, to
dropped.csv; manifest.csv and the image files are unchanged. --sheets saves
contact sheets of the nudity and graphic drops to review/.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

CLIP = "openai/clip-vit-base-patch32"
PROMPT = "a photo of {}"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Provider whose queries are human-verified and never checked for a mismatch.
TRUSTED = "openimages"
CACHE = "clip_embeddings.pt"
CONTENT = {
    "nudity": ["a photo of a naked person", "a nude body", "a pornographic photo"],
    "graphic": [
        "a photo of gore and blood",
        "a photo of a dead body",
        "a graphic injury",
    ],
}
NEUTRAL = ["a photo", "a photo of an object", "a photo of a person"]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SCORES = ["query_score", "rank", "nudity", "graphic"]


@dataclass
class Args(argparse.Namespace):
    data: Path = Path("pixel-image-data")
    # Defaults keep 94% of Open Images pairs, whose queries are known to be right.
    min_score: float = 0.22
    top_k: int = 50
    # Chosen by reviewing the dropped photos: below 0.02 photos are rarely nude;
    # below 0.0395 graphic flags are mostly harmless or mild (dead animals, make-up).
    nudity: float = 0.02
    graphic: float = 0.0395
    sheets: bool = False


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


@torch.inference_mode()
def embed_texts(
    model: CLIPModel, processor: CLIPProcessor, texts: list[str]
) -> torch.Tensor:
    """Embeds texts with CLIP.

    Args:
        model: CLIP model.
        processor: CLIP processor.
        texts: Texts to embed.

    Returns:
        A len(texts) x 512 float32 tensor on the CPU, each row of length 1.
    """
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(DEVICE)
    return normalize(
        model.text_projection(model.text_model(**inputs).pooler_output)
    ).cpu()


@torch.inference_mode()
def embed_photos(
    model: CLIPModel, processor: CLIPProcessor, paths: list[Path], batch_size: int = 256
) -> torch.Tensor:
    """Embeds photos with CLIP.

    Args:
        model: CLIP model.
        processor: CLIP processor.
        paths: Photo files.
        batch_size: Photos embedded per forward pass.

    Returns:
        A len(paths) x 512 float16 tensor on the CPU, each row of length 1.
    """
    batches: list[torch.Tensor] = []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(paths), batch_size):
            images = list(pool.map(load_image, paths[start : start + batch_size]))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            photos = model.visual_projection(
                model.vision_model(pixel_values=pixels.to(DEVICE)).pooler_output
            )
            batches.append(normalize(photos).half().cpu())
            print(f"embedded {min(start + batch_size, len(paths))}/{len(paths)}")
    return torch.cat(batches)


def drop_reason(row: dict[str, str], scores: dict[str, float], args: Args) -> str:
    """Decides whether a manifest row is dropped.

    Args:
        row: Manifest row.
        scores: The row's SCORES, from main.
        args: Cut-offs.

    Returns:
        The reason the row is dropped, or "" to keep it.
    """
    if scores["nudity"] >= args.nudity:
        return "nudity"
    if scores["graphic"] >= args.graphic:
        return "graphic"
    if row["provider"] != TRUSTED and (
        scores["query_score"] < args.min_score or scores["rank"] >= args.top_k
    ):
        return "mismatch"
    if "BY" in row["license"] and not row["author"].strip():
        return "no author"
    if EMAIL.search(row["author"]):
        return "email author"
    return ""


def save_sheets(
    dropped: list[tuple[dict[str, str], dict[str, float], str]], root: Path
) -> None:
    """Saves contact sheets of the nudity and graphic drops, highest score first.

    Args:
        dropped: (row, scores, reason) per dropped row.
        root: Dataset folder; sheets go to root/review/<reason>-<n>.png, replacing
            any sheets from an earlier run.
    """
    cell, columns, per_sheet = 128, 10, 100
    folder = root.joinpath("review")
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for reason in CONTENT:
        flagged = sorted(
            ((row, scores[reason]) for row, scores, why in dropped if why == reason),
            key=lambda item: -item[1],
        )
        for number, start in enumerate(range(0, len(flagged), per_sheet), 1):
            page = flagged[start : start + per_sheet]
            rows = -(-len(page) // columns)
            sheet = Image.new("RGB", (columns * cell, rows * (cell + 24)), "black")
            draw = ImageDraw.Draw(sheet)
            for n, (row, score) in enumerate(page):
                x, y = n % columns * cell, n // columns * (cell + 24)
                photo = load_image(root.joinpath(row["input_path"]))
                sheet.paste(photo.resize((cell - 2, cell - 2)), (x, y))
                draw.text(
                    (x + 2, y + cell), f"{score:.3f} {row['query'][:14]}", "white"
                )
                draw.text((x + 2, y + cell + 11), row["id"], "gray")
            sheet.save(folder.joinpath(f"{reason}-{number}.png"))
        print(f"saved {len(flagged)} {reason} drops to {folder}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument("--data", type=Path, default=Args.data)
    _ = parser.add_argument("--min-score", type=float, default=Args.min_score)
    _ = parser.add_argument("--top-k", type=int, default=Args.top_k)
    _ = parser.add_argument("--nudity", type=float, default=Args.nudity)
    _ = parser.add_argument("--graphic", type=float, default=Args.graphic)
    _ = parser.add_argument(
        "--sheets", action="store_true", help="save review sheets of content drops"
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    with args.data.joinpath("manifest.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        rows = list(reader)

    model = CLIPModel.from_pretrained(CLIP).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP)

    cache = args.data.joinpath(CACHE)
    saved = (
        torch.load(cache, weights_only=True)
        if cache.exists()
        else {"ids": [], "embeddings": torch.empty(0, 512, dtype=torch.float16)}
    )
    known = set(saved["ids"])
    todo = [r for r in rows if r["id"] not in known]
    if todo:
        paths = [args.data.joinpath(r["input_path"]) for r in todo]
        saved = {
            "ids": saved["ids"] + [r["id"] for r in todo],
            "embeddings": torch.cat(
                [saved["embeddings"], embed_photos(model, processor, paths)]
            ),
        }
        torch.save(saved, cache)
    index = {row_id: i for i, row_id in enumerate(saved["ids"])}
    photos = saved["embeddings"][[index[r["id"]] for r in rows]].float()

    queries = sorted({r["query"] for r in rows})
    position = {query: i for i, query in enumerate(queries)}
    texts = embed_texts(model, processor, [PROMPT.format(q) for q in queries])
    similarity = photos @ texts.T
    own = similarity[torch.arange(len(rows)), [position[r["query"]] for r in rows]]
    neutral = (photos @ embed_texts(model, processor, NEUTRAL).T).max(dim=1).values
    columns = {
        "query_score": own,
        "rank": (similarity > own[:, None]).sum(dim=1),
        **{
            name: (photos @ embed_texts(model, processor, prompts).T).max(dim=1).values
            - neutral
            for name, prompts in CONTENT.items()
        },
    }
    values = {name: column.tolist() for name, column in columns.items()}

    kept: list[dict[str, str]] = []
    dropped: list[tuple[dict[str, str], dict[str, float], str]] = []
    for i, row in enumerate(rows):
        scores = {name: values[name][i] for name in SCORES}
        reason = drop_reason(row, scores, args)
        if reason:
            dropped.append((row, scores, reason))
        else:
            kept.append(row)

    clean = args.data.joinpath("clean.csv")
    with clean.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    with args.data.joinpath("dropped.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["id", "provider", "query", "reason", *SCORES])
        for row, scores, reason in dropped:
            writer.writerow(
                [row["id"], row["provider"], row["query"], reason]
                + [round(scores[name], 4) for name in SCORES]
            )
    if args.sheets:
        save_sheets(dropped, args.data)

    reasons = Counter(reason for _, _, reason in dropped)
    print(f"wrote {len(kept)} of {len(rows)} rows to {clean}")
    print("dropped:", ", ".join(f"{reason} {n}" for reason, n in reasons.most_common()))


if __name__ == "__main__":
    main()
