"""Drops web pairs whose photo does not match its query, judged by CLIP.

Open Images queries come from human-drawn boxes, so only web pairs (Wikimedia and
Openverse search results) are scored. A web pair is kept when its photo's CLIP
similarity to "a photo of {query}" is at least --min-score and its query is among
the --top-k best-matching of every query in the manifest.

Scores are cached in clip_scores.csv, so changing the cut-offs does not rerun CLIP.
Ranks depend on the query list, so a warning is printed when the manifest's queries
have changed since a cached rank was computed. Kept rows are written to clean.csv;
manifest.csv and the image files are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
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
# Provider whose queries are human-verified and never scored.
TRUSTED = "openimages"
SCORES = "clip_scores.csv"

Scores = dict[str, tuple[float, int]]


@dataclass
class Args(argparse.Namespace):
    data: Path = Path("pixel-image-data")
    # Defaults keep 94% of Open Images pairs, whose queries are known to be right.
    min_score: float = 0.22
    top_k: int = 50


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def score_pairs(
    rows: list[dict[str, str]],
    root: Path,
    queries: list[str],
    batch_size: int = 256,
) -> list[tuple[float, int]]:
    """Scores each pair's photo against its own query with CLIP.

    Args:
        rows: Manifest rows to score.
        root: Folder the manifest paths are relative to.
        queries: Every query to rank against; must include each row's query.
        batch_size: Photos embedded per forward pass.

    Returns:
        (score, rank) per row, in order. score is the cosine similarity between
        the photo and PROMPT filled with the row's query; rank is how many other
        queries match the photo better, so 0 is the best match.
    """
    model = CLIPModel.from_pretrained(CLIP).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP)
    text = processor(
        text=[PROMPT.format(q) for q in queries], padding=True, return_tensors="pt"
    ).to(DEVICE)
    texts = model.text_projection(model.text_model(**text).pooler_output)
    texts = texts / texts.norm(dim=-1, keepdim=True)
    index = {query: i for i, query in enumerate(queries)}
    results: list[tuple[float, int]] = []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            images = list(
                pool.map(load_image, [root.joinpath(r["input_path"]) for r in batch])
            )
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            photos = model.visual_projection(
                model.vision_model(pixel_values=pixels.to(DEVICE)).pooler_output
            )
            photos = photos / photos.norm(dim=-1, keepdim=True)
            similarity = photos @ texts.T
            own = similarity[
                torch.arange(len(batch)), [index[r["query"]] for r in batch]
            ]
            ranks = (similarity > own[:, None]).sum(dim=1)
            results += zip(own.tolist(), ranks.tolist())
            print(f"scored {len(results)}/{len(rows)}")
    return results


def keep(row: dict[str, str], scores: Scores, min_score: float, top_k: int) -> bool:
    """Decides whether a manifest row stays in the clean dataset.

    Args:
        row: Manifest row.
        scores: (score, rank) per row id, from score_pairs.
        min_score: Lowest CLIP similarity a web pair may have.
        top_k: A web pair's query must rank below this.

    Returns:
        True for Open Images rows and for web rows that pass both cut-offs.
    """
    if row["provider"] == TRUSTED:
        return True
    score, rank = scores[row["id"]]
    return score >= min_score and rank < top_k


def query_list(queries: list[str]) -> str:
    """Names a query list by its size and content.

    Args:
        queries: Sorted queries.

    Returns:
        "<count>-<hash>", e.g. "462-3fa9c1d2". Lists with different queries get
        different names, even when they are the same size.
    """
    digest = hashlib.sha256("\n".join(queries).encode()).hexdigest()[:8]
    return f"{len(queries)}-{digest}"


def read_cache(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--data", type=Path, default=Args.data)
    _ = parser.add_argument("--min-score", type=float, default=Args.min_score)
    _ = parser.add_argument("--top-k", type=int, default=Args.top_k)
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    with args.data.joinpath("manifest.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        rows = list(reader)

    queries = sorted({r["query"] for r in rows})
    current = query_list(queries)
    cache = args.data.joinpath(SCORES)
    cached = read_cache(cache)
    scores: Scores = {r["id"]: (float(r["score"]), int(r["rank"])) for r in cached}
    stale = Counter(r["query_list"] for r in cached if r["query_list"] != current)
    if stale:
        then = ", ".join(f"{name.split('-')[0]} queries ({name})" for name in stale)
        print(
            f"warning: {sum(stale.values())} ranks in {cache} were computed against "
            f"{then}, but the manifest now has {len(queries)} queries ({current}).\n"
            f"Ranks depend on the query list; delete {cache} and rerun to rescore.",
            file=sys.stderr,
        )

    todo = [r for r in rows if r["provider"] != TRUSTED and r["id"] not in scores]
    if todo:
        new = score_pairs(todo, args.data, queries)
        is_new = not cache.exists()
        with cache.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(["id", "query", "score", "rank", "query_list"])
            for row, (score, rank) in zip(todo, new):
                writer.writerow([row["id"], row["query"], score, rank, current])
                scores[row["id"]] = (score, rank)

    clean = args.data.joinpath("clean.csv")
    kept = [r for r in rows if keep(r, scores, args.min_score, args.top_k)]
    with clean.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    web = Counter(r["query"] for r in rows if r["provider"] != TRUSTED)
    left = Counter(r["query"] for r in kept if r["provider"] != TRUSTED)
    print(f"kept {sum(left.values())} of {sum(web.values())} web pairs")
    print(f"wrote {len(kept)} of {len(rows)} rows to {clean}")
    worst = sorted(web, key=lambda q: left[q] / web[q])[:10]
    print("most dropped:", ", ".join(f"{q} {left[q]}/{web[q]}" for q in worst))


if __name__ == "__main__":
    main()
