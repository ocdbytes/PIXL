"""Loads the scenes of pixl-data as captions and 32x32 palette-token tensors.

The first load downloads the scenes and encodes them into tokens; the datasets
library caches the result for later loads.

    test = load("test", keep=("kind", "swapped"))[:]
    test["caption"]  # list[str]
    test["tokens"]   # (2000, 32, 32) uint8
"""

from __future__ import annotations

import os

from datasets import Array2D, Dataset, Features, Value, load_dataset

from .preprocess import encode, load_palette

REPO = "ocdbytes/pixl-data"
SIZE = 32
# Splits this large encode in parallel on the first load; smaller ones encode faster in one process than
# worker processes start.
PARALLEL_ROWS = 100_000


def load(split: str, keep: tuple[str, ...] = (), num_proc: int | None = None) -> Dataset:
    """Loads one split of the scenes as captions and token grids.

    Args:
        split: "train" or "test", or a slice such as "train[:2000]".
        keep: Other string columns to keep, e.g. ("kind", "swapped").
        num_proc: Processes that encode the images on the first load. By default,
            every core this process may use for splits of PARALLEL_ROWS rows or
            more, and one for smaller splits. The rows keep their order.

    Returns:
        A dataset in torch format with "caption", "tokens" and the keep columns.
    """
    palette = load_palette()

    def to_tokens(batch: dict[str, list]) -> dict[str, list]:
        return {"tokens": [encode(image, palette) for image in batch["image"]]}

    data = load_dataset(REPO, "scenes", split=split).select_columns(["caption", "image", *keep])
    if num_proc is None:
        # On Linux the affinity mask honours a container's --cpuset-cpus; macOS has no such call.
        if hasattr(os, "sched_getaffinity"):
            cores = len(os.sched_getaffinity(0))
        else:
            cores = os.cpu_count() or 1
        num_proc = cores if len(data) >= PARALLEL_ROWS else 1
    data = data.map(
        to_tokens,
        batched=True,
        num_proc=num_proc,
        remove_columns=["image"],
        features=Features(
            {
                "caption": Value("string"),
                **{column: Value("string") for column in keep},
                "tokens": Array2D((SIZE, SIZE), "uint8"),
            }
        ),
    )
    return data.with_format("torch")
