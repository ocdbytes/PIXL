"""Trains EncoderPixl from scratch on the four-object scenes of pixl-data.

The scenes come from the Hugging Face dataset; the first run downloads and
encodes them, and later runs read the Hugging Face cache. Training makes --epochs
passes over the train scenes. Next to the pixel loss, a hinge term asks each scene
to have a loss at least --margin nats lower under its caption than under the same
caption with two random objects' places exchanged. Every --eval-every steps it prints the loss and
position accuracy of each kind of evaluation scene, saves real and generated
preview scenes as <checkpoints>/<run>/samples-<step>.png, and saves checkpoint.pt
in the same folder. At the end a full run scores every evaluation scene.
--quick uses <run>-quick, trains for 40 steps on 2,000 train scenes, evaluates
every 20 steps, and scores every 50th evaluation scene, including at the end.

A restarted run resumes from its last saved evaluation. The checkpoint holds the
random states, settings, and fingerprints of the data, tokenizer and palette;
other saved settings or inputs stop the resume. The checkpoint directory and
device may change; numerical results can differ across hardware and software.

    uv run python -m pixl.train
    uv run python -m pixl.train --quick   # a short run that checks every step

A check on a Mac is lighter on the CPU, at low priority and on a few threads, which
leaves the GPU and most cores to everything else:

    OMP_NUM_THREADS=4 nice -n 19 uv run python -m pixl.train --quick --device cpu
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F

from .data import PLACES, load
from .model import GUIDANCE, TEMPERATURE, TOP_K, Config, EncoderPixl, encode_captions
from .preprocess import PALETTE, decode, load_palette
from .tokenizer import TOKENIZER, Tokenizer

KINDS = ("train", "new sprites", "new pair")
# Caption slot: the longest scene caption has 80 tokens, plus <bos> and <eos>.
CAPTION = 96
TRAIN_EVAL = 500  # train scenes among the evaluation scenes, next to every test scene
EVAL_SCENES = 4  # Full runs: every fourth scene during training, all scenes at the end; quick runs use 50.
EVAL_BATCH = 64
# Picks the evaluated train scenes and seeds the preview samples, whatever --seed is, so every run
# is measured on the same scenes.
EVAL_SEED = 0
# The published model on the Hugging Face Hub.
MODEL_REPO = "ocdbytes/pixl"


@dataclass
class Args(argparse.Namespace):
    run: str = "scenes-four"
    batch: int = 32
    epochs: int = 1
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup: int = 1000
    eval_every: int = 2000
    checkpoints: Path = Path("checkpoints")
    quick: bool = False
    device: str = "auto"
    margin: float = 2.0
    seed: int = 0
    caption_drop: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


# A place phrase; every scene caption has one per object.
PLACE_PHRASE = re.compile(rf"\bat the (?:{'|'.join(PLACES)})\b")


def swap_places(caption: str, rng: random.Random) -> str:
    """Exchanges the places of two random objects in a scene caption.

    Args:
        caption: A caption with a place phrase ("at the top left") for each object.
        rng: Picks the two objects.

    Returns:
        The caption with those two place phrases exchanged.
    """
    places = PLACE_PHRASE.findall(caption)
    i, j = rng.sample(range(len(places)), 2)
    places[i], places[j] = places[j], places[i]
    swapped = iter(places)
    return PLACE_PHRASE.sub(lambda _: next(swapped), caption)


def pick_device(name: str = "auto") -> str:
    """The PyTorch device to use: name itself, or for "auto" CUDA, then Apple's GPU (MPS), then the CPU."""
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def run_folder(args: Args) -> Path:
    """The folder of a run's checkpoint and sample images: <checkpoints>/<run>, or <run>-quick."""
    return args.checkpoints.joinpath(f"{args.run}-quick" if args.quick else args.run)


def clock(seconds: float) -> str:
    """Seconds as h:mm:ss."""
    return str(datetime.timedelta(seconds=round(seconds)))


def fingerprint(*parts: bytes | memoryview) -> str:
    """A short SHA-256 of the parts, which tells whether a resumed run sees the same inputs."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()[:16]


def save_samples(path: Path, rows: list[np.ndarray], palette: np.ndarray) -> None:
    """Saves rows of (n, 32, 32) token grids as one PNG, each scene drawn 4 times larger."""
    grid = np.concatenate([np.concatenate(list(row), axis=1) for row in rows], axis=0)
    image = decode(grid, palette)
    image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST).save(path)


def load_model(path: Path, device: str = "cpu") -> tuple[EncoderPixl, dict[str, Any]]:
    """Loads a trained model from a checkpoint.

    Args:
        path: A checkpoint saved by train().
        device: Where to put the model.

    Returns:
        The model in eval mode, and the whole checkpoint on the CPU: its log,
        preview, settings and, after the last step, final scores (subsampled in
        quick mode).
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = EncoderPixl(Config(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    return model, state


def load_pretrained(repo: str = MODEL_REPO, bf16: bool = False, device: str = "cpu") -> EncoderPixl:
    """Downloads a published model from the Hugging Face Hub.

    Args:
        repo: Model repo id.
        bf16: Download the half-size bf16 copy of the weights instead of the
            float32 ones. Either way the model computes in float32.
        device: Where to put the model.

    Returns:
        The model in eval mode.

    Raises:
        ValueError: If the repo's tokenizer or palette differs from this package's,
            which the model's inputs and outputs depend on.
    """
    for name, bundled in (("tokenizer.json", TOKENIZER), ("palette.json", PALETTE)):
        if Path(hf_hub_download(repo, name)).read_bytes() != bundled.read_bytes():
            raise ValueError(f"{repo}'s {name} differs from the one in this pixl package")
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))
    model = EncoderPixl(Config(**config))
    # load_state_dict copies bf16 values into the float32 parameters.
    model.load_state_dict(load_file(hf_hub_download(repo, "model-bf16.safetensors" if bf16 else "model.safetensors")))
    return model.to(device).eval()


@dataclass
class EvalScenes:
    """The evaluation pool: TRAIN_EVAL train scenes, then every test scene.

    Attributes:
        tokens: (n, 32, 32) palette tokens.
        captions: Each scene's caption.
        swapped: Each caption with its first two objects' quadrants exchanged.
        kinds: Each scene's kind, one of KINDS.
    """

    tokens: torch.Tensor
    captions: list[str]
    swapped: list[str]
    kinds: np.ndarray

    @property
    def preview(self) -> list[int]:
        """The scenes drawn at every evaluation: the first 4 train scenes and the first 4 new pairs."""
        return [int(i) for kind in ("train", "new pair") for i in np.flatnonzero(self.kinds == kind)[:4]]


def load_scenes(quick: bool) -> tuple[dict[str, Any], EvalScenes, dict[str, str]]:
    """Loads the train scenes and the evaluation pool, and fingerprints the inputs.

    Args:
        quick: Use the first 2,000 train scenes instead of all of them.

    Returns:
        The train split's columns ("tokens", "caption", "swapped"), the evaluation
        pool, and short SHA-256s of the data, tokenizer and palette.
    """
    train_set = load("train[:2000]" if quick else "train", keep=("swapped",))[:]
    test_set = load("test", keep=("kind", "swapped"))[:]
    tokens, captions = train_set["tokens"], train_set["caption"]
    picked = np.sort(np.random.default_rng(EVAL_SEED).choice(len(captions), min(TRAIN_EVAL, len(captions)), replace=False))
    scenes = EvalScenes(
        tokens=torch.cat([tokens[torch.from_numpy(picked)], test_set["tokens"]]),
        captions=[captions[i] for i in picked] + test_set["caption"],
        swapped=[train_set["swapped"][i] for i in picked] + test_set["swapped"],
        kinds=np.array(["train"] * len(picked) + test_set["kind"]),
    )
    fingerprints = {
        "data": fingerprint(
            np.ascontiguousarray(tokens.numpy()).data, np.ascontiguousarray(test_set["tokens"].numpy()).data,
            "\n".join(captions + train_set["swapped"] + test_set["caption"] + test_set["swapped"] + test_set["kind"]).encode(),
        ),
        "tokenizer": fingerprint(TOKENIZER.read_bytes()),
        "palette": fingerprint(PALETTE.read_bytes()),
    }
    return train_set, scenes, fingerprints


def build(device: str, args: Args) -> tuple[EncoderPixl, torch.optim.AdamW]:
    """A new model, and AdamW with weight decay on its matrices and embeddings only.

    PyTorch is seeded first, so the same seed gives the same initial weights.

    Args:
        device: Where to put the model.
        args: The run's settings: seed, peak_lr (the starting learning rate; training
            sets it every step), weight_decay, beta1 and beta2.
    """
    torch.manual_seed(args.seed)
    # Recompute (gradient checkpointing): batch 32 in float32 runs out of a 24 GB Mac's GPU memory without it.
    model = EncoderPixl(Config(caption=CAPTION, recompute=device != "cuda", mask_padding=True)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": args.weight_decay},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=args.peak_lr,
        betas=(args.beta1, args.beta2),
    )
    return model, optimizer


def learning_rate(step: int, steps: int, warmup: int, peak: float, low: float) -> float:
    """Linear warm-up to the peak rate by step warmup - 1, then cosine decay that reaches the low
    rate at step `steps`, one past the last training step, which ends a hair above it."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * progress))


def batch_loss(
    model: EncoderPixl,
    encode: Callable[[list[str]], torch.Tensor],
    texts: list[str],
    pixels: torch.Tensor,
    margin: float,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The training loss of one batch: pixel loss plus the position margin.

    Args:
        model: The model, in training mode.
        encode: Turns captions into caption ids on the model's device.
        texts: Each scene's caption; "" where caption dropout emptied it.
        pixels: (batch, pixels) pixel tokens on the model's device.
        margin: Nats per scene each kept caption's loss must stay below that of
            the same caption with two random objects' places exchanged; 0 skips
            the exchanged captions.
        rng: Picks the objects whose places are exchanged.

    Returns:
        The loss to minimise, and the true captions' mean pixel loss alone.
    """
    kept = [i for i, text in enumerate(texts) if text] if margin else []
    both = torch.cat([pixels, pixels[kept]])
    device = pixels.device.type
    # bf16 on NVIDIA GPUs: faster matrix multiplies and half the activation memory.
    with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
        scene_losses = model.loss(encode(texts + [swap_places(texts[i], rng) for i in kept]), both)
    right, wrong = scene_losses[: len(texts)], scene_losses[len(texts) :]
    # The margin is in nats per scene and the losses per pixel; dropped captions add 0.
    gap = (wrong - right[kept]) * pixels.shape[1]
    pixel_loss = right.mean()
    return pixel_loss + F.relu(margin - gap).sum() / (pixels.shape[1] * len(texts)), pixel_loss


@torch.no_grad()
def losses(model: EncoderPixl, encode: Callable[[list[str]], torch.Tensor], tokens: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """(n,) mean loss per pixel of scene i under texts[i], in batches of EVAL_BATCH, on the CPU.

    Runs in float32, without the bf16 autocast training uses on CUDA, so these losses and the
    logged batch loss are not strictly comparable (docs/training.md#measures).
    """
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(texts), EVAL_BATCH):
        pixels = tokens[start : start + EVAL_BATCH].flatten(1).long().to(device)
        result.append(model.loss(encode(texts[start : start + EVAL_BATCH]), pixels).cpu())
    return torch.cat(result)


def scores(
    model: EncoderPixl, encode: Callable[[list[str]], torch.Tensor], scenes: EvalScenes, every: int
) -> dict[str, float]:
    """Loss, and how often the true caption beats the swapped one (chance 0.5), per kind.

    Args:
        model: The model; in eval mode while scoring, then back in training mode.
        encode: Turns captions into caption ids on the model's device.
        scenes: The evaluation pool.
        every: Scores every this many-th scene of the pool.

    Returns:
        "<kind> loss" and "<kind> position" for each kind in KINDS.
    """
    rows = list(range(0, len(scenes.kinds), every))
    tokens = scenes.tokens[rows]
    model.eval()
    right = losses(model, encode, tokens, [scenes.captions[i] for i in rows])
    wrong = losses(model, encode, tokens, [scenes.swapped[i] for i in rows])
    model.train()
    kinds = scenes.kinds[rows]
    result = {}
    for kind in KINDS:
        mask = torch.from_numpy(kinds == kind)
        result[f"{kind} loss"] = right[mask].mean().item()
        result[f"{kind} position"] = (right[mask] < wrong[mask]).float().mean().item()
    return result


def describe(point: dict[str, float]) -> str:
    """Each kind's loss and position from scores(), as one log line part."""
    return " | ".join(f"{kind} loss {point[f'{kind} loss']:.3f} pos {point[f'{kind} position']:.3f}" for kind in KINDS)


@torch.no_grad()
def samples(model: EncoderPixl, encode: Callable[[list[str]], torch.Tensor], texts: list[str]) -> np.ndarray:
    """(len(texts), side, side) samples at the model module's TEMPERATURE, TOP_K and GUIDANCE, from EVAL_SEED.

    The random states are forked, so sampling never changes what training draws.
    """
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(EVAL_SEED)
        images = model.generate(encode(texts), temperature=TEMPERATURE, top_k=TOP_K, guidance=GUIDANCE)
    model.train()
    side = model.config.side
    return images.view(-1, side, side).cpu().numpy()


def random_states(rng: random.Random, device: str) -> dict[str, Any]:
    """The states of every random source training draws from."""
    states: dict[str, Any] = {"python": rng.getstate(), "torch": torch.get_rng_state()}
    if device == "cuda":
        states["cuda"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_random(states: dict[str, Any], rng: random.Random, device: str) -> None:
    """Restores random_states(); a GPU state saved on another kind of device is skipped."""
    rng.setstate(states["python"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and device == "cuda":
        torch.cuda.set_rng_state_all(states["cuda"])
    elif "mps" in states and device == "mps":
        torch.mps.set_rng_state(states["mps"])


def resume(
    checkpoint: Path,
    model: EncoderPixl,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    device: str,
    expected: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, float]]]:
    """Restores a run from its checkpoint, when it has one.

    Args:
        checkpoint: The run's checkpoint path.
        model: Gets the saved weights.
        optimizer: Gets the saved optimizer state.
        rng: Gets the saved Python random state.
        device: The training device.
        expected: Checkpoint key ("settings", "fingerprints", "config") -> its
            values now; the saved values must match.

    Returns:
        The steps done and the log so far; 0 and [] without a checkpoint.

    Raises:
        SystemExit: If the checkpoint was made with other settings, inputs or model shape.
    """
    if not checkpoint.exists():
        return 0, []
    # On the CPU: the model and optimizer move their tensors to the device, and PyTorch
    # takes the random states on the CPU.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, current in expected.items():
        saved = state[name]
        changed = {key: (saved.get(key), value) for key, value in current.items() if saved.get(key) != value}
        if changed:
            raise SystemExit(f"{checkpoint} was made with other {name} (saved, now): {changed}; use another --run")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_random(state["random"], rng, device)
    print(f"resumed from {checkpoint} at step {state['step']:,}")
    return state["step"], state["log"]


def save(checkpoint: Path, state: dict[str, Any]) -> None:
    """Writes a checkpoint through a temporary file, so a stop mid-write leaves the previous one whole."""
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint)


def train(args: Args) -> Path:
    """Trains one run, or resumes it from its checkpoint.

    Args:
        args: Run settings; see parse_args. The checkpoint directory and device
            are excluded from the saved settings comparison.

    Returns:
        The path of the run's checkpoint.

    Raises:
        SystemExit: If the checkpoint was made with other settings, data,
            tokenizer or palette.
    """
    device = pick_device(args.device)
    if device == "cuda":
        # Float32 matrix multiplies on the tensor cores (TF32); training itself runs in bf16.
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    folder = run_folder(args)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder.joinpath("checkpoint.pt")
    palette = load_palette()
    # Where the run is stored and what it runs on may change between resumes.
    settings = {key: value for key, value in vars(args).items() if key not in ("checkpoints", "device")}

    train_set, scenes, fingerprints = load_scenes(args.quick)
    tokens, captions = train_set["tokens"], train_set["caption"]
    # Quick mode: 40 steps, 10 of them warm-up, scoring every 50th scene every 20 steps and at the end.
    steps, warmup, eval_every, eval_scenes, final_scenes = (
        (40, 10, 20, 50, 50) if args.quick
        else (args.epochs * len(captions) // args.batch, args.warmup, args.eval_every, EVAL_SCENES, 1)
    )
    shuffle = torch.Generator().manual_seed(args.seed)
    order = torch.cat([torch.randperm(len(captions), generator=shuffle) for _ in range(args.epochs)])
    tokenizer = Tokenizer.load()
    rng = random.Random(args.seed)  # caption dropout and place swaps
    model, optimizer = build(device, args)
    print(f"device {device} | {len(captions):,} train scenes | {steps:,} steps of {args.batch} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | {folder} | {fingerprints}")
    # The model's config too, except recompute, which follows the device: a run trained without
    # padding masks must not resume with them.
    shape = {key: value for key, value in asdict(model.config).items() if key != "recompute"}
    step, log = resume(checkpoint, model, optimizer, rng, device,
                       {"settings": settings, "fingerprints": fingerprints, "config": shape})

    def encode(texts: list[str]) -> torch.Tensor:
        """(len(texts), CAPTION) caption ids on the device."""
        return encode_captions(texts, tokenizer, length=CAPTION).to(device)

    preview = scenes.preview
    preview_set = {"captions": [scenes.captions[i] for i in preview], "swapped": [scenes.swapped[i] for i in preview],
                   "tokens": scenes.tokens[preview]}

    def state(**extra: object) -> dict[str, Any]:
        """Everything a resume needs, plus the log and preview both notebooks read."""
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(model.config),
                "step": step, "log": log, "preview": preview_set, "settings": settings,
                "fingerprints": fingerprints, "random": random_states(rng, device), **extra}

    started, window, tick = time.time(), [], time.time()
    model.train()
    while step < steps:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, warmup, args.peak_lr, args.min_lr)
        rows = order[step * args.batch : (step + 1) * args.batch]
        pixels = tokens[rows].flatten(1).long().to(device)
        # Caption dropout empties some captions, so samples can use guidance.
        texts = ["" if rng.random() < args.caption_drop else captions[i] for i in rows.tolist()]
        loss, pixel_loss = batch_loss(model, encode, texts, pixels, args.margin, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        window.append(pixel_loss.item())
        step += 1
        if step % eval_every == 0 or step == steps:
            seconds = (time.time() - tick) / len(window)
            point = {"step": step, "batch loss": sum(window) / len(window), "seconds per step": seconds}
            point |= scores(model, encode, scenes, eval_scenes)
            log.append(point)
            print(" | ".join([f"step {step:,}", f"batch loss {point['batch loss']:.3f}", describe(point),
                              f"{seconds:.2f} s/step", f"elapsed {clock(time.time() - started)}",
                              f"left {clock((steps - step) * seconds)}"]), flush=True)
            drawn = samples(model, encode, preview_set["captions"])
            save_samples(folder.joinpath(f"samples-{step:06d}.png"), [preview_set["tokens"].numpy(), drawn], palette)
            save(checkpoint, state())
            if device == "mps":
                torch.mps.empty_cache()  # drop the cache the measures left behind
            window, tick = [], time.time()

    final = scores(model, encode, scenes, final_scenes)
    print("final evaluation scores: " + describe(final))
    save(checkpoint, state(final=final))
    return checkpoint


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("--run", default=Args.run, help="run name: its folder in --checkpoints")
    _ = parser.add_argument("--batch", type=int, default=Args.batch)
    _ = parser.add_argument("--epochs", type=int, default=Args.epochs, help="passes over the train scenes")
    _ = parser.add_argument("--peak-lr", type=float, default=Args.peak_lr)
    _ = parser.add_argument("--min-lr", type=float, default=Args.min_lr)
    _ = parser.add_argument("--warmup", type=int, default=Args.warmup, help="warm-up steps")
    _ = parser.add_argument("--eval-every", type=int, default=Args.eval_every)
    _ = parser.add_argument("--checkpoints", type=Path, default=Args.checkpoints)
    _ = parser.add_argument(
        "--caption-drop", type=float, default=Args.caption_drop,
        help="share of captions left empty, so samples can use guidance",
    )
    _ = parser.add_argument("--weight-decay", type=float, default=Args.weight_decay, help="on matrices and embeddings")
    _ = parser.add_argument("--beta1", type=float, default=Args.beta1, help="AdamW's first-moment decay")
    _ = parser.add_argument("--beta2", type=float, default=Args.beta2, help="AdamW's second-moment decay")
    _ = parser.add_argument("--grad-clip", type=float, default=Args.grad_clip, help="largest gradient norm")
    _ = parser.add_argument(
        "--seed", type=int, default=Args.seed,
        help="seeds the initial weights, the order of the scenes, caption dropout and place swaps; "
        "the evaluated scenes and preview samples stay the same for every seed",
    )
    _ = parser.add_argument(
        "--margin", type=float, default=Args.margin,
        help="nats per scene the true caption must beat a place-swapped one by; 0 turns it off",
    )
    _ = parser.add_argument("--quick", action="store_true", help="40 steps on 2,000 train scenes: checks every step works")
    _ = parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default=Args.device,
        help="auto: CUDA, then Apple's GPU, then the CPU",
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    # Apple GPU allocation limit: 0.9x the recommended maximum. Above 0.6x,
    # attempt garbage collection and more frequent command-buffer commits.
    # Read when the GPU allocator is first initialised.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    _ = train(parse_args())


if __name__ == "__main__":
    main()
