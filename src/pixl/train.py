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
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import load
from .model import Config, EncoderPixl, encode_captions
from .preprocess import PALETTE, decode, load_palette
from .tokenizer import TOKENIZER, Tokenizer

KINDS = ("train", "new sprites", "new pair")
# Caption slot: the longest scene caption has 80 tokens, plus <bos> and <eos>.
CAPTION = 96
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CAPTION_DROP = 0.1  # share of captions left empty, so samples can use guidance
TRAIN_EVAL = 500  # train scenes among the evaluation scenes, next to every test scene
EVAL_SCENES = 4  # Full runs: every fourth scene during training, all scenes at the end; quick runs use 50.
EVAL_BATCH = 64
TEMPERATURE, GUIDANCE = 0.7, 3.0  # for the preview samples
SEED = 0


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


# A place phrase; every scene caption has one per object.
PLACES = re.compile(r"\bat the (?:top|bottom) (?:left|right)\b")


def swap_places(caption: str, rng: random.Random) -> str:
    """Exchanges the places of two random objects in a scene caption.

    Args:
        caption: A caption with a place phrase ("at the top left") for each object.
        rng: Picks the two objects.

    Returns:
        The caption with those two place phrases exchanged.
    """
    places = PLACES.findall(caption)
    i, j = rng.sample(range(len(places)), 2)
    places[i], places[j] = places[j], places[i]
    swapped = iter(places)
    return PLACES.sub(lambda _: next(swapped), caption)


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
    eval_every, warmup, eval_scenes = (20, 10, 50) if args.quick else (args.eval_every, args.warmup, EVAL_SCENES)
    folder = run_folder(args)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder.joinpath("checkpoint.pt")
    palette = load_palette()
    # Where the run is stored and what it runs on may change between resumes.
    settings = {key: value for key, value in vars(args).items() if key not in ("checkpoints", "device")}

    # Full training uses every train scene; quick mode uses the first 2,000.
    # Evaluation combines up to TRAIN_EVAL train scenes with every test scene.
    train_set = load("scenes", "train[:2000]" if args.quick else "train", keep=("swapped",))[:]
    test_set = load("scenes", "test", keep=("kind", "swapped"))[:]
    tokens, captions = train_set["tokens"], train_set["caption"]
    picked = np.sort(np.random.default_rng(SEED).choice(len(captions), min(TRAIN_EVAL, len(captions)), replace=False))
    eval_tokens = torch.cat([tokens[torch.from_numpy(picked)], test_set["tokens"]])
    eval_captions = [captions[i] for i in picked] + test_set["caption"]
    eval_swapped = [train_set["swapped"][i] for i in picked] + test_set["swapped"]
    eval_kinds = np.array(["train"] * len(picked) + test_set["kind"])
    fingerprints = {
        "data": fingerprint(
            np.ascontiguousarray(tokens.numpy()).data, np.ascontiguousarray(test_set["tokens"].numpy()).data,
            "\n".join(captions + train_set["swapped"] + test_set["caption"] + test_set["swapped"] + test_set["kind"]).encode(),
        ),
        "tokenizer": fingerprint(TOKENIZER.read_bytes()),
        "palette": fingerprint(PALETTE.read_bytes()),
    }
    # The preview: 4 train scenes and 4 new pairs.
    preview = [int(i) for kind in ("train", "new pair") for i in np.flatnonzero(eval_kinds == kind)[:4]]
    steps = 40 if args.quick else args.epochs * len(captions) // args.batch
    shuffle = torch.Generator().manual_seed(SEED)
    order = torch.cat([torch.randperm(len(captions), generator=shuffle) for _ in range(args.epochs)])

    tokenizer = Tokenizer.load()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)  # caption dropout and place swaps
    # Recompute (gradient checkpointing): batch 32 in float32 runs out of a 24 GB Mac's GPU memory without it.
    model = EncoderPixl(Config(caption=CAPTION, recompute=device != "cuda")).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=args.peak_lr,
        betas=(0.9, 0.95),
    )
    print(f"device {device} | {len(captions):,} train scenes | {steps:,} steps of {args.batch} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | {folder} | {fingerprints}")

    def learning_rate(step: int) -> float:
        """Linear warm-up to the peak rate, then cosine decay to the minimum at the last step."""
        if step < warmup:
            return args.peak_lr * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return args.min_lr + 0.5 * (args.peak_lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    def ids(texts: list[str]) -> torch.Tensor:
        """(len(texts), CAPTION) caption ids on the device."""
        return encode_captions(texts, tokenizer, length=CAPTION).to(device)

    @torch.no_grad()
    def losses(texts: list[str], rows: list[int]) -> torch.Tensor:
        """(n,) mean loss per pixel of evaluation scene rows[i] under texts[i], on the CPU."""
        result = []
        for start in range(0, len(rows), EVAL_BATCH):
            pixels = eval_tokens[rows[start : start + EVAL_BATCH]].flatten(1).long().to(device)
            result.append(model.loss(ids(texts[start : start + EVAL_BATCH]), pixels).cpu())
        return torch.cat(result)

    def scores(every: int) -> dict[str, float]:
        """Loss, and how often the true caption beats the swapped one (chance 0.5), per kind."""
        model.eval()
        rows = list(range(0, len(eval_kinds), every))
        right = losses([eval_captions[i] for i in rows], rows)
        wrong = losses([eval_swapped[i] for i in rows], rows)
        model.train()
        kinds = eval_kinds[rows]
        result = {}
        for kind in KINDS:
            mask = torch.from_numpy(kinds == kind)
            result[f"{kind} loss"] = right[mask].mean().item()
            result[f"{kind} position"] = (right[mask] < wrong[mask]).float().mean().item()
        return result

    @torch.no_grad()
    def samples(texts: list[str]) -> np.ndarray:
        """(len(texts), 32, 32) samples at TEMPERATURE and GUIDANCE, from a fixed seed.

        The random states are forked, so sampling never changes what training draws.
        """
        model.eval()
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            images = model.generate(ids(texts), temperature=TEMPERATURE, top_k=32, guidance=GUIDANCE)
        model.train()
        return images.view(-1, 32, 32).cpu().numpy()

    def random_states() -> dict[str, Any]:
        """The states of every random source training draws from."""
        states: dict[str, Any] = {"python": rng.getstate(), "torch": torch.get_rng_state()}
        if device == "cuda":
            states["cuda"] = torch.cuda.get_rng_state_all()
        elif device == "mps":
            states["mps"] = torch.mps.get_rng_state()
        return states

    log: list[dict[str, float]] = []
    step = 0
    if checkpoint.exists():
        # On the CPU: the model and optimizer move their tensors to the device, and PyTorch
        # takes the random states on the CPU.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name, saved, current in (("settings", state["settings"], settings), ("inputs", state["fingerprints"], fingerprints)):
            changed = {key: (saved.get(key), value) for key, value in current.items() if saved.get(key) != value}
            if changed:
                raise SystemExit(f"{checkpoint} was made with other {name} (saved, now): {changed}; use another --run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, log = state["step"], state["log"]
        states = state["random"]
        rng.setstate(states["python"])
        torch.set_rng_state(states["torch"])
        if "cuda" in states and device == "cuda":
            torch.cuda.set_rng_state_all(states["cuda"])
        elif "mps" in states and device == "mps":
            torch.mps.set_rng_state(states["mps"])
        print(f"resumed from {checkpoint} at step {step:,}")

    def save(**extra: object) -> None:
        """Saves resume state and the logs/previews used by both notebooks."""
        preview_set = {"captions": [eval_captions[i] for i in preview], "swapped": [eval_swapped[i] for i in preview],
                       "tokens": eval_tokens[preview]}
        # Write a temporary file, then rename it: a stop mid-write leaves the previous checkpoint whole.
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(model.config),
                    "step": step, "log": log, "preview": preview_set, "settings": settings,
                    "fingerprints": fingerprints, "random": random_states(), **extra}, temporary)
        temporary.replace(checkpoint)

    started, window, tick = time.time(), [], time.time()
    model.train()
    while step < steps:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        rows = order[step * args.batch : (step + 1) * args.batch]
        pixels = tokens[rows].flatten(1).long().to(device)
        # Caption dropout leaves some captions empty. With a margin, each kept caption is
        # also scored with two of its objects' places exchanged.
        texts = ["" if rng.random() < CAPTION_DROP else captions[i] for i in rows.tolist()]
        kept = [i for i, text in enumerate(texts) if text] if args.margin else []
        both = torch.cat([pixels, pixels[kept]])
        # bf16 on NVIDIA GPUs: faster matrix multiplies and half the activation memory.
        with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            scene_losses = model.loss(ids(texts + [swap_places(texts[i], rng) for i in kept]), both)
        right, wrong = scene_losses[: len(texts)], scene_losses[len(texts) :]
        # The margin is in nats per scene and the losses per pixel; dropped captions add 0.
        gap = (wrong - right[kept]) * pixels.shape[1]
        pixel_loss = right.mean()
        loss = pixel_loss + F.relu(args.margin - gap).sum() / (pixels.shape[1] * len(texts))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        window.append(pixel_loss.item())
        step += 1
        if step % eval_every == 0 or step == steps:
            seconds = (time.time() - tick) / len(window)
            point = {"step": step, "batch loss": sum(window) / len(window), "seconds per step": seconds} | scores(eval_scenes)
            log.append(point)
            parts = [f"step {step:,}", f"batch loss {point['batch loss']:.3f}"]
            parts += [f"{kind} loss {point[f'{kind} loss']:.3f} pos {point[f'{kind} position']:.3f}" for kind in KINDS]
            parts += [f"{seconds:.2f} s/step", f"elapsed {clock(time.time() - started)}", f"left {clock((steps - step) * seconds)}"]
            print(" | ".join(parts), flush=True)
            save_samples(folder.joinpath(f"samples-{step:06d}.png"),
                         [eval_tokens[preview].numpy(), samples([eval_captions[i] for i in preview])], palette)
            save()
            if device == "mps":
                torch.mps.empty_cache()  # drop the cache the measures left behind
            window, tick = [], time.time()

    final = scores(eval_scenes if args.quick else 1)
    print("final evaluation scores: " + " | ".join(
        f"{kind} loss {final[f'{kind} loss']:.3f} pos {final[f'{kind} position']:.3f}" for kind in KINDS))
    save(final=final)
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
