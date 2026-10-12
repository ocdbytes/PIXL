"""Training displays for notebooks, in a dark terminal style: a live panel, log
lines and plots.

- panel(): a terminal-style text box (HTML) with progress, losses and a
  loss sparkline; update it in place with an IPython display handle.
- log_line(): a one-line log entry that stays in the
  notebook output as a record of training.
- dashboard(): a dark matplotlib figure with the loss curves, the learning rate
  and a grid of generated images next to real ones.
- filmstrip(): how the samples for each caption changed over training.
- grid(): labelled rows of images, e.g. real photos or samples at several settings.

Images are 32x32 palette-token grids (see src/preprocess.py), drawn without
smoothing so every pixel stays sharp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.preprocess import decode, load_palette

BACKGROUND = "#0e1210"
GREEN = "#9bd4a0"
DIM = "#55705a"
GRID = "#1c2620"
AMBER = "#e3b45a"
CYAN = "#7fb2c9"
FONT = ["Menlo", "DejaVu Sans Mono", "Courier New", "monospace"]
SPARKS = "▁▂▃▄▅▆▇█"
WIDTH = 66  # inner width of the panel, in characters

THEME: dict[str, Any] = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": DIM,
    "axes.labelcolor": GREEN,
    "axes.titlecolor": GREEN,
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linestyle": ":",
    "xtick.color": DIM,
    "ytick.color": DIM,
    "text.color": GREEN,
    "font.family": "monospace",
    "font.monospace": FONT,
    "font.size": 9,
    "legend.frameon": False,
    "legend.labelcolor": GREEN,
    "savefig.facecolor": BACKGROUND,
}


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """Draws the last width values as block characters, low to high."""
    values = list(values)[-width:]
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1.0
    return "".join(SPARKS[int((v - low) / span * (len(SPARKS) - 1))] for v in values)


def bar(fraction: float, width: int = 30) -> str:
    """Draws a progress bar such as ████████▒░░░░."""
    filled = fraction * width
    full = int(filled)
    partial = "▒" if filled - full >= 0.5 and full < width else ""
    return ("█" * full + partial).ljust(width, "░")


def clock(seconds: float) -> str:
    """Formats seconds as H:MM:SS."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def panel(title: str, lines: Sequence[str], colour: str = GREEN) -> str:
    """Builds the status panel as HTML.

    Args:
        title: Text in the top border.
        lines: Panel rows; each is padded or cut to the panel width.
        colour: Text colour.

    Returns:
        HTML for IPython.display.HTML.
    """
    top = f"┌─ {title} ".ljust(WIDTH + 1, "─") + "┐"
    rows = [f"│ {line[: WIDTH - 2].ljust(WIDTH - 2)} │" for line in lines]
    text = "\n".join([top, *rows, "└" + "─" * WIDTH + "┘"])
    return (
        f'<pre style="display:inline-block; margin:0; padding:8px 12px;'
        f" border-radius:6px; background:{BACKGROUND}; color:{colour};"
        f' font-family:{", ".join(FONT)}; font-size:13px; line-height:1.25">{text}</pre>'
    )


def log_line(fields: dict[str, str], highlight: bool = False) -> str:
    """Builds one log line, e.g. time 0:12:04 │ step 1,500 │ loss 3.412.

    Args:
        fields: Name -> value, shown in order.
        highlight: Mark the line, e.g. for a new best test loss.

    Returns:
        A string with grey names (ANSI colours), for print().
    """
    grey, amber, reset = "\033[90m", "\033[33m", "\033[0m"
    body = f" {grey}│{reset} ".join(f"{grey}{k}{reset} {v}" for k, v in fields.items())
    return body + (f"  {amber}best{reset}" if highlight else "")


def to_rgb(tokens: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Turns a token grid into an RGB array; transparent pixels show the background."""
    rgba = np.asarray(decode(tokens.astype(np.uint8), palette)).astype(np.float32)
    back = np.array([int(BACKGROUND[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
    alpha = rgba[..., 3:] / 255
    return (rgba[..., :3] * alpha + back * (1 - alpha)).astype(np.uint8)


def dashboard(
    train: list[dict[str, float]],
    test: list[dict[str, float]],
    samples: np.ndarray | None,
    references: np.ndarray,
    captions: list[str],
    title: str,
) -> Figure:
    """Draws the training dashboard.

    Args:
        train: Logged train points with "step", "loss" and "lr".
        test: Logged test points with "step" and "loss".
        samples: (n, 32, 32) generated tokens for captions, or None.
        references: (n, 32, 32) real tokens for the same captions.
        captions: Caption of each column.
        title: Figure title.

    Returns:
        The figure; close it with plt.close after showing it.
    """
    palette = load_palette()
    with plt.rc_context(THEME):
        figure = plt.figure(figsize=(13, 7.2))
        grid = GridSpec(
            3, len(captions), figure=figure, height_ratios=[2.2, 1, 1], hspace=0.3
        )
        figure.suptitle(title, fontsize=11, x=0.02, ha="left")

        half = len(captions) // 2
        losses = figure.add_subplot(grid[0, :half])
        if train:
            steps = [p["step"] for p in train]
            raw = [p["loss"] for p in train]
            smooth = np.convolve(
                raw, np.ones(min(10, len(raw))) / min(10, len(raw)), mode="valid"
            )
            losses.plot(steps, raw, color=DIM, linewidth=0.8, label="train")
            losses.plot(
                steps[len(steps) - len(smooth) :],
                smooth,
                color=GREEN,
                linewidth=1.8,
                label="train (smoothed)",
            )
        if test:
            losses.plot(
                [p["step"] for p in test],
                [p["loss"] for p in test],
                color=AMBER,
                marker="s",
                markersize=4,
                linewidth=1.5,
                label="test",
            )
        losses.set_title("loss")
        losses.set_xlabel("step")
        losses.legend(loc="upper right")

        rate = figure.add_subplot(grid[0, half:])
        if train:
            rate.plot(
                [p["step"] for p in train],
                [p["lr"] for p in train],
                color=CYAN,
                linewidth=1.5,
            )
        rate.set_title("learning rate")
        rate.set_xlabel("step")
        rate.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        for column, caption in enumerate(captions):
            for row, (images, label) in enumerate(
                ((references, "real"), (samples, "model"))
            ):
                axis = figure.add_subplot(grid[1 + row, column])
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                if images is not None:
                    axis.imshow(
                        to_rgb(images[column], palette), interpolation="nearest"
                    )
                if row == 0:
                    axis.set_title(caption[:14], fontsize=8, loc="center")
                if column == 0:
                    axis.set_ylabel(label)
    return figure


def filmstrip(snapshots: list[tuple[int, np.ndarray]], captions: list[str]) -> Figure:
    """Draws every caption's samples across training, one row per caption.

    Args:
        snapshots: (step, (n, 32, 32) tokens) per evaluation, in order.
        captions: Caption of each of the n samples.

    Returns:
        The figure.
    """
    palette = load_palette()
    with plt.rc_context(THEME):
        columns = len(snapshots)
        figure, axes = plt.subplots(
            len(captions),
            columns,
            figsize=(1.1 * columns + 1.5, 1.15 * len(captions)),
            squeeze=False,
        )
        figure.suptitle("Samples over training", fontsize=11, x=0.02, ha="left")
        for column, (step, images) in enumerate(snapshots):
            for row, caption in enumerate(captions):
                axis = axes[row][column]
                axis.imshow(to_rgb(images[row], palette), interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                if row == 0:
                    axis.set_title(f"{step:,}", fontsize=7, color=DIM, loc="center")
                if column == 0:
                    axis.set_ylabel(
                        caption[:12], fontsize=7, rotation=0, ha="right", va="center"
                    )
    return figure


def grid(rows: list[tuple[str, np.ndarray]], captions: list[str], title: str) -> Figure:
    """Draws labelled rows of images, one column per caption.

    Args:
        rows: (row label, (n, 32, 32) tokens) per row.
        captions: Caption of each of the n columns.
        title: Figure title.

    Returns:
        The figure.
    """
    palette = load_palette()
    with plt.rc_context(THEME):
        figure, axes = plt.subplots(
            len(rows),
            len(captions),
            figsize=(1.45 * len(captions) + 0.8, 1.6 * len(rows) + 0.5),
            squeeze=False,
        )
        figure.suptitle(title, fontsize=11, x=0.02, ha="left")
        for row, (label, images) in enumerate(rows):
            for column, caption in enumerate(captions):
                axis = axes[row][column]
                axis.imshow(to_rgb(images[column], palette), interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                if row == 0:
                    axis.set_title(caption[:14], fontsize=8, loc="center")
                if column == 0:
                    axis.set_ylabel(label)
    return figure
