"""A Gradio app that builds captions from the labels pixl knows and draws them.

Pick a sprite pack, its ground and an object for each quadrant, or start from an
example scene: the app writes the caption in the training template and draws it
with the published model. The menus offer only what vocabulary.json lists, so
every caption is one the model can read.

    uv run python -m pixl.app            # http://127.0.0.1:7860
    uv run python -m pixl.app --bf16     # downloads the half-size weights
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from importlib import resources
from typing import Any

import gradio as gr
import numpy as np
import torch
from gradio.themes import Base, GoogleFont
from PIL import Image

from .model import EncoderPixl, encode_captions
from .preprocess import decode, load_palette
from .tokenizer import Tokenizer
from .train import MODEL_REPO, load_pretrained, pick_device

# Per sprite pack: its display name, object labels with articles, and grounds ("" for none).
VOCABULARY = resources.files("pixl").joinpath("vocabulary.json")
PLACES = ("top left", "top right", "bottom left", "bottom right")
# The ground menu's entry for packs drawn without a ground.
NO_GROUND = "no ground"
# Ready-made scenes: name -> (pack, ground, objects in PLACES order).
EXAMPLES = {
    "Dungeon party": ("kenney_tiny-dungeon", "a dark dungeon floor",
                      ["a purple robed wizard", "a closed treasure chest", "a brown bat", "a red potion"]),
    "Village": ("kenney_tiny-town", "grass", ["a green pine tree", "a small wooden hut", "a wooden barrel", "a beehive"]),
    "Farmyard": ("kenney_tiny-farm", "tilled soil", ["a cow", "a sheep", "a hay bale", "a farmer"]),
    "Battlefield": ("kenney_tiny-battle", "a road",
                    ["a green heavy tank", "a blue helicopter", "a red army base", "a gray supply truck"]),
    "Naval battle": ("kenney_tiny-battle", "water",
                     ["a blue battleship", "a green gunboat", "a red landing craft", "an orange helicopter"]),
    "Shoot 'em up": ("kenney_pixel-shmup", "grass", ["a green tank", "a red armored truck", "a health pack", "a yellow bomb"]),
    "Forest": ("ansimuz_sunny-land", "rocky dirt", ["a tree stump", "a gem", "a thorny bramble", "a shrooms"]),
    "Monster lair": ("dcss_dungeon-crawl-32x32", "grass with flowers",
                     ["a skeleton hydra", "a war axe", "an altar xom", "a fire vortex"]),
    "Street": ("kenney_pixel-vehicle-pack", "", ["a yellow motorcycle", "a woman walking", "a red scooter", "a man pointing"]),
    "Space": ("grafxkid_mini-pixel-pack-3", "",
              ["a player ship", "an effects explosion", "an items power item", "an enemies alan"]),
}
START = "Dungeon party"
TOP_K = 32
SCALE = 8  # each drawn pixel becomes SCALE x SCALE on screen
HEADER = f"""
<header class="px-header">
  <div>
    <div class="px-title">pixl</div>
    <div class="px-subtitle">Text to 32×32 pixel art, drawn one pixel at a time.</div>
  </div>
  <nav>
    <a href="https://huggingface.co/{MODEL_REPO}" target="_blank">Model card</a>
    <a href="https://github.com/ocdbytes/PIXL" target="_blank">GitHub</a>
  </nav>
</header>
"""
CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.px-header { display: flex; justify-content: space-between; align-items: flex-end;
             padding: 32px 2px 22px; border-bottom: 1px solid var(--border-color-primary); }
.px-title { font-size: 30px; font-weight: 650; letter-spacing: -0.03em; line-height: 1; }
.px-subtitle { margin-top: 8px; font-size: 14px; color: var(--body-text-color-subdued); }
.px-header nav a { margin-left: 24px; font-size: 14px; text-decoration: none; color: var(--body-text-color-subdued); }
.px-header nav a:hover { color: var(--body-text-color); }
.px-section { margin: 14px 0 2px; font-size: 11px; font-weight: 600; letter-spacing: 0.09em;
              text-transform: uppercase; color: var(--body-text-color-subdued); }
.px-meta, .px-meta p { font-size: 12.5px !important; color: var(--body-text-color-subdued) !important; }
.px-caption, .px-caption p { font-family: var(--font-mono) !important; font-size: 12.5px !important;
                             line-height: 1.65 !important; color: var(--body-text-color-subdued) !important; }
.px-canvas { background: var(--background-fill-secondary) !important; border-radius: 12px !important; }
.px-accordion { border-top: 1px solid var(--border-color-primary) !important; border-radius: 0 !important; }
img { image-rendering: pixelated; }
"""


def launch_options() -> dict[str, Any]:
    """The look of the app, for Blocks.launch(): a neutral theme, its CSS, and no Gradio footer."""
    theme = Base(
        primary_hue="zinc", neutral_hue="zinc", radius_size="md", text_size="md",
        font=[GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#ffffff", body_background_fill_dark="#09090b",
        block_background_fill="#ffffff", block_background_fill_dark="#09090b",
        block_shadow="none", block_shadow_dark="none", input_shadow="none", button_primary_shadow="none",
        block_border_color="#e4e4e7", block_border_color_dark="#27272a",
        border_color_primary="#e4e4e7", border_color_primary_dark="#27272a",
        background_fill_secondary="#f4f4f5", background_fill_secondary_dark="#18181b",
        block_label_text_color="#71717a", block_label_text_color_dark="#a1a1aa",
        button_primary_background_fill="#18181b", button_primary_background_fill_hover="#3f3f46",
        button_primary_text_color="#fafafa",
        button_primary_background_fill_dark="#fafafa", button_primary_background_fill_hover_dark="#d4d4d8",
        button_primary_text_color_dark="#18181b",
        button_secondary_border_color="#e4e4e7", button_secondary_border_color_dark="#27272a",
        block_border_width="0px", form_gap_width="0px", block_padding="0px",
        body_text_color_subdued="#71717a", body_text_color_subdued_dark="#a1a1aa",
        input_background_fill="#ffffff", input_background_fill_dark="#09090b",
        input_border_color="#e4e4e7", input_border_color_dark="#27272a", input_border_width="1px",
        input_border_color_focus="#a1a1aa", input_border_color_focus_dark="#52525b",
        button_border_width="1px",
        button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#09090b",
        button_secondary_background_fill_hover="#f4f4f5", button_secondary_background_fill_hover_dark="#18181b",
        button_secondary_text_color="#27272a", button_secondary_text_color_dark="#e4e4e7",
    )
    return {"theme": theme, "css": CSS, "footer_links": []}


@dataclass
class Args(argparse.Namespace):
    bf16: bool = False
    device: str = "cpu"
    port: int = 7860
    share: bool = False


def load_vocabulary() -> dict[str, dict[str, Any]]:
    """Reads vocabulary.json: pack -> {"name", "objects", "grounds"}."""
    return json.loads(VOCABULARY.read_text(encoding="utf-8"))


def caption(objects: list[str], ground: str) -> str:
    """A caption in the training template: each object at its quadrant, in PLACES order, then the ground.

    Args:
        objects: An object label with its article for each quadrant, in PLACES order.
        ground: The ground, or "" or NO_GROUND for none.
    """
    parts = [f"{thing} at the {place}" for thing, place in zip(objects, PLACES, strict=True)]
    text = f"{', '.join(parts[:-1])} and {parts[-1]}"
    return f"{text} on {ground}" if ground and ground != NO_GROUND else text


def rotate(objects: list[str]) -> list[str]:
    """Moves each object one quadrant clockwise; objects and the result are in PLACES order."""
    top_left, top_right, bottom_left, bottom_right = objects
    return [bottom_left, top_left, bottom_right, top_right]


def build(model: EncoderPixl, vocabulary: dict[str, dict[str, Any]]) -> gr.Blocks:
    """The app's page and its event handlers, without launching it.

    Args:
        model: The model that draws, in eval mode.
        vocabulary: From load_vocabulary().
    """
    tokenizer, palette = Tokenizer.load(), load_palette()
    device = next(model.parameters()).device

    def grounds(pack: str) -> list[str]:
        return [ground or NO_GROUND for ground in vocabulary[pack]["grounds"]]

    def about(pack: str) -> str:
        """The pack menu's note: how many objects the pack offers."""
        return f"{len(vocabulary[pack]['objects']):,} objects"

    def ground_note(pack: str) -> str:
        """The ground menu's note: how many grounds the pack offers."""
        count = len(vocabulary[pack]["grounds"])
        return "Drawn without a ground" if vocabulary[pack]["grounds"] == [""] else f"{count} ground{'s' * (count > 1)}"

    def scene(pack: str, ground: str, objects: list[str]) -> list[Any]:
        """Updates that show a scene: the pack, its ground menu, the four object menus and the caption.

        The caption is written here, with the menus, rather than by each menu's own event: menus
        that change together would otherwise send each other's old values.
        """
        choices = vocabulary[pack]["objects"]
        return [
            gr.Dropdown(value=pack, info=about(pack)),
            gr.Dropdown(choices=grounds(pack), value=ground or NO_GROUND, info=ground_note(pack)),
            *(gr.Dropdown(choices=choices, value=thing) for thing in objects),
            caption(objects, ground),
        ]

    def random_scene(pack: str) -> list[Any]:
        """A random ground and four different objects from the pack."""
        return scene(pack, random.choice(grounds(pack)), random.sample(vocabulary[pack]["objects"], 4))

    def example(name: str) -> list[Any]:
        return scene(*EXAMPLES[name])

    def write(ground: str, top_left: str, top_right: str, bottom_left: str, bottom_right: str) -> str:
        objects = [top_left, top_right, bottom_left, bottom_right]
        return caption(objects, ground) if all(objects) else ""

    def turn(ground: str, top_left: str, top_right: str, bottom_left: str, bottom_right: str) -> list[str]:
        """The objects one quadrant clockwise, and their caption."""
        objects = rotate([top_left, top_right, bottom_left, bottom_right])
        return [*objects, caption(objects, ground)]

    def draw(
        ground: str, top_left: str, top_right: str, bottom_left: str, bottom_right: str,
        guidance: float, temperature: float, seed: float, new_seed: bool, samples: float,
    ) -> tuple[list[Image.Image], int, str]:
        """Draws samples of the caption.

        Returns:
            The scenes, the seed used, and a status line. The
            same seed and settings draw the same scenes.
        """
        objects = [top_left, top_right, bottom_left, bottom_right]
        if not all(objects):
            raise gr.Error("Pick an object for every quadrant: the model was trained on four-object scenes.")
        if len(set(objects)) < 4:
            raise gr.Error("Pick four different objects: training scenes never repeat a label.")
        used = random.randrange(2**31) if new_seed else int(seed)
        count = int(samples)
        ids = encode_captions([caption(objects, ground)] * count, tokenizer, length=model.config.caption)
        started = time.time()
        torch.manual_seed(used)
        tokens = model.generate(ids.to(device), temperature=temperature, top_k=TOP_K, guidance=guidance)
        images = [
            decode(grid, palette).resize((32 * SCALE, 32 * SCALE), Image.Resampling.NEAREST)
            for grid in tokens.view(-1, 32, 32).cpu().numpy().astype(np.uint8)
        ]
        status = f"{count} scene{'s' if count > 1 else ''} · {time.time() - started:.1f} s · seed {used}"
        return images, used, status

    pack_name, ground_name, objects = EXAMPLES[START]
    def section(title: str) -> None:
        gr.HTML(f'<div class="px-section">{title}</div>', padding=False)

    with gr.Blocks(title="pixl") as app:
        gr.HTML(HEADER, padding=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                section("Scene")
                start = gr.Dropdown(list(EXAMPLES), value=START, label="Preset")
                with gr.Row():
                    pack = gr.Dropdown([(entry["name"], key) for key, entry in vocabulary.items()], value=pack_name,
                                       label="Sprite pack", info=about(pack_name))
                    ground = gr.Dropdown(grounds(pack_name), value=ground_name, label="Ground",
                                         info=ground_note(pack_name))
                section("Objects")
                quadrants: list[gr.Dropdown] = []
                for row in (PLACES[:2], PLACES[2:]):
                    with gr.Row():
                        quadrants += [
                            gr.Dropdown(vocabulary[pack_name]["objects"], value=objects[PLACES.index(place)],
                                        label=place.capitalize())
                            for place in row
                        ]
                with gr.Row():
                    shuffle = gr.Button("Randomize", size="sm")
                    turning = gr.Button("Rotate clockwise", size="sm")
                with gr.Accordion("Advanced", open=False, elem_classes="px-accordion"):
                    guidance = gr.Slider(1, 6, value=3, step=0.5, label="Guidance")
                    temperature = gr.Slider(0.3, 1.2, value=0.7, step=0.05, label="Temperature")
                    samples = gr.Slider(1, 8, value=4, step=1, label="Scenes")
                    with gr.Row():
                        seed = gr.Number(0, precision=0, label="Seed")
                        new_seed = gr.Checkbox(True, label="New seed each time")
                go = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column(scale=6, min_width=420):
                section("Result")
                gallery = gr.Gallery(show_label=False, columns=2, height=600, object_fit="contain",
                                     elem_classes="px-canvas")
                status = gr.Markdown("Choose a scene, then generate.", elem_classes="px-meta")
                text = gr.Markdown(caption(objects, ground_name), elem_classes="px-caption")

        shown = [pack, ground, *quadrants, text]
        hidden = "undocumented"  # callable by the Gradio client, left out of the API page
        # Only .input events, which fire for the user's own choice and never for an update from a
        # handler: one choice runs one handler, which updates every menu it affects and the caption.
        start.input(example, inputs=start, outputs=shown, api_visibility=hidden)
        pack.input(random_scene, inputs=pack, outputs=shown, api_visibility=hidden)
        shuffle.click(random_scene, inputs=pack, outputs=shown, api_visibility=hidden)
        turning.click(turn, inputs=[ground, *quadrants], outputs=[*quadrants, text], api_visibility=hidden)
        for menu in (ground, *quadrants):
            menu.input(write, inputs=[ground, *quadrants], outputs=text, api_visibility=hidden)
        go.click(draw, inputs=[ground, *quadrants, guidance, temperature, seed, new_seed, samples],
                 outputs=[gallery, seed, status], api_name="draw")
    return app


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("--bf16", action="store_true", help="download the half-size bf16 weights")
    _ = parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default=Args.device,
        help="where the model runs; auto: CUDA, then Apple's GPU, then the CPU",
    )
    _ = parser.add_argument("--port", type=int, default=Args.port)
    _ = parser.add_argument("--share", action="store_true", help="also serve a temporary public Gradio link")
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    model = load_pretrained(bf16=args.bf16, device=pick_device(args.device))
    build(model, load_vocabulary()).launch(server_port=args.port, share=args.share, **launch_options())


if __name__ == "__main__":
    main()
