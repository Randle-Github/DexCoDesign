#!/usr/bin/env python3
"""Render five 20-hand MuJoCo pages and stitch a 10 x 10 overview."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
from PIL import Image

from view_generated_hifi import OUTPUTS, build_scene


HERE = Path(__file__).resolve().parent
PAGE_DIR = OUTPUTS / "render_pages"
FINAL = OUTPUTS / "generated_100_hands.png"
PAGE_WIDTH = 1800
PAGE_HEIGHT = 430


def render_page(page_index: int, samples: list[dict]) -> Path:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = PAGE_DIR / f"page_{page_index + 1:02d}.xml"
    image_path = PAGE_DIR / f"page_{page_index + 1:02d}.png"
    build_scene(samples, scene_path=scene_path, realized_path=None, columns=10)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 12.0
    camera.azimuth = 90.0
    camera.elevation = -38.0
    with mujoco.Renderer(model, height=PAGE_HEIGHT, width=PAGE_WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(image_path)
    print(f"rendered page {page_index + 1}/5: {image_path}", flush=True)
    return image_path


def main() -> int:
    generated = json.loads((OUTPUTS / "generated_hands.json").read_text(encoding="utf-8"))
    samples = generated["samples"]
    if len(samples) != 100:
        raise ValueError(f"Expected 100 generated hands, got {len(samples)}")
    pages = [render_page(page, samples[page * 20 : (page + 1) * 20]) for page in range(5)]
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT * len(pages)))
    for row, path in enumerate(pages):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), (0, row * PAGE_HEIGHT))
    canvas.save(FINAL)
    print(f"stitched 100-hand overview: {FINAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
