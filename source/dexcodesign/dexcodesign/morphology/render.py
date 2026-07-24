#!/usr/bin/env python3
"""Render 100 grammar-v1 hands as one 10x10 MuJoCo contact sheet."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUTS = ROOT / "artifacts" / "hand_morphology" / "generated_100"
INPUT = OUTPUTS / "compiled_hands.json"
TILE_DIR = OUTPUTS / "render_tiles"
FINAL = OUTPUTS / "hands_100.png"
WIDTH, HEIGHT = 1920, 1080
COLUMNS = 5
SPACING_X, SPACING_Y = 3.1, 3.15
COLORS = (
    ".95 .55 .22 1", ".25 .65 .98 1", ".30 .82 .58 1", ".93 .35 .47 1", ".67 .48 .98 1",
    ".98 .55 .25 1", ".20 .78 .82 1", ".88 .42 .86 1", ".56 .78 .28 1", ".98 .35 .29 1",
    ".38 .58 .85 1", ".78 .62 .42 1", ".35 .76 .52 1", ".92 .69 .08 1", ".63 .72 .86 1",
    ".86 .48 .62 1", ".34 .76 .68 1", ".76 .58 .92 1", ".91 .66 .31 1", ".42 .68 .91 1",
    ".95 .42 .62 1", ".28 .78 .72 1", ".74 .68 .30 1", ".58 .48 .92 1", ".36 .72 .92 1",
)


def fmt(values) -> str:
    return " ".join(f"{float(value):.7f}" for value in values)


def bounds(hand: dict) -> np.ndarray:
    values = []
    for node in hand["parts"]:
        mesh = node.get("compiled_mesh")
        if mesh is None:
            continue
        shifted = np.asarray(mesh["bounds"], dtype=float) + np.asarray(node["world_pos"], dtype=float)
        values.append(shifted)
    array = np.asarray(values)
    return np.asarray([array[:, 0].min(0), array[:, 1].max(0)])


def scene(hands: list[dict], tile: int) -> Path:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    path = TILE_DIR / f"tile_{tile + 1:02d}.xml"
    root = ET.Element("mujoco", {"model": f"grammar-v1 tile {tile + 1}"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": str(OUTPUTS)})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(WIDTH), "offheight": str(HEIGHT)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".39 .39 .39", "diffuse": ".74 .74 .74", "specular": ".18 .18 .18"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".09 .13 .19", "rgb2": ".01 .015 .025", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "floor_tex", "type": "2d", "builtin": "checker", "rgb1": ".18 .20 .25", "rgb2": ".08 .09 .12", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "floor_tex", "texrepeat": "18 18", "reflectance": ".10"})
    mesh_names = {}
    for hand_index, hand in enumerate(hands):
        ET.SubElement(asset, "material", {"name": f"mat_{hand_index}", "rgba": COLORS[hand_index], "roughness": ".46", "metallic": ".05"})
        for node in hand["parts"]:
            mesh = node.get("compiled_mesh")
            if mesh is None:
                continue
            name = f"h{hand_index}_p{node['id']}"
            ET.SubElement(asset, "mesh", {"name": name, "file": mesh["file"], "smoothnormal": "true"})
            mesh_names[(hand_index, int(node["id"]))] = name
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -4 30", "dir": "0 .12 -1", "directional": "true", "castshadow": "true"})
    rows = math.ceil(len(hands) / COLUMNS)
    ET.SubElement(world, "geom", {"type": "plane", "size": f"{COLUMNS * SPACING_X / 2 + 2:.2f} {rows * SPACING_Y / 2 + 2:.2f} .1", "pos": "0 0 -.03", "material": "floor", "contype": "0", "conaffinity": "0"})
    center_x = (COLUMNS - 1) * SPACING_X / 2.0
    center_y = (rows - 1) * SPACING_Y / 2.0
    for hand_index, hand in enumerate(hands):
        row, column = divmod(hand_index, COLUMNS)
        hand_bounds = bounds(hand)
        center = 0.5 * (hand_bounds[0] + hand_bounds[1])
        offset = np.asarray([
            column * SPACING_X - center_x - center[0],
            center_y - row * SPACING_Y - center[1],
            0.04 - hand_bounds[0, 2],
        ])
        parent = ET.SubElement(world, "body", {"name": f"hand_{hand_index}", "pos": fmt(offset)})
        for node in hand["parts"]:
            mesh_name = mesh_names.get((hand_index, int(node["id"])))
            if mesh_name is None:
                continue
            body = ET.SubElement(parent, "body", {"name": f"h{hand_index}_part_{node['id']}", "pos": fmt(node["world_pos"])})
            ET.SubElement(body, "geom", {"type": "mesh", "mesh": mesh_name, "material": f"mat_{hand_index}", "mass": "0", "contype": "0", "conaffinity": "0"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def render_tile(hands: list[dict], tile: int, *, camera_distance: float = 17.4) -> Path:
    xml = scene(hands, tile)
    image_path = TILE_DIR / f"tile_{tile + 1:02d}.png"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.9]
    # Five rows must remain fully visible in each tile; the earlier 14.7 view
    # clipped the nearest row after the 2x2 contact-sheet composition.
    camera.distance = camera_distance
    camera.azimuth = 90.0
    camera.elevation = -33.0
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(image_path)
    print(f"tile {tile + 1}/4: hands={len(hands)} meshes={model.nmesh} faces={int(model.mesh_facenum.sum()):,}", flush=True)
    return image_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-distance", type=float, default=None)
    args = parser.parse_args()
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    hands = payload["hands"]
    if args.hand_id:
        wanted = set(args.hand_id)
        hands = [hand for hand in hands if hand["hand_id"] in wanted]
        missing = wanted - {hand["hand_id"] for hand in hands}
        if missing:
            raise ValueError(f"unknown hand ids: {sorted(missing)}")
        output = args.output or OUTPUTS / "selected_hands.png"
        distance = args.camera_distance or max(5.8, 2.15 * math.ceil(len(hands) / COLUMNS) + 2.6)
        tile = render_tile(hands, 98, camera_distance=distance)
        with Image.open(tile) as image:
            image.convert("RGB").save(output)
        print(f"selected hands -> {output}")
        return 0
    if len(hands) != 100:
        raise ValueError(f"expected 100 hands, got {len(hands)}")
    tiles = [render_tile(hands[index * 25:(index + 1) * 25], index) for index in range(4)]
    canvas = Image.new("RGB", (WIDTH * 2, HEIGHT * 2))
    for index, path in enumerate(tiles):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), ((index % 2) * WIDTH, (index // 2) * HEIGHT))
    canvas.save(FINAL)
    print(f"stitched 10x10 contact sheet -> {FINAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
