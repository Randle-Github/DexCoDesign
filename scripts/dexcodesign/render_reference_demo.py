#!/usr/bin/env python3
"""MuJoCo renderer for the source decomposition and 20 compiled variants."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "temp/design_grammar/outputs"
WIDTH = HEIGHT = 760
COLORS = {
    "palm": ".95 .70 .20 1",
    "base": ".78 .62 .22 1",
    "other": ".70 .65 .45 1",
    "thumb": ".95 .35 .35 1",
    "index": ".25 .65 .98 1",
    "middle": ".25 .80 .50 1",
    "ring": ".65 .43 .92 1",
    "pinky": ".94 .45 .70 1",
}


def _fmt(values) -> str:
    return " ".join(f"{float(value):.7f}" for value in values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounds(payload: dict, positions: dict[int, np.ndarray]) -> np.ndarray:
    bounds = []
    for part in payload["parts"]:
        if part["mesh"] is None:
            continue
        bounds.append(np.asarray(part["mesh"]["bounds"], dtype=float) + positions[part["node_id"]])
    values = np.asarray(bounds)
    return np.asarray([values[:, 0].min(0), values[:, 1].max(0)])


def _scene(payload: dict, path: Path, exploded: bool) -> tuple[Path, np.ndarray]:
    hand = payload["hand"]
    nodes = {node["node_id"]: node for node in hand["nodes"]}
    roles = ["thumb", "index", "middle", "ring", "pinky"]
    positions = {}
    for part in payload["parts"]:
        position = np.asarray(part["world_pos"], dtype=float)
        if exploded:
            role = part["role"]
            role_shift = (roles.index(role) - 2) * 0.07 if role in roles else 0.0
            position = position * 1.12 + np.asarray([role_shift, 0.0, 0.0])
        positions[part["node_id"]] = position
    bounds = _bounds(payload, positions)
    root = ET.Element("mujoco", {"model": hand["hand_id"]})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(WIDTH), "offheight": str(HEIGHT)})
    ET.SubElement(visual, "quality", {"shadowsize": "2048", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".42 .42 .42", "diffuse": ".72 .72 .72", "specular": ".18 .18 .18"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".09 .13 .19", "rgb2": ".01 .015 .025", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "floor_tex", "type": "2d", "builtin": "checker", "rgb1": ".18 .20 .25", "rgb2": ".08 .09 .12", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "floor_tex", "texrepeat": "10 10", "reflectance": ".12"})
    for role, color in COLORS.items():
        ET.SubElement(asset, "material", {"name": f"mat_{role}", "rgba": color, "roughness": ".48", "metallic": ".05"})
    for part in payload["parts"]:
        if part["mesh"] is not None:
            ET.SubElement(asset, "mesh", {"name": f"part_{part['node_id']:02d}", "file": part["mesh"]["path"], "smoothnormal": "true"})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -4 10", "dir": "0 .15 -1", "directional": "true", "castshadow": "true"})
    ET.SubElement(world, "geom", {"type": "plane", "size": "4 4 .1", "pos": f"0 0 {bounds[0, 2] - .04:.5f}", "material": "floor", "contype": "0", "conaffinity": "0"})
    for part in payload["parts"]:
        if part["mesh"] is None:
            continue
        body = ET.SubElement(world, "body", {"pos": _fmt(positions[part["node_id"]])})
        role = nodes[part["node_id"]]["semantic_role"]
        material = f"mat_{role}" if role in COLORS else "mat_other"
        ET.SubElement(body, "geom", {"type": "mesh", "mesh": f"part_{part['node_id']:02d}", "material": material, "mass": "0", "contype": "0", "conaffinity": "0"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path, bounds


def _render(compiled_path: Path, image_path: Path, exploded: bool = False) -> None:
    payload = _load(compiled_path)
    xml_path = image_path.with_suffix(".xml")
    _, bounds = _scene(payload, xml_path, exploded)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center = 0.5 * (bounds[0] + bounds[1])
    extent = float(np.max(bounds[1] - bounds[0]))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = center
    camera.distance = max(2.7, extent * 2.25)
    camera.azimuth = 90.0
    camera.elevation = -24.0
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(image_path)


def _contact_sheet(paths: list[Path], output: Path) -> None:
    cell = 380
    columns = 5
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (columns * cell, rows * cell), "#0f1620")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS), ((index % columns) * cell, (index // columns) * cell))
    canvas.save(output)


def _pair_graph_and_mesh(graph_path: Path, render_path: Path, output: Path) -> None:
    with Image.open(graph_path) as graph, Image.open(render_path) as render:
        height = 720
        graph_width = 1080
        graph_image = graph.convert("RGB").resize((graph_width, height), Image.Resampling.LANCZOS)
        render_image = render.convert("RGB").resize((height, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (graph_width + height, height), "#101722")
        canvas.paste(graph_image, (0, 0))
        canvas.paste(render_image, (graph_width, 0))
        canvas.save(output)


def main() -> int:
    decomposition = OUTPUTS / "wuji_decomposition_mesh_parts.png"
    _render(OUTPUTS / "compiled_source_wuji/compiled.json", decomposition, exploded=True)
    structure = OUTPUTS / "reference_graphs/wuji_hand_2.png"
    with Image.open(structure) as left, Image.open(decomposition) as right:
        h = 900
        left_r = left.convert("RGB").resize((1350, h), Image.Resampling.LANCZOS)
        right_r = right.convert("RGB").resize((900, h), Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (2250, h), "#101722")
        combined.paste(left_r, (0, 0))
        combined.paste(right_r, (1350, 0))
        combined.save(OUTPUTS / "wuji_handir_and_mesh_parts.png")

    index = _load(OUTPUTS / "variant_index.json")
    images = []
    for variant in index["variants"]:
        directory = REPO_ROOT / variant["directory"]
        image_path = directory / "render.png"
        _render(directory / "compiled.json", image_path)
        _pair_graph_and_mesh(directory / "structure_graph.png", image_path, directory / "graph_and_mesh.png")
        images.append(image_path)
        print(f"rendered {variant['index']:02d}/20: {variant['name']}", flush=True)
    _contact_sheet(images, OUTPUTS / "wuji_20_variants.png")
    print(f"rendered decomposition and {len(images)} variants -> {OUTPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
