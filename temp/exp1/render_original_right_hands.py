#!/usr/bin/env python3
"""Render all 14 original right-hand visual meshes with the exp1 camera."""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMP = ROOT / "temp"
sys.path.insert(0, str(TEMP))

from visualize_right_hands import PALETTE, load_all_hands  # noqa: E402


OUTPUTS = HERE / "outputs"
MESH_DIR = OUTPUTS / "original_right_meshes"
SCENE = OUTPUTS / "original_14_right_hands.xml"
IMAGE = OUTPUTS / "original_14_right_hands.png"
WIDTH = 1800
HEIGHT = 430
COLUMNS = 10
SPACING_X = 3.05
SPACING_Y = 3.20


def canonical_original_mesh(hand):
    """Use the source visual surface without exp1 morphology deformation."""
    mesh = hand.scene.to_geometry()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    transform = np.eye(4)
    transform[:3, :3] = hand.canonical_basis.T
    mesh.apply_transform(transform)
    bounds = np.asarray(mesh.bounds, dtype=float)
    offset = np.eye(4)
    offset[:3, 3] = [-bounds[:, 0].mean(), -bounds[:, 1].mean(), -bounds[0, 2]]
    mesh.apply_transform(offset)
    mesh.fix_normals()
    return mesh


def build_scene() -> Path:
    hands = load_all_hands("right")
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": "DexCoDesign original 14 right hands"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": str(OUTPUTS)})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "90", "elevation": "-32", "offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".38 .38 .38", "diffuse": ".72 .72 .72", "specular": ".22 .22 .22"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".10 .14 .20", "rgb2": ".01 .015 .025", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "floor_tex", "type": "2d", "builtin": "checker", "rgb1": ".18 .20 .25", "rgb2": ".08 .09 .12", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "floor_tex", "texrepeat": "14 14", "reflectance": ".12"})

    for index, hand in enumerate(hands):
        mesh = canonical_original_mesh(hand)
        relative = Path("original_right_meshes") / f"{index:02d}_{hand.hand_id}.obj"
        mesh.export(OUTPUTS / relative, file_type="obj", include_normals=True, include_color=False)
        ET.SubElement(asset, "mesh", {"name": f"original_mesh_{index:02d}", "file": str(relative)})
        rgba = np.asarray(PALETTE[index % len(PALETTE)], dtype=float) / 255.0
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"original_mat_{index:02d}",
                "rgba": " ".join(f"{value:.4f}" for value in rgba),
                "roughness": ".48",
                "metallic": ".06",
            },
        )
        print(f"prepared {index + 1:02d}/14 {hand.display_name}: {len(mesh.faces):,} faces", flush=True)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 28", "dir": "0 0 -1", "directional": "true", "castshadow": "true"})
    rows = math.ceil(len(hands) / COLUMNS)
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": f"{COLUMNS * SPACING_X / 2 + 2:.2f} {rows * SPACING_Y / 2 + 2:.2f} .1",
            "pos": "0 0 -.04",
            "material": "floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    x_center = (COLUMNS - 1) * SPACING_X / 2
    y_center = (rows - 1) * SPACING_Y / 2
    for index, hand in enumerate(hands):
        row, column = divmod(index, COLUMNS)
        body = ET.SubElement(
            world,
            "body",
            {
                "name": f"original_{index:02d}_{hand.hand_id}",
                "pos": f"{column * SPACING_X - x_center:.6f} {y_center - row * SPACING_Y:.6f} .03",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": f"original_mesh_{index:02d}",
                "material": f"original_mat_{index:02d}",
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE, encoding="utf-8", xml_declaration=True)
    return SCENE


def main() -> int:
    scene = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    # Exactly the same camera as render_snapshot.py.
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 12.0
    camera.azimuth = 90.0
    camera.elevation = -38.0
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(IMAGE)
    print(f"MuJoCo OK: hands=14 meshes={model.nmesh} mesh_faces={int(model.mesh_facenum.sum()):,}")
    print(IMAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
