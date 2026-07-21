#!/usr/bin/env python3
"""Build and open one MuJoCo scene containing all registered right hands.

Run with ``mjpython`` on macOS so the Cocoa window owns the main thread:

    temp/.venv/bin/mjpython temp/visualize_all_right_hands_mujoco.py
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh

from visualize_right_hands import PALETTE, HandScene, load_all_hands


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "temp" / "cache" / "mujoco_right_hand_gallery"
MAX_FACES_PER_HAND = 140_000


def presentation_mesh(hand: HandScene) -> trimesh.Trimesh:
    """Stand a hand wrist-down, fingers-up, with a common palm direction."""
    mesh = hand.scene.to_geometry()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    # The kinematic landmarks define X=thumb side, Y=palm normal and
    # Z=wrist-to-fingertips, independent of mesh shape or neutral finger pose.
    transform = np.eye(4)
    transform[:3, :3] = hand.canonical_basis.T
    mesh.apply_transform(transform)

    # Recenter horizontally/depth-wise and rest the wrist end on the floor.
    bounds = np.asarray(mesh.bounds)
    offset = np.eye(4)
    offset[:3, 3] = [-bounds[:, 0].mean(), -bounds[:, 1].mean(), -bounds[0, 2]]
    mesh.apply_transform(offset)

    if len(mesh.faces) > MAX_FACES_PER_HAND:
        original = len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(face_count=MAX_FACES_PER_HAND, aggression=7)
        print(f"      display decimation: {original:,} -> {len(mesh.faces):,} faces", flush=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def write_scene(hands: list[HandScene], paired: bool = False) -> Path:
    scene_xml = CACHE / ("all_hand_pairs.xml" if paired else "all_right_hands.xml")
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale_mesh in CACHE.glob("[0-9][0-9]_*.obj"):
        stale_mesh.unlink()

    mujoco_node = ET.Element("mujoco", {"model": "DexCoDesign all right hands"})
    ET.SubElement(
        mujoco_node,
        "compiler",
        {"angle": "radian", "meshdir": str(CACHE), "autolimits": "true"},
    )
    ET.SubElement(
        mujoco_node,
        "option",
        {"timestep": "0.01", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
    )
    visual = ET.SubElement(mujoco_node, "visual")
    ET.SubElement(visual, "global", {"azimuth": "90", "elevation": "-30", "offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "60", "fogstart": "20", "fogend": "45"})
    ET.SubElement(visual, "headlight", {"ambient": ".35 .35 .35", "diffuse": ".75 .75 .75", "specular": ".25 .25 .25"})

    asset = ET.SubElement(mujoco_node, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": ".12 .16 .22",
            "rgb2": ".015 .02 .03",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_tex",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".16 .18 .22",
            "rgb2": ".08 .09 .12",
            "mark": "edge",
            "markrgb": ".35 .38 .45",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "floor", "texture": "floor_tex", "texrepeat": "8 8", "reflectance": ".15", "shininess": ".2"},
    )

    display_meshes: list[tuple[HandScene, str]] = []
    for index, hand in enumerate(hands):
        print(f"Preparing MuJoCo gallery mesh {index + 1:02d}/{len(hands):02d}: {hand.display_name}", flush=True)
        mesh = presentation_mesh(hand)
        filename = f"{index + 1:02d}_{hand.hand_id}_{hand.side}.obj"
        mesh.export(CACHE / filename, file_type="obj", include_normals=True, include_color=False)
        mesh_name = f"hand_mesh_{index + 1:02d}"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": filename})
        color_index = index // 2 if paired else index
        color = PALETTE[color_index % len(PALETTE)].astype(float) / 255.0
        rgba = " ".join(f"{component:.4f}" for component in color)
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"hand_material_{index + 1:02d}",
                "rgba": rgba,
                "specular": ".45",
                "shininess": ".35",
                "metallic": ".05",
                "roughness": ".55",
            },
        )
        display_meshes.append((hand, mesh_name))

    worldbody = ET.SubElement(mujoco_node, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 14", "dir": "0 0 -1", "directional": "true", "castshadow": "true", "diffuse": ".9 .9 .9"})
    ET.SubElement(worldbody, "light", {"pos": "-7 -5 8", "dir": ".5 .35 -1", "directional": "false", "castshadow": "true", "diffuse": ".55 .62 .75"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "8 10 .1",
            "pos": "0 0 -0.03",
            "material": "floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    columns = 4
    item_count = len(hands) // 2 if paired else len(hands)
    rows = math.ceil(item_count / columns)
    x_spacing = 4.4 if paired else 2.75
    y_spacing = 3.55
    x_center = (columns - 1) * x_spacing / 2.0
    y_center = (rows - 1) * y_spacing / 2.0
    for index, (hand, mesh_name) in enumerate(display_meshes):
        item_index = index // 2 if paired else index
        row, column = divmod(item_index, columns)
        pair_offset = (-1.05 if hand.side == "left" else 1.05) if paired else 0.0
        x = column * x_spacing - x_center + pair_offset
        y = y_center - row * y_spacing
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{index + 1:02d}_{hand.hand_id}_{hand.side}", "pos": f"{x:.4f} {y:.4f} 0.02"},
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"visual_{index + 1:02d}_{hand.hand_id}_{hand.side}",
                "type": "mesh",
                "mesh": mesh_name,
                "material": f"hand_material_{index + 1:02d}",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        # Transparent sites provide names through MuJoCo's native label layer.
        ET.SubElement(
            body,
            "site",
            {
                "name": f"{item_index + 1:02d}{hand.side[0].upper()}  {hand.display_name}",
                "pos": "0 0 2.18",
                "size": ".001",
                "rgba": "1 1 1 0",
                "group": "4",
            },
        )

    ET.indent(mujoco_node, space="  ")
    ET.ElementTree(mujoco_node).write(scene_xml, encoding="utf-8", xml_declaration=True)
    return scene_xml


def open_viewer(model: mujoco.MjModel, data: mujoco.MjData, paired: bool = False) -> None:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.95]
        viewer.cam.distance = 24.0 if paired else 19.5
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -30.0
        viewer.sync()
        print("\nMuJoCo window is running: left-drag rotate, right-drag pan, wheel zoom, Esc closes.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build and compile the combined MJCF without opening a window")
    parser.add_argument("--pairs", action="store_true", help="Show all 14 canonical left/right pairs in one window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs:
        left_hands = load_all_hands("left")
        right_hands = load_all_hands("right")
        hands = [hand for pair in zip(left_hands, right_hands) for hand in pair]
    else:
        hands = load_all_hands("right")
    if not hands or hands[0].hand_id != "mano":
        raise RuntimeError("Expected at least one registered right hand with MANO first")
    scene_path = write_scene(hands, paired=args.pairs)
    print(f"\nCompiling combined MuJoCo scene: {scene_path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"Combined scene compiled: bodies={model.nbody - 1}, meshes={model.nmesh}, "
        f"geoms={model.ngeom}, sites={model.nsite}",
        flush=True,
    )
    if args.check:
        return 0
    open_viewer(model, data, paired=args.pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
