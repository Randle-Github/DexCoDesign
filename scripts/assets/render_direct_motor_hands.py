#!/usr/bin/env python3
"""Render all normalized right-hand URDFs and annotate active/passive DoFs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "temp"
sys.path.insert(0, str(TEMP))

from visualize_all_right_hands_mujoco import presentation_mesh  # noqa: E402
from visualize_right_hands import (  # noqa: E402
    PALETTE,
    HandScene,
    load_all_hands,
)


ASSETS = ROOT / "assets" / "robot_hands"
DIRECT = ASSETS / "direct_motor"
REGISTRY = DIRECT / "registry.json"
DISPLAY_MESHES = DIRECT / "overview_meshes"
SCENE = DIRECT / "overview_right_hands.xml"
IMAGE = DIRECT / "overview_right_hands.png"
ORIGINAL_IMAGE = ROOT / "temp/exp1/outputs/original_14_right_hands.png"

WIDTH = 2100
HEIGHT = 940
COLUMNS = 7
ROWS = 2
SPACING_X = 3.0
SPACING_Y = 3.65
SHORT_NAMES = {
    "wuji_hand_2": "WUJI Hand 2",
    "tesollo_dg5f": "Tesollo DG5F",
    "shadow_hand_e": "Shadow Hand E",
}


def load_hands() -> tuple[list[HandScene], list[dict]]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hands = load_all_hands("right")
    labels: list[dict] = []
    for index, hand in enumerate(hands):
        hand_id = hand.hand_id
        metadata = payload["hands"][hand_id]
        entry = metadata["entries"]["right"]
        labels.append(
            {
                "hand_id": hand_id,
                "display_name": metadata["display_name"],
                "active": entry["active_dofs"],
                "passive": entry["passive_mimic_dofs"],
                "total": entry["scalar_dofs"],
            }
        )
        print(
            f"{index + 1:02d}/14 {hand_id:20s} "
            f"A={entry['active_dofs']:2d} P={entry['passive_mimic_dofs']:2d} "
            f"faces={hand.faces:,}",
            flush=True,
        )
    return hands, labels


def build_scene(hands: list[HandScene]) -> Path:
    DISPLAY_MESHES.mkdir(parents=True, exist_ok=True)
    for path in DISPLAY_MESHES.glob("*.obj"):
        path.unlink()

    root = ET.Element("mujoco", {"model": "direct motor hand overview"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": str(DIRECT)})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "azimuth": "90",
            "elevation": "-34",
            "offwidth": str(WIDTH),
            "offheight": str(HEIGHT),
        },
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": ".42 .42 .42",
            "diffuse": ".72 .72 .72",
            "specular": ".20 .20 .20",
        },
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": ".11 .15 .21",
            "rgb2": ".01 .015 .025",
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
            "rgb1": ".18 .20 .25",
            "rgb2": ".08 .09 .12",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor",
            "texture": "floor_tex",
            "texrepeat": "14 8",
            "reflectance": ".12",
        },
    )

    for index, hand in enumerate(hands):
        mesh = presentation_mesh(hand)
        relative = Path("overview_meshes") / f"{index:02d}_{hand.hand_id}.obj"
        mesh.export(
            DIRECT / relative,
            file_type="obj",
            include_normals=True,
            include_color=False,
        )
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"hand_{index:02d}", "file": str(relative)},
        )
        rgba = PALETTE[index % len(PALETTE)].astype(float) / 255.0
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"mat_{index:02d}",
                "rgba": " ".join(f"{value:.4f}" for value in rgba),
                "roughness": ".46",
                "metallic": ".05",
            },
        )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {
            "pos": "0 0 28",
            "dir": "0 0 -1",
            "directional": "true",
            "castshadow": "true",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "type": "plane",
            "size": "13 6 .1",
            "pos": "0 0 -.04",
            "material": "floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    x_center = (COLUMNS - 1) * SPACING_X / 2.0
    y_center = (ROWS - 1) * SPACING_Y / 2.0
    for index, hand in enumerate(hands):
        row, column = divmod(index, COLUMNS)
        body = ET.SubElement(
            world,
            "body",
            {
                "name": hand.hand_id,
                "pos": (
                    f"{column * SPACING_X - x_center:.6f} "
                    f"{y_center - row * SPACING_Y:.6f} .03"
                ),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": f"hand_{index:02d}",
                "material": f"mat_{index:02d}",
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE, encoding="utf-8", xml_declaration=True)
    return SCENE


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def annotate(image: Image.Image, labels: list[dict]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(22)
    value_font = font(20)
    cell_width = WIDTH / COLUMNS
    cell_height = HEIGHT / ROWS
    for index, record in enumerate(labels):
        row, column = divmod(index, COLUMNS)
        center_x = (column + 0.5) * cell_width
        top = row * cell_height + 12
        title = record["display_name"]
        value = (
            f"A {record['active']}   P {record['passive']}   "
            f"DoF {record['total']}"
        )
        title_box = draw.textbbox((0, 0), title, font=title_font)
        value_box = draw.textbbox((0, 0), value, font=value_font)
        width = max(title_box[2], value_box[2]) + 22
        draw.rounded_rectangle(
            (center_x - width / 2, top, center_x + width / 2, top + 58),
            radius=8,
            fill=(5, 8, 14, 185),
            outline=(220, 228, 240, 100),
        )
        draw.text(
            (center_x - title_box[2] / 2, top + 5),
            title,
            font=title_font,
            fill=(245, 248, 255, 255),
        )
        draw.text(
            (center_x - value_box[2] / 2, top + 32),
            value,
            font=value_font,
            fill=(210, 226, 248, 255),
        )
    image.save(IMAGE)


def direct_labels() -> list[dict]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    order = ["mano", *(hand_id for hand_id in payload["hands"] if hand_id != "mano")]
    return [
        {
            "hand_id": hand_id,
            "display_name": SHORT_NAMES.get(
                hand_id, payload["hands"][hand_id]["display_name"]
            ),
            "active": payload["hands"][hand_id]["entries"]["right"]["active_dofs"],
            "passive": payload["hands"][hand_id]["entries"]["right"]["passive_mimic_dofs"],
            "total": payload["hands"][hand_id]["entries"]["right"]["scalar_dofs"],
        }
        for hand_id in order
    ]


def compose_existing_original() -> None:
    if not ORIGINAL_IMAGE.exists():
        raise FileNotFoundError(ORIGINAL_IMAGE)
    source = Image.open(ORIGINAL_IMAGE).convert("RGB")
    canvas = Image.new("RGB", (source.width, source.height + 360), (8, 12, 19))
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = font(27)
    item_font = font(16)
    small_font = font(14)
    draw.text(
        (36, source.height + 18),
        "Direct-motor normalization — original right-hand geometry at q = 0",
        font=title_font,
        fill=(244, 248, 255, 255),
    )
    draw.text(
        (38, source.height + 55),
        "A = independent active direct motor   P = passive affine mimic   "
        "Visual order: upper row #01–#10, lower row #11–#14",
        font=small_font,
        fill=(184, 202, 226, 255),
    )
    labels = direct_labels()
    columns = 7
    cell_width = (source.width - 48) / columns
    cell_height = 116
    top = source.height + 88
    for index, record in enumerate(labels):
        row, column = divmod(index, columns)
        left = 24 + column * cell_width
        y = top + row * cell_height
        draw.rounded_rectangle(
            (left + 4, y + 4, left + cell_width - 6, y + cell_height - 8),
            radius=10,
            fill=(18, 27, 41, 235),
            outline=(96, 124, 158, 150),
        )
        name = f"#{index + 1:02d}  {record['display_name']}"
        dof = (
            f"A {record['active']}   P {record['passive']}   "
            f"Total {record['total']}"
        )
        draw.text((left + 15, y + 17), name, font=item_font, fill=(242, 246, 255, 255))
        draw.text((left + 15, y + 53), dof, font=item_font, fill=(139, 201, 255, 255))
        draw.text(
            (left + 15, y + 80),
            record["hand_id"],
            font=small_font,
            fill=(158, 170, 188, 255),
        )
    canvas.save(IMAGE)
    print(IMAGE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerender-geometry",
        action="store_true",
        help="Reload and rerender every source mesh instead of reusing the verified original gallery",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.rerender_geometry:
        compose_existing_original()
        return 0
    hands, labels = load_hands()
    scene = build_scene(hands)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.95]
    camera.distance = 15.3
    camera.azimuth = 90.0
    camera.elevation = -36.0
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render())
    annotate(image, labels)
    print(IMAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
