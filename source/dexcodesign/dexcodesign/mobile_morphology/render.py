"""Render 32 compiled mobile manipulators as four close 2x4 MuJoCo panels."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image

from .common import (
    COMPILED_ROBOTS,
    GENERATED_ROOT,
    fmt,
    numbers,
    transform,
)
from .display_pose import link_transforms


TILE_DIR = GENERATED_ROOT / "render_tiles"
FINAL = GENERATED_ROOT / "mobile_robots_32.png"
TILE_WIDTH, TILE_HEIGHT = 1600, 900
CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM = 0, 40, 1600, 900
COLUMNS, ROBOTS_PER_TILE = 4, 8
SPACING_X, SPACING_Y = 1.75, 2.15
CAMERA_LOOKAT_Z = 1.05
CAMERA_DISTANCE = 6.30
COLORS = {
    "dexmate_vega": ".18 .48 .78 1",
    "rby1": ".82 .35 .19 1",
    "galaxea_r1": ".20 .62 .42 1",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    # trimesh returns xyzw; MuJoCo expects wxyz.
    xyzw = trimesh.transformations.quaternion_from_matrix(
        np.block([[matrix, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    )
    # trimesh's helper currently returns wxyz; normalize defensively.
    result = np.asarray(xyzw, dtype=float)
    return result / np.linalg.norm(result)


@lru_cache(maxsize=256)
def mesh_bounds(path: str, scale: tuple[float, float, float]) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    return np.asarray(mesh.bounds, dtype=float) * np.asarray(scale)


def visual_records(entry: dict) -> tuple[list[dict], np.ndarray]:
    urdf = GENERATED_ROOT / entry["compiled_urdf"]
    robot = ET.parse(urdf).getroot()
    worlds = link_transforms(robot, entry["display"]["joint_positions"])
    yaw = transform(rpy=(0, 0, math.radians(entry["display"]["yaw_degrees"])))
    records = []
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for index, visual in enumerate(link.findall("visual")):
            origin = visual.find("origin")
            local = transform(
                numbers(None if origin is None else origin.get("xyz")),
                numbers(None if origin is None else origin.get("rpy")),
            )
            world = yaw @ worlds[link_name] @ local
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                continue
            shape = geometry[0]
            record = {
                "name": f"{safe_name(link_name)}_{index}",
                "kind": shape.tag,
                "world": world,
            }
            if shape.tag == "mesh":
                path = (urdf.parent / shape.get("filename", "")).resolve()
                scale = tuple(numbers(shape.get("scale"), (1, 1, 1)))
                bounds = mesh_bounds(str(path), scale)
                record.update(path=path, scale=scale, local_bounds=bounds)
            elif shape.tag == "box":
                size = numbers(shape.get("size"), (1, 1, 1))
                record.update(size=size / 2, local_bounds=np.asarray((-size / 2, size / 2)))
            elif shape.tag == "cylinder":
                radius, length = float(shape.get("radius", "1")), float(shape.get("length", "1"))
                record.update(
                    size=np.asarray((radius, length / 2)),
                    local_bounds=np.asarray(((-radius, -radius, -length / 2), (radius, radius, length / 2))),
                )
            elif shape.tag == "sphere":
                radius = float(shape.get("radius", "1"))
                record.update(size=np.asarray((radius,)), local_bounds=np.asarray(((-radius,) * 3, (radius,) * 3)))
            else:
                continue
            corners = trimesh.bounds.corners(record["local_bounds"])
            points = trimesh.transform_points(corners, world)
            record["bounds"] = np.asarray((points.min(0), points.max(0)))
            records.append(record)
    all_bounds = np.asarray([record["bounds"] for record in records])
    aggregate = np.asarray((all_bounds[:, 0].min(0), all_bounds[:, 1].max(0)))
    return records, aggregate


def build_scene(
    entries: list[dict],
    tile_index: int,
    *,
    output_dir: Path = TILE_DIR,
    columns: int = COLUMNS,
    spacing_x: float = SPACING_X,
    spacing_y: float = SPACING_Y,
    width: int = TILE_WIDTH,
    height: int = TILE_HEIGHT,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"tile_{tile_index:02d}.xml"
    root = ET.Element("mujoco", {"model": f"mobile morphology tile {tile_index}"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(width), "offheight": str(height)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".38 .38 .38", "diffuse": ".72 .72 .72", "specular": ".14 .14 .14"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".11 .15 .21", "rgb2": ".02 .03 .05", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"name": "floor_tex", "type": "2d", "builtin": "checker", "rgb1": ".23 .25 .29", "rgb2": ".11 .12 .15", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "floor_tex", "texrepeat": "14 8", "reflectance": ".08"})
    for source_id, color in COLORS.items():
        ET.SubElement(asset, "material", {"name": f"mat_{source_id}", "rgba": color, "roughness": ".44", "metallic": ".03"})

    payloads = []
    mesh_assets: dict[tuple[str, tuple[float, ...]], str] = {}
    for robot_index, entry in enumerate(entries):
        records, bounds = visual_records(entry)
        payloads.append((entry, records, bounds))
        for record in records:
            if record["kind"] != "mesh":
                continue
            key = (str(record["path"]), tuple(float(value) for value in record["scale"]))
            if key not in mesh_assets:
                name = f"mesh_{len(mesh_assets)}"
                mesh_assets[key] = name
                ET.SubElement(asset, "mesh", {"name": name, "file": key[0], "scale": fmt(key[1]), "smoothnormal": "true"})
            record["mesh_asset"] = mesh_assets[key]

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "-4 -7 12", "dir": ".25 .35 -1", "directional": "true", "castshadow": "true"})
    ET.SubElement(world, "light", {"pos": "8 2 8", "dir": "-.6 -.15 -1", "directional": "true", "diffuse": ".35 .38 .43"})
    rows = math.ceil(len(entries) / columns)
    used_columns = min(columns, len(entries))
    center_x = (used_columns - 1) * spacing_x / 2
    center_y = (rows - 1) * spacing_y / 2
    for robot_index, (entry, records, bounds) in enumerate(payloads):
        row, column = divmod(robot_index, columns)
        center = 0.5 * (bounds[0] + bounds[1])
        offset = np.asarray(
            (
                column * spacing_x - center_x - center[0],
                center_y - row * spacing_y - center[1],
                0.025 - bounds[0, 2],
            )
        )
        parent = ET.SubElement(world, "body", {"name": f"robot_{robot_index}", "pos": fmt(offset)})
        for visual_index, record in enumerate(records):
            body = ET.SubElement(
                parent,
                "body",
                {
                    "name": f"r{robot_index}_v{visual_index}",
                    "pos": fmt(record["world"][:3, 3]),
                    "quat": fmt(quaternion(record["world"][:3, :3])),
                },
            )
            attributes = {
                "type": record["kind"],
                "material": f"mat_{entry['source_id']}",
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
            }
            if record["kind"] == "mesh":
                attributes["mesh"] = record["mesh_asset"]
            else:
                attributes["size"] = fmt(record["size"])
            ET.SubElement(body, "geom", attributes)
    ET.SubElement(
        world,
        "geom",
        {
            "type": "plane",
            "size": f"{used_columns * spacing_x / 2 + 2:.3f} {rows * spacing_y / 2 + 2:.3f} .1",
            "pos": "0 0 0",
            "material": "floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def render_tile(
    entries: list[dict],
    tile_index: int,
    *,
    output_dir: Path = TILE_DIR,
    columns: int = COLUMNS,
    spacing_x: float = SPACING_X,
    spacing_y: float = SPACING_Y,
    width: int = TILE_WIDTH,
    height: int = TILE_HEIGHT,
    camera_distance: float = CAMERA_DISTANCE,
) -> Path:
    xml = build_scene(
        entries,
        tile_index,
        output_dir=output_dir,
        columns=columns,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        width=width,
        height=height,
    )
    output = output_dir / f"tile_{tile_index:02d}.png"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.orthographic = 1
    camera.lookat[:] = (0, 0, CAMERA_LOOKAT_Z)
    camera.distance = camera_distance
    camera.azimuth = 90.0
    camera.elevation = -32.0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).convert("RGB").save(output)
    print(f"tile {tile_index + 1}: robots={len(entries)} meshes={model.nmesh}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FINAL)
    parser.add_argument(
        "--stitch-only",
        action="store_true",
        help="reuse the existing MuJoCo tile renders",
    )
    args = parser.parse_args()
    payload = json.loads(COMPILED_ROBOTS.read_text(encoding="utf-8"))
    robots = payload["robots"]
    if len(robots) != 32:
        raise ValueError(f"expected 32 robots, got {len(robots)}")
    if args.stitch_only:
        tiles = [TILE_DIR / f"tile_{index:02d}.png" for index in range(4)]
        missing = [path for path in tiles if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing render tiles: {missing}")
    else:
        if TILE_DIR.exists():
            shutil.rmtree(TILE_DIR)
        tiles = [
            render_tile(robots[start:start + ROBOTS_PER_TILE], start // ROBOTS_PER_TILE)
            for start in range(0, len(robots), ROBOTS_PER_TILE)
        ]
    crop_width = CROP_RIGHT - CROP_LEFT
    crop_height = CROP_BOTTOM - CROP_TOP
    canvas = Image.new("RGB", (crop_width * 2, crop_height * 2))
    for index, tile in enumerate(tiles):
        with Image.open(tile) as image:
            cropped = image.convert("RGB").crop(
                (CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)
            )
            canvas.paste(
                cropped,
                ((index % 2) * crop_width, (index // 2) * crop_height),
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"stitched 32 robots into four 2x4 panels -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
