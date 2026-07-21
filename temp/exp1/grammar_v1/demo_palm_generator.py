#!/usr/bin/env python3
"""Generate and debug-render 3/4/5-finger attachment-conditioned palms."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from palm_generator import (
    AttachmentPatch,
    PalmGeometryParams,
    generate_palm_mesh,
    transform_from_rotation_translation,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "palm_demo"
COLORS = ((0.24, 0.72, 0.95, 0.42), (0.35, 0.86, 0.55, 0.42), (0.95, 0.55, 0.24, 0.42))
AXIS_COLORS = ((0.95, 0.12, 0.12, 1.0), (0.12, 0.90, 0.25, 1.0), (0.15, 0.35, 1.0, 1.0))


def fmt(values) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def graph_frame(x: float, z: float, yaw: float = 0.0, y: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return transform_from_rotation_translation(rotation, [x, y, z])


def demo_layout(name: str) -> tuple[list[AttachmentPatch], AttachmentPatch]:
    if name == "three_finger_radial":
        specs = [(-0.30, 0.86, -0.28), (0.0, 1.02, 0.0), (0.30, 0.86, 0.28)]
    elif name == "four_finger_asymmetric":
        specs = [(-0.34, 0.82, -0.20), (-0.12, 1.04, -0.05), (0.13, 0.98, 0.07), (0.37, 0.78, 0.33)]
    elif name == "five_finger_anthropomorphic":
        specs = [
            (-0.38, 0.84, -0.12), (-0.14, 1.02, -0.03),
            (0.10, 1.08, 0.02), (0.33, 0.94, 0.11),
            (0.47, 0.38, 1.02),
        ]
    else:
        raise ValueError(name)
    patches = [
        AttachmentPatch(
            name=f"{name}_finger_{index}",
            transform=graph_frame(x, z, yaw),
            width=0.18 if index < 4 else 0.20,
            depth=0.16,
            thickness=0.075,
            locked=True,
        )
        for index, (x, z, yaw) in enumerate(specs)
    ]
    wrist = AttachmentPatch(
        name=f"{name}_wrist",
        transform=graph_frame(0.0, 0.0),
        width=0.46,
        depth=0.18,
        thickness=0.08,
        locked=True,
    )
    return patches, wrist


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a MuJoCo quaternion in w,x,y,z order."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            result = np.asarray([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                                 (matrix[0, 1] + matrix[1, 0]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            result = np.asarray([(matrix[0, 2] - matrix[2, 0]) / scale,
                                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                                 (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            result = np.asarray([(matrix[1, 0] - matrix[0, 1]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale,
                                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return result / np.linalg.norm(result)


def add_frame_debug(world: ET.Element, transform: np.ndarray, offset: np.ndarray, prefix: str) -> None:
    origin = transform[:3, 3] + offset
    ET.SubElement(world, "geom", {
        "name": f"{prefix}_origin", "type": "sphere", "size": "0.025",
        "pos": fmt(origin), "rgba": "1 1 1 1", "contype": "0", "conaffinity": "0",
    })
    for axis, color in enumerate(AXIS_COLORS):
        endpoint = origin + 0.22 * transform[:3, axis]
        ET.SubElement(world, "geom", {
            "name": f"{prefix}_axis_{axis}", "type": "capsule", "size": "0.014",
            "fromto": fmt([*origin, *endpoint]), "rgba": fmt(color),
            "contype": "0", "conaffinity": "0",
        })


def write_debug_mjcf(records: list[dict], output_dir: Path) -> Path:
    root = ET.Element("mujoco", {"model": "attachment-conditioned palm demo"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": str(output_dir)})
    ET.SubElement(root, "option", {"gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1800", "offheight": "900"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"ambient": ".38 .38 .38", "diffuse": ".72 .72 .72"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": ".08 .11 .17", "rgb2": ".01 .015 .025", "width": "512", "height": "3072"})
    for index, record in enumerate(records):
        ET.SubElement(asset, "mesh", {"name": f"palm_{index}", "file": record["visual_file"]})
        ET.SubElement(asset, "material", {"name": f"palm_mat_{index}", "rgba": fmt(COLORS[index]), "roughness": ".42"})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -4 7", "dir": "0 .35 -1", "directional": "true"})
    ET.SubElement(world, "geom", {"type": "plane", "size": "4 2 .1", "pos": "0 0 -.18", "rgba": ".10 .12 .16 1", "contype": "0", "conaffinity": "0"})
    offsets = [np.asarray([-1.55, 0.0, 0.0]), np.zeros(3), np.asarray([1.55, 0.0, 0.0])]
    for index, (record, offset) in enumerate(zip(records, offsets)):
        body = ET.SubElement(world, "body", {"name": record["name"], "pos": fmt(offset)})
        ET.SubElement(body, "geom", {"type": "mesh", "mesh": f"palm_{index}", "material": f"palm_mat_{index}", "mass": "0", "contype": "0", "conaffinity": "0"})
        for patch_index, patch in enumerate([*record["patches"], record["wrist"]]):
            transform = patch.transform
            ET.SubElement(world, "geom", {
                "name": f"{record['name']}_patch_{patch_index}", "type": "box",
                "size": fmt([0.5 * patch.width, 0.5 * patch.thickness, 0.5 * patch.depth]),
                "pos": fmt(transform[:3, 3] + offset),
                "quat": fmt(matrix_to_quaternion(transform[:3, :3])),
                "rgba": "1 .95 .15 .78", "contype": "0", "conaffinity": "0",
            })
            add_frame_debug(world, transform, offset, f"{record['name']}_{patch_index}")
    ET.indent(root, space="  ")
    path = output_dir / "palm_layouts_debug.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def render_mjcf(xml_path: Path, output_path: Path) -> None:
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, 0.52]
    camera.distance = 5.5
    camera.azimuth = 90.0
    camera.elevation = -14.0
    with mujoco.Renderer(model, height=900, width=1800) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode", choices=("attachment_hull", "parametric_2_5d"), default="parametric_2_5d"
    )
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = PalmGeometryParams(
        thickness=0.24,
        boundary_margin=0.07,
        edge_rounding_radius=0.035,
        wrist_width=0.46,
        transverse_arch=0.025,
        longitudinal_arch=0.018,
        central_cup=0.022,
        deformation_resolution=0.24,
    )
    names = ("three_finger_radial", "four_finger_asymmetric", "five_finger_anthropomorphic")
    records = []
    metadata = {"mode": args.mode, "layouts": []}
    for name in names:
        patches, wrist = demo_layout(name)
        result = generate_palm_mesh(patches, wrist, configuration, mode=args.mode)
        visual_path = args.output_dir / f"{name}_visual.obj"
        collision_path = args.output_dir / f"{name}_collision.obj"
        result.visual_mesh.export(visual_path, file_type="obj", include_normals=True)
        result.collision_mesh.export(collision_path, file_type="obj", include_normals=True)
        records.append({
            "name": name, "patches": patches, "wrist": wrist,
            "visual_file": visual_path.name, "collision_file": collision_path.name,
        })
        metadata["layouts"].append({
            "name": name,
            "finger_count": len(patches),
            "visual_file": visual_path.name,
            "collision_file": collision_path.name,
            "attachment_frames": {key: value.tolist() for key, value in result.attachment_frames.items()},
            "generator": result.metadata,
        })
    metadata_path = args.output_dir / "palm_layouts.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    xml_path = write_debug_mjcf(records, args.output_dir)
    if args.render:
        image_path = args.output_dir / "palm_layouts_debug.png"
        render_mjcf(xml_path, image_path)
        print(f"rendered {image_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {xml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
