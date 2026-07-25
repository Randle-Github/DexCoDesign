#!/usr/bin/env python3
"""Render all EEF-free mobile manipulators in one visual-only MuJoCo scene."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "mobile_manipulators"
OUTPUT_ROOT = ROOT / "artifacts" / "mobile_manipulators"
SCENE_PATH = OUTPUT_ROOT / "mobile_manipulators_scene.xml"
IMAGE_PATH = OUTPUT_ROOT / "mujoco_mobile_manipulators.png"
WIDTH, HEIGHT = 2400, 1280


@dataclass(frozen=True)
class Robot:
    robot_id: str
    urdf: Path
    rgba: str
    joint_positions: tuple[tuple[str, float], ...] = ()
    yaw_degrees: float = -90.0


ROBOTS = (
    Robot(
        "dexmate_vega",
        ASSETS / "dexmate_vega" / "eef_free" / "robot.urdf",
        ".18 .48 .78 1",
        (
            ("torso_j1", 1.3666),
            ("torso_j2", 2.7021),
            ("torso_j3", 1.2036),
            ("L_arm_j1", math.pi / 2),
            ("R_arm_j1", -math.pi / 2),
        ),
    ),
    Robot(
        "rby1",
        ASSETS / "rby1" / "eef_free" / "robot.urdf",
        ".82 .35 .19 1",
    ),
    Robot(
        "galaxea_r1",
        ASSETS / "galaxea_r1" / "eef_free" / "robot.urdf",
        ".20 .62 .42 1",
        (
            ("left_arm_joint1", -math.pi / 2),
            ("left_arm_joint3", -math.pi),
            ("right_arm_joint1", math.pi / 2),
            ("right_arm_joint3", -math.pi),
        ),
    ),
)


def numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.asarray([float(item) for item in value.split()], dtype=float)


def transform(xyz: str | None = None, rpy: str | None = None) -> np.ndarray:
    x, y, z = numbers(xyz, (0.0, 0.0, 0.0))
    roll, pitch, yaw = numbers(rpy, (0.0, 0.0, 0.0))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = (x, y, z)
    return result


def yaw_transform(degrees: float) -> np.ndarray:
    return transform(rpy=f"0 0 {math.radians(degrees)}")


def joint_motion(joint: ET.Element, position: float) -> np.ndarray:
    kind = joint.get("type", "fixed")
    if kind == "fixed":
        return np.eye(4)
    axis_element = joint.find("axis")
    axis = numbers(
        None if axis_element is None else axis_element.get("xyz"),
        (1.0, 0.0, 0.0),
    )
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(4)
    axis /= norm
    if kind == "prismatic":
        result = np.eye(4)
        result[:3, 3] = axis * position
        return result
    if kind not in {"revolute", "continuous"}:
        raise ValueError(f"unsupported URDF joint type: {kind}")
    x, y, z = axis
    cosine, sine = math.cos(position), math.sin(position)
    one_minus_cosine = 1 - cosine
    result = np.eye(4)
    result[:3, :3] = (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )
    return result


def quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to MuJoCo's wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    result = np.asarray(values)
    return result / np.linalg.norm(result)


def fmt(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def link_transforms(
    robot: ET.Element,
    joint_positions: dict[str, float],
) -> dict[str, np.ndarray]:
    """Return link poses for an explicit display configuration."""
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    child_links = set()
    links = {link.get("name", "") for link in robot.findall("link")}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name, child_name = parent.get("link", ""), child.get("link", "")
        origin = joint.find("origin")
        local = transform(
            None if origin is None else origin.get("xyz"),
            None if origin is None else origin.get("rpy"),
        ) @ joint_motion(joint, joint_positions.get(joint.get("name", ""), 0.0))
        children.setdefault(parent_name, []).append((child_name, local))
        child_links.add(child_name)
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise ValueError(f"expected one URDF root link, found {roots}")
    worlds = {roots[0]: np.eye(4)}
    queue = [roots[0]]
    while queue:
        parent_name = queue.pop()
        for child_name, local in children.get(parent_name, ()):
            worlds[child_name] = worlds[parent_name] @ local
            queue.append(child_name)
    return worlds


def visual_records(config: Robot) -> list[dict]:
    robot = ET.parse(config.urdf).getroot()
    worlds = link_transforms(robot, dict(config.joint_positions))
    records = []
    model_rotation = yaw_transform(config.yaw_degrees)
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for index, visual in enumerate(link.findall("visual")):
            origin = visual.find("origin")
            local = transform(
                None if origin is None else origin.get("xyz"),
                None if origin is None else origin.get("rpy"),
            )
            world = model_rotation @ worlds[link_name] @ local
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                continue
            shape = geometry[0]
            record = {
                "name": f"{safe_name(link_name)}_{index}",
                "world": world,
                "kind": shape.tag,
            }
            if shape.tag == "mesh":
                path = (config.urdf.parent / shape.get("filename", "")).resolve()
                scale = numbers(shape.get("scale"), (1.0, 1.0, 1.0))
                mesh = trimesh.load_mesh(path, process=False)
                corners = trimesh.bounds.corners(np.asarray(mesh.bounds) * scale)
                record.update(path=path, scale=scale, corners=corners)
            elif shape.tag == "box":
                size = numbers(shape.get("size"), (1.0, 1.0, 1.0))
                corners = trimesh.bounds.corners(np.asarray((-size / 2, size / 2)))
                record.update(size=size / 2, corners=corners)
            elif shape.tag == "cylinder":
                radius, length = float(shape.get("radius", "1")), float(shape.get("length", "1"))
                bounds = np.asarray(((-radius, -radius, -length / 2), (radius, radius, length / 2)))
                record.update(size=np.asarray((radius, length / 2)), corners=trimesh.bounds.corners(bounds))
            elif shape.tag == "sphere":
                radius = float(shape.get("radius", "1"))
                bounds = np.asarray(((-radius,) * 3, (radius,) * 3))
                record.update(size=np.asarray((radius,)), corners=trimesh.bounds.corners(bounds))
            else:
                raise ValueError(f"unsupported visual geometry {shape.tag} in {config.urdf}")
            records.append(record)
    if not records:
        raise ValueError(f"no visual geometry in {config.urdf}")
    return records


def record_bounds(record: dict) -> np.ndarray:
    points = trimesh.transform_points(record["corners"], record["world"])
    return np.asarray((points.min(axis=0), points.max(axis=0)))


def create_scene() -> tuple[list[dict], np.ndarray]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": "eef-free mobile manipulators"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(WIDTH), "offheight": str(HEIGHT)})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": ".36 .36 .36", "diffuse": ".70 .70 .70", "specular": ".16 .16 .16"},
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": ".12 .16 .22",
            "rgb2": ".025 .035 .055",
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
            "rgb1": ".25 .27 .31",
            "rgb2": ".12 .13 .16",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "floor", "texture": "floor_tex", "texrepeat": "12 8", "reflectance": ".10"},
    )
    payloads = []
    for robot_index, config in enumerate(ROBOTS):
        records = visual_records(config)
        bounds = np.asarray([record_bounds(record) for record in records])
        aggregate = np.asarray((bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)))
        payloads.append({"config": config, "records": records, "bounds": aggregate})
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"robot_mat_{robot_index}",
                "rgba": config.rgba,
                "roughness": ".43",
                "metallic": ".04",
            },
        )
        for visual_index, record in enumerate(records):
            if record["kind"] == "mesh":
                ET.SubElement(
                    asset,
                    "mesh",
                    {
                        "name": f"robot_{robot_index}_mesh_{visual_index}",
                        "file": str(record["path"]),
                        "scale": fmt(record["scale"]),
                        "smoothnormal": "true",
                    },
                )

    widths = [payload["bounds"][1, 0] - payload["bounds"][0, 0] for payload in payloads]
    gap = max(0.52, 0.14 * max(widths))
    total_width = sum(widths) + gap * (len(payloads) - 1)
    cursor = -total_width / 2
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {"pos": "-3 -6 9", "dir": ".22 .28 -1", "directional": "true", "castshadow": "true"},
    )
    ET.SubElement(
        world,
        "light",
        {"pos": "6 1 6", "dir": "-.5 -.15 -1", "directional": "true", "diffuse": ".42 .45 .52"},
    )
    placements = []
    for robot_index, payload in enumerate(payloads):
        low, high = payload["bounds"]
        offset = np.asarray((cursor - low[0], -0.5 * (low[1] + high[1]), 0.025 - low[2]))
        cursor += widths[robot_index] + gap
        placements.append({"offset": offset, "width": widths[robot_index]})
        parent = ET.SubElement(
            world,
            "body",
            {"name": f"robot_{payload['config'].robot_id}", "pos": fmt(offset)},
        )
        for visual_index, record in enumerate(payload["records"]):
            body = ET.SubElement(
                parent,
                "body",
                {
                    "name": f"robot_{robot_index}_{record['name']}",
                    "pos": fmt(record["world"][:3, 3]),
                    "quat": fmt(quaternion(record["world"][:3, :3])),
                },
            )
            attributes = {
                "name": f"visual_{robot_index}_{visual_index}",
                "type": record["kind"],
                "material": f"robot_mat_{robot_index}",
                "mass": "0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }
            if record["kind"] == "mesh":
                attributes["mesh"] = f"robot_{robot_index}_mesh_{visual_index}"
            else:
                attributes["size"] = fmt(record["size"])
            ET.SubElement(body, "geom", attributes)

    max_height = max(payload["bounds"][1, 2] - payload["bounds"][0, 2] for payload in payloads)
    ET.SubElement(
        world,
        "geom",
        {
            "type": "plane",
            "size": f"{total_width / 2 + 1.5:.3f} 3 .1",
            "pos": "0 0 0",
            "material": "floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE_PATH, encoding="utf-8", xml_declaration=True)
    return placements, np.asarray((total_width, max_height))


def render() -> dict[str, int]:
    placements, scene_size = create_scene()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.0, 0.0, scene_size[1] * 0.48]
    camera.distance = max(5.0, scene_size[0] * 0.78)
    camera.azimuth = 90.0
    camera.elevation = -9.0
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB").save(
        IMAGE_PATH,
        quality=96,
    )
    return {
        "bodies": int(model.nbody),
        "geoms": int(model.ngeom),
        "meshes": int(model.nmesh),
        "robots": len(placements),
    }


def main() -> int:
    metrics = render()
    print(f"{IMAGE_PATH} | one scene: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
