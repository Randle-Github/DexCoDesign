#!/usr/bin/env python3
"""Create self-contained, end-effector-free mobile-manipulator URDFs.

The source assets are vendor files stored under ``assets/mobile_manipulators``.
This script never edits them in place. It removes gripper/hand subtrees,
preserves the vendor tool transform as an empty EEF mount, converts visual and
collision meshes to MuJoCo-compatible OBJ, and writes a compact manifest.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets" / "mobile_manipulators"


@dataclass(frozen=True)
class Mount:
    side: str
    parent_link: str
    source_joint: str | None = None
    xyz: str = "0 0 0"
    rpy: str = "0 0 0"


@dataclass(frozen=True)
class Robot:
    robot_id: str
    source_urdf: Path
    output_urdf: Path
    prune_roots: tuple[str, ...]
    mounts: tuple[Mount, ...]
    color: tuple[float, float, float, float]


ROBOTS = (
    Robot(
        robot_id="dexmate_vega",
        source_urdf=ASSET_ROOT / "dexmate_vega" / "source" / "vega_1.urdf",
        output_urdf=ASSET_ROOT / "dexmate_vega" / "eef_free" / "robot.urdf",
        prune_roots=(),
        mounts=(
            Mount(
                "left",
                "L_arm_l8",
                xyz="0.00019 0.00390 0.00014",
                rpy="-1.57079 1.57079 0",
            ),
            Mount(
                "right",
                "R_arm_l8",
                xyz="0.00019 -0.00390 0.00014",
                rpy="1.57079 1.57079 0",
            ),
        ),
        color=(0.18, 0.48, 0.78, 1.0),
    ),
    Robot(
        robot_id="rby1",
        source_urdf=ASSET_ROOT / "rby1" / "source" / "model.urdf",
        output_urdf=ASSET_ROOT / "rby1" / "eef_free" / "robot.urdf",
        prune_roots=("ee_right", "ee_left"),
        mounts=(
            Mount("right", "link_right_arm_6", source_joint="END_right"),
            Mount("left", "link_left_arm_6", source_joint="END_left"),
        ),
        color=(0.82, 0.35, 0.19, 1.0),
    ),
    Robot(
        robot_id="galaxea_r1",
        source_urdf=ASSET_ROOT / "galaxea_r1" / "source" / "r1_v2_1_0.urdf",
        output_urdf=ASSET_ROOT / "galaxea_r1" / "eef_free" / "robot.urdf",
        prune_roots=(
            "left_gripper_link",
            "right_gripper_link",
            "left_realsense_link",
            "right_realsense_link",
        ),
        mounts=(
            Mount(
                "left",
                "left_arm_link6",
                source_joint="left_gripper_joint",
            ),
            Mount(
                "right",
                "right_arm_link6",
                source_joint="right_gripper_joint",
            ),
        ),
        color=(0.20, 0.62, 0.42, 1.0),
    ),
)


def descendants(robot: ET.Element, roots: tuple[str, ...]) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    removed = set(roots)
    queue = list(roots)
    while queue:
        parent = queue.pop()
        for child in children.get(parent, ()):
            if child not in removed:
                removed.add(child)
                queue.append(child)
    return removed


def source_mesh_path(source_urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        suffix = filename[len("package://") :]
        suffix = suffix.split("/", 1)[1] if "/" in suffix else suffix
        return (source_urdf.parent / suffix).resolve()
    return (source_urdf.parent / filename).resolve()


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
        if not meshes:
            raise ValueError(f"no mesh geometry in {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        raise TypeError(f"unsupported mesh type in {path}: {type(loaded).__name__}")
    mesh.remove_unreferenced_vertices()
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def copy_joint_origin(joint: ET.Element | None) -> tuple[str, str]:
    if joint is None:
        return "0 0 0", "0 0 0"
    origin = joint.find("origin")
    if origin is None:
        return "0 0 0", "0 0 0"
    return origin.get("xyz", "0 0 0"), origin.get("rpy", "0 0 0")


def add_empty_mount(
    robot: ET.Element,
    mount: Mount,
    source_joints: dict[str, ET.Element],
) -> None:
    link_name = f"{mount.side}_eef_mount"
    joint_name = f"{mount.side}_eef_mount_joint"
    xyz, rpy = (
        copy_joint_origin(source_joints.get(mount.source_joint))
        if mount.source_joint
        else (mount.xyz, mount.rpy)
    )
    ET.SubElement(robot, "link", {"name": link_name})
    joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": mount.parent_link})
    ET.SubElement(joint, "child", {"link": link_name})


def prepare(config: Robot) -> dict:
    source_tree = ET.parse(config.source_urdf)
    robot = source_tree.getroot()
    source_joints = {joint.get("name", ""): joint for joint in robot.findall("joint")}
    removed_links = descendants(robot, config.prune_roots)
    removed_joints = set()

    for joint in list(robot.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is None
            or child is None
            or parent.get("link") in removed_links
            or child.get("link") in removed_links
        ):
            removed_joints.add(joint.get("name", ""))
            robot.remove(joint)
    for link in list(robot.findall("link")):
        if link.get("name") in removed_links:
            robot.remove(link)

    for element in list(robot):
        if element.tag == "transmission":
            joint = element.find("joint")
            if joint is not None and joint.get("name") in removed_joints:
                robot.remove(element)
        elif element.tag == "gazebo":
            reference = element.get("reference")
            if reference in removed_links or reference in removed_joints:
                robot.remove(element)

    for mount in config.mounts:
        add_empty_mount(robot, mount, source_joints)

    mujoco_extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        {"discardvisual": "false", "strippath": "false"},
    )

    output_root = config.output_urdf.parent
    mesh_root = output_root / "meshes"
    if output_root.exists():
        shutil.rmtree(output_root)
    mesh_root.mkdir(parents=True)
    converted: dict[Path, str] = {}
    face_count = 0

    for mesh_element in robot.findall(".//mesh"):
        filename = mesh_element.get("filename")
        if not filename:
            continue
        source_path = source_mesh_path(config.source_urdf, filename)
        if not source_path.is_file():
            raise FileNotFoundError(f"{config.robot_id}: missing {source_path}")
        if source_path not in converted:
            mesh = as_mesh(source_path)
            output_name = f"{len(converted):03d}_{source_path.stem}.obj"
            mesh.export(
                mesh_root / output_name,
                file_type="obj",
                include_normals=True,
                include_color=False,
            )
            converted[source_path] = f"meshes/{output_name}"
            face_count += int(len(mesh.faces))
        mesh_element.set("filename", converted[source_path])

    rgba = " ".join(str(value) for value in config.color)
    for visual in robot.findall(".//visual"):
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        color = material.find("color")
        if color is None:
            color = ET.SubElement(material, "color")
        color.set("rgba", rgba)

    robot.set("name", f"{config.robot_id}_eef_free")
    ET.indent(source_tree, space="  ")
    source_tree.write(config.output_urdf, encoding="utf-8", xml_declaration=True)
    return {
        "robot_id": config.robot_id,
        "source_urdf": str(config.source_urdf.relative_to(ROOT)),
        "output_urdf": str(config.output_urdf.relative_to(ROOT)),
        "source_links": len(source_joints) + 1,
        "output_links": len(robot.findall("link")),
        "output_joints": len(robot.findall("joint")),
        "removed_links": sorted(removed_links),
        "empty_eef_mounts": [f"{mount.side}_eef_mount" for mount in config.mounts],
        "converted_meshes": len(converted),
        "mesh_faces": face_count,
    }


def main() -> int:
    records = [prepare(config) for config in ROBOTS]
    manifest = {
        "schema_version": 1,
        "representation": "vendor kinematics with empty left/right EEF mounts",
        "robots": records,
    }
    path = ASSET_ROOT / "eef_free_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
