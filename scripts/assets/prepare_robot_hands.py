#!/usr/bin/env python3
"""Build deterministic, self-contained left/right hand entry points.

Upstream files under ``assets/robot_hands/*/source`` are never modified.
Generated files are placed under each asset's ``models`` directory.
"""

from __future__ import annotations

import copy
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "robot_hands"


def _fmt(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rpy_to_matrix(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matrix_to_rpy(matrix: list[list[float]]) -> list[float]:
    pitch = math.atan2(-matrix[2][0], math.hypot(matrix[0][0], matrix[1][0]))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return [roll, pitch, yaw]


def _write_xml(root: ET.Element, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def normalize_urdf(source: Path, destination: Path, replacements: dict[str, str], name: str) -> None:
    root = ET.parse(source).getroot()
    root.set("name", name)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            for old, new in replacements.items():
                filename = filename.replace(old, new)
            mesh.set("filename", filename)
    _write_xml(root, destination)


def extract_subtree_urdf(source: Path, destination: Path, root_link: str, name: str) -> None:
    upstream = ET.parse(source).getroot()
    links = {link.get("name"): link for link in upstream.findall("link")}
    joints = upstream.findall("joint")
    descendants = {root_link}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if parent.get("link") in descendants and child.get("link") not in descendants:
                descendants.add(child.get("link"))
                changed = True

    root = ET.Element("robot", {"name": name})
    for material in upstream.findall("material"):
        root.append(copy.deepcopy(material))
    for link in upstream.findall("link"):
        if link.get("name") in descendants:
            root.append(copy.deepcopy(link))
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            if parent.get("link") in descendants and child.get("link") in descendants:
                root.append(copy.deepcopy(joint))

    missing = descendants.difference(links)
    if missing:
        raise ValueError(f"Missing links while extracting {name}: {sorted(missing)}")
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", filename.replace("../meshes/", "../source/meshes/"))
    _write_xml(root, destination)


def _mirror_stl(source: Path, destination: Path, axis: int = 0) -> None:
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            output = bytearray(data[:84])
            offset = 84
            for _ in range(triangle_count):
                values = list(struct.unpack_from("<12fH", data, offset))
                normal = values[0:3]
                vertices = [values[3:6], values[6:9], values[9:12]]
                normal[axis] *= -1
                for vertex in vertices:
                    vertex[axis] *= -1
                vertices[1], vertices[2] = vertices[2], vertices[1]
                output.extend(struct.pack("<12fH", *(normal + vertices[0] + vertices[1] + vertices[2]), values[12]))
                offset += 50
            destination.write_bytes(output)
            return

    text = data.decode("utf-8")
    output_lines = []
    triangle_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("facet normal"):
            parts = stripped.split()
            vector = [float(value) for value in parts[-3:]]
            vector[axis] *= -1
            prefix = line[: len(line) - len(line.lstrip())]
            line = f"{prefix}facet normal {_fmt(vector)}"
        if stripped.startswith("vertex"):
            parts = stripped.split()
            vector = [float(value) for value in parts[-3:]]
            vector[axis] *= -1
            prefix = line[: len(line) - len(line.lstrip())]
            triangle_lines.append(f"{prefix}vertex {_fmt(vector)}")
            if len(triangle_lines) == 3:
                output_lines.extend([triangle_lines[0], triangle_lines[2], triangle_lines[1]])
                triangle_lines = []
            continue
        output_lines.append(line)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def mirror_urdf(source: Path, destination: Path, mesh_source: Path, name: str, axis: int = 0) -> None:
    root = ET.parse(source).getroot()
    root.set("name", name)
    signs = [1.0, 1.0, 1.0]
    signs[axis] = -1.0
    reflection = [[0.0] * 3 for _ in range(3)]
    for index, sign in enumerate(signs):
        reflection[index][index] = sign

    for origin in root.iter("origin"):
        xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        origin.set("xyz", _fmt([signs[i] * xyz[i] for i in range(3)]))
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        rotation = _rpy_to_matrix(rpy)
        mirrored_rotation = _matmul(_matmul(reflection, rotation), reflection)
        origin.set("rpy", _fmt(_matrix_to_rpy(mirrored_rotation)))

    for axis_element in root.iter("axis"):
        vector = [float(value) for value in axis_element.get("xyz", "1 0 0").split()]
        axis_element.set("xyz", _fmt([-signs[i] * vector[i] for i in range(3)]))

    for inertia in root.iter("inertia"):
        tensor = [
            [float(inertia.get("ixx", "0")), float(inertia.get("ixy", "0")), float(inertia.get("ixz", "0"))],
            [float(inertia.get("ixy", "0")), float(inertia.get("iyy", "0")), float(inertia.get("iyz", "0"))],
            [float(inertia.get("ixz", "0")), float(inertia.get("iyz", "0")), float(inertia.get("izz", "0"))],
        ]
        mirrored = _matmul(_matmul(reflection, tensor), reflection)
        for attribute, value in {
            "ixx": mirrored[0][0], "ixy": mirrored[0][1], "ixz": mirrored[0][2],
            "iyy": mirrored[1][1], "iyz": mirrored[1][2], "izz": mirrored[2][2],
        }.items():
            inertia.set(attribute, f"{value:.12g}")

    mesh_destination = destination.parent / "meshes"
    if mesh_destination.exists():
        shutil.rmtree(mesh_destination)
    mesh_destination.mkdir(parents=True)
    for mesh_file in mesh_source.glob("*.stl"):
        _mirror_stl(mesh_file, mesh_destination / mesh_file.name, axis=axis)
    for mesh_file in mesh_source.glob("*.STL"):
        _mirror_stl(mesh_file, mesh_destination / mesh_file.name, axis=axis)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", f"meshes/{Path(filename).name}")
    _write_xml(root, destination)


def copy_right_and_derive_left(asset: str, urdf: Path, meshes: Path, package_prefix: str) -> None:
    model_root = ASSETS / asset / "models"
    if model_root.exists():
        shutil.rmtree(model_root)
    right = model_root / "right"
    (right / "meshes").mkdir(parents=True)
    for pattern in ("*.stl", "*.STL"):
        for mesh_file in meshes.glob(pattern):
            shutil.copy2(mesh_file, right / "meshes" / mesh_file.name)
    normalize_urdf(urdf, right / "hand.urdf", {package_prefix: "meshes/"}, f"{asset}_right")
    mirror_urdf(urdf, model_root / "left" / "hand.urdf", meshes, f"{asset}_left", axis=0)


def main() -> None:
    # Vendor URDFs with ROS package URIs get portable, repository-relative entry points.
    normalize_urdf(
        ASSETS / "sharpa_wave_01/source/left_sharpa_wave/left_sharpa_wave.urdf",
        ASSETS / "sharpa_wave_01/models/left.urdf",
        {"package://left_sharpa_wave/": "../source/left_sharpa_wave/"},
        "sharpa_wave_01_left",
    )
    normalize_urdf(
        ASSETS / "sharpa_wave_01/source/right_sharpa_wave/right_sharpa_wave.urdf",
        ASSETS / "sharpa_wave_01/models/right.urdf",
        {"package://right_sharpa_wave/": "../source/right_sharpa_wave/"},
        "sharpa_wave_01_right",
    )
    for side in ("left", "right"):
        normalize_urdf(
            ASSETS / f"tesollo_dg5f/source/dg5f/dg5f_{side}.urdf",
            ASSETS / f"tesollo_dg5f/models/{side}.urdf",
            {"package://meshes/": "../source/dg5f/meshes/"},
            f"tesollo_dg5f_{side}",
        )
        normalize_urdf(
            ASSETS / f"orca_hand_v2/source/v2/models/urdf/orcahand_{side}.urdf",
            ASSETS / f"orca_hand_v2/models/{side}.urdf",
            {"package://orcahand_description/v2/": "../source/v2/"},
            f"orca_hand_v2_{side}",
        )
    for side, letter in (("left", "L"), ("right", "R")):
        normalize_urdf(
            ASSETS / f"unitree_dex5_1/source/dex5_1/Dex5-URDF-{letter}/Dex5-URDF-{letter}.urdf",
            ASSETS / f"unitree_dex5_1/models/{side}.urdf",
            {},
            f"unitree_dex5_1_{side}",
        )
        # The normalized file is two levels above its source mesh folder.
        root = ET.parse(ASSETS / f"unitree_dex5_1/models/{side}.urdf").getroot()
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", f"../source/dex5_1/Dex5-URDF-{letter}/{filename}")
        _write_xml(root, ASSETS / f"unitree_dex5_1/models/{side}.urdf")

    extract_subtree_urdf(
        ASSETS / "robotera_xhand1/source/urdf/l3_with_hand_fixedpin_xml.urdf",
        ASSETS / "robotera_xhand1/models/left.urdf",
        "left_hand_base_link",
        "robotera_xhand1_left",
    )
    extract_subtree_urdf(
        ASSETS / "robotera_xhand1/source/urdf/l3_with_hand_fixedpin_xml.urdf",
        ASSETS / "robotera_xhand1/models/right.urdf",
        "right_hand_base_link",
        "robotera_xhand1_right",
    )

    for side in ("left", "right"):
        normalize_urdf(
            ASSETS / f"allegro_hand_v5/source/allegro_hand_description/allegro_hand_description_{side}_A.urdf",
            ASSETS / f"allegro_hand_v5/models/{side}.urdf",
            {"package://allegro_hand_description/": "../source/allegro_hand_description/"},
            f"allegro_hand_v5_{side}",
        )

    copy_right_and_derive_left(
        "midas_hand",
        ASSETS / "midas_hand/source/assets/midas_description/midas_hand_urdf.urdf",
        ASSETS / "midas_hand/source/assets/meshes",
        "package://midas_hand_urdf/meshes/",
    )
    copy_right_and_derive_left(
        "ruka_v2",
        ASSETS / "ruka_v2/source/assets/robot.urdf",
        ASSETS / "ruka_v2/source/assets",
        "package://assets/",
    )


if __name__ == "__main__":
    main()
