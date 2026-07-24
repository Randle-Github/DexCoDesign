#!/usr/bin/env python3
"""Build canonical rigid-part graphs from normalized direct-motor URDFs.

The compact canonical scaffold stores semantic roles and a common upright
frame for the current source library. Geometry, complete zero-pose link
frames, joint ownership, and meshes always come from the normalized
direct-motor URDFs.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_GRAPHS = ROOT / "assets" / "robot_hands" / "morphology" / "canonical_scaffold.json"
DIRECT_ROOT = ROOT / "assets" / "robot_hands" / "direct_motor"
DIRECT_REGISTRY = DIRECT_ROOT / "registry.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "hand_morphology"
OUTPUT_GRAPHS = ARTIFACT_ROOT / "reference_graphs.json"
OUTPUT_MESHES = ARTIFACT_ROOT / "reference_rigid_parts"
# These two MJCF-derived sources used a smaller canonical length unit in the
# old semantic scaffold. This is a unit correction, not a morphology edit.
CANONICAL_UNIT_SCALE = {
    "orca_hand_v2": 1.30,
    "shadow_hand_e": 1.65,
}


def numbers(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.replace(",", " ").split()])


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def element_transform(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    return transform(
        numbers(element.get("xyz"), (0.0, 0.0, 0.0)),
        numbers(element.get("rpy"), (0.0, 0.0, 0.0)),
    )


def load_direct_urdf(path: Path) -> dict:
    robot = ET.parse(path).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {}
    children = set()
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        record = {
            "name": joint.get("name"),
            "parent": parent.get("link"),
            "child": child.get("link"),
            "origin": element_transform(joint.find("origin")),
        }
        joints[record["name"]] = record
        children.add(record["child"])
    roots = sorted(set(links) - children)
    if len(roots) != 1:
        raise ValueError(f"{path}: expected one root link, found {roots}")
    by_parent: dict[str, list[dict]] = {}
    for record in joints.values():
        by_parent.setdefault(record["parent"], []).append(record)
    world = {roots[0]: np.eye(4)}
    queue = [roots[0]]
    while queue:
        parent = queue.pop(0)
        for joint in by_parent.get(parent, []):
            world[joint["child"]] = world[parent] @ joint["origin"]
            queue.append(joint["child"])
    if set(world) != set(links):
        raise ValueError(f"{path}: disconnected URDF link graph")
    return {
        "path": path,
        "links": links,
        "joints": joints,
        "root_link": roots[0],
        "world": world,
    }


def similarity_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares proper 3-D similarity mapping row-vector points."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 3:
        raise ValueError("at least three matched frames are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1.0e-12))
    translation = target_center - scale * source_center @ rotation
    fitted = scale * source @ rotation + translation
    rms = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return scale, rotation, translation, rms


def anchor_link(part: dict, direct: dict) -> str | None:
    if int(part["id"]) == 0:
        return direct["root_link"]
    joint = direct["joints"].get(part["joint_name"])
    if joint is not None:
        return joint["child"]
    members = [name for name in part.get("member_links", []) if name in direct["links"]]
    return members[0] if members else None


def load_visual_meshes(
    part: dict,
    direct: dict,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    canonical_anchor: np.ndarray,
) -> trimesh.Trimesh | None:
    meshes = []
    for link_name in part.get("member_links", []):
        link = direct["links"].get(link_name)
        if link is None:
            continue
        for visual in link.findall("visual"):
            mesh_element = visual.find("./geometry/mesh")
            if mesh_element is None or not mesh_element.get("filename"):
                continue
            mesh_path = Path(mesh_element.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = (direct["path"].parent / mesh_path).resolve()
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            if mesh.is_empty or len(mesh.faces) == 0:
                continue
            mesh = mesh.copy()
            mesh_scale = numbers(mesh_element.get("scale"), (1.0, 1.0, 1.0))
            vertices = np.asarray(mesh.vertices, dtype=float) * mesh_scale
            local = element_transform(visual.find("origin"))
            homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
            vertices = (homogeneous @ local.T)[:, :3]
            homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
            vertices = (homogeneous @ direct["world"][link_name].T)[:, :3]
            vertices = scale * vertices @ rotation + translation
            mesh.vertices = vertices - canonical_anchor
            meshes.append(mesh)
    if not meshes:
        return None
    result = trimesh.util.concatenate(meshes)
    result.remove_unreferenced_vertices()
    return result


def repair_hand(hand: dict, direct: dict) -> tuple[dict, dict]:
    unit_scale = CANONICAL_UNIT_SCALE.get(hand["hand_id"], 1.0)
    source_points = []
    target_points = []
    matches = []
    for part in hand["parts"]:
        link_name = anchor_link(part, direct)
        if link_name is None:
            continue
        source_points.append(direct["world"][link_name][:3, 3])
        target_points.append(
            unit_scale * np.asarray(part["world_pos"], dtype=float)
        )
        matches.append((int(part["id"]), link_name))
    scale, rotation, translation, rms = similarity_fit(
        np.asarray(source_points), np.asarray(target_points)
    )
    repaired = deepcopy(hand)
    for part in repaired["parts"]:
        part["world_pos"] = (
            unit_scale * np.asarray(part["world_pos"], dtype=float)
        ).tolist()
        part["relative_pos"] = (
            unit_scale * np.asarray(part["relative_pos"], dtype=float)
        ).tolist()
        part["edge_length"] = unit_scale * float(part["edge_length"])
    meshed = []
    still_empty = []
    for part in repaired["parts"]:
        canonical_anchor = np.asarray(part["world_pos"], dtype=float)
        mesh = load_visual_meshes(
            part, direct, scale, rotation, translation, canonical_anchor
        )
        if mesh is None:
            still_empty.append(int(part["id"]))
            continue
        output = OUTPUT_MESHES / repaired["hand_id"] / f"part_{int(part['id']):02d}.obj"
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output, file_type="obj", include_normals=True, include_color=False)
        bounds = np.asarray(mesh.bounds, dtype=float)
        part["mesh"] = {
            "file": str(output.relative_to(ARTIFACT_ROOT)),
            "faces": int(len(mesh.faces)),
            "bounds": bounds.tolist(),
            "size": np.maximum(bounds[1] - bounds[0], 1.0e-4).tolist(),
            "centroid_offset": np.asarray(mesh.centroid, dtype=float).tolist(),
            "source": "normalized_direct_motor_urdf_visuals",
        }
        part["part_size"] = part["mesh"]["size"]
        part["centroid_offset"] = part["mesh"]["centroid_offset"]
        meshed.append(int(part["id"]))
    canonicalization = {
        "hand_id": repaired["hand_id"],
        "matched_frames": len(matches),
        "similarity_scale": scale,
        "similarity_rotation": rotation.tolist(),
        "similarity_translation": translation.tolist(),
        "frame_fit_rms": rms,
        "canonical_unit_scale": unit_scale,
        "meshed_parts": meshed,
        "still_empty_parts": still_empty,
    }
    repaired["geometry_source"] = "normalized_direct_motor_urdf"
    repaired["canonicalization"] = canonicalization
    return repaired, canonicalization


def main() -> int:
    source = json.loads(SOURCE_GRAPHS.read_text(encoding="utf-8"))
    registry = json.loads(DIRECT_REGISTRY.read_text(encoding="utf-8"))["hands"]
    repaired_hands = []
    canonicalizations = []
    for hand in source["hands"]:
        entry = registry[hand["hand_id"]]["entries"]["right"]
        direct_path = ROOT / "assets" / "robot_hands" / entry["path"]
        repaired, canonicalization = repair_hand(
            hand, load_direct_urdf(direct_path)
        )
        repaired_hands.append(repaired)
        canonicalizations.append(canonicalization)
        print(
            f"{hand['hand_id']:20s} "
            f"fit={canonicalization['frame_fit_rms']:.5f} "
            f"meshed={len(canonicalization['meshed_parts']):2d}/"
            f"{len(hand['parts']):2d} "
            f"empty={canonicalization['still_empty_parts']}"
        )
    payload = {
        **{key: value for key, value in source.items() if key != "hands"},
        "geometry_source": "normalized direct-motor URDF visual links",
        "hands": repaired_hands,
        "canonicalizations": canonicalizations,
    }
    OUTPUT_GRAPHS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_GRAPHS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
