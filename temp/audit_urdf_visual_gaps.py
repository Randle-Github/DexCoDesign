#!/usr/bin/env python3
"""Audit visible separation across movable parent/child joints in registered URDF hands."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from visualize_right_hands import ASSETS, REGISTRY, load_mesh_file, origin_transform, urdf_geometry


SAMPLES_PER_LINK = 3000


@dataclass
class JointGap:
    name: str
    parent: str
    child: str
    surface_gap: float
    parent_pivot_depth: float
    child_pivot_depth: float


def joint_values(root: ET.Element) -> dict[str, float]:
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    values: dict[str, float] = {}
    for name, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is not None:
            continue
        limit = joint.find("limit")
        lower = float(limit.get("lower", "-inf")) if limit is not None else -math.inf
        upper = float(limit.get("upper", "inf")) if limit is not None else math.inf
        values[name] = min(max(0.0, lower), upper)
    pending = {name for name in joints if name not in values}
    while pending:
        progressed = False
        for name in list(pending):
            mimic = joints[name].find("mimic")
            target = mimic.get("joint")
            if target not in values:
                continue
            values[name] = (
                values[target] * float(mimic.get("multiplier", "1"))
                + float(mimic.get("offset", "0"))
            )
            pending.remove(name)
            progressed = True
        if not progressed:
            raise ValueError(f"Unresolved mimic chain: {sorted(pending)}")
    return values


def motion_transform(joint: ET.Element, value: float) -> np.ndarray:
    transform = np.eye(4)
    joint_type = joint.get("type")
    axis = np.fromstring((joint.find("axis").get("xyz") if joint.find("axis") is not None else "1 0 0"), sep=" ")
    norm = np.linalg.norm(axis)
    if norm:
        axis = axis / norm
    if joint_type in {"revolute", "continuous"}:
        transform[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
    elif joint_type == "prismatic":
        transform[:3, 3] = axis * value
    return transform


def link_visual_mesh(link: ET.Element, urdf_dir: Path) -> trimesh.Trimesh | None:
    pieces: list[trimesh.Trimesh] = []
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh = urdf_geometry(geometry, urdf_dir)
        mesh.apply_transform(origin_transform(visual.find("origin")))
        pieces.append(mesh)
    if not pieces:
        return None
    return trimesh.util.concatenate(pieces)


def sample_points(mesh: trimesh.Trimesh, seed: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > SAMPLES_PER_LINK:
        rng = np.random.default_rng(seed)
        vertices = vertices[rng.choice(len(vertices), SAMPLES_PER_LINK, replace=False)]
    count = min(SAMPLES_PER_LINK, max(500, len(mesh.faces)))
    surface, _ = trimesh.sample.sample_surface(mesh, count=count, seed=seed)
    return np.vstack([vertices, surface])


def nearest_distance(first: np.ndarray, second: np.ndarray) -> float:
    return min(
        float(cKDTree(first).query(second, k=1, workers=-1)[0].min()),
        float(cKDTree(second).query(first, k=1, workers=-1)[0].min()),
    )


def audit(path: Path) -> tuple[float, list[JointGap]]:
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    values = joint_values(root)
    child_joints: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_names: set[str] = set()
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        child_joints[parent].append(joint)
        child_names.add(child)
    root_name = next(iter(set(links).difference(child_names)))

    local_meshes = {name: link_visual_mesh(link, path.parent) for name, link in links.items()}
    world_meshes: dict[str, trimesh.Trimesh] = {}
    transforms: dict[str, np.ndarray] = {}

    def visit(link_name: str, transform: np.ndarray) -> None:
        transforms[link_name] = transform
        mesh = local_meshes[link_name]
        if mesh is not None:
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            world_meshes[link_name] = mesh
        for joint in child_joints[link_name]:
            child = joint.find("child").get("link")
            child_transform = (
                transform
                @ origin_transform(joint.find("origin"))
                @ motion_transform(joint, values[joint.get("name")])
            )
            visit(child, child_transform)

    visit(root_name, np.eye(4))
    hand_mesh = trimesh.util.concatenate(list(world_meshes.values()))
    hand_scale = float(np.linalg.norm(hand_mesh.extents))
    gaps: list[JointGap] = []
    for index, joint in enumerate(joints):
        if joint.get("type") == "fixed":
            continue
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent not in world_meshes or child not in world_meshes:
            continue
        parent_points = sample_points(world_meshes[parent], seed=1000 + index * 2)
        child_points = sample_points(world_meshes[child], seed=1001 + index * 2)
        pivot = transforms[child][:3, 3]
        gaps.append(
            JointGap(
                name=joint.get("name"),
                parent=parent,
                child=child,
                surface_gap=nearest_distance(parent_points, child_points),
                parent_pivot_depth=float(cKDTree(parent_points).query(pivot)[0]),
                child_pivot_depth=float(cKDTree(child_points).query(pivot)[0]),
            )
        )
    return hand_scale, sorted(gaps, key=lambda gap: gap.surface_gap, reverse=True)


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    suspicious = 0
    for hand_id, metadata in registry["hands"].items():
        for side, entry in metadata["entries"].items():
            if entry["format"] != "urdf":
                continue
            path = ASSETS / entry["path"]
            scale, gaps = audit(path)
            limit = max(0.008, scale * 0.04)
            flagged = [gap for gap in gaps if gap.surface_gap > limit]
            suspicious += len(flagged)
            worst = gaps[:3]
            details = "; ".join(
                f"{gap.name}={gap.surface_gap * 1000:.2f}mm"
                for gap in worst
            ) or "no visual-to-visual movable joints"
            print(
                f"{'FLAG' if flagged else 'PASS'} {hand_id:22} {side:5} "
                f"scale={scale * 1000:7.1f}mm threshold={limit * 1000:5.1f}mm  {details}"
            )
            for gap in flagged:
                print(
                    f"     {gap.parent} -> {gap.child} via {gap.name}: "
                    f"gap={gap.surface_gap * 1000:.2f}mm "
                    f"pivot(parent/child)={gap.parent_pivot_depth * 1000:.2f}/"
                    f"{gap.child_pivot_depth * 1000:.2f}mm"
                )
    print(f"\nSuspicious movable parent/child visual gaps: {suspicious}")
    return 1 if suspicious else 0


if __name__ == "__main__":
    raise SystemExit(main())
