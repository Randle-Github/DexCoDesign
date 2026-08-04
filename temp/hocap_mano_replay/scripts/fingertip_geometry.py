#!/usr/bin/env python3
"""Resolve physical fingertip surface points from normalized hand URDFs.

The retargeting registry names the last articulated link, but that link's
origin is usually the DIP/IP motor center rather than the fingertip.  This
module follows any fixed descendants and then uses the terminal visual mesh
to resolve a surface point.  The returned vector is expressed in the last
articulated link frame and therefore moves rigidly with the distal segment.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _vector(element: ET.Element | None, name: str, default: str) -> np.ndarray:
    if element is None:
        return np.fromstring(default, sep=" ", dtype=np.float64)
    return np.fromstring(element.get(name, default), sep=" ", dtype=np.float64)


def _origin(element: ET.Element | None) -> tuple[np.ndarray, Rotation]:
    return (
        _vector(element, "xyz", "0 0 0"),
        Rotation.from_euler("xyz", _vector(element, "rpy", "0 0 0")),
    )


def _mesh(value: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        return trimesh.util.concatenate(tuple(value.geometry.values()))
    return value


def _geometry_points(
    visual: ET.Element,
    urdf_root: Path,
) -> np.ndarray:
    geometry = visual.find("geometry")
    if geometry is None:
        return np.empty((0, 3), dtype=np.float64)
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if not filename:
            return np.empty((0, 3), dtype=np.float64)
        if filename.startswith("package://"):
            filename = filename.split("/", 3)[-1]
        mesh_path = (urdf_root / filename).resolve()
        loaded = _mesh(trimesh.load(mesh_path, process=False))
        points = np.asarray(loaded.vertices, dtype=np.float64)
        scale = _vector(mesh, "scale", "1 1 1")
        return points * scale
    box = geometry.find("box")
    if box is not None:
        half = 0.5 * _vector(box, "size", "0 0 0")
        return np.asarray(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius", "0"))
        half_length = 0.5 * float(cylinder.get("length", "0"))
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        return np.asarray(
            [
                [radius * np.cos(angle), radius * np.sin(angle), z]
                for z in (-half_length, half_length)
                for angle in angles
            ]
        )
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "0"))
        return radius * np.vstack((np.eye(3), -np.eye(3)))
    return np.empty((0, 3), dtype=np.float64)


def _link_points(link: ET.Element, urdf_root: Path) -> np.ndarray:
    point_sets = []
    elements = link.findall("visual") or link.findall("collision")
    for element in elements:
        points = _geometry_points(element, urdf_root)
        if not len(points):
            continue
        position, rotation = _origin(element.find("origin"))
        point_sets.append(rotation.apply(points) + position)
    return (
        np.concatenate(point_sets, axis=0)
        if point_sets
        else np.empty((0, 3), dtype=np.float64)
    )


def _surface_cap(points: np.ndarray, guide: np.ndarray) -> np.ndarray:
    guide = np.asarray(guide, dtype=np.float64)
    guide_norm = float(np.linalg.norm(guide))
    if guide_norm < 1e-9:
        center = points.mean(axis=0)
        guide = center if np.linalg.norm(center) > 1e-9 else np.array([0.0, 0.0, 1.0])
    guide /= np.linalg.norm(guide)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    # The mesh principal axis supplies the actual rigid-part direction; the
    # incoming kinematic segment is used only to choose its distal sign.  This
    # matters for morphology meshes whose local coordinates were rotated while
    # their joint origins remained expressed in the common graph frame.
    if float(np.dot(principal, guide)) < 0.0:
        principal = -principal
    if eigenvalues[-1] < 1.15 * max(eigenvalues[-2], 1e-16):
        principal = guide
    projection = points @ principal
    span = float(np.ptp(projection))
    threshold = float(projection.max()) - max(0.02 * span, 2e-4)
    cap = points[projection >= threshold]
    endpoint = cap.mean(axis=0)
    # Preserve the true outer surface instead of moving the averaged cap
    # slightly inward along the distal axis.
    endpoint += (
        float(projection.max()) - float(endpoint @ principal)
    ) * principal
    return endpoint


def resolve_fingertip_offsets(
    urdf_path: Path,
    distal_links: dict[str, str],
) -> dict[str, np.ndarray]:
    """Return a physical surface offset for every registered distal link."""
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    fixed_children: dict[str, list[tuple[str, np.ndarray, Rotation]]] = {}
    incoming: dict[str, tuple[np.ndarray, Rotation]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        position, rotation = _origin(joint.find("origin"))
        incoming[child_name] = (position, rotation)
        if joint.get("type") == "fixed":
            fixed_children.setdefault(parent_name, []).append(
                (child_name, position, rotation)
            )

    result: dict[str, np.ndarray] = {}
    for finger, distal_link in distal_links.items():
        terminal_link = distal_link
        position = np.zeros(3, dtype=np.float64)
        rotation = Rotation.identity()
        last_step = incoming.get(distal_link, (np.array([0.0, 0.0, 1.0]), Rotation.identity()))[0]
        while len(fixed_children.get(terminal_link, ())) == 1:
            child, child_position, child_rotation = fixed_children[terminal_link][0]
            last_step = rotation.apply(child_position)
            position = position + last_step
            rotation = rotation * child_rotation
            terminal_link = child

        terminal_points = _link_points(links[terminal_link], urdf_path.parent)
        if len(terminal_points):
            guide_terminal = rotation.inv().apply(
                position if np.linalg.norm(position) > 1e-9 else last_step
            )
            terminal_endpoint = _surface_cap(terminal_points, guide_terminal)
            endpoint = position + rotation.apply(terminal_endpoint)
        elif np.linalg.norm(position) > 1e-9:
            endpoint = position
        else:
            base_points = _link_points(links[distal_link], urdf_path.parent)
            if not len(base_points):
                raise ValueError(
                    f"{urdf_path}: no fixed fingertip marker or geometry for "
                    f"{finger} link {distal_link}"
                )
            incoming_position, incoming_rotation = incoming.get(
                distal_link,
                (np.array([0.0, 0.0, 1.0]), Rotation.identity()),
            )
            guide = incoming_rotation.inv().apply(incoming_position)
            endpoint = _surface_cap(base_points, guide)

        if np.linalg.norm(endpoint) < 1e-6:
            raise ValueError(
                f"{urdf_path}: degenerate fingertip offset for {finger}"
            )
        result[finger] = endpoint
    return result
