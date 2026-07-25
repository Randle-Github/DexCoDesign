"""Return the source display pose for length-only morphologies."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .common import SPEC_BY_ID, numbers, transform


def joint_motion(
    joint: ET.Element, position: float
) -> np.ndarray:
    """Evaluate one unchanged source joint for visualization FK."""
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
        raise ValueError(f"unsupported joint type: {kind}")
    x, y, z = axis
    cosine, sine = math.cos(position), math.sin(position)
    cross = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    result = np.eye(4)
    result[:3, :3] = (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer(axis, axis)
        + sine * cross
    )
    return result


def link_transforms(
    robot: ET.Element,
    positions: dict[str, float],
) -> dict[str, np.ndarray]:
    """Evaluate source link frames without IK or morphology compensation."""
    links = {link.get("name", "") for link in robot.findall("link")}
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    child_names = set()
    for joint in robot.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        local = transform(
            numbers(None if origin is None else origin.get("xyz")),
            numbers(None if origin is None else origin.get("rpy")),
        ) @ joint_motion(
            joint, positions.get(joint.get("name", ""), 0.0)
        )
        children.setdefault(parent.get("link", ""), []).append(
            (child.get("link", ""), local)
        )
        child_names.add(child.get("link", ""))
    roots = sorted(links - child_names)
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {roots}")
    worlds = {roots[0]: np.eye(4)}
    queue = [roots[0]]
    while queue:
        parent = queue.pop()
        for child, local in children.get(parent, ()):
            worlds[child] = worlds[parent] @ local
            queue.append(child)
    return worlds


def solve_hanging_display_positions(
    source_id: str,
    actions: list[dict],
    generated: ET.Element,
) -> tuple[dict[str, float], dict]:
    """Length edits do not require IK or any display-joint compensation."""
    del actions, generated
    positions = dict(SPEC_BY_ID[source_id].display_joint_positions)
    return positions, {
        "method": "source_upright_pose",
        "joint_values_unchanged": True,
        "ik_used": False,
    }
