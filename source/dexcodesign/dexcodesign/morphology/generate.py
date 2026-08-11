#!/usr/bin/env python3
"""Generate source-ordered hands with length/radius-only finger edits."""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACT_ROOT = ROOT / "artifacts" / "hand_morphology"
SOURCE_GRAPHS = ARTIFACT_ROOT / "reference_graphs.json"
LIBRARY = ARTIFACT_ROOT / "mechanism_bundles.json"
GENERATED_ROOT = Path(
    os.environ.get(
        "HAND_GENERATION_ROOT", str(ARTIFACT_ROOT / "generated_100")
    )
)
OUTPUT = GENERATED_ROOT / "hand_ir.json"
AUDIT = GENERATED_ROOT / "generation_summary.json"
ASSET_ROOT = ROOT / "assets" / "robot_hands"
DIRECT_REGISTRY = ASSET_ROOT / "direct_motor" / "registry.json"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
MOVABLE_URDF_TYPES = {"revolute", "continuous", "prismatic"}
MAX_FINGERS = 5
# Do not synthesize a fifth digit on a four-finger source.  Five-finger designs
# are generated from a real five-finger palm, while three/four-finger layouts
# are grammar-valid complete-bundle removals from a four/five-finger source.
ENABLE_FINGER_ADDITION = False
FIFTH_SLOT_PITCH = 0.42
PALM_LAYOUT_MODES = (
    "source_fixed",
    "anthropomorphic",
    "symmetric",
    "asymmetric",
)
PROTECTED_TRANSMISSION_SOURCES = {
    "ability_hand",
    "orca_hand_v2",
    "shadow_hand_e",
    "midas_hand",
    "ruka_v2",
    "inspire_rh56dfx",
}
BOUNDED_PALM_LAYOUT_SOURCES = PROTECTED_TRANSMISSION_SOURCES | {
    # These source palms have attachment geometry close to the fixed mount
    # sector. Large House-style radial edits exceed the source-topology
    # deformation bound; bounded edge edits remain valid.
    "mano",
    "schunk_svh",
    "tesollo_dg5f",
}


def parse_floats(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None:
        return list(default)
    return [float(item) for item in value.replace(",", " ").split()]


def load_direct_joint_records(
    hand_id: str,
    direct_registry: dict,
) -> tuple[list[dict], dict]:
    """Read the normalized direct/mimic graph independently of visual parts.

    A joint may intentionally have no mesh-bearing rigid part. This happens for
    multi-DoF roots and virtual wrist frames, so actuation cannot be inferred
    from the mesh-part graph.
    """
    entry = direct_registry["hands"][hand_id]["entries"]["right"]
    path = ASSET_ROOT / entry["path"]
    robot = ET.parse(path).getroot()
    active = {
        joint.get("name")
        for transmission in robot.findall("transmission")
        if (joint := transmission.find("joint")) is not None
    }
    records = []
    for joint in robot.findall("joint"):
        joint_type = joint.get("type", "fixed")
        if joint_type not in MOVABLE_URDF_TYPES:
            continue
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if not name or parent is None or child is None:
            raise ValueError(f"{hand_id}: malformed movable joint")
        axis = joint.find("axis")
        origin = joint.find("origin")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        if mimic is None:
            if name not in active:
                raise ValueError(
                    f"{hand_id}/{name}: movable joint is neither active nor mimic"
                )
            actuation = "active_direct"
            master = multiplier = offset = None
        else:
            if name in active:
                raise ValueError(f"{hand_id}/{name}: joint cannot be active and mimic")
            actuation = "passive_mimic"
            master = mimic.get("joint")
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
        if joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        elif limit is None:
            raise ValueError(f"{hand_id}/{name}: movable joint range is missing")
        else:
            lower = float(limit.get("lower"))
            upper = float(limit.get("upper"))
        records.append(
            {
                "joint_name": name,
                "joint_type": joint_type,
                "parent_link": parent.get("link"),
                "child_link": child.get("link"),
                "origin_translation": parse_floats(
                    None if origin is None else origin.get("xyz"),
                    (0.0, 0.0, 0.0),
                ),
                "origin_rotation_rpy": parse_floats(
                    None if origin is None else origin.get("rpy"),
                    (0.0, 0.0, 0.0),
                ),
                "joint_axis": parse_floats(
                    None if axis is None else axis.get("xyz"),
                    (1.0, 0.0, 0.0),
                ),
                "joint_range": [lower, upper],
                "actuation": actuation,
                "mimic_master": master,
                "mimic_multiplier": multiplier,
                "mimic_offset": offset,
                "editable": False,
            }
        )
    expected = int(entry["scalar_dofs"])
    if len(records) != expected:
        raise ValueError(
            f"{hand_id}: normalized URDF has {len(records)} movable joints, "
            f"registry declares {expected}"
        )
    return records, entry


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-8 else fallback.copy()


def canonical_cyclic_roles(roles: list[str], points: np.ndarray) -> list[str]:
    """Return CCW finger order, rotated to start at thumb when present."""
    center = np.mean(np.asarray(points, dtype=float), axis=0)
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2.0 * np.pi)
    ordered = [roles[int(index)] for index in np.argsort(angles)]
    if "thumb" in ordered:
        pivot = ordered.index("thumb")
        ordered = ordered[pivot:] + ordered[:pivot]
    return ordered


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal proper rotation between two directed longitudinal axes."""
    source = normalize(source, np.asarray([0.0, 0.0, 1.0]))
    target = normalize(target, np.asarray([0.0, 0.0, 1.0]))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-8:
        return np.eye(3)
    if cosine < -1.0 + 1.0e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = normalize(np.cross(source, basis), np.asarray([1.0, 0.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.asarray([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


def frame_from_axis_direction(axis: np.ndarray, direction: np.ndarray) -> np.ndarray:
    z = normalize(direction, np.asarray([0.0, 0.0, 1.0]))
    x = axis - float(np.dot(axis, z)) * z
    if np.linalg.norm(x) < 1.0e-6:
        fallback = np.asarray([0.0, 1.0, 0.0])
        if abs(float(np.dot(fallback, z))) > 0.9:
            fallback = np.asarray([1.0, 0.0, 0.0])
        x = fallback - float(np.dot(fallback, z)) * z
    x = normalize(x, np.asarray([1.0, 0.0, 0.0]))
    y = normalize(np.cross(z, x), np.asarray([0.0, 1.0, 0.0]))
    x = normalize(np.cross(y, z), x)
    return np.column_stack([x, y, z])


def source_slot(hand: dict, bundle: dict) -> dict:
    ids = bundle["source_part_ids"]
    root_id = bundle["root_part_id"]
    root = hand["parts"][root_id]
    anchor = np.asarray(root["world_pos"], dtype=float)
    offsets = [np.asarray(hand["parts"][part_id]["world_pos"], dtype=float) - anchor for part_id in ids]
    direction = max(offsets, key=lambda value: float(np.linalg.norm(value)))
    if np.linalg.norm(direction) < 1.0e-6:
        direction = np.asarray(root["relative_pos"], dtype=float)
    axis = np.asarray(root["joint_axis"], dtype=float)
    return {"anchor": anchor, "frame": frame_from_axis_direction(axis, direction)}


def transformed_slot(slot: dict, linear: np.ndarray, rotation: np.ndarray) -> dict:
    source_frame = slot["frame"]
    return {
        "anchor": linear @ slot["anchor"],
        "frame": frame_from_axis_direction(linear @ source_frame[:, 0], linear @ source_frame[:, 2]),
        "palm_rotation": rotation,
    }


def vary_attachment_layout(
    rng: np.random.Generator,
    target_slots: dict[str, dict],
    palm_rotation: np.ndarray,
    expansion: float = 1.0,
) -> list[dict]:
    """Move graph attachment frames a small distance along the palm edge.

    The compiler uses the saved reference anchor to deform the original palm
    boundary toward the new graph anchor.  Motion is tangent to the outgoing
    finger direction, so attachment search does not push a root into the palm.
    """
    anchors = np.asarray([slot["anchor"] for slot in target_slots.values()], dtype=float)
    layout_scale = max(float(np.linalg.norm(np.ptp(anchors, axis=0))), 0.25)
    palm_normal = palm_rotation @ np.asarray([0.0, 1.0, 0.0])
    edits = []
    for role, slot in target_slots.items():
        slot["reference_anchor"] = slot["anchor"].copy()
        slot["reference_frame"] = slot["frame"].copy()
        outward = normalize(slot["frame"][:, 2], np.asarray([0.0, 0.0, 1.0]))
        tangent = normalize(np.cross(palm_normal, outward), palm_rotation[:, 0])
        limit = (0.035 if role == "thumb" else 0.045) * layout_scale
        offset = expansion * float(rng.uniform(-limit, limit))
        slot["anchor"] = slot["anchor"] + offset * tangent
        edits.append({
            "role": role,
            "edge_tangent_offset": offset,
            "edge_tangent": tangent.tolist(),
        })
    return edits


def root_footprint_width(bundle: dict, candidates: dict[str, dict], radius_scale: float = 1.0) -> float:
    """Conservative palm-edge footprint of one complete source mechanism."""
    candidate = candidates[f"{bundle['source_hand_id']}:part:{bundle['root_part_id']}"]
    mesh = candidate.get("source_mesh")
    if mesh is None:
        return 0.12
    size = np.asarray(mesh.get("size", [0.12, 0.12, 0.12]), dtype=float)
    # Canonical roots vary in orientation.  The second-largest dimension is a
    # stable footprint estimate without mistaking chain length for motor width.
    return max(float(np.sort(size)[-2]) * float(radius_scale), 0.075)


def select_source_roles(
    rng: np.random.Generator,
    source_roles: list[str],
    target_count: int,
    variant: int,
) -> list[str]:
    """Remove complete bundles while preserving useful thumb/normal diversity."""
    if target_count > len(source_roles):
        raise ValueError(f"cannot grow {source_roles} to {target_count} fingers")
    if target_count == len(source_roles):
        return list(source_roles)
    selected: list[str] = []
    if "thumb" in source_roles and target_count >= 3:
        selected.append("thumb")
    normal = [role for role in source_roles if role != "thumb"]
    # Cycle the omitted digits instead of always collapsing to thumb-index-middle.
    shift = variant % max(len(normal), 1)
    normal = normal[shift:] + normal[:shift]
    rng.shuffle(normal)
    selected.extend(normal[: target_count - len(selected)])
    return [role for role in DIGITS if role in selected]


def apply_global_palm_layout(
    rng: np.random.Generator,
    target_slots: dict[str, dict],
    selected: dict[str, dict],
    candidates: dict[str, dict],
    palm_rotation: np.ndarray,
    mode: str,
    palm_bounds_local_xz: np.ndarray | None = None,
    palm_expansion: float | None = None,
) -> dict:
    """House of Dextra style discrete-slot motor placement.

    Non-anthropomorphic hands place complete finger mechanisms on a variable
    circular radius.  Symmetric layouts use evenly spaced slots; asymmetric
    layouts rejection-sample 36 slots with a four-slot circular separation.
    The palm is compiled later from these exact graph motor footprints.
    """
    if mode not in PALM_LAYOUT_MODES:
        raise ValueError(f"unknown palm layout mode {mode!r}")
    if palm_expansion is not None and not 0.0 <= palm_expansion <= 1.0:
        raise ValueError(
            f"palm_expansion must lie in [0, 1], got {palm_expansion}"
        )
    for slot in target_slots.values():
        slot["reference_anchor"] = slot["anchor"].copy()
        slot["reference_frame"] = slot["frame"].copy()

    if mode == "source_fixed":
        return {
            "mode": mode,
            "edits": [],
            "slot_count": None,
            "mount_slots": None,
            "minimum_arc_clearance": None,
            "source_order_locked": True,
            "attachment_poses_locked": True,
            "palm_expansion": 0.0,
            "thickness_scale_locked": 1.0,
        }

    if mode == "anthropomorphic":
        resolved_expansion = 1.0 if palm_expansion is None else palm_expansion
        roles = list(target_slots)
        before = np.asarray([
            [np.dot(target_slots[role]["anchor"], palm_rotation[:, 0]),
             np.dot(target_slots[role]["anchor"], palm_rotation[:, 2])]
            for role in roles
        ])
        source_order = canonical_cyclic_roles(roles, before)
        edits = vary_attachment_layout(
            rng, target_slots, palm_rotation, resolved_expansion
        )
        after = np.asarray([
            [np.dot(target_slots[role]["anchor"], palm_rotation[:, 0]),
             np.dot(target_slots[role]["anchor"], palm_rotation[:, 2])]
            for role in roles
        ])
        target_order = canonical_cyclic_roles(roles, after)
        if target_order != source_order:
            raise ValueError("anthropomorphic attachment edit changed source finger order")
        return {
            "mode": mode,
            "edits": edits,
            "source_cyclic_roles": source_order,
            "target_cyclic_roles": target_order,
            "cyclic_order_preserved": True,
            "palm_expansion": resolved_expansion,
            "slot_count": None,
            "mount_slots": None,
            "minimum_arc_clearance": None,
            "thickness_scale_locked": 1.0,
        }

    ex = palm_rotation[:, 0]
    ey = palm_rotation[:, 1]
    ez = palm_rotation[:, 2]
    roles = list(target_slots)
    anchors = np.asarray([target_slots[role]["anchor"] for role in roles], dtype=float)
    local = np.column_stack([anchors @ ex, anchors @ ez])
    width_span = max(float(np.ptp(local[:, 0])), 0.34)
    if palm_bounds_local_xz is not None:
        palm_bounds_local_xz = np.asarray(palm_bounds_local_xz, dtype=float)
        if palm_bounds_local_xz.shape != (2, 2):
            raise ValueError("palm_bounds_local_xz must be [[xmin,zmin],[xmax,zmax]]")
        palm_extent = np.maximum(palm_bounds_local_xz[1] - palm_bounds_local_xz[0], 1.0e-4)
        palm_center = 0.5 * (palm_bounds_local_xz[0] + palm_bounds_local_xz[1])
    else:
        palm_extent = np.maximum(np.ptp(local, axis=0), [0.34, 0.34])
        palm_center = np.mean(local, axis=0)
    center_x = float(palm_center[0])
    center_z = float(palm_center[1])
    center = np.asarray([center_x, center_z])
    palm_radius = max(0.48 * float(np.max(palm_extent)), 0.52 * width_span, 0.22)
    count = len(roles)
    slot_count = 36
    min_sep_slots = 4
    mount_vector = -center
    if np.linalg.norm(mount_vector) < 0.08 * max(float(np.max(palm_extent)), 1.0e-4):
        mount_vector = -np.mean(local - center, axis=0)
    if np.linalg.norm(mount_vector) < 1.0e-8:
        mount_vector = np.asarray([0.0, -1.0])
    mount_direction = math.atan2(float(mount_vector[1]), float(mount_vector[0]))
    mount_slot = int(round((mount_direction % (2.0 * math.pi)) * slot_count / (2.0 * math.pi))) % slot_count
    mount_exclusion_slots = 3

    def outside_mount_exclusion(slot: int) -> bool:
        distance = min(
            (slot - mount_slot) % slot_count,
            (mount_slot - slot) % slot_count,
        )
        return distance > mount_exclusion_slots

    if mode == "symmetric":
        starts = []
        for start in range(slot_count):
            proposal = [
                int(round(start + i * slot_count / count)) % slot_count
                for i in range(count)
            ]
            if len(set(proposal)) == count and all(outside_mount_exclusion(slot) for slot in proposal):
                starts.append((start, proposal))
        if not starts:
            raise ValueError("no symmetric House layout can preserve the root mount exclusion sector")
        _, mount_slots = starts[int(rng.integers(len(starts)))]
    else:
        mount_slots = []
        for _ in range(5000):
            candidate = int(rng.integers(slot_count))
            if outside_mount_exclusion(candidate) and all(
                min((candidate - other) % slot_count, (other - candidate) % slot_count)
                >= min_sep_slots
                for other in mount_slots
            ):
                mount_slots.append(candidate)
                if len(mount_slots) == count:
                    break
        if len(mount_slots) != count:
            raise ValueError("could not sample House-style motor slots with minimum separation")

    source_angles = np.mod(np.arctan2(local[:, 1] - center_z, local[:, 0] - center_x), 2.0 * np.pi)
    source_order_indices = np.argsort(source_angles)
    source_order_roles = [roles[int(index)] for index in source_order_indices]
    source_order_angles = source_angles[source_order_indices]
    sorted_mount_slots = np.asarray(sorted(mount_slots), dtype=int)
    best_slots = None
    best_cost = float("inf")
    for shift in range(count):
        proposal = np.roll(sorted_mount_slots, -shift)
        proposal_angles = 2.0 * np.pi * proposal / slot_count
        delta = np.angle(np.exp(1j * (proposal_angles - source_order_angles)))
        cost = float(np.sum(delta * delta))
        if cost < best_cost:
            best_cost = cost
            best_slots = proposal
    roles = source_order_roles
    local = local[source_order_indices]
    mount_slots = best_slots.tolist()
    angles = np.asarray([2.0 * math.pi * slot / slot_count for slot in mount_slots])

    widths = np.asarray([
        root_footprint_width(selected[role], candidates) for role in roles
    ], dtype=float)
    # Source compounds occasionally include a long bracket in their AABB.  It
    # is not the motor's tangential mounting footprint, so cap the conservative
    # estimate relative to the source palm width.
    widths = np.minimum(widths, 0.32 * width_span)
    widths = np.minimum(widths, 0.28 * palm_radius)
    clearance = max(0.035 * palm_radius, 0.015)
    circular_order = np.argsort(angles)
    pair_clearances = []
    required_radius = palm_radius
    for rank, index_a in enumerate(circular_order):
        index_b = circular_order[(rank + 1) % count]
        gap = float((angles[index_b] - angles[index_a]) % (2.0 * math.pi))
        required = 0.5 * float(widths[index_a] + widths[index_b]) + clearance
        required_radius = max(required_radius, required / max(0.76 * gap, 1.0e-5))
    palm_radius = required_radius
    radii = rng.uniform(0.76 * palm_radius, 0.90 * palm_radius, size=count)
    if mode == "symmetric":
        radii[:] = float(rng.uniform(0.82, 0.88)) * palm_radius

    # House positions define the design direction, but manufacturing-oriented
    # samples should not jump all the way from a source palm to that target in
    # one edit.  Interpolate the complete attachment layout so palm shape and
    # finger roots move together.  If the source-biased proposal crowds the
    # selected motor housings, increase the House contribution only as much as
    # needed to restore pairwise clearance.
    house_centers = np.column_stack([
        center_x + radii * np.cos(angles),
        center_z + radii * np.sin(angles),
    ])
    source_centers = local.copy()
    requested_blend = (
        float(rng.uniform(0.52, 0.66))
        if palm_expansion is None
        else palm_expansion
    )

    def blended_clearance(alpha: float) -> tuple[np.ndarray, float]:
        centers = source_centers + alpha * (house_centers - source_centers)
        values = []
        for index_a in range(count):
            for index_b in range(index_a + 1, count):
                required = 0.5 * float(widths[index_a] + widths[index_b]) + clearance
                values.append(float(np.linalg.norm(centers[index_a] - centers[index_b])) - required)
        return centers, min(values, default=float("inf"))

    layout_blend = requested_blend
    blended_centers, actual_clearance = blended_clearance(layout_blend)
    if actual_clearance < -1.0e-8 and palm_expansion is None:
        for candidate_blend in np.linspace(requested_blend, 0.86, 35):
            candidate_centers, candidate_clearance = blended_clearance(float(candidate_blend))
            if candidate_clearance >= -1.0e-8:
                layout_blend = float(candidate_blend)
                blended_centers = candidate_centers
                actual_clearance = candidate_clearance
                break
        else:
            raise ValueError("source/House blended layout cannot satisfy motor clearance")
    elif actual_clearance < -1.0e-8:
        raise ValueError(
            f"palm_expansion={requested_blend:.6g} violates motor clearance "
            f"by {-actual_clearance:.6g} canonical length units"
        )

    edits = []
    for index, (role, angle, radius) in enumerate(zip(roles, angles, radii)):
        q = blended_centers[index]
        target = q[0] * ex + q[1] * ez
        # Preserve the source joint's normal coordinate exactly: palm layout
        # changes shape in the local XZ plane, never palm thickness.
        target += float(np.dot(target_slots[role]["reference_anchor"], ey)) * ey
        actual_direction = normalize(q - center, [math.cos(float(angle)), math.sin(float(angle))])
        outgoing = normalize(actual_direction[0] * ex + actual_direction[1] * ez, ez)
        source_axis = target_slots[role]["reference_frame"][:, 0]
        old = target_slots[role]["anchor"].copy()
        target_slots[role]["anchor"] = target
        target_slots[role]["frame"] = frame_from_axis_direction(source_axis, outgoing)
        edits.append({
            "role": role,
            "angle_degrees": math.degrees(float(angle)),
            "reference_anchor": old.tolist(),
            "target_anchor": target.tolist(),
            "house_target_anchor_local_xz": house_centers[index].tolist(),
            "source_house_blend": layout_blend,
            "root_footprint_width": float(widths[index]),
        })

    for rank, index_a in enumerate(circular_order):
        index_b = circular_order[(rank + 1) % count]
        gap = float((angles[index_b] - angles[index_a]) % (2.0 * math.pi))
        arc = min(float(radii[index_a]), float(radii[index_b])) * gap
        required = 0.5 * float(widths[index_a] + widths[index_b]) + clearance
        pair_clearances.append(arc - required)

    minimum = min(pair_clearances, default=float("inf"))
    if minimum < -1.0e-8:
        raise ValueError(f"layout {mode} violates motor footprint clearance: {minimum}")
    return {
        "mode": mode,
        "center_local_xz": [center_x, center_z],
        "palm_radius": palm_radius,
        "finger_radii": radii.tolist(),
        "slot_count": slot_count,
        "min_sep_slots": min_sep_slots,
        "mount_slots": mount_slots,
        "root_mount_slot": mount_slot,
        "root_mount_exclusion_slots": mount_exclusion_slots,
        "all_slots_outside_root_mount_exclusion": all(
            outside_mount_exclusion(slot) for slot in mount_slots
        ),
        "source_cyclic_roles": source_order_roles,
        "target_cyclic_roles": list(source_order_roles),
        "cyclic_order_preserved": True,
        "minimum_arc_clearance": None if not pair_clearances else minimum,
        "minimum_actual_motor_clearance": actual_clearance,
        "requested_source_house_blend": requested_blend,
        "source_house_blend": layout_blend,
        "palm_expansion": layout_blend,
        "maximum_attachment_displacement": float(np.max(
            np.linalg.norm(blended_centers - source_centers, axis=1)
        )),
        "maximum_attachment_displacement_fraction": float(
            np.max(np.linalg.norm(blended_centers - source_centers, axis=1))
            / max(float(np.max(palm_extent)), 1.0e-8)
        ),
        "edits": edits,
        "thickness_scale_locked": 1.0,
    }


def candidate_map(library: dict) -> dict[str, dict]:
    return {candidate["candidate_id"]: candidate for candidate in library["candidates"]}


def extra_slot(base_slots: dict[str, dict], palm_linear: np.ndarray) -> dict:
    normal = [(role, slot) for role, slot in base_slots.items() if role != "thumb"]
    normal.sort(key=lambda item: float(item[1]["anchor"][0]))
    outer_role, outer = normal[0]
    if len(normal) > 1:
        pitch = outer["anchor"] - normal[1][1]["anchor"]
    else:
        pitch = np.asarray([-0.20, 0.0, 0.0])
    pitch = palm_linear @ pitch
    target = transformed_slot(outer, palm_linear, np.eye(3))
    target["reference_anchor"] = target["anchor"].copy()
    target["reference_frame"] = target["frame"].copy()
    target["anchor"] = target["anchor"] + FIFTH_SLOT_PITCH * pitch
    target["frame"] = rot_z(math.radians(-7.0)) @ target["frame"]
    target["derived_from"] = outer_role
    return target


def instantiate_finger(
    output: list[dict], slot_id: int, output_role: str, target_slot: dict,
    bundle: dict, source_hand: dict, candidates: dict[str, dict],
    length_scales_by_source_part: dict[int, float], body_radius_scale: float,
    distal_radius_scale: float,
    lock_proximal_hardware: bool = False,
) -> dict:
    ids = bundle["source_part_ids"]
    block_set = set(ids)
    donor_slot = source_slot(source_hand, bundle)
    # Source part meshes are already expressed in the shared canonical palm
    # frame.  Align only the directed chain axes with the minimal rotation.
    # Using the complete joint-axis frame would introduce arbitrary 180-degree
    # roll flips when two equivalent source joint axes use opposite signs.
    rotation = rotation_from_to(donor_slot["frame"][:, 2], target_slot["frame"][:, 2])
    realized_frame = rotation @ donor_slot["frame"]
    reference_frame = target_slot.get("reference_frame", target_slot["frame"])
    reference_rotation = rotation_from_to(
        donor_slot["frame"][:, 2], reference_frame[:, 2]
    )
    realized_reference_frame = reference_rotation @ donor_slot["frame"]
    editable_ids = list(ids)
    if lock_proximal_hardware and editable_ids:
        editable_ids = editable_ids[1:]
    requested_ids = set(length_scales_by_source_part)
    if requested_ids != set(editable_ids):
        raise ValueError(
            f"{output_role}: length parameters do not match source rigid parts; "
            f"missing={sorted(set(editable_ids) - requested_ids)} "
            f"extra={sorted(requested_ids - set(editable_ids))}"
        )
    distal_id = editable_ids[-1] if editable_ids else None
    target_axis = realized_frame[:, 2]
    reference_axis = realized_reference_frame[:, 2]
    linear_by_source: dict[int, np.ndarray] = {}
    reference_linear_by_source: dict[int, np.ndarray] = {}
    for source_id in ids:
        length_scale = float(length_scales_by_source_part.get(source_id, 1.0))
        radius_scale = (
            1.0 if source_id not in editable_ids
            else float(distal_radius_scale if source_id == distal_id else body_radius_scale)
        )
        deform = radius_scale * np.eye(3) + (
            length_scale - radius_scale
        ) * np.outer(target_axis, target_axis)
        reference_deform = radius_scale * np.eye(3) + (
            length_scale - radius_scale
        ) * np.outer(reference_axis, reference_axis)
        linear_by_source[source_id] = deform @ rotation
        reference_linear_by_source[source_id] = reference_deform @ reference_rotation
    id_map: dict[int, int] = {}
    root_new_id = None
    for rank, source_id in enumerate(ids):
        source_node = source_hand["parts"][source_id]
        parent_source = source_node["parent"]
        internal = parent_source in block_set
        new_id = len(output)
        id_map[source_id] = new_id
        node_length_scale = float(length_scales_by_source_part.get(source_id, 1.0))
        node_radius_scale = (
            1.0 if source_id not in editable_ids
            else float(distal_radius_scale if source_id == distal_id else body_radius_scale)
        )
        node_linear = linear_by_source[source_id]
        node_reference_linear = reference_linear_by_source[source_id]
        if not internal:
            relative = target_slot["anchor"]
            parent = 0
            root_new_id = new_id
        else:
            relative = linear_by_source[int(parent_source)] @ np.asarray(
                source_node["relative_pos"], dtype=float
            )
            parent = id_map[int(parent_source)]
        candidate_id = f"{bundle['source_hand_id']}:part:{source_id}"
        candidate = candidates[candidate_id]
        axis = rotation @ np.asarray(source_node["joint_axis"], dtype=float)
        node_record = {
            "id": new_id,
            "parent": parent,
            "role": output_role,
            "finger_slot": slot_id,
            "joint_type": source_node["joint_type"],
            "joint_axis": axis.tolist(),
            "joint_range": source_node["joint_range"],
            "joint_name": f"slot_{slot_id}_{source_node['joint_name']}",
            "relative_pos": relative.tolist(),
            "source_relative_pos": source_node["relative_pos"],
            "source_hand_id": bundle["source_hand_id"],
            "source_part_id": source_id,
            "source_mesh": source_node.get("mesh"),
            "source_member_links": source_node.get("member_links", []),
            "candidate_id": candidate_id,
            "mechanism_bundle_id": bundle["bundle_id"],
            "motor_binding": candidate["motor_binding"],
            "compatible_candidate_ids": candidate["compatible_candidate_ids"],
            "mesh_linear": node_linear.tolist(),
            "reference_mesh_linear": node_reference_linear.tolist(),
            "length_scale": node_length_scale,
            "radius_scale": node_radius_scale,
            "length_parameter_source_part_id": (
                source_id if source_id in editable_ids else None
            ),
            "source_rank": rank,
            "protected_proximal_hardware": bool(lock_proximal_hardware and rank == 0),
            "morphology_part": "finger_root_rigid_body" if rank == 0 else "finger_link_rigid_body",
            "joint_to_joint_segment": source_id in editable_ids,
            "segment_mesh_merged": bool(
                source_id in editable_ids and source_node.get("mesh") is not None
            ),
        }
        output.append(node_record)
    return {
        "slot_id": slot_id,
        "role": output_role,
        "bundle_id": bundle["bundle_id"],
        "source_role": bundle["source_role"],
        "source_hand_id": bundle["source_hand_id"],
        "dof_count": bundle["dof_count"],
        "root_node_id": root_new_id,
        "interface_parent_palm_node_id": 0,
        "interface_child_finger_node_id": root_new_id,
        "attachment_translation": target_slot["anchor"].tolist(),
        "attachment_rotation": realized_frame.tolist(),
        "reference_attachment_translation": target_slot.get(
            "reference_anchor", target_slot["anchor"]
        ).tolist(),
        "reference_attachment_rotation": realized_reference_frame.tolist(),
        "connector_transform_applied": True,
        "proximal_hardware_locked": lock_proximal_hardware,
        "editable_source_part_ids": editable_ids,
    }


def instantiate_protected_platform(
    output: list[dict],
    source_hand: dict,
    source_bundles: list[dict],
) -> list[int]:
    """Copy every non-digit base/transmission node without any deformation."""
    digit_ids = {
        int(source_id)
        for bundle in source_bundles
        for source_id in bundle["source_part_ids"]
    }
    id_map = {0: 0}
    created = []
    for source_node in source_hand["parts"][1:]:
        source_id = int(source_node["id"])
        if source_id in digit_ids:
            continue
        parent_source = int(source_node["parent"])
        if parent_source not in id_map:
            raise ValueError(
                f"protected platform node {source_id} depends on non-platform parent {parent_source}"
            )
        new_id = len(output)
        id_map[source_id] = new_id
        created.append(new_id)
        output.append({
            "id": new_id,
            "parent": id_map[parent_source],
            "role": source_node["role"],
            "joint_type": source_node["joint_type"],
            "joint_axis": source_node["joint_axis"],
            "joint_range": source_node["joint_range"],
            "joint_name": source_node["joint_name"],
            "relative_pos": source_node["relative_pos"],
            "source_hand_id": source_hand["hand_id"],
            "source_part_id": source_id,
            "source_mesh": source_node.get("mesh"),
            "candidate_id": f"{source_hand['hand_id']}:protected_platform:{source_id}",
            "mechanism_bundle_id": f"{source_hand['hand_id']}:protected_platform",
            "motor_binding": None,
            "compatible_candidate_ids": [
                f"{source_hand['hand_id']}:protected_platform:{source_id}"
            ],
            "mesh_linear": np.eye(3).tolist(),
            "length_scale": 1.0,
            "radius_scale": 1.0,
            "protected_platform": True,
            "geometry_locked": True,
            "morphology_part": "fixed_base_or_transmission",
        })
    return created


def finalize_graph(parts: list[dict]) -> None:
    world = [np.zeros(3)]
    children: dict[int, list[int]] = defaultdict(list)
    for node in parts[1:]:
        parent = int(node["parent"])
        world.append(world[parent] + np.asarray(node["relative_pos"], dtype=float))
        children[parent].append(int(node["id"]))
    for node, position in zip(parts, world):
        node["world_pos"] = position.tolist()
        node["child_count"] = len(children[int(node["id"])])


def main() -> int:
    generation_seed = int(os.environ.get("HAND_GENERATION_SEED", "20260718"))
    palm_expansion_text = os.environ.get("HAND_PALM_EXPANSION")
    palm_expansion = (
        None if palm_expansion_text is None else float(palm_expansion_text)
    )
    if palm_expansion is not None and not 0.0 <= palm_expansion <= 1.0:
        raise ValueError(
            f"HAND_PALM_EXPANSION must lie in [0, 1], got {palm_expansion}"
        )
    graph_spec_path = os.environ.get("HAND_GRAPH_SPEC_PATH")
    if graph_spec_path is None:
        design_specs: list[dict] = [{} for _ in range(100)]
    else:
        graph_payload = json.loads(Path(graph_spec_path).read_text(encoding="utf-8"))
        design_specs = graph_payload.get("hands", [graph_payload])
        if not isinstance(design_specs, list) or not design_specs:
            raise ValueError("graph specification must contain one or more hands")
        if not all(isinstance(spec, dict) for spec in design_specs):
            raise ValueError("every graph hand specification must be an object")
    rng = np.random.default_rng(generation_seed)
    source_payload = json.loads(SOURCE_GRAPHS.read_text(encoding="utf-8"))
    sources = {hand["hand_id"]: hand for hand in source_payload["hands"]}
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    bundles = {bundle["bundle_id"]: bundle for bundle in library["bundles"]}
    candidates = candidate_map(library)
    direct_registry = json.loads(DIRECT_REGISTRY.read_text(encoding="utf-8"))
    direct_joint_records = {}
    direct_entries = {}
    for hand_id in sources:
        records, entry = load_direct_joint_records(hand_id, direct_registry)
        direct_joint_records[hand_id] = records
        direct_entries[hand_id] = entry
    eligible_palms = []
    source_coverage = {}
    for hand_id in library["palm_sources"]:
        present = [role for role in DIGITS if f"{hand_id}:{role}" in bundles]
        source_roles = [role for role in DIGITS if any(p["role"] == role for p in sources[hand_id]["parts"])]
        complete = len(present) == len(source_roles) and len(present) >= 4
        source_coverage[hand_id] = {
            "source_roles": source_roles,
            "accepted_bundle_roles": present,
            "eligible": complete,
        }
        if complete:
            eligible_palms.append(hand_id)
    # The direct-motor registry may contain other articulated assets (for
    # example experimental feet) that intentionally have no hand-morphology
    # scaffold. Only sources declared by the hand grammar require coverage.
    non_morphology_registry_sources = sorted(
        set(direct_registry["hands"]) - set(source_coverage)
    )
    ineligible_sources = sorted(
        hand_id for hand_id, record in source_coverage.items()
        if not record["eligible"]
    )
    if ineligible_sources:
        raise ValueError(
            "source coverage is incomplete: "
            f"ineligible={ineligible_sources}"
        )

    hands = []
    graph_modes = ("geometry_only",)
    for index, design_spec in enumerate(design_specs):
        mode = graph_modes[0]
        palm_spec = design_spec.get("palm", {})
        finger_spec = design_spec.get("fingers", {})
        if not isinstance(palm_spec, dict) or not isinstance(finger_spec, dict):
            raise ValueError("palm and fingers graph parameters must be objects")
        requested_layout_mode = palm_spec.get(
            "layout_mode",
            ("anthropomorphic", "symmetric", "asymmetric")[index % 3],
        )
        seed_pool = eligible_palms
        seed_id = design_spec.get(
            "source_hand", seed_pool[index % len(seed_pool)]
        )
        if seed_id not in seed_pool:
            raise ValueError(
                f"unknown or ineligible source_hand {seed_id!r}; "
                f"choose one of {seed_pool}"
            )
        seed = sources[seed_id]
        # Large radial rearrangements are incompatible with a fixed tendon or
        # underactuated transmission platform. Keep those hands on the bounded
        # anthropomorphic palm-edit path; other sources retain all three modes.
        if (
            seed_id in BOUNDED_PALM_LAYOUT_SOURCES
            and requested_layout_mode not in {"source_fixed", "anthropomorphic"}
        ):
            if graph_spec_path is not None:
                raise ValueError(
                    f"{seed_id} only permits source_fixed or anthropomorphic "
                    "palm layouts because its transmission platform is bounded"
                )
            layout_mode = "anthropomorphic"
        else:
            layout_mode = requested_layout_mode
        source_roles = [role for role in DIGITS if f"{seed_id}:{role}" in bundles]
        base_roles = list(source_roles)
        selected = {role: deepcopy(bundles[f"{seed_id}:{role}"]) for role in base_roles}
        actions = [{
            "operation": "LOCK_SOURCE_TOPOLOGY_ORDER_AND_ATTACHMENTS",
            "allowed_geometry_edits": [
                "per_finger_per_source_rigid_part_length",
                "shared_normal_body_or_distal_radius",
                "shared_thumb_body_or_distal_radius",
            ],
        }]

        if seed_id in PROTECTED_TRANSMISSION_SOURCES:
            # The root frame and the separate under-palm transmission platform
            # are hardware interfaces. Palm shape may still be edited later by
            # the region-aware compiler, but no global affine may move the base.
            sx = sy = sz = 1.0
            yaw = 0.0
        else:
            sx = float(palm_spec.get("scale_x", rng.uniform(0.93, 1.10)))
            sy = 1.0
            sz = float(palm_spec.get("scale_z", rng.uniform(0.92, 1.13)))
            yaw = float(palm_spec.get("yaw", rng.uniform(-0.16, 0.16)))
        if not 0.70 <= sx <= 1.45 or not 0.70 <= sz <= 1.45:
            raise ValueError("palm scale_x and scale_z must lie in [0.70, 1.45]")
        if not -0.35 <= yaw <= 0.35:
            raise ValueError("palm yaw must lie in [-0.35, 0.35] radians")
        rotation = rot_z(yaw)
        palm_linear = rotation @ np.diag([sx, sy, sz])
        actions.append({
            "operation": "MODIFY_PALM_SHAPE_WITH_FIXED_FINGER_ORDER",
            "scale": [sx, sy, sz],
            "yaw": yaw,
            "protected_transmission_base": seed_id in PROTECTED_TRANSMISSION_SOURCES,
        })

        root = seed["parts"][0]
        parts = [{
            "id": 0,
            "parent": None,
            "role": "palm",
            "joint_type": "fixed",
            "joint_axis": [0.0, 0.0, 0.0],
            "joint_range": [0.0, 0.0],
            "joint_name": "palm_root",
            "relative_pos": [0.0, 0.0, 0.0],
            "source_hand_id": seed_id,
            "source_part_id": 0,
            "source_mesh": root["mesh"],
            "candidate_id": f"{seed_id}:palm:0",
            "mechanism_bundle_id": f"{seed_id}:palm",
            "motor_binding": None,
            "compatible_candidate_ids": [f"{seed_id}:palm:0"],
            "mesh_linear": palm_linear.tolist(),
            "protected_transmission_root": seed_id in PROTECTED_TRANSMISSION_SOURCES,
            "global_transform_locked": seed_id in PROTECTED_TRANSMISSION_SOURCES,
            "geometry_locked": False,
            "morphology_part": "palm_parent_rigid_body",
        }]

        protected_platform_node_ids = instantiate_protected_platform(
            parts,
            seed,
            [bundles[f"{seed_id}:{role}"] for role in source_roles],
        )
        digit_source_part_ids = {
            int(source_part_id)
            for role in source_roles
            for source_part_id in bundles[f"{seed_id}:{role}"]["source_part_ids"]
        }
        digit_joint_names = {
            seed["parts"][source_part_id]["joint_name"]
            for source_part_id in digit_source_part_ids
            if seed["parts"][source_part_id]["joint_type"] != "fixed"
        }
        source_part_by_joint = {
            node["joint_name"]: int(node["id"])
            for node in seed["parts"][1:]
            if node["joint_type"] != "fixed"
        }
        generated_platform_by_source_part = {
            int(node["source_part_id"]): int(node["id"])
            for node in parts
            if node.get("protected_platform")
        }
        base_palm_joints = []
        for record in direct_joint_records[seed_id]:
            if record["joint_name"] in digit_joint_names:
                continue
            source_part_id = source_part_by_joint.get(record["joint_name"])
            generated_part_id = (
                None
                if source_part_id is None
                else generated_platform_by_source_part.get(source_part_id)
            )
            base_palm_joints.append(
                {
                    **record,
                    "source_part_id": source_part_id,
                    "mesh_part_id": generated_part_id,
                    "kinematic_only": generated_part_id is None,
                    "mesh_binding": (
                        "kinematic_only_frame"
                        if generated_part_id is None
                        else "existing_rigid_part_edge"
                    ),
                }
            )
        represented_platform_joint_names = {
            seed["parts"][source_part_id]["joint_name"]
            for source_part_id in generated_platform_by_source_part
            if seed["parts"][source_part_id]["joint_type"] != "fixed"
        }
        copied_joint_names = {
            record["joint_name"] for record in base_palm_joints
        }
        missing_platform_joints = (
            represented_platform_joint_names - copied_joint_names
        )
        if missing_platform_joints:
            raise ValueError(
                f"{seed_id}: generated base/palm graph lost represented joints "
                f"{sorted(missing_platform_joints)}"
            )

        base_slots = {role: source_slot(seed, bundles[f"{seed_id}:{role}"]) for role in base_roles}
        target_slots = {role: transformed_slot(slot, palm_linear, rotation) for role, slot in base_slots.items()}
        source_palm_bounds = np.asarray(root["mesh"]["bounds"], dtype=float)[:, [0, 2]]
        palm_bounds_local_xz = source_palm_bounds * np.asarray([sx, sz])
        design_palm_expansion = palm_spec.get(
            "expansion", palm_expansion
        )
        if design_palm_expansion is not None:
            design_palm_expansion = float(design_palm_expansion)
        palm_layout = apply_global_palm_layout(
            rng, target_slots, selected, candidates, rotation, layout_mode,
            palm_bounds_local_xz=palm_bounds_local_xz,
            palm_expansion=design_palm_expansion,
        )
        actions.append({
            "operation": "APPLY_GLOBAL_PALM_LAYOUT",
            **palm_layout,
        })
        slots = []
        realized_finger_parameters = {}
        width_spec = finger_spec.get("width_scales", {})
        if not isinstance(width_spec, dict):
            raise ValueError("fingers.width_scales must be an object")
        legacy_default_radius = finger_spec.get("default_radius_scale")
        normal_body_radius = float(width_spec.get(
            "normal_body",
            legacy_default_radius if legacy_default_radius is not None else rng.uniform(0.88, 1.14),
        ))
        normal_distal_radius = float(width_spec.get(
            "normal_distal",
            legacy_default_radius if legacy_default_radius is not None else rng.uniform(0.88, 1.14),
        ))
        thumb_body_radius = float(width_spec.get(
            "thumb_body",
            legacy_default_radius if legacy_default_radius is not None else rng.uniform(0.88, 1.14),
        ))
        thumb_distal_radius = float(width_spec.get(
            "thumb_distal",
            legacy_default_radius if legacy_default_radius is not None else rng.uniform(0.88, 1.14),
        ))
        for name, value in {
            "normal_body": normal_body_radius,
            "normal_distal": normal_distal_radius,
            "thumb_body": thumb_body_radius,
            "thumb_distal": thumb_distal_radius,
        }.items():
            if not 0.60 <= value <= 1.45:
                raise ValueError(f"{name} radius scale must lie in [0.60, 1.45]")
        for slot_id, role in enumerate(base_roles):
            bundle = selected[role]
            role_spec = finger_spec.get(role, {})
            if not isinstance(role_spec, dict):
                raise ValueError(f"finger specification for {role} must be an object")
            legacy_length = role_spec.get(
                "length_scale", finger_spec.get("default_length_scale")
            )
            source_part_ids = list(bundle["source_part_ids"])
            editable_part_ids = list(source_part_ids)
            if seed_id in PROTECTED_TRANSMISSION_SOURCES and editable_part_ids:
                editable_part_ids = editable_part_ids[1:]
            requested_lengths = role_spec.get("length_scales_by_source_part")
            if requested_lengths is None:
                if legacy_length is None:
                    length_scales_by_source_part = {
                        source_id: float(rng.uniform(0.87, 1.17))
                        for source_id in editable_part_ids
                    }
                else:
                    length_scales_by_source_part = {
                        source_id: float(legacy_length)
                        for source_id in editable_part_ids
                    }
            elif isinstance(requested_lengths, dict):
                length_scales_by_source_part = {
                    int(source_id): float(value)
                    for source_id, value in requested_lengths.items()
                }
            else:
                values = [float(value) for value in requested_lengths]
                if len(values) != len(editable_part_ids):
                    raise ValueError(
                        f"{role} needs {len(editable_part_ids)} source-part length scales"
                    )
                length_scales_by_source_part = dict(zip(
                    editable_part_ids, values, strict=True
                ))
            if set(length_scales_by_source_part) != set(editable_part_ids) or any(
                not 0.55 <= value <= 1.60
                for value in length_scales_by_source_part.values()
            ):
                raise ValueError(
                    f"{role} source-part length scales must exactly cover the editable bundle"
                )
            if role == "thumb":
                body_radius_scale = thumb_body_radius
                distal_radius_scale = thumb_distal_radius
            else:
                body_radius_scale = normal_body_radius
                distal_radius_scale = normal_distal_radius
            realized_finger_parameters[role] = {
                "length_scales_by_source_part": {
                    str(source_id): value
                    for source_id, value in length_scales_by_source_part.items()
                },
                "body_radius_scale": body_radius_scale,
                "distal_radius_scale": distal_radius_scale,
            }
            slots.append(instantiate_finger(
                parts, slot_id, role, target_slots[role], bundle,
                sources[bundle["source_hand_id"]], candidates,
                length_scales_by_source_part, body_radius_scale, distal_radius_scale,
                lock_proximal_hardware=seed_id in PROTECTED_TRANSMISSION_SOURCES,
            ))

        if len(slots) > MAX_FINGERS:
            raise ValueError(
                f"Design Grammar permits at most {MAX_FINGERS} fingers; "
                f"{seed_id} generated {len(slots)}"
            )

        finalize_graph(parts)
        platform_dof = len(base_palm_joints)
        baseline_dof = platform_dof + sum(
            bundles[f"{seed_id}:{role}"]["dof_count"] for role in base_roles
        )
        generated_dof = platform_dof + sum(slot["dof_count"] for slot in slots)
        expected_dof = int(direct_entries[seed_id]["scalar_dofs"])
        if baseline_dof != expected_dof or generated_dof != expected_dof:
            raise ValueError(
                f"{seed_id}: direct URDF DoF={expected_dof}, "
                f"baseline={baseline_dof}, generated={generated_dof}"
            )
        hands.append({
            "hand_id": design_spec.get("hand_id", f"grammar_{index + 1:03d}"),
            "seed_source": seed_id,
            "graph_parameters": {
                "source_hand": seed_id,
                "palm": {
                    "layout_mode": layout_mode,
                    "expansion": palm_layout.get("palm_expansion", 0.0),
                    "prototype_index": palm_spec.get("prototype_index"),
                    "prototype_bank_id": palm_spec.get(
                        "prototype_bank_id", f"{seed_id}:palm32"
                    ),
                    "scale_x": sx,
                    "scale_z": sz,
                    "yaw": yaw,
                },
                "fingers": realized_finger_parameters,
                "shared_width_scales": {
                    "normal_body": normal_body_radius,
                    "normal_distal": normal_distal_radius,
                    "thumb_body": thumb_body_radius,
                    "thumb_distal": thumb_distal_radius,
                },
            },
            "edit_mode": mode,
            "palm_layout": palm_layout,
            "baseline_finger_count": len(source_roles),
            "finger_count": len(slots),
            "baseline_dof": baseline_dof,
            "dof_count": generated_dof,
            "platform_dof_count": platform_dof,
            "active_dof_count": int(direct_entries[seed_id]["active_dofs"]),
            "passive_mimic_dof_count": int(
                direct_entries[seed_id]["passive_mimic_dofs"]
            ),
            "base_palm_kinematics": {
                "source_hand_id": seed_id,
                "source_urdf": direct_entries[seed_id]["path"],
                "editable": False,
                "representation": (
                    "source-copied movable-joint graph independent of mesh parts"
                ),
                "dof_count": len(base_palm_joints),
                "active_dof_count": sum(
                    joint["actuation"] == "active_direct"
                    for joint in base_palm_joints
                ),
                "passive_mimic_dof_count": sum(
                    joint["actuation"] == "passive_mimic"
                    for joint in base_palm_joints
                ),
                "kinematic_only_frame_count": sum(
                    joint["kinematic_only"] for joint in base_palm_joints
                ),
                "joints": base_palm_joints,
            },
            "protected_transmission_source": seed_id in PROTECTED_TRANSMISSION_SOURCES,
            "protected_platform_node_ids": protected_platform_node_ids,
            "palm_transform": palm_linear.tolist(),
            "grammar_actions": actions,
            "general_morphology": (
                None
                if "general_morphology_vector" not in design_spec
                else {
                    "vector": design_spec["general_morphology_vector"],
                    "vector_names": design_spec["general_morphology_vector_names"],
                    "vector_dimension": len(design_spec["general_morphology_vector"]),
                    "palm_prototype_bank_id": palm_spec.get(
                        "prototype_bank_id", f"{seed_id}:palm32"
                    ),
                    "palm_prototype_index": palm_spec.get("prototype_index"),
                }
            ),
            "finger_slots": slots,
            "parts": parts,
        })

    # Keep only fail-fast invariants required to compile a physically coherent
    # design. Detailed exploratory reports belonged to the old experiment
    # and are intentionally not part of the production pipeline.
    for hand in hands:
        part_ids = {int(node["id"]) for node in hand["parts"]}
        if len(part_ids) != len(hand["parts"]):
            raise ValueError(f"{hand['hand_id']}: duplicate part ID")
        if sum(node["parent"] is None for node in hand["parts"]) != 1:
            raise ValueError(f"{hand['hand_id']}: graph must have one root")
        if any(
            node["parent"] is not None
            and (
                int(node["parent"]) not in part_ids
                or int(node["parent"]) >= int(node["id"])
            )
            for node in hand["parts"]
        ):
            raise ValueError(f"{hand['hand_id']}: disconnected or cyclic graph")
        if max(hand["finger_count"], 0) > MAX_FINGERS:
            raise ValueError(f"{hand['hand_id']}: more than {MAX_FINGERS} fingers")
        if hand["dof_count"] != int(
            direct_entries[hand["seed_source"]]["scalar_dofs"]
        ):
            raise ValueError(f"{hand['hand_id']}: source DoF was not preserved")
        if hand["dof_count"] != (
            hand["active_dof_count"] + hand["passive_mimic_dof_count"]
        ):
            raise ValueError(f"{hand['hand_id']}: invalid active/mimic partition")
        if not hand["palm_layout"].get("cyclic_order_preserved", True):
            raise ValueError(f"{hand['hand_id']}: finger order changed")
        if not np.isclose(
            np.linalg.norm(np.asarray(hand["palm_transform"])[:, 1]), 1.0
        ):
            raise ValueError(f"{hand['hand_id']}: palm thickness changed")
        bundle_ids = [slot["bundle_id"] for slot in hand["finger_slots"]]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError(f"{hand['hand_id']}: duplicate finger bundle")
        for node in hand["parts"]:
            if node["source_hand_id"] != hand["seed_source"]:
                raise ValueError(f"{hand['hand_id']}: cross-source mesh")
            if node.get("protected_transmission_root") and not np.array_equal(
                np.asarray(node["mesh_linear"]), np.eye(3)
            ):
                raise ValueError(f"{hand['hand_id']}: protected base changed")
        for node in hand["parts"][1:]:
            if node.get("protected_platform"):
                continue
            if node["candidate_id"] not in node["compatible_candidate_ids"]:
                raise ValueError(f"{hand['hand_id']}: invalid motor/link pairing")
            if candidates[node["candidate_id"]]["bundle_id"] != node["mechanism_bundle_id"]:
                raise ValueError(f"{hand['hand_id']}: candidate ownership mismatch")
        for slot in hand["finger_slots"]:
            root = hand["parts"][int(slot["root_node_id"])]
            if slot["role"] != slot["source_role"]:
                raise ValueError(f"{hand['hand_id']}: finger role order changed")
            if not np.allclose(
                root["world_pos"], slot["attachment_translation"], atol=1.0e-10
            ):
                raise ValueError(f"{hand['hand_id']}: finger root disconnected")

    segment_endpoint_errors = []
    segment_endpoint_checks = 0
    for hand in hands:
        for child in hand["parts"][1:]:
            parent = hand["parts"][int(child["parent"])]
            if (
                child.get("mechanism_bundle_id")
                != parent.get("mechanism_bundle_id")
                or "source_relative_pos" not in child
            ):
                continue
            expected = np.asarray(parent["mesh_linear"], dtype=float) @ np.asarray(
                child["source_relative_pos"], dtype=float
            )
            actual = np.asarray(child["relative_pos"], dtype=float)
            segment_endpoint_errors.append(float(np.linalg.norm(actual - expected)))
            segment_endpoint_checks += 1
    if segment_endpoint_errors and max(segment_endpoint_errors) > 1.0e-10:
        raise ValueError("generic segment mesh endpoint and child joint disagree")

    used_source_ids = list(dict.fromkeys(hand["seed_source"] for hand in hands))
    summary = {
        "generation_seed": generation_seed,
        "source_hands": len(used_source_ids),
        "generated_hands": len(hands),
        "source_hand_ids": used_source_ids,
        "available_source_hand_ids": eligible_palms,
        "ignored_non_morphology_registry_sources": non_morphology_registry_sources,
        "finger_count_range": [
            min(hand["finger_count"] for hand in hands),
            max(hand["finger_count"] for hand in hands),
        ],
        "dof_range": [
            min(hand["dof_count"] for hand in hands),
            max(hand["dof_count"] for hand in hands),
        ],
        "layout_counts": {
            mode: sum(hand["palm_layout"]["mode"] == mode for hand in hands)
            for mode in PALM_LAYOUT_MODES
        },
        "hands_with_preserved_base_or_palm_dof": sum(
            hand["base_palm_kinematics"]["dof_count"] > 0 for hand in hands
        ),
        "joint_to_joint_segment_endpoint_checks": segment_endpoint_checks,
        "maximum_segment_endpoint_error": max(segment_endpoint_errors, default=0.0),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_seed": generation_seed,
                "hands": hands,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    AUDIT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
