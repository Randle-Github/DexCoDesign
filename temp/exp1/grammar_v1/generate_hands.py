#!/usr/bin/env python3
"""Generate source-ordered hands with length/radius-only finger edits."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STRICT = HERE.parent / "strict_v2" / "outputs"
SOURCE_GRAPHS = STRICT / "source_structure_graphs.json"
LIBRARY = HERE / "outputs" / "mechanism_bundle_library.json"
OUTPUT = HERE / "outputs" / "generated_hand_ir.json"
AUDIT = HERE / "outputs" / "grammar_audit.json"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
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
        offset = float(rng.uniform(-limit, limit))
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
) -> dict:
    """House of Dextra style discrete-slot motor placement.

    Non-anthropomorphic hands place complete finger mechanisms on a variable
    circular radius.  Symmetric layouts use evenly spaced slots; asymmetric
    layouts rejection-sample 36 slots with a four-slot circular separation.
    The palm is compiled later from these exact graph motor footprints.
    """
    if mode not in PALM_LAYOUT_MODES:
        raise ValueError(f"unknown palm layout mode {mode!r}")
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
            "thickness_scale_locked": 1.0,
        }

    if mode == "anthropomorphic":
        roles = list(target_slots)
        before = np.asarray([
            [np.dot(target_slots[role]["anchor"], palm_rotation[:, 0]),
             np.dot(target_slots[role]["anchor"], palm_rotation[:, 2])]
            for role in roles
        ])
        source_order = canonical_cyclic_roles(roles, before)
        edits = vary_attachment_layout(rng, target_slots, palm_rotation)
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
    requested_blend = float(rng.uniform(0.52, 0.66))

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
    if actual_clearance < -1.0e-8:
        for candidate_blend in np.linspace(requested_blend, 0.86, 35):
            candidate_centers, candidate_clearance = blended_clearance(float(candidate_blend))
            if candidate_clearance >= -1.0e-8:
                layout_blend = float(candidate_blend)
                blended_centers = candidate_centers
                actual_clearance = candidate_clearance
                break
        else:
            raise ValueError("source/House blended layout cannot satisfy motor clearance")

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
    length_scale: float, radius_scale: float,
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
    target_axis = realized_frame[:, 2]
    deform = radius_scale * np.eye(3) + (length_scale - radius_scale) * np.outer(target_axis, target_axis)
    linear = deform @ rotation
    reference_axis = realized_reference_frame[:, 2]
    reference_deform = (
        radius_scale * np.eye(3)
        + (length_scale - radius_scale) * np.outer(reference_axis, reference_axis)
    )
    reference_linear = reference_deform @ reference_rotation
    id_map: dict[int, int] = {}
    root_new_id = None
    for rank, source_id in enumerate(ids):
        source_node = source_hand["parts"][source_id]
        parent_source = source_node["parent"]
        internal = parent_source in block_set
        new_id = len(output)
        id_map[source_id] = new_id
        node_linear = rotation if lock_proximal_hardware and rank == 0 else linear
        node_reference_linear = (
            reference_rotation
            if lock_proximal_hardware and rank == 0
            else reference_linear
        )
        if not internal:
            relative = target_slot["anchor"]
            parent = 0
            root_new_id = new_id
        else:
            relative = linear @ np.asarray(source_node["relative_pos"], dtype=float)
            parent = id_map[int(parent_source)]
        candidate_id = f"{bundle['source_hand_id']}:part:{source_id}"
        candidate = candidates[candidate_id]
        axis = rotation @ np.asarray(source_node["joint_axis"], dtype=float)
        output.append({
            "id": new_id,
            "parent": parent,
            "role": output_role,
            "finger_slot": slot_id,
            "joint_type": source_node["joint_type"],
            "joint_axis": axis.tolist(),
            "joint_range": source_node["joint_range"],
            "joint_name": f"slot_{slot_id}_{source_node['joint_name']}",
            "relative_pos": relative.tolist(),
            "source_hand_id": bundle["source_hand_id"],
            "source_part_id": source_id,
            "source_mesh": source_node.get("mesh"),
            "candidate_id": candidate_id,
            "mechanism_bundle_id": bundle["bundle_id"],
            "motor_binding": candidate["motor_binding"],
            "compatible_candidate_ids": candidate["compatible_candidate_ids"],
            "mesh_linear": node_linear.tolist(),
            "reference_mesh_linear": node_reference_linear.tolist(),
            "length_scale": 1.0 if lock_proximal_hardware and rank == 0 else length_scale,
            "radius_scale": 1.0 if lock_proximal_hardware and rank == 0 else radius_scale,
            "source_rank": rank,
            "protected_proximal_hardware": bool(lock_proximal_hardware and rank == 0),
            "morphology_part": "finger_root_rigid_body" if rank == 0 else "finger_link_rigid_body",
        })
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
    rng = np.random.default_rng(20260718)
    source_payload = json.loads(SOURCE_GRAPHS.read_text(encoding="utf-8"))
    sources = {hand["hand_id"]: hand for hand in source_payload["hands"]}
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    bundles = {bundle["bundle_id"]: bundle for bundle in library["bundles"]}
    candidates = candidate_map(library)
    eligible_palms = []
    for hand_id in library["palm_sources"]:
        present = [role for role in DIGITS if f"{hand_id}:{role}" in bundles]
        source_roles = [role for role in DIGITS if any(p["role"] == role for p in sources[hand_id]["parts"])]
        if len(present) == len(source_roles) and len(present) >= 4:
            eligible_palms.append(hand_id)

    hands = []
    graph_modes = ("geometry_only",)
    for index in range(100):
        mode = graph_modes[0]
        requested_layout_mode = ("anthropomorphic", "symmetric", "asymmetric")[index % 3]
        seed_pool = eligible_palms
        seed_id = seed_pool[index % len(seed_pool)]
        seed = sources[seed_id]
        # Large radial rearrangements are incompatible with a fixed tendon or
        # underactuated transmission platform. Keep those hands on the bounded
        # anthropomorphic palm-edit path; other sources retain all three modes.
        layout_mode = (
            "anthropomorphic"
            if seed_id in PROTECTED_TRANSMISSION_SOURCES
            else requested_layout_mode
        )
        source_roles = [role for role in DIGITS if f"{seed_id}:{role}" in bundles]
        base_roles = list(source_roles)
        selected = {role: deepcopy(bundles[f"{seed_id}:{role}"]) for role in base_roles}
        actions = [{
            "operation": "LOCK_SOURCE_TOPOLOGY_ORDER_AND_ATTACHMENTS",
            "allowed_geometry_edits": ["finger_link_length", "finger_link_radius"],
        }]

        if seed_id in PROTECTED_TRANSMISSION_SOURCES:
            # The root frame and the separate under-palm transmission platform
            # are hardware interfaces. Palm shape may still be edited later by
            # the region-aware compiler, but no global affine may move the base.
            sx = sy = sz = 1.0
            yaw = 0.0
        else:
            sx = float(rng.uniform(0.93, 1.10))
            sy = 1.0
            sz = float(rng.uniform(0.92, 1.13))
            yaw = float(rng.uniform(-0.16, 0.16))
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

        base_slots = {role: source_slot(seed, bundles[f"{seed_id}:{role}"]) for role in base_roles}
        target_slots = {role: transformed_slot(slot, palm_linear, rotation) for role, slot in base_slots.items()}
        source_palm_bounds = np.asarray(root["mesh"]["bounds"], dtype=float)[:, [0, 2]]
        palm_bounds_local_xz = source_palm_bounds * np.asarray([sx, sz])
        palm_layout = apply_global_palm_layout(
            rng, target_slots, selected, candidates, rotation, layout_mode,
            palm_bounds_local_xz=palm_bounds_local_xz,
        )
        actions.append({
            "operation": "APPLY_GLOBAL_PALM_LAYOUT",
            **palm_layout,
        })
        slots = []
        for slot_id, role in enumerate(base_roles):
            bundle = selected[role]
            length_scale = float(rng.uniform(0.87, 1.17))
            radius_scale = float(rng.uniform(0.88, 1.14))
            slots.append(instantiate_finger(
                parts, slot_id, role, target_slots[role], bundle,
                sources[bundle["source_hand_id"]], candidates,
                length_scale, radius_scale,
                lock_proximal_hardware=seed_id in PROTECTED_TRANSMISSION_SOURCES,
            ))

        if len(slots) > MAX_FINGERS:
            raise ValueError(
                f"Design Grammar permits at most {MAX_FINGERS} fingers; "
                f"{seed_id} generated {len(slots)}"
            )

        finalize_graph(parts)
        platform_dof = sum(
            parts[node_id]["joint_type"] != "fixed"
            for node_id in protected_platform_node_ids
        )
        baseline_dof = platform_dof + sum(
            bundles[f"{seed_id}:{role}"]["dof_count"] for role in base_roles
        )
        generated_dof = platform_dof + sum(slot["dof_count"] for slot in slots)
        hands.append({
            "hand_id": f"grammar_{index + 1:03d}",
            "seed_source": seed_id,
            "edit_mode": mode,
            "palm_layout": palm_layout,
            "baseline_finger_count": len(source_roles),
            "finger_count": len(slots),
            "baseline_dof": baseline_dof,
            "dof_count": generated_dof,
            "platform_dof_count": platform_dof,
            "protected_transmission_source": seed_id in PROTECTED_TRANSMISSION_SOURCES,
            "protected_platform_node_ids": protected_platform_node_ids,
            "palm_transform": palm_linear.tolist(),
            "grammar_actions": actions,
            "finger_slots": slots,
            "parts": parts,
        })

    invalid_bindings = []
    invalid_bundle_ownership = []
    cross_source_mesh_assignments = []
    duplicate_source_bundle_assignments = []
    finger_role_permutations = []
    protected_hardware_changes = []
    attachment_pose_errors = []
    cyclic_order_violations = []
    connector_errors = []
    mesh_joint_frame_errors = []
    semantic_interface_errors = []
    connected = []
    acyclic = []
    for hand in hands:
        if not hand["palm_layout"].get("cyclic_order_preserved", True):
            cyclic_order_violations.append(hand["hand_id"])
        bundle_ids = [slot["bundle_id"] for slot in hand["finger_slots"]]
        if len(bundle_ids) != len(set(bundle_ids)):
            duplicate_source_bundle_assignments.append(hand["hand_id"])
        part_ids = {int(node["id"]) for node in hand["parts"]}
        connected.append(all(node["parent"] is None or int(node["parent"]) in part_ids for node in hand["parts"]))
        acyclic.append(all(node["parent"] is None or int(node["parent"]) < int(node["id"]) for node in hand["parts"]))
        for node in hand["parts"]:
            if node["source_hand_id"] != hand["seed_source"]:
                cross_source_mesh_assignments.append([
                    hand["hand_id"], node["id"], hand["seed_source"], node["source_hand_id"],
                ])
            linear = np.asarray(node["mesh_linear"], dtype=float)
            if node.get("protected_transmission_root") and not np.array_equal(linear, np.eye(3)):
                protected_hardware_changes.append([hand["hand_id"], node["id"]])
            if node.get("protected_proximal_hardware") and not (
                np.allclose(linear.T @ linear, np.eye(3), atol=1.0e-10)
                and np.isclose(np.linalg.det(linear), 1.0, atol=1.0e-10)
            ):
                protected_hardware_changes.append([hand["hand_id"], node["id"]])
        for node in hand["parts"][1:]:
            if node.get("protected_platform"):
                source_node = sources[hand["seed_source"]]["parts"][int(node["source_part_id"])]
                if (
                    not np.array_equal(np.asarray(node["mesh_linear"]), np.eye(3))
                    or not np.array_equal(
                        np.asarray(node["relative_pos"]),
                        np.asarray(source_node["relative_pos"]),
                    )
                ):
                    protected_hardware_changes.append([hand["hand_id"], node["id"]])
                continue
            if node["candidate_id"] not in node["compatible_candidate_ids"]:
                invalid_bindings.append([hand["hand_id"], node["id"]])
            if candidates[node["candidate_id"]]["bundle_id"] != node["mechanism_bundle_id"]:
                invalid_bundle_ownership.append([hand["hand_id"], node["id"]])
        for slot in hand["finger_slots"]:
            if slot["role"] != slot["source_role"]:
                finger_role_permutations.append([
                    hand["hand_id"], slot["slot_id"], slot["role"], slot["source_role"],
                ])
            translation_error = float(np.linalg.norm(
                np.asarray(slot["attachment_translation"])
                - np.asarray(slot["reference_attachment_translation"])
            ))
            rotation_error = float(np.linalg.norm(
                np.asarray(slot["attachment_rotation"])
                - np.asarray(slot["reference_attachment_rotation"])
            ))
            attachment_pose_errors.append(max(translation_error, rotation_error))
            root = hand["parts"][slot["root_node_id"]]
            error = float(np.linalg.norm(np.asarray(root["world_pos"]) - np.asarray(slot["attachment_translation"])))
            connector_errors.append(error)
            if (
                slot.get("interface_parent_palm_node_id") != 0
                or slot.get("interface_child_finger_node_id") != slot["root_node_id"]
                or hand["parts"][0].get("morphology_part") != "palm_parent_rigid_body"
                or root.get("morphology_part") != "finger_root_rigid_body"
            ):
                semantic_interface_errors.append([
                    hand["hand_id"], slot["slot_id"], slot["role"]
                ])
            donor_frame = source_slot(
                sources[slot["source_hand_id"]], bundles[slot["bundle_id"]]
            )["frame"]
            linear = np.asarray(root["mesh_linear"], dtype=float)
            left, _, right = np.linalg.svd(linear)
            mesh_rotation = left @ right
            mesh_joint_frame_errors.append(float(np.linalg.norm(
                mesh_rotation @ donor_frame
                - np.asarray(slot["attachment_rotation"], dtype=float)
            )))
    house_layouts = [
        hand["palm_layout"] for hand in hands
        if "source_house_blend" in hand["palm_layout"]
    ]
    audit = {
        "hands": len(hands),
        "finger_count_range": [min(h["finger_count"] for h in hands), max(h["finger_count"] for h in hands)],
        "dof_range": [min(h["dof_count"] for h in hands), max(h["dof_count"] for h in hands)],
        "designs_with_added_finger": sum(h["finger_count"] > h["baseline_finger_count"] for h in hands),
        "designs_with_removed_finger": sum(h["finger_count"] < h["baseline_finger_count"] for h in hands),
        "four_to_five_addition_disabled": not ENABLE_FINGER_ADDITION,
        "maximum_fingers_enforced": max(h["finger_count"] for h in hands) <= MAX_FINGERS,
        "designs_with_increased_dof": sum(h["dof_count"] > h["baseline_dof"] for h in hands),
        "maximum_dof_increase": max(h["dof_count"] - h["baseline_dof"] for h in hands),
        "all_graphs_connected": all(connected),
        "all_graphs_acyclic": all(acyclic),
        "invalid_motor_link_bindings": invalid_bindings,
        "invalid_bundle_ownership": invalid_bundle_ownership,
        "cross_source_mesh_assignments": cross_source_mesh_assignments,
        "all_meshes_from_seed_source": not cross_source_mesh_assignments,
        "hands_with_mixed_source_meshes": len({
            record[0] for record in cross_source_mesh_assignments
        }),
        "duplicate_source_bundle_assignments": duplicate_source_bundle_assignments,
        "all_source_bundles_used_at_most_once_per_hand": not duplicate_source_bundle_assignments,
        "finger_role_permutations": finger_role_permutations,
        "cyclic_order_violations": cyclic_order_violations,
        "source_finger_order_locked": not finger_role_permutations and not cyclic_order_violations,
        "maximum_attachment_pose_error": max(attachment_pose_errors, default=0.0),
        "attachment_poses_editable_with_order_constraint": True,
        "protected_hardware_changes": protected_hardware_changes,
        "all_protected_hardware_locked": not protected_hardware_changes,
        "protected_platform_nodes": sum(
            len(hand["protected_platform_node_ids"]) for hand in hands
        ),
        "protected_transmission_hands": sum(
            hand["protected_transmission_source"] for hand in hands
        ),
        "palm_edits_enabled": True,
        "protected_transmission_layout_policy": "anthropomorphic_only_with_locked_base_region",
        "finger_length_scale_range": [
            min(
                node["length_scale"] for hand in hands for node in hand["parts"]
                if "finger_slot" in node
            ),
            max(
                node["length_scale"] for hand in hands for node in hand["parts"]
                if "finger_slot" in node
            ),
        ],
        "finger_radius_scale_range": [
            min(
                node["radius_scale"] for hand in hands for node in hand["parts"]
                if "finger_slot" in node
            ),
            max(
                node["radius_scale"] for hand in hands for node in hand["parts"]
                if "finger_slot" in node
            ),
        ],
        "maximum_slot_connector_error": max(connector_errors, default=0.0),
        "semantic_palm_finger_interface_errors": semantic_interface_errors,
        "all_interfaces_have_explicit_palm_parent_and_finger_child": (
            not semantic_interface_errors
        ),
        "maximum_finger_mesh_joint_frame_error": max(
            mesh_joint_frame_errors, default=0.0
        ),
        "all_palm_transforms_applied_to_slots": all(
            slot["connector_transform_applied"] for hand in hands for slot in hand["finger_slots"]
        ),
        "all_palm_thickness_scales_locked": all(
            np.isclose(np.linalg.norm(np.asarray(hand["palm_transform"], dtype=float)[:, 1]), 1.0)
            for hand in hands
        ),
        "edge_attachment_layout_edits": sum(
            len(action["edits"])
            for hand in hands for action in hand["grammar_actions"]
            if action["operation"] == "APPLY_GLOBAL_PALM_LAYOUT"
        ),
        "palm_layout_mode_counts": {
            mode: sum(hand["palm_layout"]["mode"] == mode for hand in hands)
            for mode in PALM_LAYOUT_MODES
        },
        "minimum_motor_arc_clearance": min(
            (
                hand["palm_layout"]["minimum_arc_clearance"]
                for hand in hands
                if hand["palm_layout"]["minimum_arc_clearance"] is not None
            ),
            default=None,
        ),
        "minimum_actual_motor_clearance": min(
            (
                layout["minimum_actual_motor_clearance"]
                for layout in house_layouts
            ),
            default=None,
        ),
        "source_house_blend_range": None if not house_layouts else [
            min(layout["source_house_blend"] for layout in house_layouts),
            max(layout["source_house_blend"] for layout in house_layouts),
        ],
        "maximum_attachment_displacement_fraction": max(
            (layout["maximum_attachment_displacement_fraction"] for layout in house_layouts),
            default=0.0,
        ),
    }
    critical_failures = {
        "invalid_motor_link_bindings": invalid_bindings,
        "invalid_bundle_ownership": invalid_bundle_ownership,
        "cross_source_mesh_assignments": cross_source_mesh_assignments,
        "duplicate_source_bundle_assignments": duplicate_source_bundle_assignments,
        "finger_role_permutations": finger_role_permutations,
        "cyclic_order_violations": cyclic_order_violations,
        "protected_hardware_changes": protected_hardware_changes,
    }
    if any(critical_failures.values()):
        raise ValueError(f"source-locked grammar invariant failed: {critical_failures}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"schema_version": 1, "hands": hands}, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
