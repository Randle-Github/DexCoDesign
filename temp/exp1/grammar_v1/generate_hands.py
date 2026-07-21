#!/usr/bin/env python3
"""Generate 100 grammar-valid hands with large, source-bound edits.

Large edits replace complete finger mechanism bundles.  Added fingers extend
the palm, create a transformed slot connector, and attach one complete source
motor/joint/link bundle.  Palm affine transforms are applied to both the palm
mesh and every slot pose.
"""

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
# Finger-count expansion is intentionally disabled in grammar_v1.  A new slot
# needs a palm-specific mounting interface and cannot be inferred safely from
# visual finger pitch alone.  It will return only after the compiler has an
# explicit palm connector/candidate model.
ENABLE_FINGER_ADDITION = False
FIFTH_SLOT_PITCH = 0.42


def normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-8 else fallback.copy()


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


def candidate_map(library: dict) -> dict[str, dict]:
    return {candidate["candidate_id"]: candidate for candidate in library["candidates"]}


def choose_more_dof(rng: np.random.Generator, role: str, current: dict, by_role: dict[str, list[dict]]) -> dict | None:
    choices = [bundle for bundle in by_role[role] if bundle["dof_count"] > current["dof_count"]]
    if not choices:
        return None
    maximum = max(bundle["dof_count"] for bundle in choices)
    strongest = [bundle for bundle in choices if bundle["dof_count"] == maximum]
    return deepcopy(strongest[int(rng.integers(len(strongest)))])


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
    target["anchor"] = target["anchor"] + FIFTH_SLOT_PITCH * pitch
    target["frame"] = rot_z(math.radians(-7.0)) @ target["frame"]
    target["derived_from"] = outer_role
    return target


def instantiate_finger(
    output: list[dict], slot_id: int, output_role: str, target_slot: dict,
    bundle: dict, source_hand: dict, candidates: dict[str, dict],
    length_scale: float, radius_scale: float,
) -> dict:
    ids = bundle["source_part_ids"]
    block_set = set(ids)
    donor_slot = source_slot(source_hand, bundle)
    # Source part meshes are already expressed in the shared canonical palm
    # frame.  Align only the directed chain axes with the minimal rotation.
    # Using the complete joint-axis frame would introduce arbitrary 180-degree
    # roll flips when two equivalent source joint axes use opposite signs.
    rotation = rotation_from_to(donor_slot["frame"][:, 2], target_slot["frame"][:, 2])
    target_axis = target_slot["frame"][:, 2]
    deform = radius_scale * np.eye(3) + (length_scale - radius_scale) * np.outer(target_axis, target_axis)
    linear = deform @ rotation
    id_map: dict[int, int] = {}
    root_new_id = None
    for rank, source_id in enumerate(ids):
        source_node = source_hand["parts"][source_id]
        parent_source = source_node["parent"]
        internal = parent_source in block_set
        new_id = len(output)
        id_map[source_id] = new_id
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
            "mesh_linear": linear.tolist(),
            "length_scale": length_scale,
            "radius_scale": radius_scale,
            "source_rank": rank,
        })
    return {
        "slot_id": slot_id,
        "role": output_role,
        "bundle_id": bundle["bundle_id"],
        "source_role": bundle["source_role"],
        "source_hand_id": bundle["source_hand_id"],
        "dof_count": bundle["dof_count"],
        "root_node_id": root_new_id,
        "attachment_translation": target_slot["anchor"].tolist(),
        "attachment_rotation": target_slot["frame"].tolist(),
        "connector_transform_applied": True,
    }


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
    by_role: dict[str, list[dict]] = defaultdict(list)
    for bundle in bundles.values():
        by_role[bundle["source_role"]].append(bundle)

    eligible_palms = []
    for hand_id in library["palm_sources"]:
        present = [role for role in DIGITS if f"{hand_id}:{role}" in bundles]
        source_roles = [role for role in DIGITS if any(p["role"] == role for p in sources[hand_id]["parts"])]
        if len(present) == len(source_roles) and len(present) >= 4:
            eligible_palms.append(hand_id)

    hands = []
    modes = ("increase_dof", "large_bundle_change")
    four_finger_palms = [
        hand_id for hand_id in eligible_palms
        if sum(f"{hand_id}:{role}" in bundles for role in DIGITS) == 4
    ]
    if not four_finger_palms:
        raise ValueError("ADD_FINGER requires at least one grammar-valid four-finger palm")
    for index in range(100):
        mode = modes[index % len(modes)]
        add_finger = ENABLE_FINGER_ADDITION and mode in {"add_finger", "add_finger_plus_dof"}
        # The grammar has a hard maximum of five fingers.  ADD_FINGER therefore
        # starts only from a real four-finger palm and creates its fifth slot.
        seed_pool = four_finger_palms if add_finger else eligible_palms
        seed_id = seed_pool[index % len(seed_pool)]
        seed = sources[seed_id]
        base_roles = [role for role in DIGITS if f"{seed_id}:{role}" in bundles]
        selected = {role: deepcopy(bundles[f"{seed_id}:{role}"]) for role in base_roles}
        actions = []

        if mode in {"increase_dof", "add_finger_plus_dof"}:
            improvable = [role for role in base_roles if choose_more_dof(rng, role, selected[role], by_role) is not None]
            rng.shuffle(improvable)
            for role in improvable[: max(1, min(2, len(improvable)))]:
                replacement = choose_more_dof(rng, role, selected[role], by_role)
                if replacement is not None:
                    selected[role] = replacement
                    actions.append({"operation": "REPLACE_COMPLETE_SOURCE_BUNDLE", "target": role, "bundle": replacement["bundle_id"], "reason": "increase_dof"})

        if mode == "large_bundle_change":
            roles = list(base_roles)
            rng.shuffle(roles)
            for role in roles[:3]:
                alternatives = [bundle for bundle in by_role[role] if bundle["source_hand_id"] != seed_id]
                replacement = alternatives[int(rng.integers(len(alternatives)))]
                selected[role] = deepcopy(replacement)
                actions.append({"operation": "REPLACE_COMPLETE_SOURCE_BUNDLE", "target": role, "bundle": replacement["bundle_id"], "reason": "large_mechanism_change"})

        sx = float(rng.uniform(0.93, 1.10))
        if add_finger:
            sx = float(rng.uniform(1.22, 1.36))
        sy = float(rng.uniform(0.92, 1.10))
        sz = float(rng.uniform(0.92, 1.13))
        yaw = float(rng.uniform(-0.16, 0.16))
        rotation = rot_z(yaw)
        palm_linear = rotation @ np.diag([sx, sy, sz])
        actions.append({"operation": "MODIFY_PALM_AND_SLOT_CONNECTORS", "scale": [sx, sy, sz], "yaw": yaw})

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
        }]

        base_slots = {role: source_slot(seed, bundles[f"{seed_id}:{role}"]) for role in base_roles}
        target_slots = {role: transformed_slot(slot, palm_linear, rotation) for role, slot in base_slots.items()}
        slots = []
        for slot_id, role in enumerate(base_roles):
            bundle = selected[role]
            length_scale = float(rng.uniform(0.87, 1.17))
            radius_scale = float(rng.uniform(0.88, 1.14))
            slots.append(instantiate_finger(
                parts, slot_id, role, target_slots[role], bundle,
                sources[bundle["source_hand_id"]], candidates,
                length_scale, radius_scale,
            ))

        if add_finger:
            if len(base_roles) != 4 or "pinky" in base_roles:
                raise ValueError(f"ADD_FINGER may only perform 4->5, got {seed_id}: {base_roles}")
            added_role = "pinky"
            donor_role = "pinky"
            choices = sorted(by_role[donor_role], key=lambda item: (-item["dof_count"], item["bundle_id"]))
            bundle = deepcopy(choices[index % min(4, len(choices))])
            target = extra_slot(base_slots, palm_linear)
            slot_id = len(slots)
            slots.append(instantiate_finger(
                parts, slot_id, added_role, target, bundle,
                sources[bundle["source_hand_id"]], candidates,
                float(rng.uniform(0.90, 1.10)), float(rng.uniform(0.90, 1.10)),
            ))
            actions.append({
                "operation": "ADD_FINGER_WITH_COMPLETE_BUNDLE",
                "target": added_role,
                "bundle": bundle["bundle_id"],
                "palm_extended": True,
            })

        if len(slots) > MAX_FINGERS:
            raise ValueError(
                f"Design Grammar permits at most {MAX_FINGERS} fingers; "
                f"{seed_id} generated {len(slots)}"
            )

        finalize_graph(parts)
        baseline_dof = sum(bundles[f"{seed_id}:{role}"]["dof_count"] for role in base_roles)
        generated_dof = sum(slot["dof_count"] for slot in slots)
        hands.append({
            "hand_id": f"grammar_{index + 1:03d}",
            "seed_source": seed_id,
            "edit_mode": mode,
            "baseline_finger_count": len(base_roles),
            "finger_count": len(slots),
            "baseline_dof": baseline_dof,
            "dof_count": generated_dof,
            "palm_transform": palm_linear.tolist(),
            "grammar_actions": actions,
            "finger_slots": slots,
            "parts": parts,
        })

    invalid_bindings = []
    invalid_bundle_ownership = []
    connector_errors = []
    connected = []
    acyclic = []
    for hand in hands:
        part_ids = {int(node["id"]) for node in hand["parts"]}
        connected.append(all(node["parent"] is None or int(node["parent"]) in part_ids for node in hand["parts"]))
        acyclic.append(all(node["parent"] is None or int(node["parent"]) < int(node["id"]) for node in hand["parts"]))
        for node in hand["parts"][1:]:
            if node["candidate_id"] not in node["compatible_candidate_ids"]:
                invalid_bindings.append([hand["hand_id"], node["id"]])
            if candidates[node["candidate_id"]]["bundle_id"] != node["mechanism_bundle_id"]:
                invalid_bundle_ownership.append([hand["hand_id"], node["id"]])
        for slot in hand["finger_slots"]:
            root = hand["parts"][slot["root_node_id"]]
            error = float(np.linalg.norm(np.asarray(root["world_pos"]) - np.asarray(slot["attachment_translation"])))
            connector_errors.append(error)
    audit = {
        "hands": len(hands),
        "finger_count_range": [min(h["finger_count"] for h in hands), max(h["finger_count"] for h in hands)],
        "dof_range": [min(h["dof_count"] for h in hands), max(h["dof_count"] for h in hands)],
        "designs_with_added_finger": sum(h["finger_count"] > h["baseline_finger_count"] for h in hands),
        "maximum_fingers_enforced": max(h["finger_count"] for h in hands) <= MAX_FINGERS,
        "designs_with_increased_dof": sum(h["dof_count"] > h["baseline_dof"] for h in hands),
        "maximum_dof_increase": max(h["dof_count"] - h["baseline_dof"] for h in hands),
        "all_graphs_connected": all(connected),
        "all_graphs_acyclic": all(acyclic),
        "invalid_motor_link_bindings": invalid_bindings,
        "invalid_bundle_ownership": invalid_bundle_ownership,
        "maximum_slot_connector_error": max(connector_errors, default=0.0),
        "all_palm_transforms_applied_to_slots": all(
            slot["connector_transform_applied"] for hand in hands for slot in hand["finger_slots"]
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"schema_version": 1, "hands": hands}, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

