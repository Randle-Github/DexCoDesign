#!/usr/bin/env python3
"""Build a conservative source-derived motor/link grammar library.

Most public hand descriptions do not identify the physical motor model.  We
therefore never infer cross-vendor equivalence.  Repeated digit actuators inside
one source hand share a source-local motor type; link compatibility is the set
of semantic/depth-compatible links that this repeated type actually drives in
that source model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACT_ROOT = ROOT / "artifacts" / "hand_morphology"
SOURCE_GRAPHS = ARTIFACT_ROOT / "reference_graphs.json"
OUTPUT = ARTIFACT_ROOT / "mechanism_bundles.json"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
CONTAMINATED: set[tuple[str, int]] = set()


def descendant_digits(hand: dict) -> list[set[str]]:
    children: dict[int, list[int]] = defaultdict(list)
    for node in hand["parts"][1:]:
        children[int(node["parent"])].append(int(node["id"]))
    result = [set() for _ in hand["parts"]]
    for index in range(len(hand["parts"]) - 1, -1, -1):
        role = hand["parts"][index]["role"]
        if role in DIGITS:
            result[index].add(role)
        for child in children[index]:
            result[index].update(result[child])
    return result


def digit_block(hand: dict, role: str) -> list[int]:
    descendants = descendant_digits(hand)
    return [
        int(node["id"])
        for node in hand["parts"][1:]
        if node["role"] == role
        or (node["role"] == "other" and descendants[int(node["id"])] == {role})
    ]


def depth_class(role: str, rank: int, count: int) -> str:
    family = "thumb" if role == "thumb" else "normal"
    if rank == 0:
        segment = "root"
    elif rank == count - 1:
        segment = "tip"
    elif rank == 1:
        segment = "proximal"
    elif rank == count - 2:
        segment = "distal"
    else:
        segment = "middle"
    return f"{family}:{segment}"


def motor_mode(node: dict) -> str:
    name = str(node.get("joint_name", "")).lower()
    if any(token in name for token in ("spread", "yaw", "abd", "opposition")):
        return "spread_or_opposition"
    return "flexion"


def main() -> int:
    source = json.loads(SOURCE_GRAPHS.read_text(encoding="utf-8"))
    hands = {hand["hand_id"]: hand for hand in source["hands"]}
    bundles = []
    candidates = []
    rejected = []
    for hand_id, hand in hands.items():
        for role in DIGITS:
            ids = digit_block(hand, role)
            if not ids:
                continue
            # A movable kinematic frame may intentionally have no visual
            # geometry (for example MANO compound root axes). It remains part
            # of the mechanism bundle and must not make the finger ineligible.
            nonzero_empty: list[int] = []
            contaminated = [part_id for part_id in ids if (hand_id, part_id) in CONTAMINATED]
            if nonzero_empty or contaminated:
                rejected.append({
                    "bundle_id": f"{hand_id}:{role}",
                    "nonzero_empty_parts": nonzero_empty,
                    "contaminated_parts": contaminated,
                })
                continue
            block_set = set(ids)
            roots = [part_id for part_id in ids if hand["parts"][part_id]["parent"] not in block_set]
            if len(roots) != 1:
                rejected.append({"bundle_id": f"{hand_id}:{role}", "invalid_roots": roots})
                continue
            binding_ids = []
            candidate_ids = []
            for rank, part_id in enumerate(ids):
                node = hand["parts"][part_id]
                candidate_id = f"{hand_id}:part:{part_id}"
                interface = depth_class(role, rank, len(ids))
                mode = motor_mode(node)
                motor_type_id = f"{hand_id}:repeated_digit_motor:{mode}"
                binding_id = f"{hand_id}:binding:{part_id}"
                candidate_ids.append(candidate_id)
                binding_ids.append(binding_id)
                candidates.append({
                    "candidate_id": candidate_id,
                    "bundle_id": f"{hand_id}:{role}",
                    "source_hand_id": hand_id,
                    "source_part_id": part_id,
                    "semantic_role": role,
                    "interface_class": interface,
                    "source_mesh": node.get("mesh"),
                    "motor_binding": {
                        "binding_id": binding_id,
                        "motor_type_id": motor_type_id,
                        "actuation_mode": mode,
                        "evidence": "source-local repeated actuator; physical model not published in asset",
                    },
                })
            bundles.append({
                "bundle_id": f"{hand_id}:{role}",
                "source_hand_id": hand_id,
                "source_role": role,
                "source_part_ids": ids,
                "root_part_id": roots[0],
                "dof_count": sum(hand["parts"][part_id]["joint_type"] != "fixed" for part_id in ids),
                "candidate_ids": candidate_ids,
                "binding_ids": binding_ids,
                "transmission_contract": f"{hand_id}:source_digit_contract",
                "replacement_policy": "replace the complete finger bundle; never split motor/link ownership",
            })

    by_motor_interface: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate["motor_binding"]["motor_type_id"], candidate["interface_class"])
        by_motor_interface[key].append(candidate["candidate_id"])
    for candidate in candidates:
        key = (candidate["motor_binding"]["motor_type_id"], candidate["interface_class"])
        candidate["compatible_candidate_ids"] = sorted(by_motor_interface[key])

    palm_sources = [
        hand_id for hand_id, hand in hands.items()
        if hand["parts"][0].get("mesh") is not None
        and sum(bool(digit_block(hand, role)) for role in DIGITS) >= 4
    ]
    payload = {
        "schema_version": 1,
        "motor_equivalence_policy": {
            "cross_vendor": "forbidden unless an explicit physical motor model proves equivalence",
            "source_local": "same source actuator mode is treated as a repeated motor type",
            "link_compatibility": "candidate must be an original pairing of the same motor type and interface class",
        },
        "palm_sources": palm_sources,
        "bundles": bundles,
        "candidates": candidates,
        "rejected_bundles": rejected,
        "summary": {
            "source_hands": len(hands),
            "palm_sources": len(palm_sources),
            "accepted_finger_bundles": len(bundles),
            "rejected_finger_bundles": len(rejected),
            "link_candidates": len(candidates),
            "motor_types": len({c["motor_binding"]["motor_type_id"] for c in candidates}),
            "multi_candidate_motor_interfaces": sum(len(value) > 1 for value in by_motor_interface.values()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
