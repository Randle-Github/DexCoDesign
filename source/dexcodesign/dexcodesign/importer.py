"""Import audited source graphs into HandIR and source-bound bundles.

This is a migration importer.  It intentionally consumes the already audited
source structure/rigid-part library while the native URDF/MJCF/USD parser is
being rebuilt.  Nothing downstream depends on the legacy VAE or generator.
"""

from __future__ import annotations

from pathlib import Path

from .io import read_json
from .schema import (
    FingerSlotIR,
    HandIR,
    JointEdgeIR,
    LinkCandidateSpec,
    LinkNodeIR,
    MechanismBundleSpec,
    ModuleDatabase,
)

DIGIT_ROLES = ("thumb", "index", "middle", "ring", "pinky")


def _tuple3(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


def import_reference_library(repo_root: Path) -> tuple[list[HandIR], ModuleDatabase]:
    output_root = repo_root / "temp/exp1/strict_v2/outputs"
    payload = read_json(output_root / "source_structure_graphs.json")
    database = ModuleDatabase()
    hands: list[HandIR] = []

    for source in payload["hands"]:
        source_id = source["hand_id"]
        nodes: list[LinkNodeIR] = []
        joints: list[JointEdgeIR] = []
        slots: list[FingerSlotIR] = []

        for part in source["parts"]:
            part_id = int(part["id"])
            bundle_id = f"{source_id}:bundle:{part_id:02d}"
            candidate_id = f"{source_id}:candidate:{part_id:02d}" if part.get("mesh") else None
            visual_path = None
            if part.get("mesh"):
                visual_path = str(Path("temp/exp1/strict_v2/outputs") / part["mesh"]["file"])
                candidate = LinkCandidateSpec(
                    candidate_id=candidate_id,
                    bundle_id=bundle_id,
                    source_hand_id=source_id,
                    source_part_id=part_id,
                    semantic_role=part["role"],
                    visual_path=visual_path,
                    canonical_dimensions=_tuple3(part["part_size"]),
                )
                database.candidates[candidate_id] = candidate

            is_root = part["parent"] is None
            movable = part["joint_type"] not in {"fixed", "none"}
            bundle = MechanismBundleSpec(
                bundle_id=bundle_id,
                source_hand_id=source_id,
                source_joint_name=part["joint_name"],
                actuator_id=(
                    None if is_root or not movable
                    else f"{source_id}:unresolved_source_actuator:{part['joint_name']}"
                ),
                binding_status="passive" if is_root or not movable else "joint_bound_pending_native_hardware_parse",
                transmission_type="unresolved_source_actuation" if movable else "passive_rigid",
                driven_joint_names=() if is_root else (part["joint_name"],),
                link_candidate_ids=() if candidate_id is None else (candidate_id,),
            )
            database.bundles[bundle_id] = bundle
            role = part["role"]
            nodes.append(
                LinkNodeIR(
                    node_id=part_id,
                    source_node_id=part_id,
                    semantic_role=role,
                    finger_slot=DIGIT_ROLES.index(role) if role in DIGIT_ROLES else None,
                    bundle_id=bundle_id,
                    candidate_id=candidate_id,
                )
            )
            if not is_root:
                lower, upper = (float(value) for value in part["joint_range"])
                joints.append(
                    JointEdgeIR(
                        joint_id=len(joints),
                        parent_node=int(part["parent"]),
                        child_node=part_id,
                        source_joint_name=part["joint_name"],
                        joint_type=part["joint_type"],
                        origin_translation=_tuple3(part["relative_pos"]),
                        axis=_tuple3(part["joint_axis"]),
                        lower_limit=lower,
                        upper_limit=upper,
                        zero_position=0.0,
                        active=movable,
                        motor_link_binding_id=bundle_id,
                    )
                )

        for role in DIGIT_ROLES:
            roots = [
                joint for joint in joints
                if nodes[joint.child_node].semantic_role == role
                and nodes[joint.parent_node].semantic_role != role
            ]
            if not roots:
                continue
            root_joint = roots[0]
            slots.append(
                FingerSlotIR(
                    slot_id=DIGIT_ROLES.index(role),
                    role=role,
                    active=True,
                    palm_node_id=root_joint.parent_node,
                    attachment_translation=root_joint.origin_translation,
                    root_bundle_id=nodes[root_joint.child_node].bundle_id,
                )
            )

        hands.append(
            HandIR(
                hand_id=source_id,
                source_hand_id=source_id,
                handedness="right",
                nodes=nodes,
                joints=joints,
                finger_slots=slots,
                metadata={
                    "display_name": source["display_name"],
                    "source_part_count": source["part_count"],
                    "source_movable_edges": source["movable_edges"],
                    "importer": "audited_source_graph_bridge_v0.1",
                },
            )
        )
    return hands, database
