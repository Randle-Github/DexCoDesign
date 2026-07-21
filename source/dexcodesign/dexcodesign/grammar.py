"""Strict grammar and source motor-link binding validation."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from .schema import HandIR, ModuleDatabase


class GrammarError(ValueError):
    """Raised when a HandIR cannot be compiled safely."""


def validate_hand(hand: HandIR, database: ModuleDatabase, repo_root: Path | None = None) -> dict[str, int]:
    nodes = {node.node_id: node for node in hand.nodes}
    if len(nodes) != len(hand.nodes):
        raise GrammarError("duplicate node ID")
    children: dict[int, list[int]] = defaultdict(list)
    parent_of: dict[int, int] = {}
    for joint in hand.joints:
        if joint.parent_node not in nodes or joint.child_node not in nodes:
            raise GrammarError(f"joint {joint.joint_id} references an inactive node")
        if joint.child_node in parent_of:
            raise GrammarError(f"node {joint.child_node} has multiple parents")
        parent_of[joint.child_node] = joint.parent_node
        children[joint.parent_node].append(joint.child_node)
        child = nodes[joint.child_node]
        if joint.motor_link_binding_id != child.bundle_id:
            raise GrammarError("joint binding does not match the child source mechanism bundle")

    roots = [node_id for node_id in nodes if node_id not in parent_of]
    if len(roots) != 1:
        raise GrammarError(f"expected one root, got {len(roots)}")
    visited: set[int] = set()
    queue = deque(roots)
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            raise GrammarError("cycle detected")
        visited.add(node_id)
        queue.extend(children[node_id])
    if visited != set(nodes):
        raise GrammarError("disconnected graph")

    for node in hand.nodes:
        if node.bundle_id not in database.bundles:
            raise GrammarError(f"unknown bundle {node.bundle_id}")
        bundle = database.bundles[node.bundle_id]
        if bundle.source_hand_id != hand.source_hand_id:
            raise GrammarError("cross-source mechanism bundle is forbidden in this stage")
        if node.candidate_id is None:
            continue
        if node.candidate_id not in database.candidates:
            raise GrammarError(f"unknown candidate {node.candidate_id}")
        candidate = database.candidates[node.candidate_id]
        if candidate.bundle_id != bundle.bundle_id:
            raise GrammarError("candidate is not owned by selected motor bundle")
        if node.candidate_id not in bundle.link_candidate_ids:
            raise GrammarError("candidate is outside the original motor-link binding")
        low, high = candidate.deformation_bounds
        if not low <= node.length_scale <= high or not low <= node.radial_scale <= high:
            raise GrammarError("candidate deformation is outside its source-specific bounds")
        if repo_root is not None and candidate.visual_path is not None:
            if not (repo_root / candidate.visual_path).is_file():
                raise GrammarError(f"missing source candidate mesh: {candidate.visual_path}")

    movable = sum(joint.active and joint.joint_type not in {"fixed", "none"} for joint in hand.joints)
    fingers = len({node.semantic_role for node in hand.nodes if node.finger_slot is not None})
    return {"nodes": len(nodes), "joints": len(hand.joints), "dof": movable, "fingers": fingers}
