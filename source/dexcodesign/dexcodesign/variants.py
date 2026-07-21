"""Deterministic, grammar-valid WUJI variants for compiler validation."""

from __future__ import annotations

from copy import deepcopy

from .schema import HandIR

DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def _variant_specs() -> list[dict]:
    return [
        {"name": "baseline"},
        {"name": "long_fingers", "length": 1.14},
        {"name": "short_fingers", "length": 0.86},
        {"name": "thick_fingers", "radial": 1.16},
        {"name": "thin_fingers", "radial": 0.86},
        {"name": "long_thick", "length": 1.10, "radial": 1.12},
        {"name": "short_thick", "length": 0.88, "radial": 1.14},
        {"name": "wide_palm", "palm": (1.18, 1.0, 1.0), "splay": 1.12},
        {"name": "narrow_palm", "palm": (0.84, 1.0, 1.0), "splay": 0.90},
        {"name": "long_thumb", "role_length": {"thumb": 1.20}},
        {"name": "short_thick_thumb", "role_length": {"thumb": 0.82}, "role_radial": {"thumb": 1.18}},
        {"name": "four_finger_no_pinky", "roles": ("thumb", "index", "middle", "ring")},
        {"name": "four_finger_no_ring", "roles": ("thumb", "index", "middle", "pinky")},
        {"name": "three_finger_precision", "roles": ("thumb", "index", "middle")},
        {"name": "three_finger_wide", "roles": ("thumb", "index", "pinky"), "splay": 1.14},
        {"name": "fixed_distal_normal", "fix_depth": {"index": 3, "middle": 3, "ring": 3, "pinky": 3}},
        {"name": "fixed_distal_all", "fix_depth": {role: 3 for role in DIGITS}},
        {"name": "three_link_normal", "max_parts": {"index": 3, "middle": 3, "ring": 3, "pinky": 3}},
        {"name": "splayed_fingers", "splay": 1.18, "role_length": {"index": 1.05, "pinky": 0.92}},
        {
            "name": "mixed_proportions",
            "role_length": {"thumb": 1.12, "index": 1.08, "middle": 1.02, "ring": 0.94, "pinky": 0.86},
            "role_radial": {"thumb": 1.10, "index": 0.96, "middle": 1.00, "ring": 1.06, "pinky": 1.12},
        },
    ]


def _role_depths(hand: HandIR) -> dict[int, int]:
    nodes = {node.node_id: node for node in hand.nodes}
    parent = {joint.child_node: joint.parent_node for joint in hand.joints}
    result = {}
    for node in hand.nodes:
        if node.semantic_role not in DIGITS:
            result[node.node_id] = 0
            continue
        depth = 1
        cursor = node.node_id
        while cursor in parent and nodes[parent[cursor]].semantic_role == node.semantic_role:
            depth += 1
            cursor = parent[cursor]
        result[node.node_id] = depth
    return result


def _apply_variant(source: HandIR, index: int, spec: dict) -> HandIR:
    hand = deepcopy(source)
    hand.hand_id = f"wuji_variant_{index:02d}_{spec['name']}"
    roles = set(spec.get("roles", DIGITS))
    depths = _role_depths(hand)
    keep = set()
    max_parts = spec.get("max_parts", {})
    for node in hand.nodes:
        if node.semantic_role in DIGITS:
            if node.semantic_role not in roles:
                continue
            if depths[node.node_id] > max_parts.get(node.semantic_role, 99):
                continue
        keep.add(node.node_id)

    old_nodes = {node.node_id: node for node in hand.nodes if node.node_id in keep}
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(old_nodes))}
    for old_id, node in old_nodes.items():
        node.node_id = id_map[old_id]
        if node.semantic_role in DIGITS:
            node.length_scale = float(spec.get("role_length", {}).get(node.semantic_role, spec.get("length", 1.0)))
            node.radial_scale = float(spec.get("role_radial", {}).get(node.semantic_role, spec.get("radial", 1.0)))
        else:
            node.palm_scale = tuple(float(value) for value in spec.get("palm", (1.0, 1.0, 1.0)))
    hand.nodes = [old_nodes[old_id] for old_id in sorted(old_nodes)]

    joints = []
    splay = float(spec.get("splay", 1.0))
    for joint in hand.joints:
        if joint.parent_node not in keep or joint.child_node not in keep:
            continue
        parent_role = next(node.semantic_role for node in source.nodes if node.node_id == joint.parent_node)
        child_role = next(node.semantic_role for node in source.nodes if node.node_id == joint.child_node)
        translation = list(joint.origin_translation)
        child_node = old_nodes[joint.child_node]
        if child_role in DIGITS:
            if parent_role == child_role:
                translation = [value * child_node.length_scale for value in translation]
            else:
                translation[0] *= splay * float(spec.get("palm", (1.0, 1.0, 1.0))[0])
                translation[1] *= float(spec.get("palm", (1.0, 1.0, 1.0))[1])
                translation[2] *= float(spec.get("palm", (1.0, 1.0, 1.0))[2])
        joint.parent_node = id_map[joint.parent_node]
        joint.child_node = id_map[joint.child_node]
        joint.origin_translation = tuple(translation)
        role_depth = depths.get(next(old_id for old_id, new_id in id_map.items() if new_id == joint.child_node), 0)
        if spec.get("fix_depth", {}).get(child_role) == role_depth:
            joint.joint_type = "fixed"
            joint.active = False
        joint.joint_id = len(joints)
        joints.append(joint)
    hand.joints = joints
    active_roles = {node.semantic_role for node in hand.nodes if node.semantic_role in DIGITS}
    hand.finger_slots = [slot for slot in hand.finger_slots if slot.role in active_roles]
    for slot in hand.finger_slots:
        slot.active = True
        root = next(
            joint for joint in hand.joints
            if hand.nodes[joint.child_node].semantic_role == slot.role
            and hand.nodes[joint.parent_node].semantic_role != slot.role
        )
        slot.palm_node_id = root.parent_node
        slot.attachment_translation = root.origin_translation
        slot.root_bundle_id = hand.nodes[root.child_node].bundle_id
    hand.metadata = {**hand.metadata, "variant_index": index, "variant_spec": spec}
    return hand


def build_wuji_demo_variants(source: HandIR) -> list[HandIR]:
    if source.source_hand_id != "wuji_hand_2":
        raise ValueError("The reference variant set is defined for WUJI Hand 2")
    return [_apply_variant(source, index + 1, spec) for index, spec in enumerate(_variant_specs())]
