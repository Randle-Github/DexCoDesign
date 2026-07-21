#!/usr/bin/env python3
"""Convert all registered bilateral hands into canonical padded graph records."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMP = ROOT / "temp"
sys.path.insert(0, str(TEMP))

from visualize_right_hands import RIGHT_HAND_LANDMARKS, load_all_hands  # noqa: E402


AUDIT = TEMP / "hand_structure_audit.json"
OUTPUT = HERE / "outputs" / "dataset.json"
MAX_NODES = 48

ROLE_NAMES = ["palm", "base", "thumb", "index", "middle", "ring", "pinky", "sensor", "other"]
JOINT_NAMES = ["fixed", "hinge", "slide", "ball", "free", "other"]
FEATURE_NAMES = [*(f"role:{x}" for x in ROLE_NAMES), *(f"joint:{x}" for x in JOINT_NAMES), "geom_count", "has_geom", "depth"]
TARGET_NAMES = [
    "palm_width",
    "palm_length",
    "palm_thickness",
    "finger_length",
    "thumb_length",
    "finger_radius",
    "base_length",
    "tip_scale",
    "finger_splay",
    "knuckle_arc",
    "digit_count_norm",
    "segment_count_norm",
]


def basename(value: str | None) -> str:
    if not value:
        return ""
    return value.rstrip("/").rsplit("/", 1)[-1]


def graph_records(audit: dict[str, object]) -> tuple[set[str], list[tuple[str, str]], dict[str, str], dict[str, float]]:
    """Extract nodes, undirected kinematic edges, child joint type and geometry count."""
    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    joint_for_child: dict[str, str] = {}
    geometry: dict[str, float] = {}

    if "body_counts" in audit:
        for record in audit["body_counts"]:
            name = str(record["name"])
            nodes.add(name)
            geometry[name] = float(record.get("geom_count", 0))
            parent = str(record.get("parent", "world"))
            if parent != "world":
                nodes.add(parent)
                edges.append((parent, name))
    elif "link_geometry_counts" in audit:
        for record in audit["link_geometry_counts"]:
            name = str(record["name"])
            nodes.add(name)
            geometry[name] = float(record.get("visual_count", 0)) + float(record.get("collision_count", 0))
    else:
        for path in audit.get("rigid_bodies", []):
            nodes.add(str(path))
            geometry[str(path)] = 1.0

    for joint in audit.get("joints", []):
        parent = joint.get("parent", joint.get("parent_body"))
        child = joint.get("child", joint.get("body"))
        if parent is None or child is None or str(parent) == "world":
            continue
        parent, child = str(parent), str(child)
        # USD uses full prim paths while some records use display names.
        if parent not in nodes:
            candidates = [node for node in nodes if basename(node) == basename(parent)]
            parent = candidates[0] if len(candidates) == 1 else parent
        if child not in nodes:
            candidates = [node for node in nodes if basename(node) == basename(child)]
            child = candidates[0] if len(candidates) == 1 else child
        nodes.update((parent, child))
        if parent != child:
            edges.append((parent, child))
        joint_for_child[child] = str(joint.get("type", "other")).lower()

    # Deduplicate while retaining only graph members.
    edges = list(dict.fromkeys((a, b) for a, b in edges if a in nodes and b in nodes and a != b))
    return nodes, edges, joint_for_child, geometry


def canonical_order(nodes: set[str], edges: list[tuple[str, str]], palm_name: str) -> tuple[list[str], dict[str, int]]:
    adjacency = {node: set() for node in nodes}
    for parent, child in edges:
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    palm_candidates = [node for node in nodes if basename(node) == palm_name]
    start = palm_candidates[0] if palm_candidates else sorted(nodes)[0]
    order: list[str] = []
    depth: dict[str, int] = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(adjacency[node], key=lambda x: basename(x).lower()):
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)
    for node in sorted(nodes.difference(order), key=lambda x: basename(x).lower()):
        depth[node] = max(depth.values(), default=0) + 1
        order.append(node)
    if len(order) > MAX_NODES:
        raise ValueError(f"Graph has {len(order)} nodes, capacity is {MAX_NODES}")
    return order, depth


def assign_roles(
    order: list[str],
    edges: list[tuple[str, str]],
    palm_name: str,
    root_to_palm: list[str],
    specification: tuple[str, list[str], list[str], str],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    lowered = {node: basename(node).lower() for node in order}
    palm_nodes = {node for node in order if basename(node) == palm_name}
    base_names = {basename(x) for x in root_to_palm}
    for node in order:
        name = lowered[node]
        if node in palm_nodes:
            roles[node] = "palm"
        elif basename(node) in base_names or any(token in name for token in ("wrist", "forearm", "mount", "base")):
            roles[node] = "base"
        elif any(token in name for token in ("sensor", "fsr", "imu")):
            roles[node] = "sensor"
        elif "thumb" in name or name.startswith(("th_", "lh_th", "rh_th")):
            roles[node] = "thumb"
        elif "index" in name or name.startswith(("ff", "lh_ff", "rh_ff")):
            roles[node] = "index"
        elif "middle" in name or name.startswith(("mf", "lh_mf", "rh_mf")):
            roles[node] = "middle"
        elif "ring" in name or name.startswith(("rf", "lh_rf", "rh_rf")):
            roles[node] = "ring"
        elif any(token in name for token in ("pinky", "little")) or name.startswith(("lf", "lh_lf", "rh_lf")):
            roles[node] = "pinky"
        else:
            roles[node] = "other"

    graph = {node: set() for node in order}
    for a, b in edges:
        if a in graph and b in graph:
            graph[a].add(b)
            graph[b].add(a)
    palm = next(iter(palm_nodes), order[0])
    distance = {palm: 0}
    queue = deque([palm])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                queue.append(nxt)

    # Anatomical landmarks fill anonymous industrial link names (notably ORCA).
    _, proximal_names, _, thumb_tip = specification
    slots = ["index", "middle", "ring", "pinky"][: len(proximal_names)]
    for landmark, role in zip(proximal_names, slots):
        candidates = [node for node in order if basename(node) == landmark]
        if not candidates:
            continue
        start = candidates[0]
        stack = [start]
        visited: set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited or node == palm:
                continue
            visited.add(node)
            if roles[node] in {"other", role}:
                roles[node] = role
                stack.extend(nxt for nxt in graph[node] if distance.get(nxt, 0) > distance.get(node, 0))

    thumb_candidates = [node for node in order if basename(node) == thumb_tip]
    if thumb_candidates:
        # Mark the shortest path from the thumb landmark back to palm.
        queue = deque([thumb_candidates[0]])
        parent = {thumb_candidates[0]: None}
        while queue and palm not in parent:
            node = queue.popleft()
            for nxt in graph[node]:
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if palm in parent:
            cursor = palm
            path = []
            while parent[cursor] is not None:
                path.append(parent[cursor])
                cursor = parent[cursor]
            for node in path:
                if node != palm and roles[node] == "other":
                    roles[node] = "thumb"
    return roles


def make_record(hand, audit: dict[str, object], specification) -> dict[str, object]:
    nodes, edges, joint_types, geometry = graph_records(audit)
    order, depths = canonical_order(nodes, edges, str(audit["palm"]))
    roles = assign_roles(order, edges, str(audit["palm"]), list(audit["root_to_palm_links"]), specification)
    index = {node: i for i, node in enumerate(order)}

    x = np.zeros((MAX_NODES, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros(MAX_NODES, dtype=np.float32)
    adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    for row, node in enumerate(order):
        mask[row] = 1.0
        x[row, ROLE_NAMES.index(roles[node])] = 1.0
        raw_joint = joint_types.get(node, "fixed")
        if "revolute" in raw_joint or "hinge" in raw_joint or "continuous" in raw_joint:
            joint = "hinge"
        elif "prismatic" in raw_joint or "slide" in raw_joint:
            joint = "slide"
        elif "ball" in raw_joint:
            joint = "ball"
        elif "free" in raw_joint or "floating" in raw_joint:
            joint = "free"
        elif "fixed" in raw_joint:
            joint = "fixed"
        else:
            joint = "other"
        x[row, len(ROLE_NAMES) + JOINT_NAMES.index(joint)] = 1.0
        x[row, -3] = min(geometry.get(node, 0.0), 8.0) / 8.0
        x[row, -2] = float(geometry.get(node, 0.0) > 0.0)
        x[row, -1] = min(depths[node], 12) / 12.0
    for a, b in edges:
        if a in index and b in index:
            adjacency[index[a], index[b]] = 1.0
            adjacency[index[b], index[a]] = 1.0

    mesh = hand.scene.to_geometry()
    vertices = np.asarray(mesh.vertices, dtype=float)
    canonical = (vertices - hand.root_origin) @ hand.canonical_basis
    extents = np.ptp(canonical, axis=0)
    extents = np.clip(extents, 0.05, 2.5)
    role_counts = {role: sum(value == role for value in roles.values()) for role in ROLE_NAMES}
    # The audited anatomical specification is authoritative for digit slots;
    # anonymous industrial link names are only a semantic-node heuristic.
    digit_count = 1 + len(specification[1])
    nonthumb = max(digit_count - 1, 2)
    digit_nodes = sum(role_counts[role] for role in ("thumb", "index", "middle", "ring", "pinky"))
    segment_count = np.clip(round(digit_nodes / max(digit_count, 1)), 2, 4)
    root_chain = max(len(audit.get("root_to_palm_links", [])) - 1, 0)

    palm_width = float(np.clip(0.46 * extents[0], 0.50, 1.05))
    non_digit_nodes = len(order) - digit_nodes
    palm_length = float(np.clip(0.52 + 0.015 * non_digit_nodes, 0.52, 0.82))
    palm_thickness = float(np.clip(0.55 * extents[1], 0.14, 0.38))
    dof_per_digit = float(audit["scalar_dofs"]) / max(digit_count, 1)
    finger_length = float(np.clip(0.46 + 0.095 * segment_count + 0.010 * dof_per_digit, 0.64, 0.98))
    mean_digit_nodes = digit_nodes / max(digit_count, 1)
    thumb_ratio = np.clip(0.68 + 0.025 * (role_counts["thumb"] - mean_digit_nodes), 0.62, 0.78)
    thumb_length = float(np.clip(thumb_ratio * finger_length + 0.08 * extents[0], 0.46, 0.88))
    finger_radius = float(np.clip(0.050 + 0.060 * palm_thickness + 0.030 * (palm_width - 0.5), 0.060, 0.095))
    base_length = float(np.clip(0.15 + 0.055 * root_chain, 0.12, 0.42))
    target = [
        palm_width,
        palm_length,
        palm_thickness,
        finger_length,
        thumb_length,
        finger_radius,
        base_length,
        float(np.clip(0.68 + 0.05 * (segment_count - 2), 0.65, 0.82)),
        float(np.clip(0.055 + 0.018 * (dof_per_digit - 2) + 0.05 * (palm_width - 0.5), 0.035, 0.17)),
        float(np.clip(0.025 + 0.014 * (segment_count - 2) + 0.004 * abs(dof_per_digit - 3), 0.02, 0.09)),
        digit_count / 5.0,
        segment_count / 4.0,
    ]
    return {
        "hand_id": hand.hand_id,
        "side": hand.side,
        "display_name": hand.display_name,
        "format": hand.source_format,
        "node_count": len(order),
        "edge_count": int(adjacency.sum() // 2),
        "digit_count": int(digit_count),
        "segment_count": int(segment_count),
        "scalar_dofs": int(audit["scalar_dofs"]),
        "actuators": int(audit.get("actuators", audit.get("driven_joints", audit.get("transmissions", 0)))),
        "constraints": int(audit.get("equalities", 0)) + int(audit.get("mimic_joints", 0)) + int(audit.get("tendons", 0)),
        "node_names": [basename(node) for node in order],
        "node_roles": [roles[node] for node in order],
        "x": x.tolist(),
        "adjacency": adjacency.tolist(),
        "mask": mask.tolist(),
        "target": target,
    }


def main() -> int:
    audit_data = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = []
    side = "right"
    print("Loading all right-hand source assets...", flush=True)
    for hand in load_all_hands(side):
        record = make_record(hand, audit_data[hand.hand_id][side], RIGHT_HAND_LANDMARKS[hand.hand_id])
        records.append(record)
        print(
            f"  graph {hand.hand_id:20s} {side}: nodes={record['node_count']:2d} "
            f"edges={record['edge_count']:2d} digits={record['digit_count']}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "max_nodes": MAX_NODES,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} right-hand graph samples to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
