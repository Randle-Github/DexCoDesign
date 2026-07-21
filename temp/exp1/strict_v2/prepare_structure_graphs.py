#!/usr/bin/env python3
"""Create mesh-free rigid-part graphs from the low-loss source hand library.

Fixed-joint bodies are contracted into one rigid part. Every movable joint is
kept as an edge between distinct parts. Source meshes are only concatenated
per rigid part for future reattachment; they are not used by the structure
model or graph renderer.
"""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent
SOURCE = EXP1 / "outputs"
OUTPUTS = HERE / "outputs"
LIBRARY = SOURCE / "mesh_library.json"
PART_ROOT = OUTPUTS / "source_rigid_parts"
GRAPH_PATH = OUTPUTS / "source_structure_graphs.json"
ASSET_ROOT = HERE.parents[2] / "assets" / "robot_hands"
REGISTRY = ASSET_ROOT / "registry.json"

ROLES = ["palm", "base", "thumb", "index", "middle", "ring", "pinky", "other"]
JOINT_TYPES = ["fixed", "hinge", "slide", "ball"]
ROLE_PRIORITY = {role: index for index, role in enumerate(ROLES)}
DIGIT_ROLES = {"thumb", "index", "middle", "ring", "pinky"}


class DSU:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def infer_role(node: dict, hand_id: str = "") -> str:
    role = node.get("role", "other")
    name = str(node.get("name", "")).lower()
    if "thumb" in name or name.startswith(("rh_th", "lh_th", "r_thumb", "l_thumb")):
        return "thumb"
    if "index" in name or name.startswith(("rh_ff", "lh_ff", "r_index", "l_index")):
        return "index"
    # Match the digit identity before generic segment words such as "middle".
    # Examples: r_ring_finger_middle and rh_rfmiddle are ring parts, not the
    # middle finger.  This distinction is structural, so it must be correct
    # before the graph is learned.
    if "pinky" in name or "little" in name or name.startswith(("rh_lf", "lh_lf", "r_pinky", "l_pinky")):
        return "pinky"
    if "ring" in name or name.startswith(("rh_rf", "lh_rf", "r_ring", "l_ring")):
        if hand_id == "ruka_v2" and name == "knuckle_ring_ext_2":
            return "pinky"
        return "ring"
    if "middle" in name or name.startswith(("rh_mf", "lh_mf", "r_middle", "l_middle")):
        return "middle"
    return role if role in ROLES else "other"


def normalize_joint(value: str) -> str:
    value = value.lower()
    if value in {"revolute", "continuous", "hinge"}:
        return "hinge"
    if value in {"prismatic", "slide"}:
        return "slide"
    if value in {"ball", "spherical"}:
        return "ball"
    return "fixed"


def joint_axis_and_range(name: str, role: str, joint_type: str, index: int) -> tuple[list[float], list[float]]:
    lowered = name.lower()
    if joint_type == "fixed":
        return [0.0, 0.0, 0.0], [0.0, 0.0]
    if joint_type == "slide":
        return [0.0, 0.0, 1.0], [-0.25, 0.25]
    if any(token in lowered for token in ("splay", "abad", "abduction", "rota", "yaw")):
        return [0.0, 1.0, 0.0], [-0.55, 0.55]
    if "roll" in lowered:
        return [0.0, 0.0, 1.0], [-0.65, 0.65]
    if role in {"base", "palm"}:
        return ([1.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 1.0, 0.0]), [-0.65, 0.65]
    return [1.0, 0.0, 0.0], [-0.15, 1.55]


def numbers(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(item) for item in value.replace(",", " ").split()]


def urdf_joint_metadata(path: Path) -> dict[str, dict]:
    result = {}
    root = ET.parse(path).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or not child.get("link"):
            continue
        kind = normalize_joint(joint.get("type", "fixed"))
        axis_element = joint.find("axis")
        axis = numbers(axis_element.get("xyz") if axis_element is not None else None, [1.0, 0.0, 0.0])
        limit = joint.find("limit")
        if kind == "fixed":
            bounds = [0.0, 0.0]
        elif joint.get("type") == "continuous":
            bounds = [-math.pi, math.pi]
        elif limit is not None:
            bounds = [float(limit.get("lower", -math.pi)), float(limit.get("upper", math.pi))]
        else:
            bounds = [-math.pi, math.pi]
        result[child.get("link")] = {
            "name": joint.get("name", child.get("link")), "type": kind,
            "axis": axis, "range": bounds, "source": "source_urdf",
        }
    return result


def mjcf_joint_metadata(path: Path, seen: set[Path] | None = None) -> dict[str, dict]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return {}
    seen.add(path)
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    degrees = compiler is not None and compiler.get("angle", "degree").lower() == "degree"
    global_default: dict[str, str] = {}
    class_defaults: dict[str, dict[str, str]] = {}
    for default in root.iter("default"):
        joint = default.find("joint")
        if joint is None:
            continue
        target = class_defaults.setdefault(default.get("class", ""), {}) if default.get("class") else global_default
        target.update(joint.attrib)
    result = {}
    for body in root.iter("body"):
        name = body.get("name")
        joints = body.findall("joint")
        if not name or not joints:
            continue
        joint = joints[0]
        attributes = dict(global_default)
        attributes.update(class_defaults.get(joint.get("class", ""), {}))
        attributes.update(joint.attrib)
        kind = normalize_joint(attributes.get("type", "hinge"))
        axis = numbers(attributes.get("axis"), [0.0, 0.0, 1.0])
        bounds = numbers(attributes.get("range"), [-math.pi, math.pi])
        if degrees and kind in {"hinge", "ball"}:
            bounds = [math.radians(value) for value in bounds]
        result[name] = {
            "name": attributes.get("name", name), "type": kind,
            "axis": axis, "range": bounds, "source": "source_mjcf",
        }
    for include in root.iter("include"):
        filename = include.get("file")
        if filename:
            result.update(mjcf_joint_metadata(path.parent / filename, seen))
    return result


def usd_joint_metadata(path: Path) -> dict[str, dict]:
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return {}
    stage = Usd.Stage.Open(str(path))
    result = {}
    axes = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
    for prim in stage.Traverse():
        kind = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            schema, kind = UsdPhysics.RevoluteJoint(prim), "hinge"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            schema, kind = UsdPhysics.PrismaticJoint(prim), "slide"
        elif prim.IsA(UsdPhysics.SphericalJoint):
            schema, kind = UsdPhysics.SphericalJoint(prim), "ball"
        if kind is None:
            continue
        targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if not targets:
            continue
        child = targets[0].name
        axis = axes.get(str(schema.GetAxisAttr().Get()), [1.0, 0.0, 0.0])
        lower, upper = schema.GetLowerLimitAttr().Get(), schema.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            bounds = [-math.pi, math.pi] if kind != "slide" else [-0.25, 0.25]
        elif kind == "slide":
            bounds = [float(lower), float(upper)]
        else:
            # USD Physics angular limits are authored in degrees.
            bounds = [math.radians(float(lower)), math.radians(float(upper))]
        result[child] = {
            "name": prim.GetName(), "type": kind, "axis": axis,
            "range": bounds, "source": "source_usd",
        }
    return result


def load_joint_metadata() -> dict[str, dict[str, dict]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["hands"]
    result = {}
    for hand_id, record in registry.items():
        entry = record["entries"]["right"]
        path = ASSET_ROOT / entry["path"]
        if entry["format"] == "urdf":
            result[hand_id] = urdf_joint_metadata(path)
        elif entry["format"] == "mjcf":
            result[hand_id] = mjcf_joint_metadata(path)
        elif entry["format"] == "usd":
            result[hand_id] = usd_joint_metadata(path)
        else:
            result[hand_id] = {}
    return result


def choose_role(members: list[dict], boundary: dict | None, is_root: bool, hand_id: str = "") -> str:
    roles = [infer_role(node, hand_id) for node in members if node.get("role") != "sensor"]
    if is_root and "palm" in roles:
        return "palm"
    if boundary is not None:
        boundary_role = infer_role(boundary, hand_id)
        if boundary_role != "other":
            return boundary_role
    counts = Counter(roles or ["other"])
    return min(counts, key=lambda role: (-counts[role], ROLE_PRIORITY[role]))


def merge_part_mesh(hand_id: str, part_id: int, members: list[dict], anchor: np.ndarray) -> dict | None:
    meshes = []
    for node in members:
        if not node.get("mesh"):
            continue
        mesh = trimesh.load(SOURCE / node["mesh"], force="mesh", process=False)
        mesh = mesh.copy()
        mesh.apply_translation(np.asarray(node["pivot"], dtype=float) - anchor)
        meshes.append(mesh)
    if not meshes:
        return None
    combined = trimesh.util.concatenate(meshes)
    combined.remove_unreferenced_vertices()
    path = PART_ROOT / hand_id / f"part_{part_id:02d}.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(path, file_type="obj", include_normals=True, include_color=False)
    bounds = np.asarray(combined.bounds, dtype=float)
    return {
        "file": str(path.relative_to(OUTPUTS)),
        "faces": int(len(combined.faces)),
        "bounds": bounds.tolist(),
        "size": np.maximum(bounds[1] - bounds[0], 1.0e-4).tolist(),
        "centroid_offset": np.asarray(combined.centroid, dtype=float).tolist(),
    }


def contract_hand(hand_id: str, hand: dict, joint_metadata: dict[str, dict]) -> dict:
    nodes = [
        node for node in hand["nodes"]
        if not (hand_id == "schunk_svh" and str(node["name"]).startswith("R_forearm_"))
    ]
    by_index = {int(node["index"]): node for node in nodes}
    dsu = DSU(max(by_index) + 1)
    for node in nodes:
        parent = node["parent"]
        if parent is not None and normalize_joint(node["joint_type"]) == "fixed":
            dsu.union(int(parent), int(node["index"]))

    raw_groups: dict[int, list[int]] = {}
    for index in by_index:
        raw_groups.setdefault(dsu.find(index), []).append(index)
    group_of = {index: root for root, members in raw_groups.items() for index in members}

    boundaries: dict[int, tuple[int, int]] = {}
    children: dict[int, set[int]] = {root: set() for root in raw_groups}
    roots = []
    for node in nodes:
        child_index = int(node["index"])
        child_group = group_of[child_index]
        parent_index = node["parent"]
        if parent_index is None:
            roots.append(child_group)
            continue
        parent_group = group_of[int(parent_index)]
        if parent_group != child_group:
            children[parent_group].add(child_group)
            boundaries[child_group] = (int(parent_index), child_index)
    root_group = roots[0]

    anchors: dict[int, np.ndarray] = {}
    for root, member_indices in raw_groups.items():
        members = [by_index[index] for index in member_indices]
        if root == root_group:
            palm = next((node for node in members if node["role"] == "palm"), members[0])
            anchors[root] = np.asarray(palm["pivot"], dtype=float)
        else:
            anchors[root] = np.asarray(by_index[boundaries[root][1]]["pivot"], dtype=float)

    def group_key(group: int) -> tuple:
        boundary = by_index[boundaries[group][1]] if group in boundaries else None
        role = choose_role([by_index[index] for index in raw_groups[group]], boundary, group == root_group, hand_id)
        anchor = anchors[group]
        return ROLE_PRIORITY[role], -float(anchor[2]), float(anchor[0]), float(anchor[1])

    order, parent_group, depth = [], {root_group: None}, {root_group: 0}
    queue = deque([root_group])
    while queue:
        group = queue.popleft()
        order.append(group)
        for child in sorted(children[group], key=group_key):
            parent_group[child] = group
            depth[child] = depth[group] + 1
            queue.append(child)
    group_index = {group: index for index, group in enumerate(order)}

    root_anchor = anchors[root_group].copy()
    records = []
    for group in order:
        part_id = group_index[group]
        members = [by_index[index] for index in sorted(raw_groups[group])]
        boundary_node = by_index[boundaries[group][1]] if group in boundaries else None
        role = choose_role(members, boundary_node, group == root_group, hand_id)
        anchor = anchors[group]
        parent = parent_group[group]
        relative = np.zeros(3) if parent is None else anchor - anchors[parent]
        joint_type = "fixed" if boundary_node is None else normalize_joint(boundary_node["joint_type"])
        joint_name = members[0]["name"] if boundary_node is None else boundary_node["name"]
        metadata = joint_metadata.get(joint_name) if boundary_node is not None else None
        if metadata is not None:
            joint_name = metadata["name"]
            joint_type = metadata["type"]
            axis, joint_range = metadata["axis"], metadata["range"]
            metadata_source = metadata["source"]
        else:
            axis, joint_range = joint_axis_and_range(joint_name, role, joint_type, part_id)
            metadata_source = "root" if boundary_node is None else "heuristic_fallback"
        mesh = merge_part_mesh(hand_id, part_id, members, anchor)
        size = mesh["size"] if mesh else [0.10, 0.10, max(float(np.linalg.norm(relative)), 0.10)]
        centroid_offset = mesh["centroid_offset"] if mesh else [0.0, 0.0, 0.0]
        records.append(
            {
                "id": part_id,
                "parent": None if parent is None else group_index[parent],
                "role": role,
                "joint_type": joint_type,
                "joint_name": joint_name,
                "joint_axis": axis,
                "joint_range": joint_range,
                "joint_metadata_source": metadata_source,
                "joint_axis_frame": "joint_local",
                "relative_pos": relative.tolist(),
                "edge_length": float(np.linalg.norm(relative)),
                "world_pos": (anchor - root_anchor).tolist(),
                "part_size": size,
                "centroid_offset": centroid_offset,
                "depth": depth[group],
                "member_links": [node["name"] for node in members],
                "mesh": mesh,
            }
        )

    child_counts = Counter(node["parent"] for node in records if node["parent"] is not None)
    for node in records:
        node["child_count"] = int(child_counts[node["id"]])
        node["is_leaf"] = node["child_count"] == 0
    return {
        "hand_id": hand_id,
        "display_name": hand["display_name"],
        "digit_count": hand["digit_count"],
        "source_scalar_dofs": hand["scalar_dofs"],
        "part_count": len(records),
        "movable_edges": sum(node["joint_type"] != "fixed" for node in records),
        "parts": records,
    }


def main() -> int:
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    joint_metadata = load_joint_metadata()
    if PART_ROOT.exists():
        shutil.rmtree(PART_ROOT)
    hands = [contract_hand(hand_id, hand, joint_metadata.get(hand_id, {})) for hand_id, hand in library["hands"].items()]
    max_parts = max(hand["part_count"] for hand in hands)
    payload = {
        "schema_version": 1,
        "representation": "fixed-joint-contracted rigid-part tree; mesh-free structure attributes",
        "roles": ROLES,
        "joint_types": JOINT_TYPES,
        "max_parts": max_parts,
        "hands": hands,
    }
    GRAPH_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for hand in hands:
        print(
            f"{hand['hand_id']:20s}: source bodies={len(library['hands'][hand['hand_id']]['nodes']):2d} "
            f"rigid parts={hand['part_count']:2d} movable_edges={hand['movable_edges']:2d}",
            flush=True,
        )
    print(f"Wrote {GRAPH_PATH}; max_parts={max_parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

