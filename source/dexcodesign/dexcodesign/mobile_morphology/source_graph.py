"""Convert the three EEF-free URDFs into mesh-filled canonical link graphs."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from . import SCHEMA_VERSION
from .common import (
    ARTIFACT_ROOT,
    REFERENCE_GRAPHS,
    ROOT,
    SPECS,
    numbers,
    rpy_matrix,
    transform,
)


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def build_graph(spec) -> dict:
    robot = ET.parse(spec.urdf).getroot()
    links = {link.get("name", ""): link for link in robot.findall("link")}
    arm_links = {name for pair in spec.arm_link_pairs for name in pair}
    mutable_links = arm_links | set(spec.torso_links)
    joints = []
    child_links = set()
    by_parent: dict[str, list[dict]] = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if parent is None or child is None:
            continue
        record = {
            "name": joint.get("name", ""),
            "type": joint.get("type", "fixed"),
            "parent": parent.get("link", ""),
            "child": child.get("link", ""),
            "origin_xyz": numbers(None if origin is None else origin.get("xyz")).tolist(),
            "origin_rpy": numbers(None if origin is None else origin.get("rpy")).tolist(),
        }
        axis = joint.find("axis")
        if axis is not None:
            record["axis"] = numbers(axis.get("xyz"), (1, 0, 0)).tolist()
        joints.append(record)
        by_parent.setdefault(record["parent"], []).append(record)
        child_links.add(record["child"])
    roots = sorted(set(links) - child_links)
    if len(roots) != 1:
        raise ValueError(f"{spec.urdf}: expected one root, found {roots}")

    worlds = {roots[0]: np.eye(4)}
    queue = [roots[0]]
    while queue:
        parent = queue.pop(0)
        for joint in by_parent.get(parent, ()):
            worlds[joint["child"]] = worlds[parent] @ transform(
                joint["origin_xyz"], joint["origin_rpy"]
            )
            queue.append(joint["child"])
    if set(worlds) != set(links):
        raise ValueError(f"{spec.urdf}: graph is disconnected")

    node_records = []
    for node_id, (name, link) in enumerate(links.items()):
        visuals = []
        local_visual_vertices = []
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh_element = visual.find("./geometry/mesh")
            if mesh_element is None:
                continue
            source_path = (spec.urdf.parent / mesh_element.get("filename", "")).resolve()
            mesh = trimesh.load_mesh(source_path, process=False)
            visual_origin = visual.find("origin")
            mesh_scale = numbers(
                mesh_element.get("scale"), (1, 1, 1)
            )
            origin_xyz = numbers(
                None
                if visual_origin is None
                else visual_origin.get("xyz")
            )
            origin_rpy = numbers(
                None
                if visual_origin is None
                else visual_origin.get("rpy")
            )
            local_visual_vertices.append(
                (
                    np.asarray(mesh.vertices, dtype=float) * mesh_scale
                )
                @ rpy_matrix(origin_rpy).T
                + origin_xyz
            )
            visuals.append(
                {
                    "visual_index": visual_index,
                    "source_mesh": relative_to_root(source_path),
                    "mesh_scale": mesh_scale.tolist(),
                    "origin_xyz": origin_xyz.tolist(),
                    "origin_rpy": origin_rpy.tolist(),
                    "faces": int(len(mesh.faces)),
                    "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                }
            )
        connector_geometry = []
        if local_visual_vertices:
            vertices = np.concatenate(local_visual_vertices)
            for joint in by_parent.get(name, ()):
                connector = np.asarray(
                    joint["origin_xyz"], dtype=float
                )
                span = float(np.linalg.norm(connector))
                if span < 1.0e-9:
                    connector_geometry.append(
                        {
                            "joint": joint["name"],
                            "span": 0.0,
                            "transverse_diameter": None,
                            "slenderness": 0.0,
                        }
                    )
                    continue
                axis = connector / span
                seed = (
                    np.asarray((1.0, 0.0, 0.0))
                    if abs(float(axis[0])) < 0.9
                    else np.asarray((0.0, 1.0, 0.0))
                )
                transverse_1 = np.cross(axis, seed)
                transverse_1 /= np.linalg.norm(transverse_1)
                transverse_2 = np.cross(axis, transverse_1)
                diameter = max(
                    float(np.ptp(vertices @ transverse_1)),
                    float(np.ptp(vertices @ transverse_2)),
                )
                connector_geometry.append(
                    {
                        "joint": joint["name"],
                        "span": span,
                        "transverse_diameter": diameter,
                        "slenderness": span / max(diameter, 1.0e-9),
                    }
                )
        node_records.append(
            {
                "id": node_id,
                "name": name,
                "role": (
                    "arm"
                    if name in arm_links
                    else "torso"
                    if name in spec.torso_links
                    else "immutable"
                ),
                "immutable": name not in mutable_links,
                "world_zero": worlds[name].tolist(),
                "visuals": visuals,
                "connector_geometry": connector_geometry,
                "outgoing_joints": [joint["name"] for joint in by_parent.get(name, ())],
            }
        )
    return {
        "source_id": spec.source_id,
        "source_urdf": relative_to_root(spec.urdf),
        "root_link": roots[0],
        "links": node_records,
        "joints": joints,
        "display": {
            "yaw_degrees": spec.display_yaw_degrees,
            "joint_positions": dict(spec.display_joint_positions),
        },
        "audit": {
            "links": len(node_records),
            "joints": len(joints),
            "meshed_links": sum(bool(link["visuals"]) for link in node_records),
            "immutable_links": sum(link["immutable"] for link in node_records),
        },
    }


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    graphs = [build_graph(spec) for spec in SPECS]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "URDF link-joint graph filled with source visual meshes",
        "graphs": graphs,
        "summary": {
            "robots": len(graphs),
            "links": sum(graph["audit"]["links"] for graph in graphs),
            "joints": sum(graph["audit"]["joints"] for graph in graphs),
        },
    }
    REFERENCE_GRAPHS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
