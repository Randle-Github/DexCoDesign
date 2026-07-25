"""Build the length-only grammar for supported mobile manipulators."""

from __future__ import annotations

import json

import numpy as np

from . import SCHEMA_VERSION
from .common import GRAMMAR, REFERENCE_GRAPHS, SPECS, transform


def joint_motion(joint: dict, position: float) -> np.ndarray:
    result = np.eye(4)
    if joint["type"] == "fixed":
        return result
    axis = np.asarray(joint.get("axis", (1, 0, 0)), dtype=float)
    axis /= np.linalg.norm(axis)
    if joint["type"] == "prismatic":
        result[:3, 3] = position * axis
        return result
    if joint["type"] not in {"revolute", "continuous"}:
        raise ValueError(f"unsupported joint type: {joint['type']}")
    x, y, z = axis
    cosine, sine = np.cos(position), np.sin(position)
    cross = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    result[:3, :3] = (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer(axis, axis)
        + sine * cross
    )
    return result


def reference_worlds(spec, graph: dict) -> dict[str, np.ndarray]:
    """Evaluate link frames in the original upright display pose."""
    positions = dict(spec.display_joint_positions)
    by_parent: dict[str, list[dict]] = {}
    children = set()
    links = {link["name"] for link in graph["links"]}
    for joint in graph["joints"]:
        by_parent.setdefault(joint["parent"], []).append(joint)
        children.add(joint["child"])
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"{spec.source_id}: expected one root")
    worlds = {roots[0]: np.eye(4)}
    queue = [roots[0]]
    while queue:
        parent = queue.pop(0)
        for joint in by_parent.get(parent, ()):
            worlds[joint["child"]] = (
                worlds[parent]
                @ transform(joint["origin_xyz"], joint["origin_rpy"])
                @ joint_motion(joint, positions.get(joint["name"], 0.0))
            )
            queue.append(joint["child"])
    return worlds


def directional_length_contract(
    link: str,
    outgoing: dict[str, list[dict]],
    worlds: dict[str, np.ndarray],
    world_direction: np.ndarray,
) -> dict | None:
    """Use one source-pose world direction for mesh and graph-edge scaling."""
    rotation = worlds[link][:3, :3]
    local_axis = rotation.T @ np.asarray(world_direction, dtype=float)
    candidates = list(outgoing.get(link, ()))
    if not candidates:
        return None
    connector = max(
        candidates,
        key=lambda joint: abs(
            float(
                world_direction
                @ (
                    rotation
                    @ np.asarray(joint["origin_xyz"], dtype=float)
                )
            )
        ),
    )
    connector_vector = np.asarray(connector["origin_xyz"], dtype=float)
    span = abs(
        float(world_direction @ (rotation @ connector_vector))
    )
    if span < 1.0e-6:
        return None
    return {
        "link": link,
        "deformation_axis": local_axis.tolist(),
        "reference_world_direction": np.asarray(
            world_direction, dtype=float
        ).tolist(),
        "reference_direction_span": span,
        "connector_joint": connector["name"],
        "connector_vector": connector_vector.tolist(),
        "connector_is_movable": connector["type"]
        in {"revolute", "continuous", "prismatic"},
        "reference_pose": "original upright display pose",
        "transforms_all_outgoing_edges": True,
    }


def shoulder_carrier_contract(
    spec,
    graph: dict,
    incoming: dict[str, dict],
    worlds: dict[str, np.ndarray],
) -> dict:
    """Find the nearest meshed common ancestor carrying both shoulders."""
    joint_by_name = {joint["name"]: joint for joint in graph["joints"]}
    shoulder_joints = [
        joint_by_name[name] for name in spec.arm_joint_pairs[0]
    ]
    shoulder_parents = [joint["parent"] for joint in shoulder_joints]

    def ancestors(link: str) -> list[str]:
        result = [link]
        while link in incoming:
            link = incoming[link]["parent"]
            result.append(link)
        return result

    left_chain = ancestors(shoulder_parents[0])
    right_ancestors = set(ancestors(shoulder_parents[1]))
    carrier = next(link for link in left_chain if link in right_ancestors)
    link_records = {link["name"]: link for link in graph["links"]}
    while not link_records[carrier]["visuals"]:
        if carrier not in incoming:
            raise ValueError(
                f"{spec.source_id}: shoulder carrier has no meshed ancestor"
            )
        carrier = incoming[carrier]["parent"]

    attachment_joint_names = []
    for shoulder_joint in shoulder_joints:
        path = [shoulder_joint["name"]]
        parent = shoulder_joint["parent"]
        while parent != carrier:
            edge = incoming[parent]
            path.append(edge["name"])
            parent = edge["parent"]
        attachment_joint_names.extend(reversed(path))
    attachment_joint_names = list(dict.fromkeys(attachment_joint_names))
    world_lateral = np.asarray((0.0, 1.0, 0.0))
    attachment_edges = []
    for name in attachment_joint_names:
        joint = joint_by_name[name]
        parent_rotation = worlds[joint["parent"]][:3, :3]
        attachment_edges.append(
            {
                "joint": name,
                "parent_link": joint["parent"],
                "deformation_axis": (
                    parent_rotation.T @ world_lateral
                ).tolist(),
            }
        )
    carrier_axis = worlds[carrier][:3, :3].T @ world_lateral
    return {
        "link": carrier,
        "side": "midline",
        "deformation_axis": carrier_axis.tolist(),
        "reference_world_direction": world_lateral.tolist(),
        "reference_pose": "original upright display pose",
        "attachment_edges": attachment_edges,
        "shoulder_anchor_joints": list(spec.arm_joint_pairs[0]),
    }


def build_robot_grammar(spec, graph: dict) -> dict:
    outgoing: dict[str, list[dict]] = {}
    incoming: dict[str, dict] = {}
    for joint in graph["joints"]:
        outgoing.setdefault(joint["parent"], []).append(joint)
        incoming[joint["child"]] = joint
    link_names = {link["name"] for link in graph["links"]}
    joint_names = {joint["name"] for joint in graph["joints"]}
    link_world = reference_worlds(spec, graph)

    length_groups = []
    for left, right in spec.arm_link_pairs[1:]:
        if left not in link_names or right not in link_names:
            continue
        left_contract = directional_length_contract(
            left, outgoing, link_world, np.asarray((0.0, 0.0, 1.0))
        )
        right_contract = directional_length_contract(
            right, outgoing, link_world, np.asarray((0.0, 0.0, 1.0))
        )
        if left_contract is None or right_contract is None:
            continue
        terminal_only = not (
            left_contract["connector_is_movable"]
            and right_contract["connector_is_movable"]
        )
        length_groups.append(
            {
                "group_id": f"arm_length:{left}|{right}",
                "kind": "bilateral_arm_pair",
                "factors": [1.5] if terminal_only else [0.5, 1.5],
                "members": [
                    {**left_contract, "side": "left"},
                    {**right_contract, "side": "right"},
                ],
            }
        )
    for link in spec.torso_links:
        contract = directional_length_contract(
            link, outgoing, link_world, np.asarray((0.0, 0.0, 1.0))
        )
        if contract is not None:
            length_groups.append(
                {
                    "group_id": f"torso_length:{link}",
                    "kind": "midline_single",
                    "factors": (
                        [0.5, 1.5]
                        if contract["connector_is_movable"]
                        else [1.5]
                    ),
                    "members": [{**contract, "side": "midline"}],
                }
            )

    shoulder_width_contract = shoulder_carrier_contract(
        spec, graph, incoming, link_world
    )
    shoulder_width_groups = [
        {
            "group_id": (
                f"shoulder_width:{shoulder_width_contract['link']}"
            ),
            "kind": "midline_shoulder_carrier",
            "members": [shoulder_width_contract],
        }
    ]

    mutable_links = {
        member["link"]
        for group in length_groups
        for member in group["members"]
    } | {shoulder_width_contract["link"]}
    mutable_joints = {
        joint["name"]
        for group in length_groups
        for member in group["members"]
        for joint in outgoing.get(member["link"], ())
    }
    mutable_joints |= {
        edge["joint"]
        for edge in shoulder_width_contract["attachment_edges"]
    }
    return {
        "source_id": spec.source_id,
        "root_link": graph["root_link"],
        "productions": {
            "vertical_link_length": {
                "factors": [0.5, 1.5],
                "scale_other_axes": 1.0,
                "reference_direction": (
                    "world +Z in original upright pose"
                ),
                "implementation": (
                    "root-outward connector-span rig: lock the proximal "
                    "mesh cap, move the distal cap with its graph edge, "
                    "and deform only the middle span"
                ),
                "runtime_mesh_deformation": False,
                "groups": length_groups,
            },
            "shoulder_width": {
                "factors": [0.75, 1.25],
                "scale_other_axes": 1.0,
                "reference_direction": (
                    "world +/-Y in original upright pose"
                ),
                "changes_only_shoulder_carrier_width": True,
                "moves_complete_left_right_attachment_paths": True,
                "runtime_mesh_deformation": False,
                "groups": shoulder_width_groups,
            },
        },
        "hard_constraints": {
            "default_deny": True,
            "base_is_immutable": True,
            "immutable_links": sorted(link_names - mutable_links),
            "immutable_joints": sorted(joint_names - mutable_joints),
            "joint_origin_rpy_is_immutable": True,
            "joint_axis_is_immutable": True,
            "arm_edits_are_atomic_bilateral_pairs": True,
            "bilateral_mirror_plane": "y=0",
            "ordinary_length_changes_transverse_size": False,
            "ordinary_length_world_direction": (
                "original-pose vertical only"
            ),
            "only_lateral_exception": "shoulder_width",
            "eef_is_empty": True,
        },
    }


def main() -> int:
    graphs_payload = json.loads(
        REFERENCE_GRAPHS.read_text(encoding="utf-8")
    )
    graph_by_id = {
        graph["source_id"]: graph for graph in graphs_payload["graphs"]
    }
    grammars = [
        build_robot_grammar(spec, graph_by_id[spec.source_id])
        for spec in SPECS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "grammar_id": "mobile-manipulator-grammar-v11-length-only",
        "semantics": (
            "default-deny graph rewrite grammar; base frozen; arm length "
            "edits are bilateral; ordinary length is original-pose vertical "
            "only; the meshed common shoulder carrier alone permits lateral "
            "width; joint RPY, joint axes, and all arm-rotation productions "
            "are excluded"
        ),
        "robots": grammars,
    }
    GRAMMAR.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                robot["source_id"]: {
                    name: len(production["groups"])
                    for name, production in robot[
                        "productions"
                    ].items()
                }
                for robot in grammars
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
