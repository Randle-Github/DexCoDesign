"""Compile cached graph rewrites to complete URDFs without runtime mesh edits."""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION
from .common import (
    COMPILED_ROBOTS,
    GENERATED_ROOT,
    GRAMMAR,
    PRECOMPUTED_VARIANTS,
    ROBOT_IR,
    ROOT,
    SPEC_BY_ID,
    fmt,
    longitudinal_linear,
    numbers,
    rpy_matrix,
)
from .preprocess import variant_key
from .display_pose import solve_hanging_display_positions


def ensure_origin(element: ET.Element) -> ET.Element:
    origin = element.find("origin")
    if origin is None:
        origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        element.insert(0, origin)
    return origin


def apply_cached_mesh_variant(
    link: ET.Element,
    variant: dict,
) -> int:
    """Point mesh owners at a precomputed hand-style rigid-node variant."""
    applied = 0
    for record in variant["mesh_owners"]:
        owners = link.findall(record["owner_kind"])
        owner = owners[int(record["owner_index"])]
        mesh = owner.find("./geometry/mesh")
        if mesh is None:
            raise ValueError(
                f"{link.get('name')}: cached owner is no longer a mesh"
            )
        origin = ensure_origin(owner)
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        mesh.set("filename", str((ROOT / record["file"]).resolve()))
        mesh.set("scale", "1 1 1")
        applied += 1
    return applied


def transform_nonmesh_geometry(
    link: ET.Element,
    linear: np.ndarray,
    longitudinal_axis: np.ndarray,
    factor: float,
) -> None:
    """Apply representable portions of the graph map to primitive geometry."""
    for owner in (*link.findall("visual"), *link.findall("collision")):
        if owner.find("./geometry/mesh") is not None:
            continue
        origin = ensure_origin(owner)
        xyz = numbers(origin.get("xyz"))
        rotation = rpy_matrix(numbers(origin.get("rpy")))
        origin.set("xyz", fmt(linear @ xyz))
        geometry = owner.find("geometry")
        if geometry is None or len(geometry) != 1:
            continue
        shape = geometry[0]
        if shape.tag in {"capsule", "cylinder"}:
            local_axis = rotation[:, 2]
            if abs(float(local_axis @ longitudinal_axis)) >= 0.95:
                shape.set(
                    "length",
                    f"{float(shape.get('length', '0')) * factor:.10g}",
                )
        elif shape.tag == "box":
            size = numbers(shape.get("size"), (1, 1, 1))
            for index in range(3):
                if abs(float(rotation[:, index] @ longitudinal_axis)) >= 0.95:
                    size[index] *= factor
            shape.set("size", fmt(size))


def transform_inertial(link: ET.Element, linear: np.ndarray) -> None:
    """Uniform-density affine mass/COM/inertia update in the link frame."""
    inertial = link.find("inertial")
    if inertial is None:
        return
    origin = ensure_origin(inertial)
    origin_xyz = numbers(origin.get("xyz"))
    origin_rotation = rpy_matrix(numbers(origin.get("rpy")))
    origin.set("xyz", fmt(linear @ origin_xyz))
    determinant = float(np.linalg.det(linear))
    mass = inertial.find("mass")
    if mass is not None:
        mass.set(
            "value",
            f"{float(mass.get('value', '0')) * determinant:.10g}",
        )
    inertia = inertial.find("inertia")
    if inertia is None:
        return
    source = np.asarray(
        (
            (
                float(inertia.get("ixx", "0")),
                float(inertia.get("ixy", "0")),
                float(inertia.get("ixz", "0")),
            ),
            (
                float(inertia.get("ixy", "0")),
                float(inertia.get("iyy", "0")),
                float(inertia.get("iyz", "0")),
            ),
            (
                float(inertia.get("ixz", "0")),
                float(inertia.get("iyz", "0")),
                float(inertia.get("izz", "0")),
            ),
        )
    )
    source_link = origin_rotation @ source @ origin_rotation.T
    second_moment = 0.5 * np.trace(source_link) * np.eye(3) - source_link
    transformed_moment = (
        determinant * linear @ second_moment @ linear.T
    )
    transformed_link = (
        np.trace(transformed_moment) * np.eye(3) - transformed_moment
    )
    transformed = (
        origin_rotation.T @ transformed_link @ origin_rotation
    )
    inertia.attrib.update(
        {
            "ixx": f"{transformed[0, 0]:.10g}",
            "ixy": f"{transformed[0, 1]:.10g}",
            "ixz": f"{transformed[0, 2]:.10g}",
            "iyy": f"{transformed[1, 1]:.10g}",
            "iyz": f"{transformed[1, 2]:.10g}",
            "izz": f"{transformed[2, 2]:.10g}",
        }
    )


def transform_outgoing_edges(
    robot: ET.Element,
    link_name: str,
    linear: np.ndarray,
) -> int:
    """Apply exactly the same map as the mesh to every outgoing graph edge."""
    transformed = 0
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        if parent is None or parent.get("link") != link_name:
            continue
        origin = ensure_origin(joint)
        origin.set("xyz", fmt(linear @ numbers(origin.get("xyz"))))
        transformed += 1
    return transformed


def transform_selected_edges(
    joints: dict[str, ET.Element],
    attachment_edges: list[dict],
    factor: float,
) -> int:
    """Scale only the fixed paths from one carrier mesh to both shoulders."""
    transformed = 0
    for edge in attachment_edges:
        joint = joints[edge["joint"]]
        linear = longitudinal_linear(
            edge["deformation_axis"], factor
        )
        origin = ensure_origin(joint)
        origin.set("xyz", fmt(linear @ numbers(origin.get("xyz"))))
        transformed += 1
    return transformed


def semantic_signature(element: ET.Element) -> tuple:
    """Geometry/frame signature that deliberately ignores rewritten mesh paths."""
    records = []
    for owner in (*element.findall("visual"), *element.findall("collision")):
        origin = owner.find("origin")
        geometry = owner.find("geometry")
        shape = None if geometry is None or not len(geometry) else geometry[0]
        records.append(
            (
                owner.tag,
                tuple(numbers(None if origin is None else origin.get("xyz"))),
                tuple(numbers(None if origin is None else origin.get("rpy"))),
                None if shape is None else shape.tag,
                None
                if shape is None
                else tuple(
                    sorted(
                        (key, value)
                        for key, value in shape.attrib.items()
                        if key != "filename"
                    )
                ),
            )
        )
    return tuple(records)


def rewrite_mesh_paths(
    robot: ET.Element,
    source_dir: Path,
    output_dir: Path,
) -> None:
    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        source = Path(filename)
        if not source.is_absolute():
            source = (source_dir / source).resolve()
        mesh.set("filename", os.path.relpath(source, output_dir))


def compile_robot(
    robot_ir: dict,
    grammar: dict,
    variants: dict,
) -> dict:
    spec = SPEC_BY_ID[robot_ir["source_id"]]
    source_root = ET.parse(spec.urdf).getroot()
    robot = deepcopy(source_root)
    links = {link.get("name", ""): link for link in robot.findall("link")}
    joints = {joint.get("name", ""): joint for joint in robot.findall("joint")}
    immutable_links = set(grammar["hard_constraints"]["immutable_links"])
    immutable_joints = set(grammar["hard_constraints"]["immutable_joints"])
    before_links = {
        name: semantic_signature(links[name]) for name in immutable_links
    }
    before_joints = {
        name: (
            tuple(
                numbers(
                    None
                    if joints[name].find("origin") is None
                    else joints[name].find("origin").get("xyz")
                )
            ),
            tuple(
                numbers(
                    None
                    if joints[name].find("origin") is None
                    else joints[name].find("origin").get("rpy")
                )
            ),
        )
        for name in immutable_joints
    }
    output_dir = GENERATED_ROOT / "robots" / robot_ir["robot_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    applied = []
    cached_mesh_owners = outgoing_edges = 0
    for action in robot_ir["actions"]:
        if action["production"] == "vertical_link_length":
            factor = float(action["factor"])
            for member in action["members"]:
                link_name = member["link"]
                if link_name in immutable_links:
                    raise ValueError(
                        f"{robot_ir['robot_id']}: immutable link edit"
                    )
                key = variant_key(
                    robot_ir["source_id"],
                    action["production"],
                    link_name,
                    factor,
                )
                if key not in variants:
                    raise KeyError(f"missing precomputed variant: {key}")
                linear = longitudinal_linear(
                    member["deformation_axis"], factor
                )
                cached_mesh_owners += apply_cached_mesh_variant(
                    links[link_name], variants[key]
                )
                transform_nonmesh_geometry(
                    links[link_name],
                    linear,
                    np.asarray(member["deformation_axis"], dtype=float),
                    factor,
                )
                transform_inertial(links[link_name], linear)
                outgoing_edges += transform_outgoing_edges(
                    robot, link_name, linear
                )
            applied.append(action["group_id"])
        elif action["production"] == "shoulder_width":
            factor = float(action["factor"])
            member = action["members"][0]
            link_name = member["link"]
            if link_name in immutable_links:
                raise ValueError(
                    f"{robot_ir['robot_id']}: immutable shoulder carrier"
                )
            key = variant_key(
                robot_ir["source_id"],
                action["production"],
                link_name,
                factor,
            )
            if key not in variants:
                raise KeyError(f"missing precomputed variant: {key}")
            linear = longitudinal_linear(
                member["deformation_axis"], factor
            )
            cached_mesh_owners += apply_cached_mesh_variant(
                links[link_name], variants[key]
            )
            transform_nonmesh_geometry(
                links[link_name],
                linear,
                np.asarray(member["deformation_axis"], dtype=float),
                factor,
            )
            transform_inertial(links[link_name], linear)
            outgoing_edges += transform_selected_edges(
                joints, member["attachment_edges"], factor
            )
            applied.append(action["group_id"])
        else:
            raise ValueError(
                f"unknown production: {action['production']}"
            )

    after_links = {
        name: semantic_signature(links[name]) for name in immutable_links
    }
    after_joints = {
        name: (
            tuple(
                numbers(
                    None
                    if joints[name].find("origin") is None
                    else joints[name].find("origin").get("xyz")
                )
            ),
            tuple(
                numbers(
                    None
                    if joints[name].find("origin") is None
                    else joints[name].find("origin").get("rpy")
                )
            ),
        )
        for name in immutable_joints
    }
    if before_links != after_links or before_joints != after_joints:
        raise ValueError(
            f"{robot_ir['robot_id']}: base/immutable structure changed"
        )

    rewrite_mesh_paths(robot, spec.urdf.parent, output_dir)
    robot.set("name", robot_ir["robot_id"])
    display_positions, display_pose_audit = (
        solve_hanging_display_positions(
            robot_ir["source_id"],
            robot_ir["actions"],
            robot,
        )
    )
    output_path = output_dir / "robot.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(
        output_path, encoding="utf-8", xml_declaration=True
    )
    return {
        **robot_ir,
        "compiled_urdf": str(output_path.relative_to(GENERATED_ROOT)),
        "display": {
            "yaw_degrees": spec.display_yaw_degrees,
            "joint_positions": display_positions,
            "pose_audit": display_pose_audit,
        },
        "compile_audit": {
            "applied_productions": applied,
            "base_and_immutable_semantics_unchanged": True,
            "runtime_mesh_deformation": False,
            "precomputed_mesh_owners_referenced": cached_mesh_owners,
            "mesh_and_graph_edge_share_one_connector_rig": True,
            "outgoing_edges_transformed": outgoing_edges,
            "joint_origin_rpy_unchanged": True,
            "joint_axes_unchanged": True,
            "dynamic_inertial_resynthesis": True,
        },
    }


def main() -> int:
    ir = json.loads(ROBOT_IR.read_text(encoding="utf-8"))
    grammar_payload = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    grammar_by_id = {
        grammar["source_id"]: grammar
        for grammar in grammar_payload["robots"]
    }
    variant_payload = json.loads(
        PRECOMPUTED_VARIANTS.read_text(encoding="utf-8")
    )
    variants = variant_payload["variants"]
    robot_root = GENERATED_ROOT / "robots"
    if robot_root.exists():
        shutil.rmtree(robot_root)
    robots = [
        compile_robot(
            robot,
            grammar_by_id[robot["source_id"]],
            variants,
        )
        for robot in ir["robots"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "grammar_id": ir["grammar_id"],
        "precomputed_cache_version": variant_payload["cache_version"],
        "robots": robots,
        "summary": {
            "robots": len(robots),
            "valid_urdfs": sum(
                (GENERATED_ROOT / robot["compiled_urdf"]).is_file()
                for robot in robots
            ),
            "base_immutable_audits_passed": sum(
                robot["compile_audit"][
                    "base_and_immutable_semantics_unchanged"
                ]
                for robot in robots
            ),
            "arm_symmetry_audits_passed": sum(
                robot["audit"]["arm_edits_bilateral"]
                for robot in robots
            ),
            "runtime_mesh_deformation": False,
            "shared_precomputed_variant_cache": True,
            "dynamic_inertial_resynthesis": True,
        },
    }
    COMPILED_ROBOTS.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
