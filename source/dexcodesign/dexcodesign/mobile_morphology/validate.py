"""Independent validation of compiled length-only mobile manipulators."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from .common import (
    COMPILED_ROBOTS,
    GENERATED_ROOT,
    GRAMMAR,
    PRECOMPUTED_VARIANTS,
    ROOT,
    SPEC_BY_ID,
    longitudinal_linear,
    numbers,
    rpy_matrix,
)
from .generate import MODES, validate_action
from .mesh_compiler import semantic_signature
from .preprocess import variant_key


REPORT = GENERATED_ROOT / "validation_report.json"


def joint_origin(joint: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    origin = joint.find("origin")
    return (
        numbers(None if origin is None else origin.get("xyz")),
        numbers(None if origin is None else origin.get("rpy")),
    )


def joint_axis(joint: ET.Element) -> np.ndarray:
    axis = joint.find("axis")
    return numbers(
        None if axis is None else axis.get("xyz"),
        (1.0, 0.0, 0.0),
    )


def joint_topology(joint: ET.Element) -> tuple[str, str, str]:
    parent, child = joint.find("parent"), joint.find("child")
    if parent is None or child is None:
        raise ValueError(f"incomplete joint {joint.get('name')}")
    return (
        joint.get("type", "fixed"),
        parent.get("link", ""),
        child.get("link", ""),
    )


def root_link(robot: ET.Element) -> str:
    links = {link.get("name", "") for link in robot.findall("link")}
    children = {
        child.get("link", "")
        for joint in robot.findall("joint")
        if (child := joint.find("child")) is not None
    }
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {roots}")
    return roots[0]


def assert_close(actual, expected, message: str) -> None:
    if not np.allclose(actual, expected, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError(
            f"{message}: expected {expected}, got {actual}"
        )


def expected_joint_xyz(
    source_joints: dict[str, ET.Element],
    actions: list[dict],
) -> dict[str, np.ndarray]:
    expected = {
        name: joint_origin(joint)[0]
        for name, joint in source_joints.items()
    }
    for action in actions:
        factor = float(action["factor"])
        if action["production"] == "vertical_link_length":
            for member in action["members"]:
                linear = longitudinal_linear(
                    member["deformation_axis"], factor
                )
                for name, joint in source_joints.items():
                    parent = joint.find("parent")
                    if (
                        parent is not None
                        and parent.get("link") == member["link"]
                    ):
                        expected[name] = linear @ expected[name]
        elif action["production"] == "shoulder_width":
            for edge in action["members"][0]["attachment_edges"]:
                linear = longitudinal_linear(
                    edge["deformation_axis"], factor
                )
                expected[edge["joint"]] = (
                    linear @ expected[edge["joint"]]
                )
        else:
            raise ValueError(
                f"orientation production survived: "
                f"{action['production']}"
            )
    return expected


def validate_cached_meshes(
    entry: dict,
    generated_links: dict[str, ET.Element],
    variants: dict,
    urdf_dir,
) -> None:
    for action in entry["actions"]:
        factor = float(action["factor"])
        for member in action["members"]:
            key = variant_key(
                entry["source_id"],
                action["production"],
                member["link"],
                factor,
            )
            if key not in variants:
                raise KeyError(f"missing cached mesh variant: {key}")
            link = generated_links[member["link"]]
            for record in variants[key]["mesh_owners"]:
                owners = link.findall(record["owner_kind"])
                owner = owners[int(record["owner_index"])]
                mesh = owner.find("./geometry/mesh")
                if mesh is None:
                    raise ValueError(
                        f"{entry['robot_id']}: cached mesh owner missing"
                    )
                actual = (
                    urdf_dir / mesh.get("filename", "")
                ).resolve()
                expected = (ROOT / record["file"]).resolve()
                if actual != expected:
                    raise ValueError(
                        f"{entry['robot_id']}: wrong cached mesh "
                        f"for {member['link']}"
                    )


def validate_inertials(
    robot_id: str, links: dict[str, ET.Element]
) -> None:
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or float(mass.get("value", "0")) <= 0:
            raise ValueError(
                f"{robot_id}: nonpositive mass on {name}"
            )
        if inertia is None:
            continue
        matrix = np.asarray(
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
        if float(np.linalg.eigvalsh(matrix).min()) <= 0:
            raise ValueError(
                f"{robot_id}: nonpositive inertia on {name}"
            )


def validate_compiled(
    entry: dict, grammar: dict, variants: dict
) -> dict:
    robot_id = entry["robot_id"]
    validate_path = GENERATED_ROOT / entry["compiled_urdf"]
    generated = ET.parse(validate_path).getroot()
    source_path = SPEC_BY_ID[entry["source_id"]].urdf
    source = ET.parse(source_path).getroot()
    generated_links = {
        link.get("name", ""): link
        for link in generated.findall("link")
    }
    source_links = {
        link.get("name", ""): link for link in source.findall("link")
    }
    generated_joints = {
        joint.get("name", ""): joint
        for joint in generated.findall("joint")
    }
    source_joints = {
        joint.get("name", ""): joint
        for joint in source.findall("joint")
    }
    if set(generated_links) != set(source_links) or set(
        generated_joints
    ) != set(source_joints):
        raise ValueError(f"{robot_id}: source topology changed")
    if root_link(generated) != grammar["root_link"]:
        raise ValueError(f"{robot_id}: root/base link changed")

    for action in entry["actions"]:
        validate_action(action, grammar)
        if action["production"] not in MODES:
            raise ValueError(
                f"{robot_id}: forbidden orientation action"
            )

    for mesh in generated.findall(".//mesh"):
        path = (
            validate_path.parent / mesh.get("filename", "")
        ).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{robot_id}: missing mesh {path}"
            )

    for name in grammar["hard_constraints"]["immutable_links"]:
        if semantic_signature(
            generated_links[name]
        ) != semantic_signature(source_links[name]):
            raise ValueError(
                f"{robot_id}: immutable link changed: {name}"
            )
        generated_inertial = generated_links[name].find("inertial")
        source_inertial = source_links[name].find("inertial")
        if (
            None
            if generated_inertial is None
            else ET.tostring(generated_inertial)
        ) != (
            None
            if source_inertial is None
            else ET.tostring(source_inertial)
        ):
            raise ValueError(
                f"{robot_id}: immutable inertia changed: {name}"
            )

    expected_xyz = expected_joint_xyz(
        source_joints, entry["actions"]
    )
    for name, generated_joint in generated_joints.items():
        source_joint = source_joints[name]
        if joint_topology(generated_joint) != joint_topology(source_joint):
            raise ValueError(
                f"{robot_id}: joint topology changed: {name}"
            )
        generated_xyz, generated_rpy = joint_origin(generated_joint)
        _, source_rpy = joint_origin(source_joint)
        assert_close(
            generated_xyz,
            expected_xyz[name],
            f"{robot_id} joint xyz {name}",
        )
        assert_close(
            rpy_matrix(generated_rpy),
            rpy_matrix(source_rpy),
            f"{robot_id} joint RPY changed {name}",
        )
        assert_close(
            joint_axis(generated_joint),
            joint_axis(source_joint),
            f"{robot_id} joint axis changed {name}",
        )

    validate_cached_meshes(
        entry,
        generated_links,
        variants,
        validate_path.parent,
    )
    validate_inertials(robot_id, generated_links)
    expected_display = dict(
        SPEC_BY_ID[entry["source_id"]].display_joint_positions
    )
    if entry["display"]["joint_positions"] != expected_display:
        raise ValueError(
            f"{robot_id}: display joint values changed"
        )
    pose_audit = entry["display"]["pose_audit"]
    if (
        pose_audit.get("method") != "source_upright_pose"
        or pose_audit.get("ik_used") is not False
    ):
        raise ValueError(
            f"{robot_id}: stale rotation/IK display logic remains"
        )
    return {
        "robot_id": robot_id,
        "source_id": entry["source_id"],
        "topology_preserved": True,
        "mesh_paths_resolve": True,
        "base_immutable": True,
        "bilateral_arm_constraints": True,
        "one_axis_length_only": True,
        "ordinary_length_original_pose_vertical_only": True,
        "shoulder_width_only_lateral_exception": True,
        "connector_span_rig_preserves_interfaces": True,
        "joint_origin_rpy_unchanged": True,
        "joint_axes_unchanged": True,
        "orientation_productions_absent": True,
        "display_pose_uses_no_ik": True,
        "inertials_positive": True,
    }


def main() -> int:
    payload = json.loads(
        COMPILED_ROBOTS.read_text(encoding="utf-8")
    )
    grammar_payload = json.loads(
        GRAMMAR.read_text(encoding="utf-8")
    )
    grammar_by_id = {
        grammar["source_id"]: grammar
        for grammar in grammar_payload["robots"]
    }
    variants = json.loads(
        PRECOMPUTED_VARIANTS.read_text(encoding="utf-8")
    )["variants"]
    records = [
        validate_compiled(
            entry,
            grammar_by_id[entry["source_id"]],
            variants,
        )
        for entry in payload["robots"]
    ]
    keys = [
        key
        for key in records[0]
        if key not in {"robot_id", "source_id"}
    ] if records else []
    report = {
        "schema_version": 1,
        "robots": records,
        "summary": {
            "robots": len(records),
            **{
                key: sum(bool(record[key]) for record in records)
                for key in keys
            },
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
