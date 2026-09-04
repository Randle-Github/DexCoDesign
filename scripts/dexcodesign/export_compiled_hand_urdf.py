#!/usr/bin/env python3
"""Export a compiled morphology graph as a physical left-hand URDF.

The morphology compiler works in a canonical right-hand coordinate system.
This exporter applies the source audit's inverse similarity transform and a
proper left/right reflection.  Every rigid part receives both visual and
collision geometry; topology, joint ranges, and scalar DoF count are retained.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_GRAPHS = (
    REPO_ROOT / "artifacts" / "hand_morphology" / "reference_graphs.json"
)
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"


def _mesh(value: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        return trimesh.util.concatenate(tuple(value.geometry.values()))
    return value


def _inverse_source_linear(source_hand: str) -> tuple[float, np.ndarray]:
    payload = json.loads(REFERENCE_GRAPHS.read_text(encoding="utf-8"))
    source = next(
        hand for hand in payload["hands"] if hand["hand_id"] == source_hand
    )
    # Current preprocessing names this record ``canonicalization``; accept the
    # earlier cache key so existing content-addressed artifacts remain usable.
    audit = source.get("canonicalization", source.get("direct_geometry_audit"))
    if audit is None:
        raise ValueError(
            f"{source_hand}: reference graph lacks canonicalization metadata"
        )
    return (
        float(audit["similarity_scale"]),
        np.asarray(audit["similarity_rotation"], dtype=np.float64),
    )


def _rpy_matrix(text: str | None) -> np.ndarray:
    values = np.fromstring(text or "0 0 0", sep=" ", dtype=np.float64)
    roll, pitch, yaw = values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _canonical_joint_axes(
    source_hand: str,
    source_rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flatten source joint-local axes into the canonical graph frame."""
    urdf = DIRECT_ROOT / source_hand / "right" / "hand.urdf"
    root = ET.parse(urdf).getroot()
    links = {link.get("name") for link in root.findall("link")}
    children = set()
    by_parent: dict[str, list[tuple[ET.Element, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        children.add(child_name)
        by_parent.setdefault(parent_name, []).append((joint, child_name))
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"{urdf}: expected one root link, found {roots}")

    world_rotation = {roots[0]: np.eye(3, dtype=np.float64)}
    axes: dict[str, np.ndarray] = {}
    queue = [roots[0]]
    while queue:
        parent_name = queue.pop(0)
        for joint, child_name in by_parent.get(parent_name, ()):
            origin = joint.find("origin")
            local_rotation = _rpy_matrix(
                None if origin is None else origin.get("rpy")
            )
            child_rotation = world_rotation[parent_name] @ local_rotation
            world_rotation[child_name] = child_rotation
            axis_element = joint.find("axis")
            if axis_element is not None and joint.get("type") != "fixed":
                local_axis = np.fromstring(
                    axis_element.get("xyz", "1 0 0"),
                    sep=" ",
                    dtype=np.float64,
                )
                world_axis = child_rotation @ local_axis
                canonical_axis = world_axis @ source_rotation
                canonical_axis /= np.linalg.norm(canonical_axis)
                axes[str(joint.get("name"))] = canonical_axis
            queue.append(child_name)
    return axes


def _source_part_joint_names(source_hand: str) -> dict[int, str]:
    payload = json.loads(REFERENCE_GRAPHS.read_text(encoding="utf-8"))
    source = next(
        hand for hand in payload["hands"] if hand["hand_id"] == source_hand
    )
    return {
        int(part["id"]): str(part["joint_name"])
        for part in source["parts"]
    }


def _vector_text(vector: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in vector)


def _add_geometry(
    link: ET.Element,
    kind: str,
    mesh_file: str,
) -> None:
    # Deliberately omit the URDF element name.  The legacy visualization scene
    # strips geoms whose imported name contains "collision"; anonymous URDF
    # collision geoms follow the normalized source-hand convention and remain
    # available for the physical-scene contact pass.
    element = ET.SubElement(link, kind)
    ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(element, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": mesh_file})
    if kind == "visual":
        material = ET.SubElement(element, "material", {"name": "demo_blue"})
        ET.SubElement(material, "color", {"rgba": "0.16 0.58 0.82 1"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("compiled", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--compiled-hand-id",
        help="select one hand from a multi-hand compiled artifact",
    )
    parser.add_argument("--hand-id", default="wuji_exaggerated_demo")
    parser.add_argument("--display-name")
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help=(
            "export collision geometry only for headless physical search; "
            "the selected morphology can be re-exported with visuals later"
        ),
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help="export links/joints/inertials without geometry for parametric USD",
    )
    args = parser.parse_args()

    payload = json.loads(args.compiled.read_text(encoding="utf-8"))
    if args.compiled_hand_id is None:
        if len(payload["hands"]) != 1:
            raise ValueError(
                "multi-hand artifact requires --compiled-hand-id"
            )
        hand = payload["hands"][0]
    else:
        matches = [
            candidate
            for candidate in payload["hands"]
            if candidate["hand_id"] == args.compiled_hand_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one compiled hand {args.compiled_hand_id!r}, "
                f"found {len(matches)}"
            )
        hand = matches[0]
    source_hand = hand["seed_source"]
    scale, source_rotation = _inverse_source_linear(source_hand)

    # Reflection is a polar-vector transform.  Revolute axes are axial
    # vectors, so they additionally receive det(reflection)=-1 below.
    reflection = np.diag((-1.0, 1.0, 1.0))
    polar_linear = source_rotation.T @ reflection
    axial_linear = -polar_linear
    canonical_joint_axes = _canonical_joint_axes(source_hand, source_rotation)
    source_part_joint_names = _source_part_joint_names(source_hand)

    hand_root = args.output_root / args.hand_id / "left"
    mesh_root = hand_root / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    compiled_root = args.compiled.parent

    robot = ET.Element("robot", {"name": args.hand_id})
    link_names: dict[int, str] = {}
    leaf_by_role: dict[str, str] = {}
    children = {int(part["id"]): 0 for part in hand["parts"]}
    for part in hand["parts"]:
        if part["parent"] is not None:
            children[int(part["parent"])] += 1

    for part in hand["parts"]:
        part_id = int(part["id"])
        role = str(part["role"])
        link_name = f"part_{part_id:02d}_{role}"
        link_names[part_id] = link_name
        if children[part_id] == 0 and role != "palm":
            leaf_by_role[role] = link_name

        link = ET.SubElement(robot, "link", {"name": link_name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        mass = 0.35 if role == "palm" else 0.035
        inertia = 8e-4 if role == "palm" else 1.5e-5
        ET.SubElement(inertial, "mass", {"value": f"{mass:g}"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{inertia:g}",
                "ixy": "0",
                "ixz": "0",
                "iyy": f"{inertia:g}",
                "iyz": "0",
                "izz": f"{inertia:g}",
            },
        )

        if args.skeleton_only:
            continue
        compiled_mesh = part["compiled_mesh"]
        if compiled_mesh is None:
            raise ValueError(
                f"part {part_id} has no compiled mesh without --skeleton-only"
            )
        visual_source = compiled_root / compiled_mesh["file"]
        collision_source = compiled_root / compiled_mesh.get(
            "collision_file", compiled_mesh["file"]
        )
        exported_files: dict[str, str] = {}
        mesh_sources = (
            # Preserve the established generated-hand collision surface. The
            # physics-only path retains the same vertices while avoiding a
            # duplicate visual mesh export.
            (("collision", visual_source),)
            if args.physics_only
            else (("visual", visual_source), ("collision", collision_source))
        )
        for kind, source_path in mesh_sources:
            source_mesh = _mesh(trimesh.load(source_path, process=False))
            vertices = (
                np.asarray(source_mesh.vertices, dtype=np.float64)
                @ polar_linear
                / scale
            )
            faces = np.asarray(source_mesh.faces, dtype=np.int64)[:, ::-1]
            exported = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )
            destination = mesh_root / f"part_{part_id:02d}_{kind}.obj"
            exported.export(destination)
            exported_files[kind] = f"meshes/{destination.name}"
        if args.physics_only:
            _add_geometry(link, "collision", exported_files["collision"])
        else:
            # Keep the MuJoCo-compatible visual path for general exported assets.
            # Physics-only consumers use the collision-only branch above.
            for kind in ("visual", "collision"):
                _add_geometry(link, kind, exported_files["visual"])

    for part in hand["parts"]:
        parent_id = part["parent"]
        if parent_id is None:
            continue
        part_id = int(part["id"])
        joint_kind = {
            "hinge": "revolute",
            "slide": "prismatic",
            "fixed": "fixed",
        }.get(part["joint_type"], part["joint_type"])
        joint = ET.SubElement(
            robot,
            "joint",
            {"name": str(part["joint_name"]), "type": joint_kind},
        )
        ET.SubElement(joint, "parent", {"link": link_names[int(parent_id)]})
        ET.SubElement(joint, "child", {"link": link_names[part_id]})
        position = np.asarray(part["relative_pos"], dtype=np.float64)
        position = position @ polar_linear / scale
        ET.SubElement(
            joint,
            "origin",
            {"xyz": _vector_text(position), "rpy": "0 0 0"},
        )
        if joint_kind != "fixed":
            source_joint_name = source_part_joint_names.get(
                int(part.get("source_part_id", part["id"])),
                str(part["joint_name"]),
            )
            axis = canonical_joint_axes.get(
                source_joint_name,
                np.asarray(part["joint_axis"], dtype=np.float64),
            )
            axis = axis @ axial_linear
            axis /= np.linalg.norm(axis)
            ET.SubElement(joint, "axis", {"xyz": _vector_text(axis)})
            lower, upper = part["joint_range"]
            ET.SubElement(
                joint,
                "limit",
                {
                    "lower": f"{float(lower):.12g}",
                    "upper": f"{float(upper):.12g}",
                    "effort": "15",
                    "velocity": "5",
                },
            )
            ET.SubElement(joint, "dynamics", {"damping": "0.2", "friction": "0.01"})

    expected_roles = {"thumb", "index", "middle", "ring", "pinky"}
    if set(leaf_by_role) != expected_roles:
        raise ValueError(f"unexpected fingertip roles: {sorted(leaf_by_role)}")
    mujoco_extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        {
            "strippath": "false",
            "discardvisual": "false",
            "balanceinertia": "true",
        },
    )
    ET.indent(robot, space="  ")
    urdf_path = hand_root / "hand.urdf"
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    metadata = {
        "hand_id": args.hand_id,
        "display_name": args.display_name or args.hand_id,
        "source_hand": source_hand,
        "urdf": str(urdf_path),
        "tips": leaf_by_role,
        "active_dofs": sum(
            part["joint_type"] != "fixed" for part in hand["parts"]
        ),
        "passive_mimic_dofs": 0,
        "all_parts_have_visual_and_collision": not (
            args.physics_only or args.skeleton_only
        ),
        "all_parts_have_collision": not args.skeleton_only,
        "physics_only": bool(args.physics_only),
        "skeleton_only": bool(args.skeleton_only),
        "link_names": {str(key): value for key, value in link_names.items()},
        "mirror_semantics": "left polar geometry + left axial revolute axes",
    }
    metadata_path = args.output_root / "runtime_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
