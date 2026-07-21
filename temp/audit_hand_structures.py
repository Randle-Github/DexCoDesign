#!/usr/bin/env python3
"""Audit base/palm topology and geometry complexity of every registered hand."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "robot_hands"
REGISTRY = ASSETS / "registry.json"

PALM_NAMES = {
    "mano": {"left": "left_palm", "right": "right_palm"},
    "ability_hand": {"left": "base", "right": "base"},
    "schunk_svh": {"left": "left_hand_link", "right": "right_hand_base_link"},
    "wuji_hand_2": {"left": "l_base_link", "right": "r_base_link"},
    "sharpa_wave_01": {"left": "left_hand_C_MC", "right": "right_hand_C_MC"},
    "tesollo_dg5f": {"left": "ll_dg_palm", "right": "rl_dg_palm"},
    "unitree_dex5_1": {"left": "base_link00L", "right": "base_link00"},
    "robotera_xhand1": {"left": "left_hand_link", "right": "right_hand_link"},
    "orca_hand_v2": {"left": "L-Carpals_719fff8c", "right": "R-Carpals_8d1f1041"},
    "shadow_hand_e": {"left": "lh_palm", "right": "rh_palm"},
    "allegro_hand_v5": {"left": "palm_link", "right": "palm_link"},
    "midas_hand": {"left": "palm_base", "right": "palm_base"},
    "ruka_v2": {"left": "backhand", "right": "backhand"},
    "inspire_rh56dfx": {"left": "base", "right": "base"},
}


def source_path(entry: dict[str, object]) -> Path:
    return (ASSETS / str(entry.get("source_path", entry["path"]))).resolve()


def urdf_audit(path: Path, palm_name: str) -> dict[str, object]:
    robot = ET.parse(path).getroot()
    links = {node.get("name"): node for node in robot.findall("link")}
    joints = robot.findall("joint")
    child_to_joint: dict[str, ET.Element] = {}
    children: set[str] = set()
    for joint in joints:
        child = joint.find("child")
        if child is not None:
            child_to_joint[child.get("link")] = joint
            children.add(child.get("link"))
    roots = set(links).difference(children)
    if len(roots) != 1:
        raise ValueError(f"URDF roots={sorted(roots)} in {path}")
    root_name = next(iter(roots))

    chain_links = [palm_name]
    chain_joints: list[str] = []
    cursor = palm_name
    while cursor != root_name:
        joint = child_to_joint[cursor]
        chain_joints.append(f"{joint.get('name')}[{joint.get('type')}]")
        parent = joint.find("parent")
        assert parent is not None
        cursor = parent.get("link")
        chain_links.append(cursor)
    chain_links.reverse()
    chain_joints.reverse()

    joint_types = Counter(joint.get("type", "unknown") for joint in joints)
    movable = sum(joint_types[k] for k in ("revolute", "continuous", "prismatic"))
    movable += 3 * joint_types["planar"] + 6 * joint_types["floating"]
    mimic = sum(joint.find("mimic") is not None for joint in joints)
    joint_records: list[dict[str, object]] = []
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        mimic_node = joint.find("mimic")
        joint_records.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent.get("link") if parent is not None else None,
                "child": child.get("link") if child is not None else None,
                "mimic": mimic_node.get("joint") if mimic_node is not None else None,
            }
        )
    visual_types: Counter[str] = Counter()
    collision_types: Counter[str] = Counter()
    multi_visual_links = 0
    multi_collision_links = 0
    link_records: list[dict[str, object]] = []
    for link in links.values():
        visuals = link.findall("visual")
        collisions = link.findall("collision")
        multi_visual_links += len(visuals) > 1
        multi_collision_links += len(collisions) > 1
        for tag, nodes, counter in (("visual", visuals, visual_types), ("collision", collisions, collision_types)):
            del tag
            for node in nodes:
                geometry = node.find("geometry")
                if geometry is None or len(geometry) == 0:
                    counter["missing"] += 1
                else:
                    counter[geometry[0].tag] += 1
        link_records.append(
            {
                "name": link.get("name"),
                "visual_count": len(visuals),
                "collision_count": len(collisions),
            }
        )

    return {
        "root": root_name,
        "palm": palm_name,
        "root_to_palm_links": chain_links,
        "root_to_palm_joints": chain_joints,
        "links": len(links),
        "joint_types": dict(joint_types),
        "scalar_dofs": movable,
        "mimic_joints": mimic,
        "transmissions": len(robot.findall("transmission")),
        "visual_types": dict(visual_types),
        "collision_types": dict(collision_types),
        "multi_visual_links": multi_visual_links,
        "multi_collision_links": multi_collision_links,
        "joints": joint_records,
        "link_geometry_counts": link_records,
    }


def mjcf_joint_type(model: mujoco.MjModel, joint_id: int) -> str:
    value = int(model.jnt_type[joint_id])
    return {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }[value]


def mjcf_audit(path: Path, palm_name: str) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world" for body_id in range(model.nbody)]
    if palm_name not in names:
        raise KeyError(f"MJCF palm {palm_name!r} not found in {path}")
    cursor = names.index(palm_name)
    chain_ids: list[int] = []
    while cursor > 0:
        chain_ids.append(cursor)
        cursor = int(model.body_parentid[cursor])
    chain_ids.reverse()

    chain_links = [names[body_id] for body_id in chain_ids]
    chain_joints: list[str] = []
    for body_id in chain_ids:
        start = int(model.body_jntadr[body_id])
        count = int(model.body_jntnum[body_id])
        for joint_id in range(start, start + count):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            chain_joints.append(f"{name}[{mjcf_joint_type(model, joint_id)}]")

    joint_types = Counter(mjcf_joint_type(model, joint_id) for joint_id in range(model.njnt))
    geom_types = Counter(mujoco.mjtGeom(int(value)).name.replace("mjGEOM_", "").lower() for value in model.geom_type)
    multi_geom_bodies = sum(int(model.body_geomnum[body_id]) > 1 for body_id in range(1, model.nbody))
    joint_records: list[dict[str, object]] = []
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        joint_records.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}",
                "type": mjcf_joint_type(model, joint_id),
                "body": names[body_id],
                "parent_body": names[int(model.body_parentid[body_id])],
            }
        )
    body_records: list[dict[str, object]] = []
    for body_id in range(1, model.nbody):
        body_records.append(
            {
                "name": names[body_id],
                "parent": names[int(model.body_parentid[body_id])],
                "joint_count": int(model.body_jntnum[body_id]),
                "geom_count": int(model.body_geomnum[body_id]),
            }
        )
    return {
        "root": chain_links[0] if chain_links else palm_name,
        "palm": palm_name,
        "root_to_palm_links": chain_links,
        "root_to_palm_joints": chain_joints,
        "links": model.nbody - 1,
        "joint_types": dict(joint_types),
        "scalar_dofs": model.nv,
        "actuators": model.nu,
        "tendons": model.ntendon,
        "equalities": model.neq,
        "geom_types": dict(geom_types),
        "multi_geom_bodies": multi_geom_bodies,
        "joints": joint_records,
        "body_counts": body_records,
    }


def usd_audit(path: Path, palm_name: str) -> dict[str, object]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open {path}")
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    rigid = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    by_name = {prim.GetName(): prim for prim in rigid}
    if palm_name not in by_name:
        # Some upstream files author rigid-body APIs on Xforms reached through
        # references; fall back to every traversed prim with the requested name.
        matches = [prim for prim in prims if prim.GetName() == palm_name]
        if not matches:
            raise KeyError(f"USD palm {palm_name!r} not found in {path}")
        palm_prim = matches[0]
    else:
        palm_prim = by_name[palm_name]

    child_to_joint: dict[str, tuple[str | None, Usd.Prim]] = {}
    joint_types: Counter[str] = Counter()
    drive_count = 0
    joint_records: list[dict[str, object]] = []
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        parent = str(body0[0]) if body0 else None
        child = str(body1[0]) if body1 else None
        if child:
            child_to_joint[child] = (parent, prim)
        joint_types[prim.GetTypeName()] += 1
        driven = any(attribute.GetName().startswith("drive:") for attribute in prim.GetAttributes())
        drive_count += driven
        joint_records.append(
            {
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "parent": parent,
                "child": child,
                "driven": driven,
            }
        )

    chain_links = [palm_prim.GetName()]
    chain_joints: list[str] = []
    cursor = str(palm_prim.GetPath())
    visited: set[str] = set()
    while cursor in child_to_joint and cursor not in visited:
        visited.add(cursor)
        parent, joint_prim = child_to_joint[cursor]
        chain_joints.append(f"{joint_prim.GetName()}[{joint_prim.GetTypeName()}]")
        if not parent:
            break
        parent_prim = stage.GetPrimAtPath(parent)
        chain_links.append(parent_prim.GetName() if parent_prim else parent)
        cursor = parent
    chain_links.reverse()
    chain_joints.reverse()

    return {
        "root": chain_links[0],
        "palm": palm_name,
        "root_to_palm_links": chain_links,
        "root_to_palm_joints": chain_joints,
        "links": len(rigid),
        "joint_types": dict(joint_types),
        "scalar_dofs": sum(prim.GetTypeName() in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"} for prim in joints),
        "driven_joints": drive_count,
        "mesh_prims": len(meshes),
        "joints": joint_records,
        "rigid_bodies": [str(prim.GetPath()) for prim in rigid],
    }


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    output: dict[str, dict[str, object]] = {}
    for hand_id, metadata in registry["hands"].items():
        output[hand_id] = {}
        for side, entry in metadata["entries"].items():
            path = source_path(entry)
            palm = PALM_NAMES[hand_id][side]
            if entry["format"] == "urdf":
                audit = urdf_audit(path, palm)
            elif entry["format"] == "mjcf":
                audit = mjcf_audit(path, palm)
            elif entry["format"] == "usd":
                audit = usd_audit(path, palm)
            else:
                raise ValueError(entry["format"])
            audit["format"] = entry["format"]
            audit["source_path"] = str(path.relative_to(ROOT))
            output[hand_id][side] = audit
            path_text = " -> ".join(audit["root_to_palm_links"])
            joints_text = ", ".join(audit["root_to_palm_joints"]) or "none"
            print(
                f"{hand_id:20s} {side:5s} {entry['format']:4s} "
                f"links={audit['links']:2d} dofs={audit['scalar_dofs']:2d} "
                f"root→palm: {path_text}; joints: {joints_text}"
            )

    report = ROOT / "temp" / "hand_structure_audit.json"
    report.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
