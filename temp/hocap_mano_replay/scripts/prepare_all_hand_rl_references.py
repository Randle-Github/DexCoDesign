#!/usr/bin/env python3
"""Prepare generic residual-RL assets and references for every robot hand.

Each source hand receives a six-joint virtual wrist in front of its original
root link. The reference contains that wrist pose plus all articulated finger
joints from the reviewed all-hands IK trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
REGISTRY = DIRECT_ROOT / "registry.json"
RETARGET_ROOT = REPO_ROOT / "artifacts" / "all_hands_success_action_retarget"
DEFAULT_CAPTURE = (
    REPO_ROOT
    / "temp"
    / "hocap_mano_replay"
    / "data"
    / "subset"
    / "subject_7"
    / "20231022_192832"
    / "isaaclab_reference.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts" / "isaaclab_all_hands_residual" / "prepared"
)

sys.path.insert(0, str(SCRIPT_ROOT))
from retarget_all_hands import (  # noqa: E402
    TIP_LINKS,
    assign_full_qpos,
    assign_wrist_pose,
    build_hand,
)


ROOT_JOINT_NAMES = (
    "root_pos_x",
    "root_pos_y",
    "root_pos_z",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
)


def sanitize(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not result or result[0].isdigit():
        result = f"n_{result}"
    return result


def unique_mapping(names: list[str], prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for original in names:
        base = f"{prefix}{sanitize(original)}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapping[original] = candidate
        used.add(candidate)
    return mapping


def tiny_inertial_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1e-8"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1e-12",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1e-12",
            "iyz": "0",
            "izz": "1e-12",
        },
    )
    return link


def virtual_joint(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    axis: str,
) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "axis", {"xyz": axis})
    if joint_type == "prismatic":
        ET.SubElement(
            joint,
            "limit",
            {"lower": "-3", "upper": "3", "effort": "1000", "velocity": "4"},
        )
    else:
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": "-6.28",
                "upper": "6.28",
                "effort": "1000",
                "velocity": "20",
            },
        )
    ET.SubElement(joint, "dynamics", {"damping": "0", "friction": "0"})
    return joint


def prepare_wrapped_urdf(source: Path, output: Path) -> dict[str, object]:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", f"{sanitize(root.get('name', source.parent.name))}_rl")
    links = root.findall("link")
    joints = root.findall("joint")
    original_link_names = [link.get("name") for link in links]
    original_joint_names = [joint.get("name") for joint in joints]
    if any(name is None for name in original_link_names + original_joint_names):
        raise ValueError(f"Unnamed link or joint in {source}")
    link_map = unique_mapping(
        [str(name) for name in original_link_names], "hand__"
    )
    joint_map = unique_mapping(
        [str(name) for name in original_joint_names], "finger__"
    )
    mimic_relations: dict[str, dict[str, float | str]] = {}
    for joint in joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        original = str(joint.get("name"))
        source_joint = mimic.get("joint")
        if source_joint not in joint_map:
            raise ValueError(
                f"Mimic source {source_joint!r} for {original!r} is missing in {source}"
            )
        mimic_relations[joint_map[original]] = {
            "source": joint_map[source_joint],
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }

    child_links = {
        joint.find("child").get("link")
        for joint in joints
        if joint.find("child") is not None
    }
    root_links = [
        str(name) for name in original_link_names if name not in child_links
    ]
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link in {source}, got {root_links}")
    original_root_link = root_links[0]

    mesh_alias_root = output.parent / "mesh_aliases"
    mesh_directory_aliases: dict[Path, Path] = {}
    mesh_alias_index = 0
    for link in links:
        original = str(link.get("name"))
        link.set("name", link_map[original])
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(
                inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"}
            )
            ET.SubElement(inertial, "mass", {"value": "1e-8"})
            ET.SubElement(
                inertial,
                "inertia",
                {
                    "ixx": "1e-12",
                    "ixy": "0",
                    "ixz": "0",
                    "iyy": "1e-12",
                    "iyz": "0",
                    "izz": "1e-12",
                },
            )
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            mesh_source = (source.parent / filename).resolve()
            source_directory = mesh_source.parent
            if source_directory not in mesh_directory_aliases:
                alias_directory = (
                    mesh_alias_root / f"source_{len(mesh_directory_aliases):02d}"
                )
                alias_directory.mkdir(parents=True, exist_ok=True)
                for sibling in source_directory.iterdir():
                    if not sibling.is_file():
                        continue
                    sibling_alias = alias_directory / sibling.name
                    if sibling_alias.is_symlink() or sibling_alias.exists():
                        sibling_alias.unlink()
                    sibling_alias.symlink_to(sibling.resolve())
                mesh_directory_aliases[source_directory] = alias_directory
            alias_directory = mesh_directory_aliases[source_directory]
            mesh_alias = alias_directory / (
                f"mesh_{mesh_alias_index:04d}_{sanitize(mesh_source.stem)}"
                f"{mesh_source.suffix.lower()}"
            )
            mesh_alias_index += 1
            if mesh_alias.is_symlink() or mesh_alias.exists():
                mesh_alias.unlink()
            mesh_alias.symlink_to(mesh_source)
            mesh.set("filename", str(mesh_alias.absolute()))
        for element in link.findall(".//*[@name]"):
            element.set("name", sanitize(element.get("name")))

    for joint in joints:
        original = str(joint.get("name"))
        joint.set("name", joint_map[original])
        parent = joint.find("parent")
        child = joint.find("child")
        assert parent is not None and child is not None
        parent.set("link", link_map[parent.get("link")])
        child.set("link", link_map[child.get("link")])
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint") in joint_map:
            mimic.set("joint", joint_map[mimic.get("joint")])

    world_link = "rl_world"
    virtual_links = [f"rl_virtual_{index}" for index in range(5)]
    root.insert(0, ET.Element("link", {"name": world_link}))
    insert_index = 1
    for name in virtual_links:
        root.insert(insert_index, tiny_inertial_link(name))
        insert_index += 1

    parents = [world_link, *virtual_links]
    children = [*virtual_links, link_map[original_root_link]]
    joint_types = ("prismatic", "prismatic", "prismatic", "revolute", "revolute", "revolute")
    axes = ("1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1")
    for name, joint_type, parent, child, axis in zip(
        ROOT_JOINT_NAMES, joint_types, parents, children, axes
    ):
        root.insert(
            insert_index,
            virtual_joint(name, joint_type, parent, child, axis),
        )
        insert_index += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "link_map": link_map,
        "joint_map": joint_map,
        "root_link": link_map[original_root_link],
        "mimic_relations": mimic_relations,
    }


def fingertip_reference(
    hand,
    q_trajectory: np.ndarray,
    wrist_positions: np.ndarray,
    wrist_quaternions: np.ndarray,
    policy_fingers: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.empty((len(q_trajectory), len(policy_fingers), 7), dtype=np.float32)
    wrapper_positions = np.empty((len(q_trajectory), 3), dtype=np.float64)
    wrapper_quaternions = np.empty((len(q_trajectory), 4), dtype=np.float64)
    for frame_index, q in enumerate(q_trajectory):
        assign_full_qpos(hand, q)
        assign_wrist_pose(
            hand,
            wrist_positions[frame_index],
            Rotation.from_quat(wrist_quaternions[frame_index]),
        )
        mujoco.mj_forward(hand.model, hand.data)
        wrapper_positions[frame_index] = hand.data.mocap_pos[
            hand.wrapper_mocap_id
        ]
        wrapper_quaternions[frame_index] = np.roll(
            hand.data.mocap_quat[hand.wrapper_mocap_id], -1
        )
        for finger_index, finger in enumerate(policy_fingers):
            body_id = hand.tip_ids[finger]
            result[frame_index, finger_index, :3] = hand.data.xpos[body_id]
            result[frame_index, finger_index, 3:] = hand.data.xquat[body_id]
    return result, wrapper_positions, wrapper_quaternions


def prepare_hand(
    hand_id: str,
    display_name: str,
    capture: np.lib.npyio.NpzFile,
    output_root: Path,
) -> dict[str, object]:
    trajectory_path = RETARGET_ROOT / hand_id / "retargeted_trajectory.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)
    trajectory = np.load(trajectory_path)
    frame_count = min(len(capture["hand_q"]), len(trajectory["qpos"]))
    if frame_count != len(capture["hand_q"]):
        raise ValueError(
            f"{hand_id} only has {frame_count}/{len(capture['hand_q'])} reference frames"
        )

    source_urdf = DIRECT_ROOT / hand_id / "left" / "hand.urdf"
    hand_root = output_root / hand_id
    urdf_path = hand_root / "hand_rl.urdf"
    mapping = prepare_wrapped_urdf(source_urdf, urdf_path)

    source_tip_position = {
        finger: np.zeros(3, dtype=np.float64) for finger in TIP_LINKS[hand_id]
    }
    source_tip_rotation = {
        finger: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        for finger in TIP_LINKS[hand_id]
    }
    # identity_wrist_map avoids a meaningless align_vectors call when only
    # model topology and body ids are needed here. Use nonzero dummy vectors.
    for index, finger in enumerate(source_tip_position):
        source_tip_position[finger][index % 3] = 1.0
    hand = build_hand(
        hand_id,
        display_name,
        source_tip_position,
        source_tip_rotation,
        identity_wrist_map=True,
    )
    source_qpos_ids = trajectory["qpos_ids"].astype(int)
    qpos_to_joint = {
        int(hand.model.jnt_qposadr[joint_id]): hand.model.joint(joint_id).name
        for joint_id in range(hand.model.njnt)
    }
    source_joint_names = [
        qpos_to_joint[int(qpos_id)] for qpos_id in source_qpos_ids
    ]
    finger_joint_names = [
        mapping["joint_map"][name] for name in source_joint_names
    ]

    q_trajectory = trajectory["qpos"][:frame_count].astype(np.float64)
    q_column_by_joint = {
        name: index for index, name in enumerate(finger_joint_names)
    }
    mimic_relations = mapping["mimic_relations"]
    for mimic_joint, relation in mimic_relations.items():
        if mimic_joint not in q_column_by_joint:
            continue
        source_joint = str(relation["source"])
        if source_joint not in q_column_by_joint:
            raise ValueError(
                f"{hand_id}: retargeted trajectory lacks active mimic source "
                f"{source_joint} for {mimic_joint}"
            )
        q_trajectory[:, q_column_by_joint[mimic_joint]] = (
            float(relation["multiplier"])
            * q_trajectory[:, q_column_by_joint[source_joint]]
            + float(relation["offset"])
        )

    wrist_positions = trajectory["wrist_position"][:frame_count].astype(np.float64)
    wrist_quaternions = trajectory["wrist_quaternion_xyzw"][:frame_count].astype(
        np.float64
    )
    policy_fingers = ("thumb", "index")
    (
        fingertip_pose,
        root_positions,
        root_quaternions,
    ) = fingertip_reference(
        hand,
        q_trajectory,
        wrist_positions,
        wrist_quaternions,
        policy_fingers,
    )
    root_euler = np.unwrap(
        Rotation.from_quat(root_quaternions).as_euler("XYZ"), axis=0
    )
    wrapped_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    all_joint_names = [
        wrapped_model.joint(joint_id).name
        for joint_id in range(wrapped_model.njnt)
    ]
    action_joint_names = [
        name for name in all_joint_names if name not in mimic_relations
    ]
    action_index = {name: index for index, name in enumerate(action_joint_names)}
    action_to_control = np.zeros(
        (len(all_joint_names), len(action_joint_names)), dtype=np.float32
    )

    def resolve_action_row(
        joint_name: str, resolving: frozenset[str] = frozenset()
    ) -> np.ndarray:
        if joint_name in action_index:
            row = np.zeros(len(action_joint_names), dtype=np.float32)
            row[action_index[joint_name]] = 1.0
            return row
        if joint_name in resolving:
            raise ValueError(f"{hand_id}: mimic cycle involving {joint_name}")
        relation = mimic_relations[joint_name]
        return float(relation["multiplier"]) * resolve_action_row(
            str(relation["source"]), resolving | {joint_name}
        )

    for joint_index, name in enumerate(all_joint_names):
        action_to_control[joint_index] = resolve_action_row(name)

    hand_q = np.zeros((frame_count, len(all_joint_names)), dtype=np.float32)
    reference_values = {
        **{
            name: root_positions[:, index]
            for index, name in enumerate(ROOT_JOINT_NAMES[:3])
        },
        **{
            name: root_euler[:, index]
            for index, name in enumerate(ROOT_JOINT_NAMES[3:])
        },
        **{
            name: q_trajectory[:, index]
            for index, name in enumerate(finger_joint_names)
        },
    }
    for joint_index, name in enumerate(all_joint_names):
        if name in reference_values:
            hand_q[:, joint_index] = reference_values[name]
    link_map = mapping["link_map"]
    tip_links = TIP_LINKS[hand_id]
    other_fingers = [
        finger
        for finger in ("index", "middle", "ring", "pinky")
        if finger in tip_links
    ]
    middle_finger = "middle" if "middle" in tip_links else "index"
    reference_path = hand_root / "reference.npz"
    np.savez_compressed(
        reference_path,
        hand_id=np.asarray(hand_id),
        display_name=np.asarray(display_name),
        joint_names=np.asarray(all_joint_names),
        action_joint_names=np.asarray(action_joint_names),
        action_to_control_matrix=action_to_control,
        hand_q=hand_q,
        hand_ctrl=hand_q,
        object_pose_wxyz=capture["object_pose_wxyz"][:frame_count].astype(np.float32),
        fingertip_pose_wxyz=fingertip_pose,
        fingertip_link_names=np.asarray(
            [link_map[tip_links[finger]] for finger in policy_fingers]
        ),
        fingertip_offsets=np.zeros((2, 3), dtype=np.float32),
        thumb_contact_link_names=np.asarray([link_map[tip_links["thumb"]]]),
        other_finger_contact_link_names=np.asarray(
            [link_map[tip_links[finger]] for finger in other_fingers]
        ),
        palm_body_name=np.asarray(mapping["root_link"]),
        middle_tip_body_name=np.asarray(link_map[tip_links[middle_finger]]),
        fps=np.asarray(30.0, dtype=np.float32),
    )
    manifest = {
        "hand_id": hand_id,
        "display_name": display_name,
        "urdf": str(urdf_path),
        "reference": str(reference_path),
        "frames": frame_count,
        "action_dim": len(action_joint_names),
        "control_dim": int(hand_q.shape[1]),
        "active_finger_joint_count": len(action_joint_names) - len(ROOT_JOINT_NAMES),
        "passive_mimic_joint_count": len(mimic_relations),
        "finger_joint_count": len(all_joint_names) - len(ROOT_JOINT_NAMES),
        "retargeted_joint_count": len(finger_joint_names),
        "default_zero_joint_names": [
            name
            for name in all_joint_names
            if name not in reference_values
        ],
        "thumb_contact_link_names": [link_map[tip_links["thumb"]]],
        "other_finger_contact_link_names": [
            link_map[tip_links[finger]] for finger in other_fingers
        ],
    }
    (hand_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hands", nargs="*", default=None)
    args = parser.parse_args()

    capture = np.load(args.capture)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hand_ids = [hand_id for hand_id in registry["hands"] if hand_id != "mano"]
    if args.hands:
        requested = set(args.hands)
        hand_ids = [hand_id for hand_id in hand_ids if hand_id in requested]
    manifests = {}
    for index, hand_id in enumerate(hand_ids, 1):
        manifest = prepare_hand(
            hand_id,
            registry["hands"][hand_id]["display_name"],
            capture,
            args.output_dir,
        )
        manifests[hand_id] = manifest
        print(
            f"[{index:02d}/{len(hand_ids)}] {hand_id}: "
            f"{manifest['frames']} frames, {manifest['action_dim']} actions",
            flush=True,
        )
    output = args.output_dir / "registry.json"
    output.write_text(json.dumps({"hands": manifests}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
