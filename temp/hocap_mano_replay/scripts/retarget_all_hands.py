#!/usr/bin/env python3
"""Retarget the verified HO-Cap replay to every direct-motor left hand.

The intentionally small IK objective contains only:
  * the bottom wrist orientation target (wrist translation has no target), and
  * each available fingertip position and terminal-link orientation.

No intermediate keypoints, contacts, or collision terms are used.  A damped
least-squares solve is warm-started from the preceding frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
DIRECT_ROOT = SOURCE_DIRECT_ROOT
REGISTRY = DIRECT_ROOT / "registry.json"
SEQUENCE_ROOT = (
    EXPERIMENT_ROOT / "data" / "subset" / "subject_7" / "20231022_192832"
)
OBJECT_MESH = (
    EXPERIMENT_ROOT
    / "data"
    / "subset"
    / "models"
    / "G04_1"
    / "cleaned_mesh_2000.obj"
)
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"
CACHE_ROOT = ARTIFACT_ROOT / "all_hands_ik_cache"
MANO_COMMAND_REFERENCE = SEQUENCE_ROOT / "isaaclab_reference.npz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_mujoco import (  # noqa: E402
    LOCAL_HAND_TO_MANO,
    mano_fingertip_offsets,
)
from fingertip_geometry import resolve_fingertip_offsets  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_TIPS = {
    "thumb": "left_thumb3",
    "index": "left_index3",
    "middle": "left_middle3",
    "ring": "left_ring3",
    "pinky": "left_pinky3",
}
HOCAP_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_JOINTS_WORLD = SEQUENCE_ROOT / "hand_joints_3d_left.npy"
TIP_LINKS: dict[str, dict[str, str]] = {
    "mano": MANO_TIPS,
    "ability_hand": {
        f: f"{f}_L2" for f in FINGERS
    },
    "schunk_svh": {
        f: f"left_hand_{f}_distal" for f in FINGERS
    },
    "wuji_hand_2": {
        "thumb": "l_thumb_distal",
        "index": "l_index_finger_distal",
        "middle": "l_middle_finger_distal",
        "ring": "l_ring_finger_distal",
        "pinky": "l_pinky_distal",
    },
    "sharpa_wave_01": {
        f: f"left_{f}_DP" for f in FINGERS
    },
    "tesollo_dg5f": {
        f: f"ll_dg_{i}_4" for i, f in enumerate(FINGERS, 1)
    },
    "unitree_dex5_1": {
        f: f"Link_{i}4L" for i, f in enumerate(FINGERS, 1)
    },
    "robotera_xhand1": {
        "thumb": "left_hand_thumb_rota_link2",
        "index": "left_hand_index_rota_link2",
        "middle": "left_hand_mid_link2",
        "ring": "left_hand_ring_link2",
        "pinky": "left_hand_pinky_link2",
    },
    "orca_hand_v2": {
        "thumb": "T-DP_307db3cc",
        "index": "I-FingerTipAssembly_ed91b18a",
        "middle": "M-FingerTipAssembly_34afb748",
        "ring": "M-FingerTipAssembly_424a8e75",
        "pinky": "P-FingerTipAssembly_cd219176",
    },
    "shadow_hand_e": {
        "thumb": "lh_thdistal",
        "index": "lh_ffdistal",
        "middle": "lh_mfdistal",
        "ring": "lh_rfdistal",
        "pinky": "lh_lfdistal",
    },
    "allegro_hand_v5": {
        "thumb": "link_15_0",
        "index": "link_3_0",
        "middle": "link_7_0",
        "ring": "link_11_0",
    },
    "midas_hand": {
        "thumb": "thumb_dip",
        "index": "index_dip_link",
        "middle": "middle_dip_link",
        "ring": "ring_dip_link",
    },
    "ruka_v2": {
        "thumb": "thumb___joint_3",
        "index": "finger___joint_3",
        "middle": "finger___joint_3_2",
        "ring": "finger___joint_3_3",
        "pinky": "pinky___joint_3",
    },
    "inspire_rh56dfx": {
        "thumb": "thumb_distal",
        "index": "index_intermediate",
        "middle": "middle_intermediate",
        "ring": "ring_intermediate",
        "pinky": "pinky_intermediate",
    },
    "supr_female_foot": {
        "thumb": "big_toe",
        "index": "toe_2",
        "middle": "toe_3",
        "ring": "toe_4",
        "pinky": "toe_5",
    },
}


@dataclass
class HandIK:
    hand_id: str
    display_name: str
    scene_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    wrist_id: int
    tip_ids: dict[str, int]
    tip_offsets: dict[str, np.ndarray] | None
    tip_directions: dict[str, np.ndarray]
    full_dof_ids: np.ndarray
    full_qpos_ids: np.ndarray
    dof_ids: np.ndarray
    qpos_ids: np.ndarray
    active_to_full: np.ndarray
    full_qpos_offset: np.ndarray
    active_lower_limits: np.ndarray
    active_upper_limits: np.ndarray
    mimic_relations: dict[str, tuple[str, float, float]]
    map_rotation: Rotation
    size_ratio: float
    neutral_tip_rot: dict[str, Rotation]
    neutral_ref_rot: dict[str, Rotation]
    wrapper_mocap_id: int
    object_mocap_id: int


def body_ancestors(model: mujoco.MjModel, body_id: int) -> list[int]:
    result = []
    while body_id:
        result.append(body_id)
        body_id = int(model.body_parentid[body_id])
    result.append(0)
    return result


def lowest_common_ancestor(model: mujoco.MjModel, body_ids: list[int]) -> int:
    paths = [body_ancestors(model, body_id) for body_id in body_ids]
    common = set(paths[0]).intersection(*map(set, paths[1:]))
    return max(common, key=lambda body_id: len(body_ancestors(model, body_id)))


def make_scene(hand_id: str) -> Path:
    hand_urdf = DIRECT_ROOT / hand_id / "left" / "hand.urdf"
    output = CACHE_ROOT / f"{hand_id}.xml"
    imported = mujoco.MjModel.from_xml_path(str(hand_urdf))
    mujoco.mj_saveLastXML(str(output), imported)
    tree = ET.parse(output)
    root = tree.getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    # SUPR's source URDF declares meshdir="meshes", so MuJoCo's round-trip
    # exporter stores bare OBJ basenames. Other converted hands retain their
    # "meshes/..." asset paths and therefore use the hand directory itself.
    mesh_root = (
        hand_urdf.parent / "meshes"
        if hand_id == "supr_female_foot"
        else hand_urdf.parent
    )
    compiler.set("meshdir", str(mesh_root))
    world = root.find("worldbody")
    asset = root.find("asset")
    assert world is not None and asset is not None

    wrapper = ET.Element("body", {"name": "retarget_wrapper", "mocap": "true"})
    for body in list(world.findall("body")):
        world.remove(body)
        wrapper.append(body)
    # MuJoCo folds a fixed URDF root link into worldbody, leaving its palm
    # geoms there while child links remain bodies.  The root geoms must travel
    # with the same free wrist wrapper or the palm visually detaches.
    for geom in list(world.findall("geom")):
        world.remove(geom)
        wrapper.append(geom)
    world.append(wrapper)

    # Render only the actual visual meshes. Duplicate collision meshes make
    # the normalized hands look as if small joint cylinders were added.
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if "collision" in geom.get("name", "").lower():
                body.remove(geom)

    asset.append(ET.Element("mesh", {"name": "hocap_object", "file": str(OBJECT_MESH)}))
    obj = ET.SubElement(world, "body", {"name": "hocap_object_body", "mocap": "true"})
    ET.SubElement(
        obj,
        "geom",
        {
            "name": "hocap_object_geom",
            "type": "mesh",
            "mesh": "hocap_object",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.96 0.48 0.14 1",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "table",
            "type": "plane",
            "size": "0.7 0.55 0.02",
            "pos": "0 0 -0.006",
            "rgba": "0.72 0.74 0.78 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "960", "offheight": "720"})
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.78 0.78 0.78", "ambient": "0.42 0.42 0.42"},
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def rotation_matrix(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def supr_fingertip_offsets() -> dict[str, np.ndarray]:
    """Return distal surface points of the five articulated SUPR toe meshes."""
    mesh_root = DIRECT_ROOT / "supr_female_foot" / "left" / "meshes"
    link_names = {
        "thumb": "big_toe",
        "index": "toe_2",
        "middle": "toe_3",
        "ring": "toe_4",
        "pinky": "toe_5",
    }
    offsets: dict[str, np.ndarray] = {}
    for finger, link_name in link_names.items():
        vertices = []
        mesh_path = mesh_root / f"left_{link_name}.obj"
        for line in mesh_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
        points = np.asarray(vertices, dtype=np.float64)
        if not len(points):
            raise ValueError(f"No vertices in {mesh_path}")
        distal_x = float(points[:, 0].max())
        distal_cap = points[points[:, 0] >= distal_x - 5e-4]
        offset = distal_cap.mean(axis=0)
        offset[0] = distal_x
        offsets[finger] = offset
    return offsets


def relative_tip_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wrist_id: int,
    tip_ids: dict[str, int],
    tip_offsets: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Rotation]]:
    wrist_p = data.xpos[wrist_id].copy()
    wrist_r = rotation_matrix(data, wrist_id)
    positions = {}
    rotations = {}
    for finger, body_id in tip_ids.items():
        tip_world = data.xpos[body_id]
        if tip_offsets is not None:
            tip_world = (
                tip_world
                + rotation_matrix(data, body_id) @ tip_offsets[finger]
            )
        positions[finger] = wrist_r.T @ (tip_world - wrist_p)
        rotations[finger] = Rotation.from_matrix(
            wrist_r.T @ rotation_matrix(data, body_id)
        )
    return positions, rotations


def direction_rotation(
    direction: np.ndarray,
    preferred_y: np.ndarray,
) -> Rotation:
    """Construct a frame whose +Z is the physical distal direction."""
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    x_axis = np.cross(preferred_y, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(fallback, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    return Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))


def reference_trajectory(
    mano_pose: np.ndarray,
    hand_joints_world: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Build EgoEngine-style wrist and fingertip targets from official 21 points.

    HO-Cap's first three MANO channels provide the exact global wrist
    orientation.  Finger positions and distal directions come from the
    official ``hand_joints_3d`` labels, avoiding any interpretation of the
    45 PCA coefficients.
    """
    if hand_joints_world.shape != (len(mano_pose), 21, 3):
        raise ValueError(
            "Expected hand_joints_world with shape "
            f"({len(mano_pose)}, 21, 3), got {hand_joints_world.shape}"
        )
    wrist_pos = hand_joints_world[:, 0].astype(np.float64)
    wrist_rotations = [
        Rotation.from_rotvec(pose[:3]) * LOCAL_HAND_TO_MANO
        for pose in mano_pose
    ]
    wrist_quat = np.asarray([rotation.as_quat() for rotation in wrist_rotations])
    positions = {finger: [] for finger in FINGERS}
    rotations = {finger: [] for finger in FINGERS}

    for frame_id, wrist_rotation in enumerate(wrist_rotations):
        wrist_y = wrist_rotation.apply(np.array([0.0, 1.0, 0.0]))
        points = hand_joints_world[frame_id]
        for finger in FINGERS:
            _, _, distal_id, tip_id = HOCAP_JOINTS[finger]
            # Every finger uses one common semantic frame: +Z points from the
            # last labelled joint to the physical fingertip.  The old code
            # used +X for the thumb and -Z for the other fingers, so matching
            # robot link quaternions did not align the distal segments.
            tip_rotation = direction_rotation(
                points[tip_id] - points[distal_id], wrist_y
            )
            positions[finger].append(
                wrist_rotation.inv().apply(points[tip_id] - wrist_pos[frame_id])
            )
            rotations[finger].append(
                (wrist_rotation.inv() * tip_rotation).as_quat()
            )

    return (
        wrist_pos,
        wrist_quat,
        {k: np.asarray(v) for k, v in positions.items()},
        {k: np.asarray(v) for k, v in rotations.items()},
    )


def mano_command_reference_trajectory(
    reference_path: Path = MANO_COMMAND_REFERENCE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Build retarget targets from the saved pre-RL MANO command trajectory.

    This deliberately evaluates ``hand_ctrl`` through the reviewed MANO URDF
    instead of returning to the raw HO-Cap keypoints.  Morphology search can
    therefore hold the human action command and object trajectory fixed while
    changing only the robot-hand reshape vector.
    """
    reference = np.load(reference_path)
    if "hand_ctrl" not in reference or "joint_names" not in reference:
        raise ValueError(
            f"{reference_path} must contain hand_ctrl and joint_names"
        )
    commands = reference["hand_ctrl"].astype(np.float64)
    joint_names = reference["joint_names"].tolist()
    if commands.shape != (len(commands), len(joint_names)):
        raise ValueError(
            f"invalid MANO command shape {commands.shape} for "
            f"{len(joint_names)} joints"
        )

    # Generated-hand evaluators temporarily redirect DIRECT_ROOT to an
    # isolated runtime registry.  The human command source must always remain
    # the canonical MANO asset, independent of the candidate being evaluated.
    global DIRECT_ROOT
    candidate_direct_root = DIRECT_ROOT
    try:
        DIRECT_ROOT = SOURCE_DIRECT_ROOT
        scene = make_scene("mano")
    finally:
        DIRECT_ROOT = candidate_direct_root
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_ids = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in joint_names],
        dtype=int,
    )
    tip_ids = {
        finger: int(model.body(link_name).id)
        for finger, link_name in MANO_TIPS.items()
    }
    offsets = mano_fingertip_offsets()
    directions = {
        finger: offset / (np.linalg.norm(offset) + 1e-12)
        for finger, offset in offsets.items()
    }
    wrist_id = lowest_common_ancestor(model, list(tip_ids.values()))
    wrapper_mocap_id = int(
        model.body_mocapid[model.body("retarget_wrapper").id]
    )
    data.mocap_pos[wrapper_mocap_id] = 0.0
    data.mocap_quat[wrapper_mocap_id] = (1.0, 0.0, 0.0, 0.0)

    wrist_positions: list[np.ndarray] = []
    wrist_quaternions: list[np.ndarray] = []
    positions = {finger: [] for finger in FINGERS}
    rotations = {finger: [] for finger in FINGERS}
    for command in commands:
        data.qpos[qpos_ids] = command
        mujoco.mj_forward(model, data)
        wrist_position = data.xpos[wrist_id].copy()
        wrist_rotation = Rotation.from_matrix(rotation_matrix(data, wrist_id))
        wrist_positions.append(wrist_position)
        wrist_quaternions.append(wrist_rotation.as_quat())
        for finger in FINGERS:
            body_id = tip_ids[finger]
            body_rotation = rotation_matrix(data, body_id)
            tip_position = data.xpos[body_id] + body_rotation @ offsets[finger]
            tip_direction = body_rotation @ directions[finger]
            positions[finger].append(
                wrist_rotation.inv().apply(tip_position - wrist_position)
            )
            local_direction = wrist_rotation.inv().apply(tip_direction)
            rotations[finger].append(
                direction_rotation(local_direction, np.array([0.0, 1.0, 0.0])).as_quat()
            )

    return (
        np.asarray(wrist_positions),
        np.asarray(wrist_quaternions),
        {finger: np.asarray(values) for finger, values in positions.items()},
        {finger: np.asarray(values) for finger, values in rotations.items()},
    )


def urdf_mimic_relations(
    hand_id: str,
) -> dict[str, tuple[str, float, float]]:
    """Return ``mimic = multiplier * source + offset`` relations."""
    urdf = DIRECT_ROOT / hand_id / "left" / "hand.urdf"
    root = ET.parse(urdf).getroot()
    relations: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        joint_name = joint.get("name")
        source_name = mimic.get("joint")
        if not joint_name or not source_name:
            raise ValueError(f"{hand_id}: malformed mimic joint in {urdf}")
        relations[joint_name] = (
            source_name,
            float(mimic.get("multiplier", "1")),
            float(mimic.get("offset", "0")),
        )
    return relations


def path_dofs(
    model: mujoco.MjModel,
    wrist_id: int,
    tip_ids: dict[str, int],
    mimic_relations: dict[str, tuple[str, float, float]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    joint_ids: set[int] = set()
    for tip_id in tip_ids.values():
        body_id = tip_id
        while body_id != wrist_id and body_id:
            start = int(model.body_jntadr[body_id])
            count = int(model.body_jntnum[body_id])
            joint_ids.update(range(start, start + count))
            body_id = int(model.body_parentid[body_id])
    path_joint_ids = []
    supported_joint_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    for joint_id in sorted(joint_ids):
        # MuJoCo 3.12 exposes ``model.jnt_type`` entries as NumPy integers.
        # Although scalar equality with ``mjtJoint`` works, tuple membership
        # can evaluate incorrectly and previously filtered out every MANO
        # hinge joint.  Compare normalized integer values instead.
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in supported_joint_types:
            continue
        path_joint_ids.append(joint_id)
    if not path_joint_ids:
        raise RuntimeError(
            "No hinge or slide joints found between the wrist and fingertips; "
            "cannot perform articulated-hand IK"
        )

    full_joint_names = [model.joint(joint_id).name for joint_id in path_joint_ids]
    full_name_set = set(full_joint_names)
    active_joint_names = [
        name for name in full_joint_names if name not in mimic_relations
    ]
    active_index = {
        name: index for index, name in enumerate(active_joint_names)
    }

    def resolve_affine(
        joint_name: str,
        resolving: frozenset[str] = frozenset(),
    ) -> tuple[np.ndarray, float]:
        if joint_name in active_index:
            row = np.zeros(len(active_joint_names), dtype=np.float64)
            row[active_index[joint_name]] = 1.0
            return row, 0.0
        if joint_name in resolving:
            raise ValueError(
                f"mimic cycle while resolving {joint_name}: {sorted(resolving)}"
            )
        if joint_name not in mimic_relations:
            raise ValueError(f"{joint_name} is neither active nor a mimic joint")
        source_name, multiplier, offset = mimic_relations[joint_name]
        if source_name not in full_name_set:
            raise ValueError(
                f"mimic source {source_name!r} for {joint_name!r} "
                "is outside the fingertip kinematic paths"
            )
        source_row, source_offset = resolve_affine(
            source_name, resolving | {joint_name}
        )
        return (
            multiplier * source_row,
            multiplier * source_offset + offset,
        )

    active_to_full = np.empty(
        (len(full_joint_names), len(active_joint_names)), dtype=np.float64
    )
    full_qpos_offset = np.empty(len(full_joint_names), dtype=np.float64)
    for index, name in enumerate(full_joint_names):
        active_to_full[index], full_qpos_offset[index] = resolve_affine(name)

    full_dof_ids = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in path_joint_ids],
        dtype=int,
    )
    full_qpos_ids = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in path_joint_ids],
        dtype=int,
    )
    active_joint_ids = [
        path_joint_ids[full_joint_names.index(name)]
        for name in active_joint_names
    ]
    active_dof_ids = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in active_joint_ids],
        dtype=int,
    )
    active_qpos_ids = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in active_joint_ids],
        dtype=int,
    )

    active_lower = np.full(len(active_joint_names), -np.inf, dtype=np.float64)
    active_upper = np.full(len(active_joint_names), np.inf, dtype=np.float64)
    for full_index, joint_id in enumerate(path_joint_ids):
        if not model.jnt_limited[joint_id]:
            continue
        nonzero = np.flatnonzero(np.abs(active_to_full[full_index]) > 1e-12)
        if len(nonzero) != 1:
            raise ValueError(
                f"{full_joint_names[full_index]} does not resolve to one active joint"
            )
        active_id = int(nonzero[0])
        coefficient = active_to_full[full_index, active_id]
        offset = full_qpos_offset[full_index]
        lower, upper = model.jnt_range[joint_id]
        mapped_lower = (lower - offset) / coefficient
        mapped_upper = (upper - offset) / coefficient
        if mapped_lower > mapped_upper:
            mapped_lower, mapped_upper = mapped_upper, mapped_lower
        active_lower[active_id] = max(active_lower[active_id], mapped_lower)
        active_upper[active_id] = min(active_upper[active_id], mapped_upper)
    if np.any(active_lower > active_upper):
        bad = np.flatnonzero(active_lower > active_upper)
        raise ValueError(
            "mimic-constrained joint limits are empty for "
            f"{[active_joint_names[index] for index in bad]}"
        )
    return (
        full_dof_ids,
        full_qpos_ids,
        active_dof_ids,
        active_qpos_ids,
        active_to_full,
        full_qpos_offset,
        active_lower,
        active_upper,
    )


def apply_mimic_positions(hand: HandIK) -> None:
    """Expand active coordinates into all path joints before FK."""
    active_q = hand.data.qpos[hand.qpos_ids]
    hand.data.qpos[hand.full_qpos_ids] = (
        hand.active_to_full @ active_q + hand.full_qpos_offset
    )


def assign_full_qpos(hand: HandIK, full_qpos: np.ndarray) -> None:
    """Assign a stored full trajectory while preserving active coordinates."""
    if full_qpos.shape != (len(hand.full_qpos_ids),):
        raise ValueError(
            f"{hand.hand_id}: expected {len(hand.full_qpos_ids)} qpos values, "
            f"got {full_qpos.shape}"
        )
    hand.data.qpos[hand.full_qpos_ids] = full_qpos
    apply_mimic_positions(hand)


def expanded_qpos(hand: HandIK) -> np.ndarray:
    apply_mimic_positions(hand)
    return hand.data.qpos[hand.full_qpos_ids].copy()


def build_hand(
    hand_id: str,
    display_name: str,
    ref_pos0: dict[str, np.ndarray],
    ref_rot0: dict[str, np.ndarray],
    *,
    identity_wrist_map: bool = False,
) -> HandIK:
    scene = make_scene(hand_id)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    tip_ids = {
        finger: model.body(link).id for finger, link in TIP_LINKS[hand_id].items()
    }
    if hand_id == "mano":
        tip_offsets = mano_fingertip_offsets()
    elif hand_id == "supr_female_foot":
        tip_offsets = supr_fingertip_offsets()
    else:
        tip_offsets = resolve_fingertip_offsets(
            DIRECT_ROOT / hand_id / "left" / "hand.urdf",
            TIP_LINKS[hand_id],
        )
    tip_directions = {
        finger: offset / np.linalg.norm(offset)
        for finger, offset in tip_offsets.items()
    }
    wrist_id = lowest_common_ancestor(model, list(tip_ids.values()))
    mimic_relations = urdf_mimic_relations(hand_id)
    (
        full_dof_ids,
        full_qpos_ids,
        dof_ids,
        qpos_ids,
        active_to_full,
        full_qpos_offset,
        active_lower_limits,
        active_upper_limits,
    ) = path_dofs(model, wrist_id, tip_ids, mimic_relations)
    data.qpos[full_qpos_ids] = (
        active_to_full @ data.qpos[qpos_ids] + full_qpos_offset
    )
    mujoco.mj_forward(model, data)
    neutral_p, neutral_r = relative_tip_poses(
        model,
        data,
        wrist_id,
        tip_ids,
        tip_offsets,
    )
    fingers = tuple(tip_ids)
    reference_vectors = np.stack([ref_pos0[f] for f in fingers])
    robot_vectors = np.stack([neutral_p[f] for f in fingers])
    unit_reference = reference_vectors / np.linalg.norm(
        reference_vectors, axis=1, keepdims=True
    )
    unit_robot = robot_vectors / np.linalg.norm(robot_vectors, axis=1, keepdims=True)
    map_rotation, _ = Rotation.align_vectors(unit_robot, unit_reference)
    if identity_wrist_map:
        map_rotation = Rotation.identity()
    size_ratio = float(
        np.median(
            np.linalg.norm(robot_vectors, axis=1)
            / np.linalg.norm(reference_vectors, axis=1)
        )
    )
    wrapper_mocap_id = int(model.body_mocapid[model.body("retarget_wrapper").id])
    object_mocap_id = int(model.body_mocapid[model.body("hocap_object_body").id])
    return HandIK(
        hand_id=hand_id,
        display_name=display_name,
        scene_path=scene,
        model=model,
        data=data,
        wrist_id=wrist_id,
        tip_ids=tip_ids,
        tip_offsets=tip_offsets,
        tip_directions=tip_directions,
        full_dof_ids=full_dof_ids,
        full_qpos_ids=full_qpos_ids,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        active_to_full=active_to_full,
        full_qpos_offset=full_qpos_offset,
        active_lower_limits=active_lower_limits,
        active_upper_limits=active_upper_limits,
        mimic_relations=mimic_relations,
        map_rotation=map_rotation,
        size_ratio=size_ratio,
        neutral_tip_rot=neutral_r,
        neutral_ref_rot={
            f: Rotation.from_quat(ref_rot0[f]) for f in fingers
        },
        wrapper_mocap_id=wrapper_mocap_id,
        object_mocap_id=object_mocap_id,
    )


def orientation_error(target: Rotation, current_matrix: np.ndarray) -> np.ndarray:
    return (target * Rotation.from_matrix(current_matrix).inv()).as_rotvec()


def direction_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Minimal rotation vector taking current unit direction onto target."""
    target = target / (np.linalg.norm(target) + 1e-12)
    current = current / (np.linalg.norm(current) + 1e-12)
    cross = np.cross(current, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    if cross_norm < 1e-10:
        if dot >= 0.0:
            return np.zeros(3, dtype=np.float64)
        fallback = np.cross(current, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback) < 1e-8:
            fallback = np.cross(current, np.array([0.0, 1.0, 0.0]))
        return np.pi * fallback / np.linalg.norm(fallback)
    return np.arctan2(cross_norm, dot) * cross / cross_norm


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def assign_wrist_pose(
    hand: HandIK,
    wrist_position: np.ndarray,
    wrist_rotation: Rotation,
) -> None:
    """Move the mocap wrapper so the actual wrist body has the requested pose."""
    model, data = hand.model, hand.data
    # Temporarily use identity wrapper to measure the internal wrist transform.
    data.mocap_pos[hand.wrapper_mocap_id] = 0.0
    data.mocap_quat[hand.wrapper_mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    local_p = data.xpos[hand.wrist_id].copy()
    local_r = Rotation.from_matrix(rotation_matrix(data, hand.wrist_id))
    wrapper_r = wrist_rotation * local_r.inv()
    wrapper_p = wrist_position - wrapper_r.apply(local_p)
    data.mocap_pos[hand.wrapper_mocap_id] = wrapper_p
    data.mocap_quat[hand.wrapper_mocap_id] = np.roll(wrapper_r.as_quat(), 1)


def solve_frame(
    hand: HandIK,
    reference_wrist_position: np.ndarray,
    reference_wrist_rotation: Rotation,
    ref_pos: dict[str, np.ndarray],
    ref_rot: dict[str, np.ndarray],
    iterations: int,
    wrist_position: np.ndarray,
    wrist_rotation: Rotation,
    *,
    initial_frame: bool = False,
) -> tuple[float, float, float, np.ndarray, Rotation]:
    model, data = hand.model, hand.data
    fingers = tuple(hand.tip_ids)
    target_wrist_r = reference_wrist_rotation * hand.map_rotation.inv()
    target_p = {
        f: reference_wrist_position + reference_wrist_rotation.apply(ref_pos[f])
        for f in fingers
    }
    target_world_direction = {}
    for f in fingers:
        target_local_direction = Rotation.from_quat(ref_rot[f]).apply(
            np.array([0.0, 0.0, 1.0])
        )
        target_world_direction[f] = reference_wrist_rotation.apply(
            target_local_direction
        )

    # The wrist orientation is an observed target, while wrist translation is
    # deliberately left free.  EgoEngine's retargeting uses the same
    # semantics: optimize translation and articulated joints against the
    # fingertip poses while preserving the measured wrist frame.
    wrist_rotation = target_wrist_r
    frame_start_q = data.qpos[hand.qpos_ids].copy()
    damping = 0.025
    # One radian of distal-axis error is treated like fifteen centimeters of
    # fingertip position error.  This is intentionally strong: the direction
    # is a physical final-segment constraint, not a cosmetic link-frame hint.
    orientation_length = 0.15
    temporal_regularization = 0.0 if initial_frame else 0.015
    max_frame_joint_delta = np.inf if initial_frame else 0.20
    for _ in range(iterations):
        assign_wrist_pose(hand, wrist_position, wrist_rotation)
        apply_mimic_positions(hand)
        mujoco.mj_forward(model, data)
        rows = []
        errors = []
        for f in fingers:
            body_id = hand.tip_ids[f]
            point = data.xpos[body_id].copy()
            if hand.tip_offsets is not None:
                point += (
                    rotation_matrix(data, body_id)
                    @ hand.tip_offsets[f]
                )
            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac_p, jac_r, point, body_id)
            lever = point - data.xpos[hand.wrist_id]
            rows.append(
                np.hstack(
                    (
                        np.eye(3),
                        jac_p[:, hand.full_dof_ids]
                        @ hand.active_to_full,
                    )
                )
            )
            errors.append(target_p[f] - point)
            current_direction = rotation_matrix(data, body_id) @ hand.tip_directions[f]
            direction_projector = np.eye(3) - np.outer(
                target_world_direction[f], target_world_direction[f]
            )
            rows.append(
                np.hstack(
                    (
                        np.zeros((3, 3)),
                        orientation_length
                        * direction_projector
                        @ (
                            jac_r[:, hand.full_dof_ids]
                            @ hand.active_to_full
                        ),
                    )
                )
            )
            errors.append(
                orientation_length
                * direction_error(
                    target_world_direction[f], current_direction
                )
            )
        if temporal_regularization > 0.0:
            rows.append(
                np.hstack(
                    (
                        np.zeros((len(hand.dof_ids), 3)),
                        temporal_regularization * np.eye(len(hand.dof_ids)),
                    )
                )
            )
            errors.append(
                temporal_regularization
                * (frame_start_q - data.qpos[hand.qpos_ids])
            )
        jacobian = np.vstack(rows)
        error = np.concatenate(errors)
        lhs = jacobian @ jacobian.T
        dq = jacobian.T @ np.linalg.solve(
            lhs + damping * damping * np.eye(len(error)), error
        )
        max_abs = float(np.max(np.abs(dq))) if len(dq) else 0.0
        if max_abs > 0.12:
            dq *= 0.12 / max_abs
        wrist_position += dq[:3]
        data.qpos[hand.qpos_ids] += dq[3:]
        data.qpos[hand.qpos_ids] = np.clip(
            data.qpos[hand.qpos_ids],
            frame_start_q - max_frame_joint_delta,
            frame_start_q + max_frame_joint_delta,
        )
        data.qpos[hand.qpos_ids] = np.clip(
            data.qpos[hand.qpos_ids],
            hand.active_lower_limits,
            hand.active_upper_limits,
        )
        if np.linalg.norm(error) < 5e-4:
            break

    assign_wrist_pose(hand, wrist_position, wrist_rotation)
    apply_mimic_positions(hand)
    mujoco.mj_forward(model, data)
    position_errors, rotation_errors = [], []
    for f in fingers:
        body_id = hand.tip_ids[f]
        point = data.xpos[body_id].copy()
        if hand.tip_offsets is not None:
            point += rotation_matrix(data, body_id) @ hand.tip_offsets[f]
        current_direction = rotation_matrix(data, body_id) @ hand.tip_directions[f]
        position_errors.append(np.linalg.norm(target_p[f] - point))
        rotation_errors.append(
            np.linalg.norm(
                direction_error(
                    target_world_direction[f], current_direction
                )
            )
        )
    wrist_error = np.linalg.norm(
        orientation_error(target_wrist_r, rotation_matrix(data, hand.wrist_id))
    )
    return (
        float(np.mean(position_errors)),
        float(np.mean(rotation_errors)),
        float(wrist_error),
        wrist_position,
        wrist_rotation,
    )


def configure_colors(model: mujoco.MjModel, color: tuple[float, ...]) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name not in {"hocap_object_geom", "table"}:
            model.geom_rgba[geom_id] = color


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def annotate(tile: np.ndarray, title: str, reference: bool = False) -> np.ndarray:
    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image, "RGBA")
    title_box = draw.textbbox((0, 0), title, font=font(17))
    draw.rounded_rectangle(
        (8, 7, title_box[2] + 27, 40),
        radius=7,
        fill=(5, 8, 14, 190),
    )
    draw.text((16, 11), title, font=font(17), fill="white")
    if reference:
        draw.rectangle(
            (2, 2, image.width - 3, image.height - 3),
            outline=(255, 42, 42, 255),
            width=6,
        )
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=14)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=270)
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument(
        "--reference-source",
        choices=("hocap_targets", "mano_command"),
        default="hocap_targets",
        help=(
            "retarget either the raw reviewed HO-Cap targets or FK targets "
            "from the saved pre-RL MANO hand_ctrl command trajectory"
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Render previously solved trajectories without rerunning IK",
    )
    parser.add_argument(
        "--initial-trajectory",
        type=Path,
        help=(
            "optional full-qpos trajectory used as a per-frame IK warm start; "
            "joint count and sampled frame count must match the target hand"
        ),
    )
    args = parser.parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mano_pose = np.load(SEQUENCE_ROOT / "mano_pose_left.npy")
    hand_joints_world = np.load(HAND_JOINTS_WORLD)
    object_pose = np.load(SEQUENCE_ROOT / "object_pose_G04_1.npy")
    if args.reference_source == "mano_command":
        wrist_pos, wrist_quat, ref_positions, ref_rotations = (
            mano_command_reference_trajectory()
        )
    else:
        wrist_pos, wrist_quat, ref_positions, ref_rotations = reference_trajectory(
            mano_pose, hand_joints_world
        )
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hand_ids = [hand_id for hand_id in registry["hands"] if hand_id != "mano"]
    hands = [
        build_hand(
            hand_id,
            registry["hands"][hand_id]["display_name"],
            {f: ref_positions[f][0] for f in FINGERS},
            {f: ref_rotations[f][0] for f in FINGERS},
        )
        for hand_id in hand_ids
    ]

    frame_ids = np.arange(0, len(wrist_pos), args.stride, dtype=int)
    initial_trajectory = None
    initial_wrist_position = None
    initial_wrist_quaternion = None
    if args.initial_trajectory is not None:
        with np.load(args.initial_trajectory) as initial:
            initial_trajectory = initial["qpos"].astype(np.float64)
            if "wrist_position" in initial:
                initial_wrist_position = initial["wrist_position"].astype(np.float64)
            if "wrist_quaternion_xyzw" in initial:
                initial_wrist_quaternion = initial[
                    "wrist_quaternion_xyzw"
                ].astype(np.float64)
        if len(initial_trajectory) != len(frame_ids):
            raise ValueError(
                "warm-start trajectory has "
                f"{len(initial_trajectory)} frames, expected {len(frame_ids)}"
            )
    diagnostics = {
        "reference_source": (
            "pre-RL MANO hand_ctrl forward kinematics"
            if args.reference_source == "mano_command"
            else "HO-Cap official hand_joints_3d"
        ),
        "surface_projection": False,
        "fingertip_position_semantics": "physical terminal mesh surface",
        "fingertip_orientation_semantics": (
            "distal joint center to physical fingertip axis"
        ),
    }
    for index, hand in enumerate(hands, 1):
        cache_path = CACHE_ROOT / f"{hand.hand_id}_ik.npz"
        if args.reuse_cache:
            if not cache_path.exists():
                raise FileNotFoundError(cache_path)
            continue
        q_trajectory, solved_wrist_p, solved_wrist_q = [], [], []
        p_errors, r_errors, wrist_errors = [], [], []
        wrist_position = wrist_pos[frame_ids[0]].copy()
        wrist_rotation = (
            Rotation.from_quat(wrist_quat[frame_ids[0]])
            * hand.map_rotation.inv()
        )
        for trajectory_index, frame_id in enumerate(frame_ids):
            if initial_trajectory is not None:
                if initial_trajectory.shape[1] != len(hand.full_qpos_ids):
                    raise ValueError(
                        f"{hand.hand_id}: warm-start qpos width "
                        f"{initial_trajectory.shape[1]} does not match "
                        f"{len(hand.full_qpos_ids)}"
                    )
                assign_full_qpos(hand, initial_trajectory[trajectory_index])
                if initial_wrist_position is not None:
                    wrist_position = initial_wrist_position[trajectory_index].copy()
                if initial_wrist_quaternion is not None:
                    wrist_rotation = Rotation.from_quat(
                        initial_wrist_quaternion[trajectory_index]
                    )
            (
                p_error,
                r_error,
                wrist_error,
                wrist_position,
                wrist_rotation,
            ) = solve_frame(
                hand,
                wrist_pos[frame_id],
                Rotation.from_quat(wrist_quat[frame_id]),
                {f: ref_positions[f][frame_id] for f in FINGERS},
                {f: ref_rotations[f][frame_id] for f in FINGERS},
                args.iterations,
                wrist_position,
                wrist_rotation,
                initial_frame=trajectory_index == 0,
            )
            q_trajectory.append(expanded_qpos(hand))
            solved_wrist_p.append(wrist_position.copy())
            solved_wrist_q.append(wrist_rotation.as_quat())
            p_errors.append(p_error)
            r_errors.append(r_error)
            wrist_errors.append(wrist_error)
        q_trajectory = np.asarray(q_trajectory)
        np.savez_compressed(
            cache_path,
            frame_ids=frame_ids,
            qpos_ids=hand.full_qpos_ids,
            active_qpos_ids=hand.qpos_ids,
            qpos=q_trajectory,
            wrist_position=np.asarray(solved_wrist_p),
            wrist_quaternion_xyzw=np.asarray(solved_wrist_q),
        )
        diagnostics[hand.hand_id] = {
            "fingers": list(hand.tip_ids),
            "wrist_body": hand.model.body(hand.wrist_id).name,
            "ik_dofs": int(len(hand.dof_ids)),
            "neutral_hand_to_mano_size_ratio": hand.size_ratio,
            "mean_tip_position_error_mm": 1000.0 * float(np.mean(p_errors)),
            "mean_tip_orientation_error_deg": float(
                np.rad2deg(np.mean(r_errors))
            ),
            "mean_wrist_orientation_error_deg": float(
                np.rad2deg(np.mean(wrist_errors))
            ),
        }
        print(
            f"[{index:02d}/{len(hands)}] {hand.hand_id}: "
            f"{diagnostics[hand.hand_id]['mean_tip_position_error_mm']:.1f} mm, "
            f"{diagnostics[hand.hand_id]['mean_tip_orientation_error_deg']:.1f} deg",
            flush=True,
        )

    diagnostics_path = ARTIFACT_ROOT / "all_hands_ik_diagnostics.json"
    if not args.reuse_cache:
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
    if args.solve_only:
        print(diagnostics_path)
        return

    # Put the verified MANO replay in the first tile.  It is the reference,
    # not another IK result, so its original 22 finger channels and exact
    # corrected wrist trajectory are used directly.
    mano_hand = build_hand(
        "mano",
        "MANO",
        {f: ref_positions[f][0] for f in FINGERS},
        {f: ref_rotations[f][0] for f in FINGERS},
    )
    hands = [mano_hand, *hands]

    palette = [
        (0.16, 0.58, 0.82, 1.0), (0.89, 0.35, 0.28, 1.0),
        (0.28, 0.72, 0.49, 1.0), (0.72, 0.47, 0.85, 1.0),
        (0.92, 0.67, 0.20, 1.0),
    ]
    renderers, cameras, trajectories = [], [], []
    solved_wrist_positions, solved_wrist_rotations = [], []
    for index, hand in enumerate(hands):
        configure_colors(hand.model, palette[index % len(palette)])
        renderers.append(
            mujoco.Renderer(
                hand.model, height=args.tile_height, width=args.tile_width
            )
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (-0.09, -0.05, 0.20)
        camera.distance = 0.54
        camera.azimuth = 132
        camera.elevation = -28
        cameras.append(camera)
        if hand.hand_id == "mano":
            trajectories.append(finger_q[frame_ids])
            solved_wrist_positions.append(wrist_pos[frame_ids])
            solved_wrist_rotations.append(wrist_quat[frame_ids])
        else:
            saved = np.load(CACHE_ROOT / f"{hand.hand_id}_ik.npz")
            trajectories.append(saved["qpos"])
            solved_wrist_positions.append(saved["wrist_position"])
            solved_wrist_rotations.append(saved["wrist_quaternion_xyzw"])

    columns, rows = 4, 4
    width, height = columns * args.tile_width, rows * args.tile_height
    output_video = ARTIFACT_ROOT / "hocap_all_hands_ik.mp4"
    preview_path = ARTIFACT_ROOT / "hocap_all_hands_ik_preview.png"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(output_video),
        ],
        stdin=subprocess.PIPE,
    )
    for output_frame, frame_id in enumerate(frame_ids):
        canvas = np.full((height, width, 3), 13, dtype=np.uint8)
        object_frame = object_pose[frame_id]
        for index, hand in enumerate(hands):
            assign_full_qpos(hand, trajectories[index][output_frame])
            assign_wrist_pose(
                hand,
                solved_wrist_positions[index][output_frame],
                Rotation.from_quat(
                    solved_wrist_rotations[index][output_frame]
                ),
            )
            hand.data.mocap_pos[hand.object_mocap_id] = object_frame[4:]
            hand.data.mocap_quat[hand.object_mocap_id] = np.roll(
                object_frame[:4], 1
            )
            mujoco.mj_forward(hand.model, hand.data)
            renderers[index].update_scene(hand.data, camera=cameras[index])
            tile = annotate(
                renderers[index].render(),
                hand.display_name,
                reference=hand.hand_id == "mano",
            )
            row, column = divmod(index, columns)
            canvas[
                row * args.tile_height:(row + 1) * args.tile_height,
                column * args.tile_width:(column + 1) * args.tile_width,
            ] = tile
        if output_frame == len(frame_ids) // 2:
            Image.fromarray(canvas).save(preview_path)
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.write(canvas.tobytes())
        if output_frame % 20 == 0:
            print(f"render {output_frame + 1}/{len(frame_ids)}", flush=True)
    assert ffmpeg.stdin is not None
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    for renderer in renderers:
        renderer.close()
    print(output_video)
    print(preview_path)
    print(diagnostics_path)


if __name__ == "__main__":
    main()
