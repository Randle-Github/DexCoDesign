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
DIRECT_ROOT = REPO_ROOT / "assets" / "robot_hands" / "direct_motor"
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_mujoco import (  # noqa: E402
    HOCAP_MANO_ROOT_OFFSET_WORLD,
    LOCAL_HAND_TO_MANO,
    mano_fingertip_offsets,
    surface_snapped_hand_trajectory,
)


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_TIPS = {
    "thumb": "left_thumb3",
    "index": "left_index3",
    "middle": "left_middle3",
    "ring": "left_ring3",
    "pinky": "left_pinky3",
}
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
    dof_ids: np.ndarray
    qpos_ids: np.ndarray
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
    compiler.set("meshdir", str(hand_urdf.parent))
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
    ET.SubElement(visual, "global", {"offwidth": "360", "offheight": "270"})
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


def reference_trajectory(
    mano_pose: np.ndarray,
    finger_q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Return exact wrist world poses and MANO tip poses in wrist coordinates."""
    urdf = DIRECT_ROOT / "mano" / "left" / "hand.urdf"
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    wrist_id = model.body("left_palm").id
    tip_ids = {f: model.body(name).id for f, name in MANO_TIPS.items()}
    tip_offsets = mano_fingertip_offsets()
    root_names = (
        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_rot_x", "left_rot_y", "left_rot_z",
    )
    finger_names = (
        "left_j_index1y", "left_j_index1z", "left_j_index2", "left_j_index3",
        "left_j_middle1y", "left_j_middle1z", "left_j_middle2", "left_j_middle3",
        "left_j_pinky1y", "left_j_pinky1z", "left_j_pinky2", "left_j_pinky3",
        "left_j_ring1y", "left_j_ring1z", "left_j_ring2", "left_j_ring3",
        "left_j_thumb1x", "left_j_thumb1y", "left_j_thumb1z",
        "left_j_thumb2y", "left_j_thumb2z", "left_j_thumb3",
    )
    for name in root_names:
        data.qpos[model.joint(name).qposadr[0]] = 0.0
    finger_qpos = [model.joint(name).qposadr[0] for name in finger_names]
    positions = {finger: [] for finger in FINGERS}
    rotations = {finger: [] for finger in FINGERS}
    for q in finger_q:
        data.qpos[finger_qpos] = q
        mujoco.mj_forward(model, data)
        p, r = relative_tip_poses(
            model, data, wrist_id, tip_ids, tip_offsets
        )
        for finger in FINGERS:
            positions[finger].append(p[finger])
            rotations[finger].append(r[finger].as_quat())

    wrist_pos = mano_pose[:, 48:51] + HOCAP_MANO_ROOT_OFFSET_WORLD
    wrist_quat = np.asarray(
        [
            (Rotation.from_rotvec(pose[:3]) * LOCAL_HAND_TO_MANO).as_quat()
            for pose in mano_pose
        ]
    )
    return (
        wrist_pos,
        wrist_quat,
        {k: np.asarray(v) for k, v in positions.items()},
        {k: np.asarray(v) for k, v in rotations.items()},
    )


def path_dofs(
    model: mujoco.MjModel,
    wrist_id: int,
    tip_ids: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids: set[int] = set()
    for tip_id in tip_ids.values():
        body_id = tip_id
        while body_id != wrist_id and body_id:
            start = int(model.body_jntadr[body_id])
            count = int(model.body_jntnum[body_id])
            joint_ids.update(range(start, start + count))
            body_id = int(model.body_parentid[body_id])
    dofs, qpos = [], []
    for joint_id in sorted(joint_ids):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        dofs.append(int(model.jnt_dofadr[joint_id]))
        qpos.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(dofs, dtype=int), np.asarray(qpos, dtype=int)


def build_hand(
    hand_id: str,
    display_name: str,
    ref_pos0: dict[str, np.ndarray],
    ref_rot0: dict[str, np.ndarray],
) -> HandIK:
    scene = make_scene(hand_id)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    tip_ids = {
        finger: model.body(link).id for finger, link in TIP_LINKS[hand_id].items()
    }
    wrist_id = lowest_common_ancestor(model, list(tip_ids.values()))
    mujoco.mj_forward(model, data)
    neutral_p, neutral_r = relative_tip_poses(model, data, wrist_id, tip_ids)
    fingers = tuple(tip_ids)
    reference_vectors = np.stack([ref_pos0[f] for f in fingers])
    robot_vectors = np.stack([neutral_p[f] for f in fingers])
    unit_reference = reference_vectors / np.linalg.norm(
        reference_vectors, axis=1, keepdims=True
    )
    unit_robot = robot_vectors / np.linalg.norm(robot_vectors, axis=1, keepdims=True)
    map_rotation, _ = Rotation.align_vectors(unit_robot, unit_reference)
    size_ratio = float(
        np.median(
            np.linalg.norm(robot_vectors, axis=1)
            / np.linalg.norm(reference_vectors, axis=1)
        )
    )
    dof_ids, qpos_ids = path_dofs(model, wrist_id, tip_ids)
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
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
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
) -> tuple[float, float, float, np.ndarray, Rotation]:
    model, data = hand.model, hand.data
    fingers = tuple(hand.tip_ids)
    target_wrist_r = reference_wrist_rotation * hand.map_rotation.inv()
    target_p = {
        f: reference_wrist_position + reference_wrist_rotation.apply(ref_pos[f])
        for f in fingers
    }
    target_local_r = {}
    for f in fingers:
        ref_current = Rotation.from_quat(ref_rot[f])
        delta = ref_current * hand.neutral_ref_rot[f].inv()
        mapped_delta = hand.map_rotation * delta * hand.map_rotation.inv()
        target_local_r[f] = mapped_delta * hand.neutral_tip_rot[f]

    damping = 0.035
    orientation_length = 0.025
    for _ in range(iterations):
        assign_wrist_pose(hand, wrist_position, wrist_rotation)
        mujoco.mj_forward(model, data)
        rows = [
            np.hstack(
                (
                    np.zeros((3, 3)),
                    orientation_length * np.eye(3),
                    np.zeros((3, len(hand.dof_ids))),
                )
            )
        ]
        errors = [
            orientation_length
            * orientation_error(target_wrist_r, rotation_matrix(data, hand.wrist_id))
        ]
        for f in fingers:
            body_id = hand.tip_ids[f]
            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac_p, jac_r, body_id)
            lever = data.xpos[body_id] - data.xpos[hand.wrist_id]
            rows.append(
                np.hstack(
                    (
                        np.eye(3),
                        -skew(lever),
                        jac_p[:, hand.dof_ids],
                    )
                )
            )
            errors.append(target_p[f] - data.xpos[body_id])
            target_world_r = target_wrist_r * target_local_r[f]
            rows.append(
                np.hstack(
                    (
                        np.zeros((3, 3)),
                        orientation_length * np.eye(3),
                        orientation_length * jac_r[:, hand.dof_ids],
                    )
                )
            )
            errors.append(
                orientation_length
                * orientation_error(
                    target_world_r, rotation_matrix(data, body_id)
                )
            )
        jacobian = np.vstack(rows)
        error = np.concatenate(errors)
        lhs = jacobian @ jacobian.T
        dq = jacobian.T @ np.linalg.solve(
            lhs + damping * damping * np.eye(len(error)), error
        )
        max_abs = float(np.max(np.abs(dq))) if len(dq) else 0.0
        if max_abs > 0.18:
            dq *= 0.18 / max_abs
        wrist_position += dq[:3]
        wrist_rotation = Rotation.from_rotvec(dq[3:6]) * wrist_rotation
        data.qpos[hand.qpos_ids] += dq[6:]
        for qpos_id, dof_id in zip(hand.qpos_ids, hand.dof_ids):
            joint_id = int(model.dof_jntid[dof_id])
            if model.jnt_limited[joint_id]:
                data.qpos[qpos_id] = np.clip(
                    data.qpos[qpos_id],
                    model.jnt_range[joint_id, 0],
                    model.jnt_range[joint_id, 1],
                )
        if np.linalg.norm(error) < 5e-4:
            break

    assign_wrist_pose(hand, wrist_position, wrist_rotation)
    mujoco.mj_forward(model, data)
    position_errors, rotation_errors = [], []
    for f in fingers:
        body_id = hand.tip_ids[f]
        target_world_r = target_wrist_r * target_local_r[f]
        position_errors.append(np.linalg.norm(target_p[f] - data.xpos[body_id]))
        rotation_errors.append(
            np.linalg.norm(
                orientation_error(
                    target_world_r, rotation_matrix(data, body_id)
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
        "--reuse-cache",
        action="store_true",
        help="Render previously solved trajectories without rerunning IK",
    )
    args = parser.parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mano_pose = np.load(SEQUENCE_ROOT / "mano_pose_left.npy")
    object_pose = np.load(SEQUENCE_ROOT / "object_pose_G04_1.npy")
    finger_q, projection_diagnostics = surface_snapped_hand_trajectory(
        mano_pose, object_pose
    )
    wrist_pos, wrist_quat, ref_positions, ref_rotations = reference_trajectory(
        mano_pose, finger_q
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

    frame_ids = np.arange(0, len(mano_pose), args.stride, dtype=int)
    diagnostics = {
        "mano_fingertip_surface_projection": projection_diagnostics
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
        for frame_id in frame_ids:
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
            )
            q_trajectory.append(hand.data.qpos[hand.qpos_ids].copy())
            solved_wrist_p.append(wrist_position.copy())
            solved_wrist_q.append(wrist_rotation.as_quat())
            p_errors.append(p_error)
            r_errors.append(r_error)
            wrist_errors.append(wrist_error)
        q_trajectory = np.asarray(q_trajectory)
        np.savez_compressed(
            cache_path,
            frame_ids=frame_ids,
            qpos_ids=hand.qpos_ids,
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
            hand.data.qpos[hand.qpos_ids] = trajectories[index][output_frame]
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
