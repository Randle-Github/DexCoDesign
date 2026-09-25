#!/usr/bin/env python3
"""Prepare five compact ARCTIC trajectories for the direct-motor MANO assets.

ARCTIC stores 15 local MANO axis-angle rotations while the simulator asset uses
22 scalar anatomical joints.  The conversion below preserves the measured
world wrist transform and projects each local rotation onto the supported
abduction/flexion channels.  It is intended for dataset inspection/reset
visualization; later task retargeting should use the project IK pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SELECTED = (
    "009_scissors_s08.npz",
    "004_laptop_s09.npz",
    "000_box_s08.npz",
    "005_microwave_s02.npz",
    "010_waffleiron_s05.npz",
)
FINGER_ORDER = ("index", "middle", "pinky", "ring")
MANO_JOINT_BLOCK = {"index": 0, "middle": 3, "pinky": 6, "ring": 9, "thumb": 12}
URDF_CANONICAL_ROTATION = {
    "left": Rotation.from_euler("y", -np.pi / 2.0),
    "right": Rotation.from_euler("y", np.pi / 2.0),
}
LOCAL_HAND_TO_MANO = Rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
) * Rotation.from_rotvec(np.array([-np.pi / 2.0, 0.0, 0.0]))
# MANO transl is not the wrist joint.  This neutral J0 is the measured offset
# used by the existing MANO replay pipeline; shape-dependent changes are small.
MANO_J0 = np.array([-0.09675, 0.0063, 0.0061], dtype=np.float64)


def joint_names(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    )


def clip_to_joint(model: mujoco.MjModel, name: str, value: np.ndarray) -> np.ndarray:
    joint = model.joint(name)
    lo, hi = model.jnt_range[joint.id]
    return np.clip(value, lo, hi)


def scalar_finger_pose(model: mujoco.MjModel, pose: np.ndarray, side: str) -> np.ndarray:
    """Project full MANO local rotations into the simulator's 22 channels."""
    rotations = pose.reshape(len(pose), 15, 3).astype(np.float64)
    values: dict[str, np.ndarray] = {}
    mirror = -1.0 if side == "left" else 1.0
    for finger in FINGER_ORDER:
        start = MANO_JOINT_BLOCK[finger]
        mcp, pip, dip = (rotations[:, start + offset] for offset in range(3))
        # MANO's local Z component primarily describes MCP spread; remaining
        # magnitude is the flexion that the reduced scalar articulation keeps.
        abduct = mirror * mcp[:, 2]
        mcp_flex = np.linalg.norm(mcp[:, :2], axis=1)
        pip_flex = np.linalg.norm(pip, axis=1)
        dip_flex = np.linalg.norm(dip, axis=1)
        for suffix, raw in (("1y", abduct), ("1z", mcp_flex), ("2", pip_flex), ("3", dip_flex)):
            name = f"{side}_j_{finger}{suffix}"
            values[name] = clip_to_joint(model, name, raw)

    thumb = rotations[:, MANO_JOINT_BLOCK["thumb"] : MANO_JOINT_BLOCK["thumb"] + 3]
    t0, t1, t2 = thumb[:, 0], thumb[:, 1], thumb[:, 2]
    thumb_values = {
        "thumb1x": np.abs(t0[:, 0]),
        "thumb1y": mirror * t0[:, 2],
        "thumb1z": t0[:, 1],
        "thumb2y": mirror * t1[:, 2],
        "thumb2z": np.linalg.norm(t1[:, :2], axis=1),
        "thumb3": np.linalg.norm(t2, axis=1),
    }
    for suffix, raw in thumb_values.items():
        name = f"{side}_j_{suffix}"
        values[name] = clip_to_joint(model, name, raw)

    names = joint_names(model)
    q = np.zeros((len(pose), model.nq), dtype=np.float64)
    for name, raw in values.items():
        q[:, model.joint(name).qposadr[0]] = raw
    return q


def convert_hand(urdf: Path, source: np.lib.npyio.NpzFile, side: str, selection: slice):
    model = mujoco.MjModel.from_xml_path(str(urdf))
    pose = source[f"{side}_pose_axis_angle"][selection]
    orient = source[f"{side}_global_orient_axis_angle"][selection]
    translation = source[f"{side}_translation_m"][selection]
    q = scalar_finger_pose(model, pose, side)
    root_rotation = Rotation.from_rotvec(orient) * LOCAL_HAND_TO_MANO
    wrist = translation + Rotation.from_rotvec(orient).apply(MANO_J0)
    canonical = URDF_CANONICAL_ROTATION[side]
    local_translation = canonical.inv().apply(wrist)
    local_euler = (canonical.inv() * root_rotation).as_euler("XYZ")
    roots = np.column_stack((local_translation, local_euler))
    for column, suffix in enumerate(("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")):
        q[:, model.joint(f"{side}_{suffix}").qposadr[0]] = roots[:, column]
    return q.astype(np.float32), joint_names(model)


def interaction_window(source: np.lib.npyio.NpzFile) -> slice:
    obj = source["object_root_position_m"]
    left = source["left_translation_m"]
    right = source["right_translation_m"]
    distance = np.minimum(np.linalg.norm(left - obj, axis=1), np.linalg.norm(right - obj, axis=1))
    active = np.flatnonzero(distance < 0.48)
    if not len(active):
        return slice(0, len(obj))
    start = max(0, int(active[0]) - 15)
    stop = min(len(obj), int(active[-1]) + 16)
    return slice(start, stop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark_100",
        help="Benchmark subdirectory to convert (for example benchmark_100_table_fixed).",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/arctic_mano_preview"))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[3]
    mano = repo / "assets" / "robot_hands" / "direct_motor" / "mano"
    output.mkdir(parents=True, exist_ok=True)
    index = []

    for preview_id, filename in enumerate(SELECTED):
        source_path = root / args.benchmark_dir / filename
        if not source_path.is_file():
            continue
        source = np.load(source_path)
        metadata = json.loads(str(source["metadata_json"]))
        selection = interaction_window(source)
        name = metadata["object_id"]
        preview = output / f"{preview_id:02d}_{name}"
        if preview.exists():
            shutil.rmtree(preview)
        preview.mkdir(parents=True)
        payload = {}
        for side in ("left", "right"):
            q, names = convert_hand(mano / side / "hand.urdf", source, side, selection)
            payload[f"{side}_q"] = q
            payload[f"{side}_joint_names"] = names
        payload["object_root_position_m"] = source["object_root_position_m"][selection]
        payload["object_root_quaternion_wxyz"] = source["object_root_quaternion_wxyz"][selection]
        payload["object_joint_position_rad"] = source["object_joint_position_rad"][selection]

        positions = np.concatenate(
            (source["left_translation_m"][selection], source["right_translation_m"][selection],
             source["object_root_position_m"][selection]), axis=0
        )
        center = positions.mean(axis=0)
        span = np.maximum(positions.max(axis=0) - positions.min(axis=0), 0.35)
        distance = float(max(0.72, np.linalg.norm(span[:2]) * 1.25))
        scene = {
            "camera_target": center.tolist(),
            "camera_eye": (center + np.array([distance, distance, 0.65 * distance])).tolist(),
            "ground_z": float(positions[:, 2].min() - 0.22),
        }
        metadata.update({"source_file": filename, "source_frame_start": selection.start,
                         "source_frame_stop": selection.stop, "frames": len(payload["left_q"]),
                         "scene": scene, "hand_projection": "MANO axis-angle to 22-DoF anatomical projection"})
        payload["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(preview / "trajectory.npz", **payload)
        (preview / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        index.append({"directory": preview.name, **metadata})
        print(f"ARCTIC_PREVIEW_READY object={name} frames={metadata['frames']}", flush=True)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
