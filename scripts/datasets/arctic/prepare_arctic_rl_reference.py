#!/usr/bin/env python3
"""Convert one retargeted ARCTIC trajectory into the residual-RL schema.

The hand representation is identical to the rigid-object task.  Articulated
objects add only their internal joint positions to the existing 7D root pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT / "artifacts" / "arctic_mano_preview" / "00_scissors" / "trajectory.npz"
)
DEFAULT_HAND_URDF = (
    REPO_ROOT / "assets" / "robot_hands" / "direct_motor" / "mano" / "left" / "hand.urdf"
)
TIP_SUFFIXES = ("thumb3", "index3")
URDF_CANONICAL_ROTATION = {
    "left": Rotation.from_euler("y", -np.pi / 2.0),
    "right": Rotation.from_euler("y", np.pi / 2.0),
}


def resample_linear(values: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    result = np.column_stack(
        [np.interp(new_t, old_t, flat[:, column]) for column in range(flat.shape[1])]
    )
    return result.reshape((len(new_t), *values.shape[1:])).astype(np.float32)


def resample_quaternion_wxyz(
    values: np.ndarray, old_t: np.ndarray, new_t: np.ndarray
) -> np.ndarray:
    rotation = Rotation.from_quat(values[:, [1, 2, 3, 0]])
    xyzw = Slerp(old_t, rotation)(new_t).as_quat()
    return xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def fingertip_offset(mesh_root: Path, link_name: str, side: str) -> np.ndarray:
    """Return the center of the distal mesh cap in the link-local frame."""

    mesh = trimesh.load(
        mesh_root / f"{link_name}_visual.obj", process=False, force="mesh"
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    distal_sign = 1.0 if side == "left" else -1.0
    projection = distal_sign * vertices[:, 0]
    distal_projection = float(projection.max())
    cap = vertices[projection >= distal_projection - 5.0e-4]
    offset = cap.mean(axis=0)
    offset[0] = distal_sign * distal_projection
    return offset.astype(np.float32)


def fingertip_poses(
    hand_urdf: Path,
    hand_q: np.ndarray,
    joint_names: list[str],
    tip_names: tuple[str, str],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate thumb/index poses from the exact direct-motor MANO model."""

    model = mujoco.MjModel.from_xml_path(str(hand_urdf))
    data = mujoco.MjData(model)
    qpos_indices = np.asarray(
        [model.joint(name).qposadr[0] for name in joint_names], dtype=np.int32
    )
    body_indices = [model.body(name).id for name in tip_names]
    mesh_root = hand_urdf.parent / "meshes"
    offsets = np.stack(
        [fingertip_offset(mesh_root, name, side) for name in tip_names]
    )
    poses = np.empty((len(hand_q), len(tip_names), 7), dtype=np.float32)

    for frame_index, q in enumerate(hand_q):
        data.qpos[qpos_indices] = q
        mujoco.mj_forward(model, data)
        for tip_index, (body_index, offset) in enumerate(
            zip(body_indices, offsets, strict=True)
        ):
            rotation = np.asarray(data.xmat[body_index]).reshape(3, 3)
            poses[frame_index, tip_index, :3] = (
                data.xpos[body_index] + rotation @ offset
            )
            poses[frame_index, tip_index, 3:] = data.xquat[body_index]
    return poses, offsets


def build_reference(
    trajectory_path: Path,
    hand_urdf: Path,
    side: str,
    object_joint_name: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    target_fps: float = 30.0,
    time_scale: float = 1.0,
    world_translation: np.ndarray | None = None,
    root_control_lookahead_frames: int = 0,
    finger_control_lookahead_frames: int = 0,
    root_control_bias: np.ndarray | None = None,
    hand_root_offset: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build and validate one articulated-object training reference."""

    with np.load(trajectory_path, allow_pickle=False) as source:
        selection = slice(start_frame, end_frame)
        hand_q = np.asarray(source[f"{side}_q"][selection], dtype=np.float32)
        joint_names = source[f"{side}_joint_names"].astype(str).tolist()
        root_position = np.asarray(
            source["object_root_position_m"][selection], dtype=np.float32
        )
        root_quaternion = np.asarray(
            source["object_root_quaternion_wxyz"][selection], dtype=np.float32
        )
        object_joint = np.asarray(
            source["object_joint_position_rad"][selection], dtype=np.float32
        ).reshape(len(hand_q), -1)
        metadata = json.loads(str(source["metadata_json"]))

    source_fps = float(metadata.get("fps", 10.0))
    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive")
    if len(hand_q) > 1 and (
        not np.isclose(source_fps, target_fps) or not np.isclose(time_scale, 1.0)
    ):
        old_t = np.arange(len(hand_q), dtype=np.float64) / source_fps * time_scale
        new_count = int(round(old_t[-1] * target_fps)) + 1
        new_t = np.arange(new_count, dtype=np.float64) / target_fps
        new_t[-1] = old_t[-1]
        # Root rotations are Euler channels; unwrap them before interpolation
        # to avoid artificial 2*pi jumps. All remaining scalar joints are
        # interpolated within the already validated URDF joint ranges.
        hand_q = hand_q.copy()
        hand_q[:, 3:6] = np.unwrap(hand_q[:, 3:6], axis=0)
        hand_q = resample_linear(hand_q, old_t, new_t)
        root_position = resample_linear(root_position, old_t, new_t)
        root_quaternion = resample_quaternion_wxyz(root_quaternion, old_t, new_t)
        object_joint = resample_linear(object_joint, old_t, new_t)

    translation = np.zeros(3, dtype=np.float32)
    if world_translation is not None:
        translation = np.asarray(world_translation, dtype=np.float32)
        if translation.shape != (3,):
            raise ValueError("world_translation must contain exactly xyz")
        local_translation = URDF_CANONICAL_ROTATION[side].inv().apply(translation)
        hand_q[:, :3] += local_translation.astype(np.float32)
        root_position += translation

    root_offset = np.zeros(3, dtype=np.float32)
    if hand_root_offset is not None:
        root_offset = np.asarray(hand_root_offset, dtype=np.float32)
        if root_offset.shape != (3,):
            raise ValueError("hand_root_offset must contain exactly xyz")
        hand_q[:, :3] += root_offset

    if len(joint_names) != hand_q.shape[1]:
        raise ValueError("hand joint names and trajectory dimensions disagree")
    if object_joint.shape[1] != 1:
        raise ValueError(
            f"Scissors reference must have one internal joint, got {object_joint.shape}"
        )
    if not (
        len(root_position) == len(root_quaternion) == len(object_joint) == len(hand_q)
    ):
        raise ValueError("hand, root-pose, and articulation trajectories differ in length")

    tip_names = tuple(f"{side}_{suffix}" for suffix in TIP_SUFFIXES)
    tips, offsets = fingertip_poses(
        hand_urdf, hand_q, joint_names, tip_names, side
    )
    if root_control_lookahead_frames < 0 or finger_control_lookahead_frames < 0:
        raise ValueError("control lookahead must be non-negative")
    hand_ctrl = hand_q.copy()
    frame_ids = np.arange(len(hand_q))
    root_source = np.minimum(
        frame_ids + root_control_lookahead_frames, len(hand_q) - 1
    )
    finger_source = np.minimum(
        frame_ids + finger_control_lookahead_frames, len(hand_q) - 1
    )
    # ``hand_q`` remains the physical reference used by observations and
    # rewards.  Only the position-actuator command is phase-advanced to cancel
    # predictable critically damped tracking lag at contact-rich motion.
    hand_ctrl[:, :6] = hand_q[root_source, :6]
    hand_ctrl[:, 6:] = hand_q[finger_source, 6:]
    control_bias = np.zeros(3, dtype=np.float32)
    if root_control_bias is not None:
        control_bias = np.asarray(root_control_bias, dtype=np.float32)
        if control_bias.shape != (3,):
            raise ValueError("root_control_bias must contain exactly xyz")
        hand_ctrl[:, :3] += control_bias
    object_pose = np.concatenate((root_position, root_quaternion), axis=1)
    payload = {
        "joint_names": np.asarray(joint_names),
        "hand_q": hand_q,
        "hand_ctrl": hand_ctrl,
        "object_pose_wxyz": object_pose.astype(np.float32),
        "object_joint_position_rad": object_joint,
        "object_joint_names": np.asarray([object_joint_name]),
        "fingertip_pose_wxyz": tips,
        "fingertip_link_names": np.asarray(tip_names),
        "fingertip_offsets": offsets,
        "fps": np.asarray(target_fps, dtype=np.float32),
    }
    report = {
        "schema": "dexcodesign.residual_reference.articulated.v1",
        "source": str(trajectory_path.resolve()),
        "side": side,
        "frames": int(len(hand_q)),
        "source_frame_range": [start_frame, end_frame],
        "source_fps": source_fps,
        "target_fps": target_fps,
        "time_scale": time_scale,
        "root_control_lookahead_frames": root_control_lookahead_frames,
        "finger_control_lookahead_frames": finger_control_lookahead_frames,
        "root_control_bias": control_bias.tolist(),
        "hand_root_offset": root_offset.tolist(),
        "world_translation_m": translation.tolist(),
        "hand_dof": int(hand_q.shape[1]),
        "object_joint_names": [object_joint_name],
        "object_joint_range_rad": [
            float(object_joint.min()),
            float(object_joint.max()),
        ],
        "object_pose_representation": (
            "root_xyz + root_quaternion_wxyz + internal_joint_positions"
        ),
    }
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hand-urdf", type=Path, default=DEFAULT_HAND_URDF)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--object-joint-name", default="articulation")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Stretch trajectory duration by this factor while keeping target FPS fixed.",
    )
    parser.add_argument("--root-control-lookahead-frames", type=int, default=0)
    parser.add_argument("--finger-control-lookahead-frames", type=int, default=0)
    parser.add_argument(
        "--root-control-bias",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Constant feed-forward bias for the three virtual root position commands.",
    )
    parser.add_argument(
        "--hand-root-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Offset the hand reference root only; the object pose is unchanged.",
    )
    parser.add_argument(
        "--world-translation", type=float, nargs=3, metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.trajectory.with_name("isaaclab_reference.npz")
    payload, report = build_reference(
        args.trajectory.resolve(),
        args.hand_urdf.resolve(),
        args.side,
        args.object_joint_name,
        args.start_frame,
        args.end_frame,
        args.target_fps,
        args.time_scale,
        np.asarray(args.world_translation, dtype=np.float32),
        args.root_control_lookahead_frames,
        args.finger_control_lookahead_frames,
        np.asarray(args.root_control_bias, dtype=np.float32),
        np.asarray(args.hand_root_offset, dtype=np.float32),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "ARCTIC_RL_REFERENCE_READY "
        f"frames={report['frames']} hand_dof={report['hand_dof']} "
        f"object_joint_range_rad={report['object_joint_range_rad']} "
        f"output={output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
