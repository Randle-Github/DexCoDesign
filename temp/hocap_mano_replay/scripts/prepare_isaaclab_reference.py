#!/usr/bin/env python3
"""Export the reviewed HO-Cap/MANO replay as an Isaac Lab reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_mujoco import (  # noqa: E402
    HAND_URDF,
    MANO_TIP_LINKS,
    URDF_CANONICAL_ROTATION,
    mano_fingertip_offsets,
)
from retarget_all_hands import (  # noqa: E402
    CACHE_ROOT,
    FINGERS,
    build_hand,
    reference_trajectory,
    solve_frame,
)


DEFAULT_SEQUENCE_ROOT = (
    EXPERIMENT_ROOT / "data" / "subset" / "subject_7" / "20231022_192832"
)


ROOT_JOINT_NAMES = (
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_rot_x",
    "left_rot_y",
    "left_rot_z",
)
FINGER_JOINT_NAMES = (
    "left_j_index1y",
    "left_j_index1z",
    "left_j_index2",
    "left_j_index3",
    "left_j_middle1y",
    "left_j_middle1z",
    "left_j_middle2",
    "left_j_middle3",
    "left_j_pinky1y",
    "left_j_pinky1z",
    "left_j_pinky2",
    "left_j_pinky3",
    "left_j_ring1y",
    "left_j_ring1z",
    "left_j_ring2",
    "left_j_ring3",
    "left_j_thumb1x",
    "left_j_thumb1y",
    "left_j_thumb1z",
    "left_j_thumb2y",
    "left_j_thumb2z",
    "left_j_thumb3",
)
POLICY_FINGERS = ("thumb", "index")


def reference_fingertip_poses(
    hand_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the reference thumb/index tip poses in the direct-motor model."""
    model = mujoco.MjModel.from_xml_path(str(HAND_URDF))
    data = mujoco.MjData(model)
    joint_names = ROOT_JOINT_NAMES + FINGER_JOINT_NAMES
    qpos_ids = np.asarray(
        [model.joint(name).qposadr[0] for name in joint_names],
        dtype=int,
    )
    link_names = np.asarray([MANO_TIP_LINKS[finger] for finger in POLICY_FINGERS])
    body_ids = [model.body(link_name).id for link_name in link_names]
    offsets_by_finger = mano_fingertip_offsets()
    offsets = np.stack([offsets_by_finger[finger] for finger in POLICY_FINGERS])
    poses = np.empty((len(hand_q), len(POLICY_FINGERS), 7), dtype=np.float32)

    for frame_id, q in enumerate(hand_q):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        for tip_id, (body_id, offset) in enumerate(zip(body_ids, offsets)):
            rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
            poses[frame_id, tip_id, :3] = data.xpos[body_id] + rotation @ offset
            # MuJoCo and Isaac Lab both expose body quaternions as wxyz.
            poses[frame_id, tip_id, 3:] = data.xquat[body_id]

    return poses, link_names, offsets.astype(np.float32)


def build_reference(
    sequence_root: Path = DEFAULT_SEQUENCE_ROOT,
    object_pose_path: Path | None = None,
    iterations: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    mano_pose = np.load(sequence_root / "mano_pose_left.npy")
    hand_joints_world = np.load(sequence_root / "hand_joints_3d_left.npy")
    if object_pose_path is None:
        metadata_path = sequence_root / "subset.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        object_pose_path = sequence_root / f"object_pose_{metadata['object_id']}.npy"
    object_pose = np.load(object_pose_path)
    wrist_pos, wrist_quat, ref_positions, ref_rotations = reference_trajectory(
        mano_pose, hand_joints_world
    )

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    hand = build_hand(
        "mano",
        "MANO",
        {finger: ref_positions[finger][0] for finger in FINGERS},
        {finger: ref_rotations[finger][0] for finger in FINGERS},
        identity_wrist_map=True,
    )
    finger_qpos = np.asarray(
        [
            hand.model.joint(name).qposadr[0]
            for name in FINGER_JOINT_NAMES
        ],
        dtype=int,
    )
    solved_finger_q = []
    solved_wrist_pos = []
    solved_wrist_quat = []
    position_errors = []
    rotation_errors = []
    wrist_errors = []
    wrist_position = wrist_pos[0].astype(np.float64).copy()
    wrist_rotation = Rotation.from_quat(wrist_quat[0])

    for frame_id in range(len(mano_pose)):
        (
            position_error,
            rotation_error,
            wrist_error,
            wrist_position,
            wrist_rotation,
        ) = solve_frame(
            hand,
            wrist_pos[frame_id],
            Rotation.from_quat(wrist_quat[frame_id]),
            {finger: ref_positions[finger][frame_id] for finger in FINGERS},
            {finger: ref_rotations[finger][frame_id] for finger in FINGERS},
            iterations,
            wrist_position,
            wrist_rotation,
            initial_frame=frame_id == 0,
        )
        solved_finger_q.append(hand.data.qpos[finger_qpos].copy())
        solved_wrist_pos.append(wrist_position.copy())
        solved_wrist_quat.append(wrist_rotation.as_quat())
        position_errors.append(position_error)
        rotation_errors.append(rotation_error)
        wrist_errors.append(wrist_error)

    solved_finger_q = np.asarray(solved_finger_q)
    solved_wrist_pos = np.asarray(solved_wrist_pos)
    solved_wrist_rotation = Rotation.from_quat(np.asarray(solved_wrist_quat))
    local_translation = URDF_CANONICAL_ROTATION.inv().apply(solved_wrist_pos)
    local_rotation = URDF_CANONICAL_ROTATION.inv() * solved_wrist_rotation
    local_euler_xyz = np.unwrap(local_rotation.as_euler("XYZ"), axis=0)
    hand_q = np.concatenate(
        (local_translation, local_euler_xyz, solved_finger_q),
        axis=1,
    )
    fingertip_pose_wxyz, fingertip_link_names, fingertip_offsets = (
        reference_fingertip_poses(hand_q)
    )

    # HO-Cap stores object pose as xyzw quaternion followed by translation.
    object_pose_wxyz = np.concatenate(
        (
            object_pose[:, 4:7],
            object_pose[:, 3:4],
            object_pose[:, 0:3],
        ),
        axis=1,
    )
    return (
        {
            "joint_names": np.asarray(ROOT_JOINT_NAMES + FINGER_JOINT_NAMES),
            "hand_q": hand_q.astype(np.float32),
            # EgoEngine uses a separately named control reference.  The MANO
            # articulation has position actuators, so the temporally
            # warm-started IK target is the initial ctrl_ref.
            "hand_ctrl": hand_q.astype(np.float32),
            "object_pose_wxyz": object_pose_wxyz.astype(np.float32),
            "fingertip_pose_wxyz": fingertip_pose_wxyz,
            "fingertip_link_names": fingertip_link_names,
            "fingertip_offsets": fingertip_offsets,
            "fps": np.asarray(30.0, dtype=np.float32),
        },
        {
            "source": "HO-Cap official hand_joints_3d",
            "surface_projection": False,
            "ik_iterations_per_frame": iterations,
            "mean_fingertip_position_error_mm": (
                1000.0 * float(np.mean(position_errors))
            ),
            "max_fingertip_position_error_mm": (
                1000.0 * float(np.max(position_errors))
            ),
            "mean_fingertip_orientation_error_rad": float(
                np.mean(rotation_errors)
            ),
            "mean_wrist_orientation_error_rad": float(
                np.mean(wrist_errors)
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT
    )
    parser.add_argument("--object-pose", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()
    output = args.output or args.sequence_root / "isaaclab_reference.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    reference, diagnostics = build_reference(
        sequence_root=args.sequence_root,
        object_pose_path=args.object_pose,
        iterations=args.iterations,
    )
    np.savez_compressed(output, **reference)
    diagnostics_path = output.with_suffix(".retargeting.json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(reference['hand_q'])} frames, "
        f"{reference['hand_q'].shape[1]} joints to {output}"
    )
    print(diagnostics_path)


if __name__ == "__main__":
    main()
