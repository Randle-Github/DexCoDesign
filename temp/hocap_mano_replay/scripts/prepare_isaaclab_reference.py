#!/usr/bin/env python3
"""Export the reviewed HO-Cap/MANO replay as an Isaac Lab reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_mujoco import (  # noqa: E402
    HOCAP_MANO_ROOT_OFFSET_WORLD,
    LOCAL_HAND_TO_MANO,
    SEQUENCE_ROOT,
    URDF_CANONICAL_ROTATION,
    reduced_hand_trajectory,
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


def build_reference() -> dict[str, np.ndarray]:
    mano_pose = np.load(SEQUENCE_ROOT / "mano_pose_left.npy")
    object_pose = np.load(SEQUENCE_ROOT / "object_pose_G04_1.npy")
    finger_q, _ = reduced_hand_trajectory(mano_pose, object_pose)

    world_translation = mano_pose[:, 48:51] + HOCAP_MANO_ROOT_OFFSET_WORLD
    local_translation = URDF_CANONICAL_ROTATION.inv().apply(world_translation)
    world_rotation = Rotation.from_rotvec(mano_pose[:, :3]) * LOCAL_HAND_TO_MANO
    local_rotation = URDF_CANONICAL_ROTATION.inv() * world_rotation
    local_euler_xyz = local_rotation.as_euler("XYZ")
    hand_q = np.concatenate((local_translation, local_euler_xyz, finger_q), axis=1)

    # HO-Cap stores object pose as xyzw quaternion followed by translation.
    object_pose_wxyz = np.concatenate(
        (
            object_pose[:, 4:7],
            object_pose[:, 3:4],
            object_pose[:, 0:3],
        ),
        axis=1,
    )
    return {
        "joint_names": np.asarray(ROOT_JOINT_NAMES + FINGER_JOINT_NAMES),
        "hand_q": hand_q.astype(np.float32),
        "object_pose_wxyz": object_pose_wxyz.astype(np.float32),
        "fps": np.asarray(30.0, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=SEQUENCE_ROOT / "isaaclab_reference.npz",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = build_reference()
    np.savez_compressed(args.output, **reference)
    print(
        f"Saved {len(reference['hand_q'])} frames, "
        f"{reference['hand_q'].shape[1]} joints to {args.output}"
    )


if __name__ == "__main__":
    main()
