#!/usr/bin/env python3
"""Convert existing HO-Cap task clips to the common pose trajectory schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory_schema import save_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("temp/hocap_mano_replay/data/tasks"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/hocap_v1/canonical"))
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    converted = []
    for task in sorted(args.tasks.glob("*/subset.json")):
        directory = task.parent
        info = json.loads(task.read_text())
        joints_path = directory / "hand_joints_3d_left.npy"
        object_paths = sorted(directory.glob("object_pose_*.npy"))
        if not joints_path.is_file() or len(object_paths) != 1:
            continue
        left = np.asarray(np.load(joints_path), dtype=np.float32)
        pose_xyzw_xyz = np.asarray(np.load(object_paths[0]), dtype=np.float32)
        count = min(len(left), len(pose_xyzw_xyz))
        hands = np.full((count, 2, 21, 3), np.nan, dtype=np.float32)
        hands[:, 0] = left[:count]
        valid = np.zeros((count, 2), dtype=bool)
        valid[:, 0] = np.isfinite(left[:count]).all(axis=(1, 2))
        object_root = np.column_stack(
            (pose_xyzw_xyz[:count, 4:7], pose_xyzw_xyz[:count, [3, 0, 1, 2]])
        )[:, None, :]
        destination = args.output / info["task_id"] / "trajectory.npz"
        save_trajectory(
            destination,
            hand_joints_m=hands,
            hand_valid=valid,
            object_root_pose_wxyz=object_root,
            object_joint_positions_rad=np.empty((count, 1, 0), dtype=np.float32),
            frame_indices=np.arange(count, dtype=np.int64),
            fps=args.fps,
            metadata={
                "dataset": "HO-Cap", "sequence_id": info["sequence"],
                "task_id": info["task_id"],
                "hand_representation": "source 21 world-space joints; left hand only",
                "objects": [{"object_id": info["object_id"], "role": "manipulated_object", "joint_names": [], "joint_types": []}],
            },
        )
        converted.append({"task_id": info["task_id"], "path": str(destination)})
        print(f"HOCAP_CANONICAL_READY {info['task_id']}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "HO-Cap", "records": converted}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
