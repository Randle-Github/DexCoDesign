#!/usr/bin/env python3
"""Convert the selected TACO benchmark to the canonical pose schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "common"))
from trajectory_schema import save_trajectory  # noqa: E402


def pose_wxyz(matrices: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.column_stack((matrices[:, :3, 3], xyzw[:, [3, 0, 1, 2]])).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/taco_v1"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--source-fps", type=float, default=30.0)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "canonical_100")).resolve()
    records = [json.loads(line) for line in (root / "manifests" / "selected_100.jsonl").read_text().splitlines() if line]
    converted = []
    for index, record in enumerate(records):
        sequence = record["sequence_id"]
        hands = np.load(root / "raw" / "hand_poses_3d" / sequence / "hand_joints.npy")
        object_files = sorted((root / "raw" / "object_poses" / sequence).glob("*.npy"))
        count = min(len(hands), *(len(np.load(path, mmap_mode="r")) for path in object_files))
        frame_ids = np.arange(0, count, args.stride, dtype=np.int64)
        roots, objects = [], []
        for path in object_files:
            role, object_id = path.stem.split("_", 1)
            roots.append(pose_wxyz(np.load(path)[frame_ids]))
            objects.append({"object_id": object_id, "role": role, "joint_names": [], "joint_types": []})
        destination = output / f"{index:03d}" / "trajectory.npz"
        save_trajectory(
            destination,
            hand_joints_m=np.asarray(hands[frame_ids], dtype=np.float32),
            hand_valid=np.isfinite(hands[frame_ids]).all(axis=(2, 3)),
            object_root_pose_wxyz=np.stack(roots, axis=1),
            object_joint_positions_rad=np.empty((len(frame_ids), len(roots), 0), dtype=np.float32),
            frame_indices=frame_ids,
            fps=args.source_fps / args.stride,
            metadata={
                "dataset": "TACO", "sequence_id": sequence,
                "hand_representation": "source 21 world-space joints", "objects": objects,
                "action": record["action"],
            },
        )
        converted.append({"sequence_id": sequence, "path": str(destination.relative_to(root))})
        print(f"TACO_CANONICAL_READY {index + 1:03d}/{len(records)} {sequence}", flush=True)
    (output / "manifest.json").write_text(
        json.dumps({"schema": "dexcodesign.pose_dataset.v1", "dataset": "TACO", "records": converted}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
