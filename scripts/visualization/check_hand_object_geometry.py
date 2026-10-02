#!/usr/bin/env python3
"""Read-only spot check of canonical hand/object geometry; never changes poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    data = np.load(args.trajectory, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    np.random.seed(0)
    samples, _ = trimesh.sample.sample_surface(mesh, 20000)
    tree = cKDTree(samples)
    hand = np.asarray(data["hand_joints_m"], dtype=np.float64)
    valid = np.asarray(data["hand_valid"], dtype=bool)
    root = np.asarray(data["object_root_pose_wxyz"][:, 0], dtype=np.float64)
    frames = np.unique(np.linspace(0, len(root) - 1, min(args.frames, len(root)), dtype=int))
    report = {"dataset": metadata["dataset"], "sequence": metadata["sequence_id"],
              "frames": len(root), "sampled_frames": len(frames),
              "mesh_bounds_m": mesh.bounds.tolist(), "mesh_watertight": bool(mesh.is_watertight)}
    for convention in ("canonical_rotation", "official_gigahands_transpose"):
        gaps = []
        for frame in frames:
            points = hand[frame, valid[frame]].reshape(-1, 3)
            if not len(points):
                continue
            pose = root[frame]
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            if convention == "official_gigahands_transpose":
                rotation = rotation.T
            local = (points - pose[:3]) @ rotation
            distances = tree.query(local)[0]
            gaps.append(float(np.min(distances)))
        gaps = np.asarray(gaps)
        report[convention] = {
            "min_joint_to_mesh_median_m": float(np.median(gaps)),
            "min_joint_to_mesh_p10_m": float(np.percentile(gaps, 10)),
            "frames_min_joint_within_20mm": int(np.count_nonzero(gaps < 0.02)),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
