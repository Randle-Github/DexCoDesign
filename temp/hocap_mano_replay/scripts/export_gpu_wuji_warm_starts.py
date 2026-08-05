#!/usr/bin/env python3
"""Materialize top GPU proposal trajectories as exact-IK warm starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.proposal) as data:
        indices = data["top_indices"].astype(np.int64)
        vectors = data["top_vectors"].astype(np.float64)
        qpos = data["top_qpos"].astype(np.float64)
        wrist_position = data["top_wrist_position"].astype(np.float64)
        wrist_quaternion = data["top_wrist_quaternion_xyzw"].astype(np.float64)
        frame_ids = data["frame_ids"].astype(np.int64)
        joint_names = data["joint_names"].astype(str)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, candidate_index in enumerate(indices):
        candidate = args.output_root / f"proposal_{rank:04d}_sample_{candidate_index:06d}"
        candidate.mkdir(parents=True, exist_ok=True)
        trajectory = candidate / "gpu_warm_start.npz"
        np.savez_compressed(
            trajectory,
            frame_ids=frame_ids,
            qpos=qpos[rank],
            wrist_position=wrist_position[rank],
            wrist_quaternion_xyzw=wrist_quaternion[rank],
            source_joint_names=joint_names,
        )
        row = {
            "rank": rank,
            "sample_index": int(candidate_index),
            "vector": vectors[rank].tolist(),
            "trajectory": str(trajectory.resolve()),
        }
        (candidate / "proposal.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        manifest.append(row)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"GPU_WUJI_WARM_STARTS_EXPORTED count={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
