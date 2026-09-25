#!/usr/bin/env python3
"""Retarget canonical ARCTIC 21-point hands to the direct-motor MANO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.taco.prepare_taco_mano_previews import solve_hand  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-trajectory", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--hand-urdf", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument(
        "--priority-target-ids",
        type=int,
        nargs="*",
        default=(),
        help="Canonical MANO point IDs whose contact geometry must dominate IK.",
    )
    parser.add_argument("--priority-weight", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hand_urdf = args.hand_urdf or (
        REPO_ROOT / "assets" / "robot_hands" / "direct_motor" / "mano"
        / args.side / "hand.urdf"
    )
    with np.load(args.canonical_trajectory, allow_pickle=False) as source:
        sides = source["hand_sides"].astype(str).tolist()
        side_index = sides.index(args.side)
        targets = np.asarray(source["hand_joints_m"][:, side_index], dtype=np.float32)
        q, joint_names, diagnostics = solve_hand(
            hand_urdf.resolve(), targets, args.side, stride=1, iterations=args.iterations,
            priority_target_ids=set(args.priority_target_ids),
            priority_weight=args.priority_weight,
        )
        object_root = np.asarray(source["object_root_pose_wxyz"][:, 0], dtype=np.float32)
        object_joint = np.asarray(source["object_joint_positions_rad"][:, 0], dtype=np.float32)
        fps = float(np.asarray(source["fps"]))
        metadata = json.loads(str(source["metadata_json"]))

    metadata.update(
        {
            "fps": fps,
            "hand_retargeting": "21-point damped-least-squares IK",
            "hand_retargeting_diagnostics": diagnostics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            f"{args.side}_q": q,
            f"{args.side}_joint_names": joint_names,
            "object_root_position_m": object_root[:, :3],
            "object_root_quaternion_wxyz": object_root[:, 3:],
            "object_joint_position_rad": object_joint,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )
    print(
        "ARCTIC_IK_TRAJECTORY_READY "
        f"side={args.side} frames={len(q)} "
        f"mean_error_mm={diagnostics['mean_keypoint_error_mm']:.3f} "
        f"max_error_mm={diagnostics['max_frame_mean_keypoint_error_mm']:.3f} "
        f"output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
