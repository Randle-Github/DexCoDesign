#!/usr/bin/env python3
"""Prepare synchronized left/right MANO references for one ARCTIC trajectory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--target-fps", type=float, default=30.0)
    args = parser.parse_args()

    source = args.canonical_trajectory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}
    for side in ("left", "right"):
        hand_urdf = (
            REPO_ROOT
            / "assets/robot_hands/direct_motor/mano"
            / side
            / "hand.urdf"
        )
        ik_path = output / f"trajectory_ik_{side}.npz"
        reference_path = output / f"isaaclab_reference_{side}.npz"
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/datasets/arctic/prepare_arctic_ik_trajectory.py"),
                "--canonical-trajectory",
                str(source),
                "--side",
                side,
                "--iterations",
                str(args.iterations),
                "--output",
                str(ik_path),
            ]
        )
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/datasets/arctic/prepare_arctic_rl_reference.py"),
                "--trajectory",
                str(ik_path),
                "--hand-urdf",
                str(hand_urdf),
                "--side",
                side,
                "--target-fps",
                str(args.target_fps),
                "--output",
                str(reference_path),
            ]
        )
        records[f"{side}_reference"] = str(reference_path)

    with np.load(records["left_reference"]) as left, np.load(
        records["right_reference"]
    ) as right:
        if not np.allclose(
            left["object_pose_wxyz"], right["object_pose_wxyz"], atol=1.0e-6
        ):
            raise RuntimeError("Left/right references disagree on object trajectory")
        if not np.allclose(
            left["object_joint_position_rad"],
            right["object_joint_position_rad"],
            atol=1.0e-6,
        ):
            raise RuntimeError("Left/right references disagree on articulation")
        frames = len(left["hand_q"])

    manifest = {
        "schema": "dexcodesign.arctic_bimanual_reference.v1",
        "canonical_trajectory": str(source),
        "frames": frames,
        **records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"ARCTIC_BIMANUAL_REFERENCE_READY frames={frames} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
