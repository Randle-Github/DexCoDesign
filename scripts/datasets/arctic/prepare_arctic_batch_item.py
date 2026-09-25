#!/usr/bin/env python3
"""Prepare one tabletop-aligned ARCTIC benchmark item for right-hand residual RL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "datasets" / "arctic_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "artifacts" / "arctic_rl_100"
    )
    args = parser.parse_args()

    sources = sorted((args.root / "canonical_100_tabletop").glob("*/trajectory.npz"))
    if not 0 <= args.index < len(sources):
        raise IndexError(f"index {args.index} outside prepared set of {len(sources)}")
    source = sources[args.index]
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    object_id = str(metadata["objects"][0]["object_id"])
    sample_id = source.parent.name
    destination = args.output / sample_id
    destination.mkdir(parents=True, exist_ok=True)
    ik_path = destination / "trajectory_ik_right.npz"
    reference_path = destination / "isaaclab_reference_right_30hz.npz"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/datasets/arctic/prepare_arctic_ik_trajectory.py"),
            "--canonical-trajectory",
            str(source),
            "--side",
            "right",
            "--iterations",
            str(args.iterations),
            "--output",
            str(ik_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/datasets/arctic/prepare_arctic_rl_reference.py"),
            "--trajectory",
            str(ik_path),
            "--hand-urdf",
            str(REPO_ROOT / "assets/robot_hands/direct_motor/mano/right/hand.urdf"),
            "--side",
            "right",
            "--target-fps",
            "30",
            "--output",
            str(reference_path),
        ],
        check=True,
    )
    record = {
        "index": args.index,
        "sample_id": sample_id,
        "object_id": object_id,
        "source": str(source),
        "reference": str(reference_path),
    }
    (destination / "record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"ARCTIC_BATCH_ITEM_READY index={args.index} sample={sample_id} "
        f"object={object_id} reference={reference_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
