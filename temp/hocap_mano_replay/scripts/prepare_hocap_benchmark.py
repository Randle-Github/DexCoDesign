#!/usr/bin/env python3
"""Download and prepare every task in the compact HO-Cap benchmark manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(*args: object) -> None:
    command = [sys.executable, *map(str, args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/benchmark_tasks.json"
    )
    parser.add_argument("--ik-iterations", type=int, default=32)
    args = parser.parse_args()
    run(SCRIPTS / "download_minimal.py")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        task_id = str(task["task_id"])
        task_root = ROOT / "data/tasks" / task_id
        # The first extraction provides calibration needed to transform labels.
        run(
            SCRIPTS / "prepare_hocap_task_subset.py",
            "--manifest", args.manifest,
            "--task-id", task_id,
        )
        if task.get("role") == "existing_baseline":
            canonical = ROOT / "data/subset" / task["sequence"]
            for name in (
                "hand_joints_3d_left.npy",
                "hand_joints_3d_left.json",
                "isaaclab_reference.npz",
                "isaaclab_reference.retargeting.json",
            ):
                shutil.copy2(canonical / name, task_root / name)
            print(f"reuse verified baseline reference: {task_root}", flush=True)
            continue
        full_labels = task_root / "hand_joints_3d_left_full.npy"
        run(
            SCRIPTS / "download_label_subset.py",
            "--sequence", task["sequence"],
            "--camera", task["camera"],
            "--hand-slot", task["hand_slot"],
            "--extrinsics", ROOT / "data/tasks/calibration/extrinsics/extrinsics_20231014.yaml",
            "--output", full_labels,
        )
        run(
            SCRIPTS / "prepare_hocap_task_subset.py",
            "--manifest", args.manifest,
            "--task-id", task_id,
            "--labels-full", full_labels,
        )
        run(
            SCRIPTS / "prepare_isaaclab_reference.py",
            "--sequence-root", task_root,
            "--iterations", args.ik_iterations,
            "--output", task_root / "isaaclab_reference.npz",
        )


if __name__ == "__main__":
    main()
