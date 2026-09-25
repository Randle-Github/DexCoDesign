#!/usr/bin/env python3
"""Retarget five complete canonical ARCTIC sequences to the MANO robot assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "taco"))
from prepare_taco_mano_previews import solve_hand  # noqa: E402


TARGETS = {
    "00_scissors": "s08/scissors_use_01",
    "01_laptop": "s09/laptop_use_04",
    "02_box": "s08/box_use_01",
    "03_microwave": "s02/microwave_use_01_retake",
    "04_waffleiron": "s05/waffleiron_use_01",
}


def process_one(args: tuple[str, str, str, str, int]) -> dict:
    directory, sequence_id, root_text, output_text, iterations = args
    root, output = Path(root_text), Path(output_text)
    manifest = json.loads((root / "canonical_100" / "manifest.json").read_text())
    record = next(item for item in manifest["records"] if item["sequence_id"] == sequence_id)
    source = np.load(root / record["path"], allow_pickle=False)
    metadata = json.loads(str(source["metadata_json"]))
    mano_root = REPO / "assets" / "robot_hands" / "direct_motor" / "mano"
    preview = output / directory
    preview.mkdir(parents=True, exist_ok=True)
    payload = {}
    diagnostics = {}
    for side_index, side in enumerate(("left", "right")):
        q, names, report = solve_hand(
            mano_root / side / "hand.urdf",
            source["hand_joints_m"][:, side_index],
            side,
            stride=1,
            iterations=iterations,
        )
        payload[f"{side}_q"] = q
        payload[f"{side}_joint_names"] = names
        diagnostics[side] = report

    roots = source["object_root_pose_wxyz"][:, 0]
    payload["object_root_position_m"] = roots[:, :3]
    payload["object_root_quaternion_wxyz"] = roots[:, 3:]
    object_q = source["object_joint_positions_rad"]
    payload["object_joint_position_rad"] = object_q[:, 0, 0]

    positions = np.concatenate(
        (source["hand_joints_m"].reshape(-1, 3), roots[:, :3]), axis=0
    )
    center = positions.mean(axis=0)
    span = np.maximum(positions.max(axis=0) - positions.min(axis=0), 0.35)
    distance = float(max(0.72, np.linalg.norm(span[:2]) * 1.25))
    scene = {
        "camera_target": center.tolist(),
        "camera_eye": (center + np.array([distance, distance, 0.65 * distance])).tolist(),
        "ground_z": float(positions[:, 2].min() - 0.22),
    }
    metadata.update(
        {
            "object_id": directory.split("_", 1)[1],
            "frames": int(len(roots)), "scene": scene, "exact_mano_ik": diagnostics,
            "hand_projection": "complete official MANO 21-joint trajectory -> direct-motor URDF IK",
        }
    )
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(preview / "trajectory.npz", **payload)
    (preview / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"directory": directory, "sequence_id": sequence_id, "frames": len(roots), "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/arctic_mano_preview"))
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [
        (directory, sequence, str(args.root.resolve()), str(args.output.resolve()), args.iterations)
        for directory, sequence in TARGETS.items()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(process_one, jobs))
    records.sort(key=lambda item: item["directory"])
    (args.output / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(
            f"ARCTIC_FULL_PREVIEW_READY {record['directory']} frames={record['frames']} "
            f"left_mm={record['diagnostics']['left']['mean_keypoint_error_mm']:.2f} "
            f"right_mm={record['diagnostics']['right']['mean_keypoint_error_mm']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
