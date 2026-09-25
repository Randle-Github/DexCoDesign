#!/usr/bin/env python3
"""Replace the five ARCTIC previews with exact MANO-joint IK trajectories.

The episodic annotations contain the official fitted MANO 21-joint positions
and absolute source-frame indices.  Left/right episodes are paired by temporal
overlap, then retargeted to the direct-motor MANO URDF with the same keypoint IK
used by the TACO pipeline.  Object motion is sliced with the original frame
indices, so hand and object motion stay synchronized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "datasets" / "taco"))
from prepare_taco_mano_previews import solve_hand  # noqa: E402


SEQUENCES = {
    "00_scissors": "s08_scissors_use_01",
    "01_laptop": "s09_laptop_use_04",
    "02_box": "s08_box_use_01",
    "03_microwave": "s02_microwave_use_01_retake",
    "04_waffleiron": "s05_waffleiron_use_01",
}


def load_episode(path: Path) -> dict:
    item = np.load(path, allow_pickle=True).item()
    side = item["anno_type"]
    hand = item[side]
    frames = np.asarray(item["video_decode_frame"], dtype=np.int64)
    joints = np.asarray(hand["joints_worldspace"], dtype=np.float64)
    kept = np.asarray(hand.get("kept_frames", np.ones(len(frames))), dtype=bool)
    count = min(len(frames), len(joints), len(kept))
    frames, joints, kept = frames[:count], joints[:count], kept[:count]
    valid = kept & np.isfinite(joints).all(axis=(1, 2))
    fitting = np.asarray(hand.get("fitting_err", np.zeros(count)), dtype=np.float64)[:count]
    return {
        "path": path,
        "side": side,
        "frames": frames[valid],
        "joints": joints[valid],
        "error": float(np.nanmean(fitting[valid])) if valid.any() else np.inf,
    }


def select_pair(annotation_root: Path, sequence: str) -> tuple[dict, dict, np.ndarray]:
    episodes = [load_episode(path) for path in annotation_root.rglob(f"*{sequence}_ep_*.npy")]
    left = [episode for episode in episodes if episode["side"] == "left"]
    right = [episode for episode in episodes if episode["side"] == "right"]
    candidates = []
    for lhs in left:
        for rhs in right:
            common = np.intersect1d(lhs["frames"], rhs["frames"])
            if len(common) < 12:
                continue
            # Prefer useful clip length, then the official fitting residual.
            score = len(common) - 0.25 * (lhs["error"] + rhs["error"])
            candidates.append((score, lhs, rhs, common))
    if not candidates:
        raise RuntimeError(f"No overlapping left/right MANO episodes for {sequence}")
    _, lhs, rhs, common = max(candidates, key=lambda item: item[0])
    return lhs, rhs, common


def samples_at(episode: dict, frames: np.ndarray) -> np.ndarray:
    lookup = {int(frame): index for index, frame in enumerate(episode["frames"])}
    return np.asarray([episode["joints"][lookup[int(frame)]] for frame in frames])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path, default=Path("artifacts/arctic_exact_annotations")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/arctic_mano_preview"))
    parser.add_argument("--iterations", type=int, default=24)
    args = parser.parse_args()

    annotation_root = args.annotations.resolve()
    output = args.output.resolve()
    mano_root = REPO / "assets" / "robot_hands" / "direct_motor" / "mano"
    reports = []

    for directory, sequence in SEQUENCES.items():
        preview = output / directory
        trajectory_path = preview / "trajectory.npz"
        old = np.load(trajectory_path)
        payload = {key: old[key] for key in old.files if key != "metadata_json"}
        metadata = json.loads((preview / "metadata.json").read_text())

        left, right, common = select_pair(annotation_root, sequence)
        # ARCTIC source is 30 Hz and the compact object track is 10 Hz.
        object_indices = np.rint(common / 3.0).astype(np.int64)
        object_length = len(payload["object_root_position_m"])
        valid = (object_indices >= 0) & (object_indices < object_length)
        common, object_indices = common[valid], object_indices[valid]
        # Preserve the source-frame order but render at 10 Hz without duplicates.
        unique = np.r_[True, np.diff(object_indices) != 0]
        common, object_indices = common[unique], object_indices[unique]

        diagnostics = {}
        for episode, side in ((left, "left"), (right, "right")):
            targets = samples_at(episode, common)
            q, names, report = solve_hand(
                mano_root / side / "hand.urdf", targets, side, stride=1,
                iterations=args.iterations,
            )
            payload[f"{side}_q"] = q
            payload[f"{side}_joint_names"] = names
            diagnostics[side] = {
                **report,
                "annotation": episode["path"].name,
                "official_fitting_error": episode["error"],
            }

        for key in (
            "object_root_position_m",
            "object_root_quaternion_wxyz",
            "object_joint_position_rad",
        ):
            payload[key] = payload[key][object_indices]

        metadata.update(
            {
                "frames": int(len(common)),
                "source_frame_start": int(common[0]),
                "source_frame_stop": int(common[-1] + 1),
                "hand_projection": "official MANO 21-joint world-space targets -> direct-motor URDF IK",
                "exact_mano_ik": diagnostics,
            }
        )
        payload["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(trajectory_path, **payload)
        (preview / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        reports.append({"directory": directory, "sequence": sequence, **diagnostics})
        print(
            f"ARCTIC_EXACT_READY sequence={sequence} frames={len(common)} "
            f"left_mm={diagnostics['left']['mean_keypoint_error_mm']:.2f} "
            f"right_mm={diagnostics['right']['mean_keypoint_error_mm']:.2f}",
            flush=True,
        )

    (output / "exact_ik_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
