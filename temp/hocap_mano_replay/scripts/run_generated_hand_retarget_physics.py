#!/usr/bin/env python3
"""Retarget and physically simulate one generated hand without RL."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))

import retarget_all_hands as retarget  # noqa: E402
import simulate_retargeted_all_hands as physics  # noqa: E402


def prepare_retarget(
    runtime_root: Path,
    *,
    iterations: int = 18,
    reuse_ik: bool = False,
) -> tuple[dict[str, object], Path]:
    """Retarget the fixed pre-RL MANO command trajectory to one generated hand."""
    runtime_root = runtime_root.resolve()
    metadata = json.loads(
        (runtime_root / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    hand_id = str(metadata["hand_id"])
    output_root = runtime_root.parent / "retarget_physics"
    ik_root = output_root / "ik"
    cache_root = ik_root / "cache"
    retarget_root = output_root / "retargeted"
    for path in (ik_root, cache_root, retarget_root):
        path.mkdir(parents=True, exist_ok=True)

    registry = {
        "hands": {
            hand_id: {
                "display_name": metadata["display_name"],
                "entries": {
                    "left": {
                        "format": "urdf",
                        "path": f"{hand_id}/left/hand.urdf",
                        "active_dofs": metadata["active_dofs"],
                        "passive_mimic_dofs": metadata["passive_mimic_dofs"],
                        "scalar_dofs": metadata["active_dofs"]
                        + metadata["passive_mimic_dofs"],
                    }
                },
            }
        }
    }
    registry_path = runtime_root / "registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    retarget.DIRECT_ROOT = runtime_root
    retarget.REGISTRY = registry_path
    retarget.ARTIFACT_ROOT = ik_root
    retarget.CACHE_ROOT = cache_root
    retarget.TIP_LINKS[hand_id] = metadata["tips"]
    ik_trajectory = cache_root / f"{hand_id}_ik.npz"
    if not reuse_ik or not ik_trajectory.is_file():
        original_argv = sys.argv
        try:
            sys.argv = [
                "retarget_all_hands.py",
                "--stride",
                "1",
                "--iterations",
                str(iterations),
                "--reference-source",
                "mano_command",
                "--solve-only",
            ]
            retarget.main()
        finally:
            sys.argv = original_argv
    hand_retarget_root = retarget_root / hand_id
    hand_retarget_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = hand_retarget_root / "retargeted_trajectory.npz"
    shutil.copyfile(ik_trajectory, trajectory_path)
    return metadata, trajectory_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--reuse-ik", action="store_true")
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    metadata, trajectory_path = prepare_retarget(
        runtime_root,
        iterations=args.iterations,
        reuse_ik=args.reuse_ik,
    )
    hand_id = str(metadata["hand_id"])
    output_root = runtime_root.parent / "retarget_physics"
    rollout_root = output_root / "physical_rollout"
    physics_cache = output_root / "physics_cache"
    for path in (rollout_root, physics_cache):
        path.mkdir(parents=True, exist_ok=True)

    physics.PHYSICS_CACHE = physics_cache
    captured = np.load(physics.DEFAULT_CAPTURE)
    object_reference = captured["object_pose_wxyz"].astype(np.float64)
    result = physics.simulate_hand(
        hand_id,
        metadata["display_name"],
        trajectory_path,
        object_reference,
        rollout_root / hand_id,
        args.fps,
        args.width,
        args.height,
    )
    report = {
        "hand": metadata,
        "retarget": {
            "trajectory": str(trajectory_path),
            "frames": int(len(np.load(trajectory_path)["qpos"])),
            "rl_used": False,
        },
        "physics": {
            **result,
            "object_reset_after_frame_zero": False,
            "hand_object_contact": True,
            "object_table_contact": True,
            "hand_table_contact": False,
            "hand_self_collision": False,
        },
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(result["video"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
