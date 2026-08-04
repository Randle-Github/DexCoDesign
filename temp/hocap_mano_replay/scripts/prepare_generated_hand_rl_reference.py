#!/usr/bin/env python3
"""Prepare one generated-hand reference for the generic Isaac Lab RL task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import prepare_all_hand_rl_references as prepare  # noqa: E402
import retarget_all_hands as retarget  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=prepare.DEFAULT_CAPTURE)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    trajectory = args.trajectory.resolve()
    output_dir = args.output_dir.resolve()
    metadata = json.loads(
        (runtime_root / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    hand_id = metadata["hand_id"]
    retarget_root = output_dir.parent / "generated_retarget_inputs"
    hand_retarget_root = retarget_root / hand_id
    hand_retarget_root.mkdir(parents=True, exist_ok=True)
    target_trajectory = hand_retarget_root / "retargeted_trajectory.npz"
    target_trajectory.write_bytes(trajectory.read_bytes())

    retarget.DIRECT_ROOT = runtime_root
    retarget.CACHE_ROOT = output_dir.parent / "generated_ik_scene_cache"
    retarget.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    retarget.TIP_LINKS[hand_id] = metadata["tips"]
    prepare.DIRECT_ROOT = runtime_root
    prepare.RETARGET_ROOT = retarget_root
    prepare.TIP_LINKS[hand_id] = metadata["tips"]

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = np.load(args.capture)
    manifest = prepare.prepare_hand(
        hand_id,
        metadata["display_name"],
        capture,
        output_dir,
    )
    registry_path = output_dir / "generated_registry.json"
    registry_path.write_text(
        json.dumps({"hands": {hand_id: manifest}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(registry_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
