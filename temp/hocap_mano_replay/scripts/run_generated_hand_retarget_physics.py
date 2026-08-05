#!/usr/bin/env python3
"""Retarget and physically simulate one generated hand without RL."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
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
    initial_trajectory: Path | None = None,
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
        if initial_trajectory is not None and iterations == 0:
            # The high-throughput GPU solver already produced the complete
            # retargeted trajectory. Materialize it directly instead of
            # repeating 446 frames of MuJoCo CPU IK for every top-k hand.
            with np.load(initial_trajectory) as initial:
                qpos = initial["qpos"].astype(np.float64)
                wrist_position = initial["wrist_position"].astype(np.float64)
                wrist_quaternion = initial[
                    "wrist_quaternion_xyzw"
                ].astype(np.float64)
                source_names = (
                    initial["source_joint_names"].astype(str).tolist()
                    if "source_joint_names" in initial
                    else None
                )
                frame_ids = (
                    initial["frame_ids"].astype(np.int64)
                    if "frame_ids" in initial
                    else np.arange(len(qpos), dtype=np.int64)
                )
            scene = retarget.make_scene(hand_id)
            model = mujoco.MjModel.from_xml_path(str(scene))
            scalar_joint_ids = [
                joint_id
                for joint_id in range(model.njnt)
                if int(model.jnt_type[joint_id])
                in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                )
            ]
            by_name = {
                str(model.joint(joint_id).name): joint_id
                for joint_id in scalar_joint_ids
            }
            if source_names is not None and all(
                name in by_name for name in source_names
            ):
                ordered_joint_ids = [by_name[name] for name in source_names]
            else:
                ordered_joint_ids = scalar_joint_ids
            if len(ordered_joint_ids) != qpos.shape[1]:
                raise ValueError(
                    f"{hand_id}: GPU qpos width {qpos.shape[1]} does not "
                    f"match runtime scalar joints {len(ordered_joint_ids)}"
                )
            qpos_ids = np.asarray(
                [int(model.jnt_qposadr[joint_id]) for joint_id in ordered_joint_ids],
                dtype=np.int64,
            )
            np.savez_compressed(
                ik_trajectory,
                frame_ids=frame_ids,
                qpos_ids=qpos_ids,
                active_qpos_ids=qpos_ids,
                qpos=qpos,
                wrist_position=wrist_position,
                wrist_quaternion_xyzw=wrist_quaternion,
                backend=np.asarray("gpu_direct_no_cpu_ik"),
            )
        else:
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
                if initial_trajectory is not None:
                    sys.argv.extend(
                        ["--initial-trajectory", str(initial_trajectory.resolve())]
                    )
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
    parser.add_argument("--initial-trajectory", type=Path)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    metadata, trajectory_path = prepare_retarget(
        runtime_root,
        iterations=args.iterations,
        reuse_ik=args.reuse_ik,
        initial_trajectory=args.initial_trajectory,
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
