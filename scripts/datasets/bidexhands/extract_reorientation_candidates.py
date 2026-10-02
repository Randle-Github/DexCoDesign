"""Extract a few Shadow-hand cube reorientation candidates from one HDF5 shard.

The Vintix mirror contains simulator observations, not directly validated demos.
Offsets here follow the official Bi-DexHands ``shadow_hand_re_orientation.py``.
Keep both cubes, both 24-DOF hands and source actions; do not trust the mirrored
left fingertip observation until checked against forward kinematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _quat_change(first: np.ndarray, last: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    last = last / np.linalg.norm(last)
    return float(2 * np.arccos(np.clip(abs(np.dot(first, last)), 0, 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    candidates = []
    with h5py.File(args.shard) as source:
        for name in sorted(source, key=lambda item: int(item.split("-")[0])):
            group = source[name]
            obs = group["proprio_observation"][:]
            action = group["action"][:]
            reward = group["reward"][:]
            step = group["step_num"][:]
            if obs.shape[1] != 422 or action.shape[1] != 40:
                raise ValueError(f"Unexpected Shadow reorientation shape in {name}")
            bounds = np.r_[0, np.flatnonzero(np.diff(step) != 1) + 1, len(step)]
            for start, end in zip(bounds[:-1], bounds[1:]):
                if end - start < 50 or step[start] != 0:
                    continue
                trajectory = obs[start:end]
                if not np.isfinite(trajectory).all():
                    continue
                pose = trajectory[:, 374:381]
                other_pose = trajectory[:, 398:405]
                if not (np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1, atol=.03)
                        and np.allclose(np.linalg.norm(other_pose[:, 3:7], axis=1), 1, atol=.03)):
                    continue
                change = _quat_change(pose[0, 3:7], pose[-1, 3:7])
                other_change = _quat_change(other_pose[0, 3:7], other_pose[-1, 3:7])
                right_tips = trajectory[:, 72:137].reshape(-1, 5, 13)[:, :, :3]
                right_dist = np.linalg.norm(right_tips - pose[:, None, :3], axis=-1).min(axis=1)
                contact_fraction = float(np.mean(right_dist < .07))
                if change < .35 or contact_fraction < .35:
                    continue
                score = change + .25 * other_change + contact_fraction
                candidates.append((score, name, int(start), int(end), change,
                                   other_change, contact_fraction))
        candidates.sort(reverse=True)
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index, (_, name, start, end, change, other_change, contact) in enumerate(
            candidates[:args.count]
        ):
            group = source[name]
            obs = group["proprio_observation"][start:end]
            path = args.output / f"reorientation_{index:03d}.npz"
            np.savez_compressed(
                path,
                right_qpos_scaled=obs[:, :24],
                left_qpos_scaled=obs[:, 187:211],
                right_fingertip_state_world=obs[:, 72:137].reshape(-1, 5, 13),
                left_fingertip_state_world_unverified=obs[:, 259:324].reshape(-1, 5, 13),
                object_pose_xyzw=obs[:, 374:381],
                object2_pose_xyzw=obs[:, 398:405],
                goal_pose_xyzw=obs[:, 387:394],
                goal2_pose_xyzw=obs[:, 411:418],
                action=group["action"][start:end],
                reward=group["reward"][start:end],
                step_num=group["step_num"][start:end],
                fps=np.asarray(60),
            )
            manifest.append({
                "file": path.name,
                "source_shard": str(args.shard),
                "source_group": name,
                "source_interval_in_group": [start, end],
                "frames": end-start,
                "right_cube_rotation_rad": change,
                "left_cube_rotation_rad": other_change,
                "right_fingertip_near_cube_fraction": contact,
                "object_asset": "assets/urdf/objects/cube_multicolor.urdf",
                "status": "candidate_not_physics_replayed",
            })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"qualifying_episodes": len(candidates), "selected": manifest}, indent=2))


if __name__ == "__main__":
    main()
