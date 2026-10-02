"""Collect compact PenSpin trajectories from its unmodified official test policy.

Run this from the upstream PenSpin repository with the same Hydra overrides as
``scripts/vis_teacher.sh``.  The collector replaces only the infinite test
loop: it records simulator states before each action and keeps the best complete
episodes.  It does not modify the policy, task, physics, or reward.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import isaacgym  # Isaac Gym requires this import to precede torch.
import numpy as np
import torch


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _episode_score(episode: dict[str, np.ndarray]) -> float:
    # The requested task is spinning, not merely holding the pen.  Reject very
    # short/fallen rollouts and rank by actual object angular motion.
    if len(episode["reward"]) < 100:
        return -np.inf
    if np.min(episode["object_pose"][:, 2]) < 0.605:
        return -np.inf
    return float(np.mean(np.abs(episode["object_angular_velocity"][:, 2])))


def _collect(self) -> None:
    output = Path(os.environ["PENSPIN_OUTPUT"])
    count = int(os.environ.get("PENSPIN_COUNT", "20"))
    max_steps = int(os.environ.get("PENSPIN_MAX_STEPS", "5000"))
    output.mkdir(parents=True, exist_ok=True)

    self.set_eval()
    obs_dict = self.env.reset()
    n_envs = int(self.env.num_envs)
    pending: list[dict[str, list[np.ndarray]]] = [{} for _ in range(n_envs)]
    completed: list[dict[str, np.ndarray]] = []
    keys = (
        "hand_qpos",
        "hand_root_pose",
        "object_pose",
        "object_linear_velocity",
        "object_angular_velocity",
        "fingertips_world",
        "object_asset_id",
        "object_scale",
        "action",
        "reward",
    )

    for step in range(max_steps):
        with torch.no_grad():
            if self.ppo_config.distill:
                raise RuntimeError("Official oracle checkpoint, not student, is required")
            point_cloud = obs_dict["point_cloud_info"]
            if self.normalize_point_cloud:
                point_cloud = self.point_cloud_mean_std(
                    point_cloud.reshape(-1, 3)
                ).reshape((obs_dict["obs"].shape[0], -1, 3))
            inputs = {
                "obs": self.running_mean_std(obs_dict["obs"]),
                "priv_info": (
                    self.priv_mean_std(obs_dict["priv_info"])
                    if self.normalize_priv
                    else obs_dict["priv_info"]
                ),
                "proprio_hist": obs_dict["proprio_hist"],
                "point_cloud_info": point_cloud,
            }
            action, extrin, _ = self.model.act_inference(inputs)
            action = torch.clamp(action, -1.0, 1.0)

        env = self.env
        state = {
            "hand_qpos": _numpy(env.allegro_hand_dof_pos),
            "hand_root_pose": _numpy(env.root_state_tensor[env.hand_indices, :7]),
            "object_pose": _numpy(env.root_state_tensor[env.object_indices, :7]),
            "object_linear_velocity": _numpy(env.object_linvel),
            "object_angular_velocity": _numpy(env.object_angvel),
            "fingertips_world": _numpy(env.fingertip_pos).reshape(n_envs, 4, 3),
            "object_asset_id": _numpy(env.object_type_at_env),
            "object_scale": _numpy(env.priv_info_buf[:, 3]),
            "action": _numpy(action),
        }
        obs_dict, reward, done, _ = env.step(action, extrin_record=extrin)
        state["reward"] = _numpy(reward)
        done_np = _numpy(done).astype(bool)
        for env_id in range(n_envs):
            buffer = pending[env_id]
            for key in keys:
                buffer.setdefault(key, []).append(state[key][env_id])
            if done_np[env_id]:
                episode = {key: np.asarray(buffer[key]) for key in keys}
                episode["source_env_id"] = np.asarray(env_id)
                episode["source_step"] = np.asarray(step)
                # Official Isaac Gym test mode may signal resets immediately
                # after env.reset(); these are not motion episodes.
                if len(episode["reward"]) >= 50:
                    completed.append(episode)
                pending[env_id] = {}
        if step % 100 == 0:
            print(f"step={step} completed={len(completed)}", flush=True)
        if len(completed) >= max(count * 5, n_envs):
            break

    ranked = sorted(completed, key=_episode_score, reverse=True)
    good = [episode for episode in ranked if np.isfinite(_episode_score(episode))]
    if len(good) < count:
        print(
            "episode diagnostics",
            {
                "completed": len(completed),
                "length_ge_100": sum(len(e["reward"]) >= 100 for e in completed),
                "height_ok": sum(np.min(e["object_pose"][:, 2]) >= .605 for e in completed),
                "top_lengths": sorted((len(e["reward"]) for e in completed), reverse=True)[:10],
                "top_min_z": sorted((float(np.min(e["object_pose"][:, 2])) for e in completed), reverse=True)[:10],
                "top_spin": sorted((float(np.mean(np.abs(e["object_angular_velocity"][:, 2]))) for e in completed), reverse=True)[:10],
            },
            flush=True,
        )
        raise RuntimeError(
            f"Only {len(good)} valid episodes of {len(completed)} completed; "
            "not claiming that 20 successful PenSpin trajectories were collected"
        )
    manifest = []
    for index, episode in enumerate(good[:count]):
        path = output / f"penspin_{index:03d}.npz"
        np.savez_compressed(path, **episode)
        manifest.append(
            {
                "file": path.name,
                "frames": int(len(episode["reward"])),
                "mean_abs_z_angular_velocity": _episode_score(episode),
                "mean_reward": float(np.mean(episode["reward"])),
                "source_env_id": int(episode["source_env_id"]),
                "object_asset_id": int(episode["object_asset_id"][0]),
                "object_asset_name": self.env.object_type_list[int(episode["object_asset_id"][0])],
                "object_scale": float(episode["object_scale"][0]),
                "control_fps": float(1.0 / (self.env.dt * self.env.control_freq_inv)),
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} PenSpin trajectories to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    args, hydra_args = parser.parse_known_args()
    os.environ["PENSPIN_OUTPUT"] = args.output
    os.environ["PENSPIN_COUNT"] = str(args.count)
    os.environ["PENSPIN_MAX_STEPS"] = str(args.max_steps)

    from penspin.algo.ppo.ppo import PPO

    PPO.test = _collect
    import train
    from hydra import compose, initialize_config_dir

    upstream = Path(train.__file__).resolve().parent
    with initialize_config_dir(config_dir=str(upstream / "configs"), version_base="1.1"):
        config = compose(config_name="config", overrides=hydra_args)
        train.main.__wrapped__(config)


if __name__ == "__main__":
    main()
