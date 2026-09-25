#!/usr/bin/env python3
"""Force an ARCTIC object through a MANO fingertip to audit PhysX contacts."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(
        "DexCoDesign-MANO-Residual-Direct-v0", device=args.device,
        num_envs=1, use_fabric=False,
    )
    cfg.articulate_mode = True
    env = gym.make("DexCoDesign-MANO-Residual-Direct-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    thumb_ids, _ = raw.hand.find_bodies("right_thumb3")
    thumb_position = raw.hand.data.body_pos_w[:, thumb_ids[0]].clone()
    object_pose = raw.object.data.root_pose_w.clone()
    object_pose[:, :3] = thumb_position
    object_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=raw.device)
    raw.object.write_root_pose_to_sim(object_pose)
    raw.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw.device))
    maximum = 0.0
    for _ in range(12):
        env.step(torch.zeros((1, raw.cfg.action_space.shape[0]), device=raw.device))
        maximum = max(maximum, float(raw._contact_sensor_force(raw._all_hand_contact_sensors).max()))
    print(f"FORCED_COLLISION_MAX_FORCE_N={maximum:.9g}", flush=True)
    if maximum <= 0.0:
        raise RuntimeError("Forced hand-object overlap produced zero PhysX contact force")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
