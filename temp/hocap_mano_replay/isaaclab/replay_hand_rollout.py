#!/usr/bin/env python3
"""Render an exactly captured all-hand residual-RL rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexCoDesign-Hand-Residual-Direct-Play-v0")
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, None)
def main(env_cfg, _agent_cfg) -> None:
    data = np.load(args_cli.trajectory)
    hand_q = data["hand_q"].astype(np.float32)
    object_pose = data["object_pose_wxyz"].astype(np.float32)
    pose_reward = data["pose_reward"].astype(np.float64)
    contact_reward = data["contact_reward"].astype(np.float64)
    metadata = json.loads(str(data["metadata_json"]))
    if hand_q.ndim != 2 or object_pose.shape != (len(hand_q), 7):
        raise ValueError(
            f"Invalid captured trajectory: hand_q={hand_q.shape}, "
            f"object_pose={object_pose.shape}"
        )

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args_cli.output_dir / "farthest_rollout.mp4"
    reward_csv_path = args_cli.output_dir / "farthest_pose_reward.csv"
    reward_png_path = args_cli.output_dir / "farthest_pose_reward.png"
    summary_path = args_cli.output_dir / "rollout_summary.json"

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 15.0
    env_cfg.randomize_start_phase = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw = env.unwrapped
    env.reset()
    if hand_q.shape[1] != raw.hand.num_joints:
        raise ValueError(
            f"Captured {hand_q.shape[1]} joints but {raw.hand.num_joints} "
            f"were imported for {metadata.get('hand_id')}"
        )

    device = raw.device
    zero_joint_velocity = torch.zeros((1, hand_q.shape[1]), device=device)
    zero_object_velocity = torch.zeros((1, 6), device=device)
    env_origin = raw.scene.env_origins[0]
    writer = imageio.get_writer(
        video_path,
        fps=30,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        raw.render(recompute=True)
        raw.render(recompute=True)
        for frame_index in range(len(hand_q)):
            q = torch.as_tensor(
                hand_q[frame_index], device=device, dtype=torch.float32
            ).unsqueeze(0)
            pose = torch.as_tensor(
                object_pose[frame_index], device=device, dtype=torch.float32
            ).unsqueeze(0)
            pose[:, :3] += env_origin
            raw.hand.write_joint_state_to_sim(q, zero_joint_velocity)
            raw.hand.set_joint_position_target(q)
            raw.object.write_root_pose_to_sim(pose)
            raw.object.write_root_velocity_to_sim(zero_object_velocity)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(dt=0.0)
            writer.append_data(raw.render(recompute=True))
    finally:
        writer.close()

    accumulated_pose_reward = np.cumsum(pose_reward)
    with reward_csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.writer(stream)
        csv_writer.writerow(("timestep", "pose_reward", "accumulated_pose_reward"))
        csv_writer.writerows(
            zip(range(len(pose_reward)), pose_reward, accumulated_pose_reward)
        )

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        np.arange(len(accumulated_pose_reward)),
        accumulated_pose_reward,
        color="#2563eb",
        linewidth=2.2,
    )
    plt.xlabel("trajectory timestep")
    plt.ylabel("accumulated pose-only reward")
    plt.title(
        f"{metadata.get('hand_id', 'hand')}: "
        f"{metadata.get('status', 'farthest')} rollout"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(reward_png_path, dpi=180)
    plt.close()

    summary = {
        **metadata,
        "frames": len(hand_q),
        "pose_tracking_return_recomputed": float(accumulated_pose_reward[-1]),
        "contact_return_recomputed": float(contact_reward.sum()),
        "video": str(video_path),
        "trajectory": str(args_cli.trajectory),
        "render_semantics": (
            "direct replay of simulator states captured during training"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "HAND_CAPTURED_ROLLOUT_RENDERED "
        f"hand_id={metadata.get('hand_id')} video={video_path} "
        f"frames={len(hand_q)}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
