#!/usr/bin/env python3
"""Render an exactly captured successful MANO training trajectory.

This script never evaluates a policy or advances physics. It writes the saved
hand and object state for each frame directly into Isaac Lab and renders that
state, so the video is the captured successful rollout itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexCoDesign-MANO-Residual-Direct-Play-v0")
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
    if hand_q.shape != (446, 28) or object_pose.shape != (446, 7):
        raise ValueError(
            f"Expected captured 446-frame trajectory, got "
            f"hand_q={hand_q.shape}, object_pose={object_pose.shape}"
        )
    if metadata["final_phase"] != 445:
        raise ValueError(f"Trajectory is not successful: {metadata}")

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args_cli.output_dir / "successful_rollout.mp4"
    reward_csv_path = args_cli.output_dir / "successful_pose_reward.csv"
    reward_png_path = args_cli.output_dir / "successful_pose_reward.png"
    summary_path = args_cli.output_dir / "successful_rollout_summary.json"

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 15.0
    env_cfg.randomize_start_phase = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw = env.unwrapped
    env.reset()

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
            frame = raw.render(recompute=True)
            writer.append_data(frame)
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
    plt.title("Captured successful rollout: pose-only accumulated reward")
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
        "render_semantics": "direct replay of captured simulator states; no policy evaluation",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "MANO_CAPTURED_TRAJECTORY_RENDERED "
        f"video={video_path} frames={len(hand_q)}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac/RTX shutdown has hung on this cluster after successful renders.
    # All outputs are flushed above, so let the OS release the isolated process.
    os._exit(0)


if __name__ == "__main__":
    main()
