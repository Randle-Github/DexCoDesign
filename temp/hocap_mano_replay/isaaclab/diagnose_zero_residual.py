#!/usr/bin/env python3
"""Replay the MANO reference with exactly zero residual and record failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexCoDesign-MANO-Residual-Direct-Eval-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _experiment_cfg: dict) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    actions = torch.zeros(
        (1, raw.cfg.action_space),
        dtype=torch.float32,
        device=raw.device,
    )
    trace = []

    with torch.inference_mode():
        for _ in range(raw._reference_length + 2):
            _, reward, terminated, truncated, _ = env.step(actions)
            (
                thumb_contact,
                other_contact,
                pinch_contact,
                thumb_force,
                other_force,
            ) = raw._compute_pinch_contact()
            trace.append(
                {
                    "phase": int(raw._last_evaluated_phase[0].item()),
                    "position_error_m": float(
                        raw._object_position_error[0].item()
                    ),
                    "rotation_error_rad": float(
                        raw._object_rotation_error[0].item()
                    ),
                    "total_reward": float(torch.as_tensor(reward)[0].item()),
                    "thumb_contact": bool(thumb_contact[0].item()),
                    "other_contact": bool(other_contact[0].item()),
                    "pinch_contact": bool(pinch_contact[0].item()),
                    "thumb_force_n": float(thumb_force[0].item()),
                    "other_force_n": float(other_force[0].item()),
                }
            )
            if bool(torch.as_tensor(terminated).any().item()) or bool(
                torch.as_tensor(truncated).any().item()
            ):
                break

    last = trace[-1]
    result = {
        "steps": len(trace),
        "last_phase": last["phase"],
        "terminated": bool(
            last["position_error_m"] > raw.cfg.object_failure_distance
            or last["rotation_error_rad"]
            > raw.cfg.object_failure_orientation
        ),
        "pinch_contact_steps": sum(row["pinch_contact"] for row in trace),
        "thumb_contact_steps": sum(row["thumb_contact"] for row in trace),
        "other_contact_steps": sum(row["other_contact"] for row in trace),
        "last": last,
        "trace": trace,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
