#!/usr/bin/env python3
"""Evaluate whether a MANO residual checkpoint reaches the final reference step."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--algorithm", default="PPO")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


evaluation_exit_code = 1


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    global evaluation_exit_code

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(Path(args_cli.checkpoint).resolve()))
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    agent_act_requires_states = "states" in inspect.signature(runner.agent.act).parameters
    reference_last_phase = raw_env.unwrapped._reference_length - 1
    result = {
        "success": False,
        "steps": 0,
        "final_phase": 0,
        "reference_last_phase": reference_last_phase,
        "terminated": False,
        "truncated": False,
    }
    action_abs_sum = torch.zeros(raw_env.unwrapped.action_dim, device=raw_env.unwrapped.device)
    action_abs_max = torch.zeros_like(action_abs_sum)
    action_near_limit_count = torch.zeros_like(action_abs_sum)
    target_joint_limit_clip_count = torch.zeros_like(action_abs_sum)
    action_sample_count = 0

    with torch.inference_mode():
        for timestep in range(reference_last_phase + 2):
            if agent_act_requires_states:
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            executed_actions = torch.clamp(actions, -1.0, 1.0)
            action_abs = executed_actions[0].abs()
            action_abs_sum += action_abs
            action_abs_max = torch.maximum(action_abs_max, action_abs)
            action_near_limit_count += (action_abs >= 0.95).to(torch.float32)
            phase_before_step = raw_env.unwrapped.phase_buf[0]
            unclamped_targets = (
                raw_env.unwrapped.reference_hand_ctrl[phase_before_step]
                + raw_env.unwrapped.residual_scale * executed_actions[0]
            )
            target_joint_limit_clip_count += (
                (unclamped_targets < raw_env.unwrapped.joint_lower_limits[0])
                | (unclamped_targets > raw_env.unwrapped.joint_upper_limits[0])
            ).to(torch.float32)
            action_sample_count += 1
            obs, _, terminated, truncated, _ = env.step(actions)

            terminated_now = bool(torch.as_tensor(terminated).any().item())
            truncated_now = bool(torch.as_tensor(truncated).any().item())
            phase = int(raw_env.unwrapped._last_evaluated_phase[0].item())
            result.update(
                {
                    "steps": timestep + 1,
                    "final_phase": phase,
                    "terminated": terminated_now,
                    "truncated": truncated_now,
                }
            )
            if terminated_now or truncated_now:
                result["success"] = (
                    not terminated_now
                    and truncated_now
                    and phase >= reference_last_phase
                )
                break

    result["position_error_m"] = float(
        raw_env.unwrapped._object_position_error[0].item()
    )
    result["rotation_error_rad"] = float(
        raw_env.unwrapped._object_rotation_error[0].item()
    )
    action_abs_mean = action_abs_sum / max(action_sample_count, 1)
    action_near_limit_fraction = action_near_limit_count / max(action_sample_count, 1)
    target_joint_limit_clip_fraction = (
        target_joint_limit_clip_count / max(action_sample_count, 1)
    )
    residual_abs_max = action_abs_max * raw_env.unwrapped.residual_scale
    joint_names = list(raw_env.unwrapped.hand.joint_names)

    def group_summary(start: int, stop: int) -> dict[str, float]:
        return {
            "action_abs_mean": float(action_abs_mean[start:stop].mean().item()),
            "action_abs_max": float(action_abs_max[start:stop].amax().item()),
            "near_action_limit_fraction": float(
                action_near_limit_fraction[start:stop].mean().item()
            ),
            "residual_abs_max": float(residual_abs_max[start:stop].amax().item()),
            "joint_limit_clip_fraction": float(
                target_joint_limit_clip_fraction[start:stop].mean().item()
            ),
        }

    result["residual_diagnostics"] = {
        "action_sample_count": action_sample_count,
        "root_translation": group_summary(0, 3),
        "root_rotation": group_summary(3, 6),
        "fingers": group_summary(6, len(joint_names)),
        "per_joint": {
            name: {
                "action_abs_mean": float(action_abs_mean[index].item()),
                "action_abs_max": float(action_abs_max[index].item()),
                "near_action_limit_fraction": float(
                    action_near_limit_fraction[index].item()
                ),
                "residual_abs_max": float(residual_abs_max[index].item()),
                "joint_limit_clip_fraction": float(
                    target_joint_limit_clip_fraction[index].item()
                ),
            }
            for index, name in enumerate(joint_names)
        },
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("MANO_EVAL_SUCCESS" if result["success"] else "MANO_EVAL_FAILED")
    evaluation_exit_code = 0 if result["success"] else 2
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    raise SystemExit(evaluation_exit_code)
