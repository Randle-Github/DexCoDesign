#!/usr/bin/env python3
"""Evaluate whether a MANO residual checkpoint reaches the final reference step."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--algorithm", default="PPO")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.output == Path(".") or (
    args_cli.output.exists() and args_cli.output.is_dir()
):
    parser.error(
        "--output must name a JSON file, not a directory. If using a shell "
        "variable, define it before launching evaluation."
    )
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from omegaconf import OmegaConf
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


evaluation_exit_code = 1


def restore_saved_env_scalars(env_cfg, path: Path) -> set[str]:
    """Safely restore scalar fields from a saved Isaac Lab environment YAML.

    Isaac Lab's ``dump_yaml`` emits Python-specific tags for values such as
    NumPy dtypes and Gym spaces. Composing the YAML node tree avoids unsafe
    construction and lets evaluation restore only type-compatible scalars.
    """

    with path.open(encoding="utf-8") as stream:
        root = yaml.compose(stream)

    restored_fields: set[str] = set()

    def decode_scalar(node: yaml.ScalarNode):
        if node.tag == "tag:yaml.org,2002:str":
            return node.value
        if node.tag in {
            "tag:yaml.org,2002:null",
            "tag:yaml.org,2002:bool",
            "tag:yaml.org,2002:int",
            "tag:yaml.org,2002:float",
        }:
            return yaml.safe_load(node.value)
        return None

    def restore(target, node, namespace: str = "") -> None:
        if not isinstance(node, yaml.MappingNode):
            return
        for key_node, value_node in node.value:
            key = key_node.value
            field_path = f"{namespace}.{key}" if namespace else key
            if isinstance(target, dict):
                if key not in target:
                    continue
                current = target[key]
            else:
                if not hasattr(target, key):
                    continue
                current = getattr(target, key)

            if isinstance(value_node, yaml.MappingNode):
                restore(current, value_node, field_path)
                continue
            if not isinstance(value_node, yaml.ScalarNode):
                continue

            value = decode_scalar(value_node)
            compatible = (
                (current is None and value is None)
                or (isinstance(current, bool) and isinstance(value, bool))
                or (
                    isinstance(current, int)
                    and not isinstance(current, bool)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                )
                or (
                    isinstance(current, float)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
                or (isinstance(current, str) and isinstance(value, str))
            )
            if not compatible:
                continue
            if isinstance(current, float):
                value = float(value)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)
            restored_fields.add(field_path)

    restore(env_cfg, root)
    return restored_fields


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    global evaluation_exit_code

    if args_cli.num_envs < 1:
        raise ValueError(f"--num_envs must be positive, got {args_cli.num_envs}")

    # Restore the environment saved beside the checkpoint, while retaining
    # behavior belonging specifically to the registered Eval task. Explicit
    # Hydra overrides and regular CLI arguments take precedence afterward.
    eval_only_settings = {
        name: getattr(env_cfg, name)
        for name in (
            "object_failure_distance",
            "object_failure_orientation",
            "randomize_start_phase",
            "log_rollout_diagnostics",
        )
        if hasattr(env_cfg, name)
    }
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    log_dir = checkpoint_path.parent.parent
    saved_env_cfg_path = log_dir / "params" / "env.yaml"
    if saved_env_cfg_path.is_file():
        restored_env_fields = restore_saved_env_scalars(
            env_cfg, saved_env_cfg_path
        )
        if (
            hasattr(env_cfg, "disable_hand_support_collisions")
            and "disable_hand_support_collisions" not in restored_env_fields
        ):
            # Runs created before this option was introduced have no recorded
            # collision choice. Preserve their historical intended behavior.
            env_cfg.disable_hand_support_collisions = True
            print(
                "[INFO] Saved environment predates the hand-support collision "
                "setting; defaulting disable_hand_support_collisions=True."
            )
        for name, value in eval_only_settings.items():
            setattr(env_cfg, name, value)

        env_override_args = [
            override.removeprefix("env.")
            for override in hydra_args
            if override.startswith("env.")
        ]
        if env_override_args:
            explicit_env_cfg = OmegaConf.to_container(
                OmegaConf.from_dotlist(env_override_args), resolve=True
            )
            env_cfg.from_dict(explicit_env_cfg)
        print(
            "[INFO] Loaded saved environment configuration "
            f"({len(restored_env_fields)} scalar fields) from: "
            f"{saved_env_cfg_path}"
        )
    else:
        print(
            "[WARNING] No saved environment configuration found at "
            f"{saved_env_cfg_path}. Using the registered default configuration."
        )

    # Non-Hydra command-line options have final precedence.
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # A training job may intentionally exit as soon as the environment saves
    # its first successful rollout. Evaluation must instead receive the final
    # transition so it can write a JSON summary matching the captured NPZ.
    os.environ["HAND_EXIT_AFTER_SUCCESS_CAPTURE"] = "0"
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(checkpoint_path))
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    agent_act_requires_states = "states" in inspect.signature(runner.agent.act).parameters
    reference_last_phase = raw_env.unwrapped._reference_length - 1
    num_envs = args_cli.num_envs
    device = raw_env.unwrapped.device
    action_dim = raw_env.unwrapped.action_dim
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    successful = torch.zeros_like(finished)
    final_phase = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    final_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    final_terminated = torch.zeros_like(finished)
    final_truncated = torch.zeros_like(finished)
    final_position_error = torch.full((num_envs,), float("nan"), device=device)
    final_rotation_error = torch.full((num_envs,), float("nan"), device=device)

    metric_shape = (num_envs, action_dim)
    action_abs_sum = torch.zeros(metric_shape, device=device)
    action_abs_max = torch.zeros_like(action_abs_sum)
    action_near_limit_count = torch.zeros_like(action_abs_sum)
    reference_joint_limit_clip_count = torch.zeros_like(action_abs_sum)
    residual_joint_limit_clip_count = torch.zeros_like(action_abs_sum)
    target_joint_limit_clip_count = torch.zeros_like(action_abs_sum)
    action_sample_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    with torch.inference_mode():
        for timestep in range(reference_last_phase + 2):
            if agent_act_requires_states:
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            executed_actions = torch.clamp(actions, -1.0, 1.0)
            active = ~finished
            active_column = active[:, None]
            action_abs = executed_actions.abs()
            action_abs_sum += action_abs * active_column
            action_abs_max = torch.where(
                active_column,
                torch.maximum(action_abs_max, action_abs),
                action_abs_max,
            )
            action_near_limit_count += (
                (action_abs >= 0.95) & active_column
            ).to(torch.float32)
            phase_before_step = raw_env.unwrapped.phase_buf.clone()
            reference_targets = raw_env.unwrapped._reference_at(
                raw_env.unwrapped.reference_hand_ctrl, phase_before_step
            )
            unclamped_targets = (
                reference_targets
                + raw_env.unwrapped.residual_scale * executed_actions
            )
            reference_outside_limits = (
                (reference_targets < raw_env.unwrapped.joint_lower_limits)
                | (reference_targets > raw_env.unwrapped.joint_upper_limits)
            )
            target_outside_limits = (
                (unclamped_targets < raw_env.unwrapped.joint_lower_limits)
                | (unclamped_targets > raw_env.unwrapped.joint_upper_limits)
            )
            reference_joint_limit_clip_count += (
                reference_outside_limits & active_column
            ).to(torch.float32)
            residual_joint_limit_clip_count += (
                target_outside_limits & ~reference_outside_limits & active_column
            ).to(torch.float32)
            target_joint_limit_clip_count += (
                target_outside_limits & active_column
            ).to(torch.float32)
            action_sample_count += active.to(torch.long)
            obs, _, terminated, truncated, _ = env.step(actions)

            terminated_now = torch.as_tensor(terminated, device=device).reshape(-1).bool()
            truncated_now = torch.as_tensor(truncated, device=device).reshape(-1).bool()
            phase_now = raw_env.unwrapped._last_evaluated_phase.clone()
            newly_finished = active & (terminated_now | truncated_now)
            if newly_finished.any():
                final_phase[newly_finished] = phase_now[newly_finished]
                final_step[newly_finished] = timestep + 1
                final_terminated[newly_finished] = terminated_now[newly_finished]
                final_truncated[newly_finished] = truncated_now[newly_finished]
                final_position_error[newly_finished] = (
                    raw_env.unwrapped._object_position_error[newly_finished]
                )
                final_rotation_error[newly_finished] = (
                    raw_env.unwrapped._object_rotation_error[newly_finished]
                )
                successful[newly_finished] = (
                    ~terminated_now[newly_finished]
                    & truncated_now[newly_finished]
                    & (phase_now[newly_finished] >= reference_last_phase)
                )
                finished |= newly_finished
            if bool(finished.all().item()):
                break

    # The loop is expected to finish every environment at either a failure or
    # the end of the reference. Preserve useful diagnostics if one does not.
    unfinished = ~finished
    if unfinished.any():
        final_phase[unfinished] = raw_env.unwrapped._last_evaluated_phase[unfinished]
        final_step[unfinished] = reference_last_phase + 2
        final_position_error[unfinished] = raw_env.unwrapped._object_position_error[unfinished]
        final_rotation_error[unfinished] = raw_env.unwrapped._object_rotation_error[unfinished]

    successful_ids = successful.nonzero(as_tuple=False).flatten()
    if successful_ids.numel():
        representative_env_id = int(successful_ids[0].item())
    else:
        representative_env_id = int(torch.argmax(final_phase).item())

    per_env = [
        {
            "env_id": env_id,
            "success": bool(successful[env_id].item()),
            "steps": int(final_step[env_id].item()),
            "final_phase": int(final_phase[env_id].item()),
            "terminated": bool(final_terminated[env_id].item()),
            "truncated": bool(final_truncated[env_id].item()),
            "position_error_m": float(final_position_error[env_id].item()),
            "rotation_error_rad": float(final_rotation_error[env_id].item()),
        }
        for env_id in range(num_envs)
    ]
    result = {
        "success": bool(successful.any().item()),
        "success_criterion": "at_least_one_environment",
        "num_envs": num_envs,
        "num_successes": int(successful.sum().item()),
        "success_rate": float(successful.to(torch.float32).mean().item()),
        "successful_env_ids": [int(value.item()) for value in successful_ids],
        "representative_env_id": representative_env_id,
        "steps": per_env[representative_env_id]["steps"],
        "final_phase": per_env[representative_env_id]["final_phase"],
        "reference_last_phase": reference_last_phase,
        "terminated": per_env[representative_env_id]["terminated"],
        "truncated": per_env[representative_env_id]["truncated"],
        "position_error_m": per_env[representative_env_id]["position_error_m"],
        "rotation_error_rad": per_env[representative_env_id]["rotation_error_rad"],
        "per_env": per_env,
    }

    representative_samples = action_sample_count[representative_env_id].clamp(min=1)
    action_abs_mean = action_abs_sum[representative_env_id] / representative_samples
    action_abs_max = action_abs_max[representative_env_id]
    action_near_limit_fraction = (
        action_near_limit_count[representative_env_id] / representative_samples
    )
    target_joint_limit_clip_fraction = (
        target_joint_limit_clip_count[representative_env_id] / representative_samples
    )
    reference_joint_limit_clip_fraction = (
        reference_joint_limit_clip_count[representative_env_id]
        / representative_samples
    )
    residual_joint_limit_clip_fraction = (
        residual_joint_limit_clip_count[representative_env_id]
        / representative_samples
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
            "reference_joint_limit_clip_fraction": float(
                reference_joint_limit_clip_fraction[start:stop].mean().item()
            ),
            "residual_induced_joint_limit_clip_fraction": float(
                residual_joint_limit_clip_fraction[start:stop].mean().item()
            ),
        }

    result["residual_diagnostics"] = {
        "representative_env_id": representative_env_id,
        "action_sample_count": int(action_sample_count[representative_env_id].item()),
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
                "reference_joint_limit_clip_fraction": float(
                    reference_joint_limit_clip_fraction[index].item()
                ),
                "residual_induced_joint_limit_clip_fraction": float(
                    residual_joint_limit_clip_fraction[index].item()
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
