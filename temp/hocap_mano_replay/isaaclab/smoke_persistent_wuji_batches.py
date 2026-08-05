#!/usr/bin/env python3
"""Verify multiple heterogeneous PhysX batches in one Isaac Kit process."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--task", default="DexCoDesign-Hand-Residual-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.manifest = args_cli.manifest.resolve()
os.environ["DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"] = str(args_cli.manifest)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config


def rollout(env) -> dict[str, float | int]:
    raw = env.unwrapped
    env.reset()
    actions = torch.zeros(
        (raw.num_envs, raw.action_dim), dtype=torch.float32, device=raw.device
    )
    active = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    reward = torch.zeros(raw.num_envs, dtype=torch.float32, device=raw.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(raw._reference_length + 2):
            _, step_reward, terminated, truncated, _ = env.step(actions)
            reward[active] += torch.as_tensor(step_reward, device=raw.device)[active]
            active &= ~(
                torch.as_tensor(terminated, device=raw.device)
                | torch.as_tensor(truncated, device=raw.device)
            )
            if not active.any():
                break
    return {
        "rollout_seconds": time.perf_counter() - start,
        "best_reward": float(reward.max().item()),
        "best_phase": int(raw._last_evaluated_phase.max().item()),
    }


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg) -> None:
    manifest = json.loads(args_cli.manifest.read_text(encoding="utf-8"))
    count = len(manifest["hand_usd_paths"])
    results = []
    process_start = time.perf_counter()
    for repeat in range(args_cli.repeats):
        if repeat:
            sim_utils.create_new_stage()
        cfg = copy.deepcopy(env_cfg)
        cfg.scene.num_envs = count
        cfg.scene.replicate_physics = False
        cfg.scene.clone_in_fabric = False
        start = time.perf_counter()
        env = gym.make(args_cli.task, cfg=cfg)
        initialization_seconds = time.perf_counter() - start
        result = rollout(env)
        result.update(repeat=repeat, initialization_seconds=initialization_seconds)
        results.append(result)
        print("WUJI_PERSISTENT_BATCH " + json.dumps(result), flush=True)
        env.close()
    payload = {
        "same_isaac_process": True,
        "repeats": args_cli.repeats,
        "candidate_count": count,
        "process_seconds": time.perf_counter() - process_start,
        "results": results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

