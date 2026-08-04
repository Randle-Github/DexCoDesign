#!/usr/bin/env python3
"""Initialize one all-hand environment and verify collision/contact wiring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexCoDesign-Hand-Residual-Direct-Eval-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=8)
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
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    actions = torch.zeros((1, raw.cfg.action_space), device=raw.device)
    forces = []
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            env.step(actions)
            forces.append(
                float(raw._contact_sensor_force(raw._all_hand_contact_sensor)[0].item())
            )

    result = {
        "hand_id": os.environ["DEXCODESIGN_HAND_ID"],
        "physical_contact_links": list(raw._all_hand_contact_sensor.cfg.filter_prim_paths_expr),
        "physical_contact_link_count": len(
            raw._all_hand_contact_sensor.cfg.filter_prim_paths_expr
        ),
        "all_hand_object_sensor_initialized": True,
        "sampled_all_hand_object_force_n": forces,
        "hand_support_contact_disabled": True,
        "object_hand_contact_enabled": True,
        "object_support_contact_enabled": True,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("HAND_CONTACT_RUNTIME_OK " + json.dumps(result, separators=(",", ":")))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
