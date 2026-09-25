#!/usr/bin/env python3
"""Test whether residual hand actions can physically rotate the ARCTIC object."""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=369)
parser.add_argument("--steps", type=int, default=45)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(
        "DexCoDesign-MANO-Residual-Direct-Play-v0",
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=False,
    )
    cfg.articulate_mode = True
    cfg.articulated_object_fix_root_link = True
    cfg.disable_object_failure_termination = True
    cfg.randomize_start_phase = False
    cfg.residual_root_position_scale = 0.015
    cfg.residual_root_rotation_scale = 0.04
    cfg.residual_finger_scale = 0.30
    env = gym.make("DexCoDesign-MANO-Residual-Direct-Play-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()

    phase_value = min(max(args.phase, 0), raw._reference_length - args.steps - 2)
    env_ids = torch.arange(raw.num_envs, device=raw.device)
    phases = torch.full_like(env_ids, phase_value)
    raw.phase_buf[:] = phases
    raw.reference_time_buf[:] = phases.to(torch.float32) / raw._reference_fps

    hand_source = (
        raw.initialization_hand_q
        if raw.initialization_hand_q is not None
        else raw.reference_hand_q
    )
    hand_q = raw._reference_at(hand_source, phases, env_ids)
    hand_qd = raw._reference_velocity_at(hand_source, phases, env_ids)
    hand_root = raw.hand.data.default_root_state.clone()
    hand_root[:, :3] += raw.scene.env_origins
    hand_root[:, 7:] = 0.0
    raw.hand.write_root_pose_to_sim(hand_root[:, :7])
    raw.hand.write_root_velocity_to_sim(hand_root[:, 7:])
    raw.hand.write_joint_state_to_sim(hand_q, hand_qd)

    object_pose = raw._reference_at(raw.reference_object_pose, phases, env_ids).clone()
    object_pose[:, :3] += raw.scene.env_origins
    object_q_source = (
        raw.initialization_object_joint
        if raw.initialization_object_joint is not None
        else raw.reference_object_joint
    )
    object_q = raw._reference_at(object_q_source, phases, env_ids)
    object_qd = raw._reference_velocity_at(object_q_source, phases, env_ids)
    raw.object.write_root_pose_to_sim(object_pose)
    raw.object.write_root_velocity_to_sim(
        torch.zeros((raw.num_envs, 6), device=raw.device)
    )
    raw.object.write_joint_state_to_sim(object_q, object_qd)

    actions = torch.zeros((raw.num_envs, raw.action_dim), device=raw.device)
    random_count = raw.num_envs - 1
    actions[1:] = 2.0 * torch.rand(
        (random_count, raw.action_dim), device=raw.device
    ) - 1.0
    # The virtual wrist is already aligned by the physically reachable reset.
    # Isolate whether finger residuals can generate hinge torque.
    actions[:, raw._root_position_action_indices] = 0.0
    actions[:, raw._root_rotation_action_indices] = 0.0

    initial_q = raw.object.data.joint_pos[:, 0].clone()
    max_force = torch.zeros(raw.num_envs, device=raw.device)
    for _ in range(args.steps):
        env.step(actions)
        max_force = torch.maximum(
            max_force,
            raw._contact_sensor_force(raw._all_hand_contact_sensors),
        )
    final_q = raw.object.data.joint_pos[:, 0].clone()
    final_phase = raw.phase_buf.clone()
    target_q = raw._reference_at(raw.reference_object_joint, final_phase)[:, 0]
    error = (final_q - target_q).abs()
    delta = final_q - initial_q
    best = torch.topk(error, min(10, raw.num_envs), largest=False)
    print(
        "ARCTIC_ACTION_CONTROLLABILITY "
        f"phase={phase_value} steps={args.steps} num_envs={raw.num_envs} "
        f"zero_delta_rad={float(delta[0]):.9g} "
        f"random_delta_min_rad={float(delta[1:].min()):.9g} "
        f"random_delta_max_rad={float(delta[1:].max()):.9g} "
        f"random_delta_mean_rad={float(delta[1:].mean()):.9g} "
        f"zero_error_rad={float(error[0]):.9g} "
        f"best_error_rad={float(best.values[0]):.9g} "
        f"force_median_n={float(torch.median(max_force)):.9g} "
        f"force_p95_n={float(torch.quantile(max_force, 0.95)):.9g} "
        f"force_max_n={float(max_force.max()):.9g}",
        flush=True,
    )
    for rank, index in enumerate(best.indices.cpu().tolist()):
        active = [
            (raw._action_joint_names[j], float(actions[index, j]))
            for j in range(raw.action_dim)
            if abs(float(actions[index, j])) > 0.35
        ]
        print(
            f"CONTROLLABILITY_BEST rank={rank} env={index} "
            f"initial_q={float(initial_q[index]):.6f} "
            f"final_q={float(final_q[index]):.6f} "
            f"target_q={float(target_q[index]):.6f} "
            f"error={float(error[index]):.6f} max_force={float(max_force[index]):.3f} "
            f"active_actions={active}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
