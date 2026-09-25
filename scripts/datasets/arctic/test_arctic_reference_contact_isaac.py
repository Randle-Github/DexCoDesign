#!/usr/bin/env python3
"""Audit whether an ARCTIC reference contains physical MANO-object contact.

Each parallel Isaac environment is initialized at a different reference frame.
The test writes the reference joint/object states directly, without a policy or
position-controller tracking error, and then takes one simulation step.  A
zero result therefore diagnoses retarget/reference geometry, not PPO.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(
        "DexCoDesign-MANO-Residual-Direct-Play-v0",
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=False,
    )
    cfg.articulate_mode = True
    cfg.scissors_handle_contact_gate = True
    env = gym.make("DexCoDesign-MANO-Residual-Direct-Play-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()

    count = min(args.num_envs, raw._reference_length)
    env_ids = torch.arange(count, device=raw.device, dtype=torch.long)
    phases = torch.linspace(
        0, raw._reference_length - 1, count, device=raw.device
    ).round().to(torch.long)
    raw.phase_buf[env_ids] = phases

    hand_q = raw._reference_at(raw.reference_hand_q, phases, env_ids)
    raw.hand.write_joint_state_to_sim(
        hand_q, torch.zeros_like(hand_q), env_ids=env_ids
    )
    raw.hand.set_joint_position_target(hand_q, env_ids=env_ids)
    raw.joint_targets[env_ids] = hand_q

    object_pose = raw._reference_at(raw.reference_object_pose, phases, env_ids).clone()
    object_pose[:, :3] += raw.scene.env_origins[env_ids]
    raw.object.write_root_pose_to_sim(object_pose, env_ids)
    raw.object.write_root_velocity_to_sim(
        torch.zeros((count, 6), device=raw.device), env_ids
    )
    object_q = raw._reference_at(raw.reference_object_joint, phases, env_ids)
    raw.object.write_joint_state_to_sim(
        object_q, torch.zeros_like(object_q), env_ids=env_ids
    )

    # One normal environment step updates PhysX contacts and all contact
    # reporters.  The states above are exact at the beginning of that step.
    env.step(torch.zeros((args.num_envs, raw.cfg.action_space.shape[0]), device=raw.device))
    thumb = raw._contact_sensor_force(raw._thumb_contact_sensors)[:count]
    other = raw._contact_sensor_force(raw._other_finger_contact_sensors)[:count]
    all_hand = raw._contact_sensor_force(raw._all_hand_contact_sensors)[:count]
    pinch = (thumb > 0.0) & (other > 0.0)
    hole_thumb, hole_other, hole_pinch, _, _ = (
        raw._compute_scissors_handle_contact()
    )
    index_hole_frames = {}
    for name in ("right_index1z", "right_index2", "right_index3"):
        sensor = raw._articulated_contact_sensors[name]
        contact_pos_w = sensor.data.contact_pos_w[:count, 0]
        finite = torch.isfinite(contact_pos_w).all(dim=-1).any(dim=-1)
        index_hole_frames[name] = int(finite.sum())

    top = torch.topk(all_hand, min(12, count))
    top_pairs = [
        (int(phases[index]), float(force))
        for force, index in zip(top.values.cpu(), top.indices.cpu(), strict=True)
    ]
    contact_phases = phases[all_hand > 0.0].cpu().tolist()
    pinch_phases = phases[pinch].cpu().tolist()
    print(
        "REFERENCE_CONTACT_AUDIT "
        f"frames={count} max_all_force_n={float(all_hand.max()):.9g} "
        f"max_thumb_force_n={float(thumb.max()):.9g} "
        f"max_other_force_n={float(other.max()):.9g} "
        f"pinch_frames={int(pinch.sum())} contact_frames={int((all_hand > 0).sum())} "
        f"hole_thumb_frames={int(hole_thumb[:count].sum())} "
        f"hole_other_frames={int(hole_other[:count].sum())} "
        f"hole_pinch_frames={int(hole_pinch[:count].sum())} "
        f"index_mesh_contact_frames={index_hole_frames} "
        f"first_contact_phases={contact_phases[:16]} "
        f"first_pinch_phases={pinch_phases[:16]} top_phase_force={top_pairs}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
