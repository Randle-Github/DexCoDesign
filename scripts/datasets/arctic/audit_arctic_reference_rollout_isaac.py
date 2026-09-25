#!/usr/bin/env python3
"""Run an exact zero-residual reference rollout and report physical tracking."""

import argparse
import os

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--wrist-stiffness", type=float, default=1000.0)
parser.add_argument("--finger-stiffness", type=float, default=300.0)
parser.add_argument("--fix-object-root", action="store_true")
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
        num_envs=1,
        use_fabric=False,
    )
    cfg.articulate_mode = True
    cfg.articulated_object_fix_root_link = args.fix_object_root
    cfg.hand_cfg.actuators["wrist"].stiffness = args.wrist_stiffness
    cfg.hand_cfg.actuators["wrist"].damping = 2.0 * args.wrist_stiffness**0.5
    cfg.hand_cfg.actuators["fingers"].stiffness = args.finger_stiffness
    cfg.hand_cfg.actuators["fingers"].damping = 2.0 * args.finger_stiffness**0.5
    env = gym.make("DexCoDesign-MANO-Residual-Direct-Play-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    object_masses = raw.object.root_physx_view.get_masses()[0]
    object_inertias = raw.object.root_physx_view.get_inertias()[0]
    print(
        "ARTICULATED_OBJECT_PHYSICS "
        f"body_names={raw.object.body_names} "
        f"masses_kg={object_masses.detach().cpu().tolist()} "
        f"inertias={object_inertias.detach().cpu().tolist()} "
        f"joint_names={raw.object.joint_names} "
        f"soft_joint_limits={raw.object.data.soft_joint_pos_limits[0].detach().cpu().tolist()}",
        flush=True,
    )
    zero = torch.zeros((1, raw.cfg.action_space.shape[0]), device=raw.device)

    q_errors = []
    q_signed_errors = []
    q_actual_trace = []
    q_reference_trace = []
    object_errors = []
    object_q_actual_trace = []
    object_q_reference_trace = []
    object_position_errors = []
    object_rotation_errors = []
    object_pose_actual_trace = []
    object_pose_reference_trace = []
    all_forces = []
    thumb_forces = []
    other_forces = []
    per_link_forces = {
        name: [] for name in raw._articulated_contact_sensors
    }
    phases = []
    for _ in range(raw._reference_length - 1):
        env.step(zero)
        phase = raw.phase_buf.clone()
        target_q = raw._reference_at(raw.reference_hand_q, phase)
        signed_q_error = raw.hand.data.joint_pos - target_q
        q_actual_trace.append(raw.hand.data.joint_pos[0].clone())
        q_reference_trace.append(target_q[0].clone())
        q_signed_errors.append(signed_q_error[0])
        q_errors.append(signed_q_error.abs()[0])
        target_object_q = raw._reference_at(raw.reference_object_joint, phase)
        object_q_actual_trace.append(raw.object.data.joint_pos[0].clone())
        object_q_reference_trace.append(target_object_q[0].clone())
        object_errors.append(
            (raw.object.data.joint_pos - target_object_q).abs().mean()
        )
        target_object_pose = raw._reference_at(raw.reference_object_pose, phase)
        actual_object_position = (
            raw.object.data.root_pos_w - raw.scene.env_origins
        )
        object_pose_actual_trace.append(
            torch.cat((actual_object_position, raw.object.data.root_quat_w), dim=-1)[0]
        )
        object_pose_reference_trace.append(target_object_pose[0].clone())
        object_position_errors.append(
            torch.linalg.vector_norm(
                actual_object_position - target_object_pose[:, :3], dim=-1
            )[0]
        )
        quaternion_dot = torch.sum(
            raw.object.data.root_quat_w * target_object_pose[:, 3:7], dim=-1
        ).abs().clamp(max=1.0)
        object_rotation_errors.append((2.0 * torch.acos(quaternion_dot))[0])
        all_forces.append(raw._contact_sensor_force(raw._all_hand_contact_sensors)[0])
        thumb_forces.append(raw._contact_sensor_force(raw._thumb_contact_sensors)[0])
        other_forces.append(raw._contact_sensor_force(raw._other_finger_contact_sensors)[0])
        for name, sensor in raw._articulated_contact_sensors.items():
            per_link_forces[name].append(raw._contact_sensor_force(sensor)[0])
        phases.append(phase[0])

    phase = torch.stack(phases)
    q_error = torch.stack(q_errors)
    q_signed_error = torch.stack(q_signed_errors)
    object_error = torch.stack(object_errors)
    object_position_error = torch.stack(object_position_errors)
    object_rotation_error = torch.stack(object_rotation_errors)
    all_force = torch.stack(all_forces)
    thumb_force = torch.stack(thumb_forces)
    other_force = torch.stack(other_forces)
    pinch = (thumb_force > 0.0) & (other_force > 0.0)
    top = torch.topk(all_force, min(12, len(all_force)))
    top_pairs = [
        (int(phase[index]), float(force))
        for force, index in zip(top.values.cpu(), top.indices.cpu(), strict=True)
    ]
    link_force_summary = sorted(
        (
            (
                name,
                float(torch.stack(values).max()),
                float(torch.stack(values).mean()),
                int((torch.stack(values) > 0.0).sum()),
            )
            for name, values in per_link_forces.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    per_joint_mean = q_error.mean(dim=0)
    worst = torch.topk(per_joint_mean, min(8, len(per_joint_mean)))
    worst_joints = [
        (raw.hand.joint_names[int(index)], float(error))
        for error, index in zip(worst.values.cpu(), worst.indices.cpu(), strict=True)
    ]
    trace_path = os.environ.get("ARCTIC_AUDIT_TRACE_PATH")
    if trace_path:
        np.savez_compressed(
            trace_path,
            phase=phase.cpu().numpy(),
            hand_q_actual=torch.stack(q_actual_trace).cpu().numpy(),
            hand_q_reference=torch.stack(q_reference_trace).cpu().numpy(),
            object_joint_error=object_error.cpu().numpy(),
            object_joint_actual=torch.stack(object_q_actual_trace).cpu().numpy(),
            object_joint_reference=torch.stack(object_q_reference_trace).cpu().numpy(),
            object_pose_actual_wxyz=torch.stack(object_pose_actual_trace).cpu().numpy(),
            object_pose_reference_wxyz=torch.stack(object_pose_reference_trace).cpu().numpy(),
            object_position_error_m=object_position_error.cpu().numpy(),
            object_rotation_error_rad=object_rotation_error.cpu().numpy(),
            all_hand_object_force_n=all_force.cpu().numpy(),
            thumb_object_force_n=thumb_force.cpu().numpy(),
            other_finger_object_force_n=other_force.cpu().numpy(),
            joint_names=np.asarray(raw.hand.joint_names),
        )
        print(f"REFERENCE_ROLLOUT_TRACE_SAVED path={trace_path}", flush=True)
    print(
        "REFERENCE_ROLLOUT_AUDIT "
        f"wrist_kp={args.wrist_stiffness:g} finger_kp={args.finger_stiffness:g} "
        f"steps={len(phase)} q_mae_mean_rad={float(q_error.mean()):.9g} "
        f"q_mae_max_rad={float(q_error.max()):.9g} "
        f"root_position_mae_m={float(q_error[:, :3].mean()):.9g} "
        f"root_position_signed_mean={q_signed_error[:, :3].mean(dim=0).cpu().tolist()} "
        f"root_position_signed_contact_window="
        f"{q_signed_error[len(phase)//10:len(phase)//2, :3].mean(dim=0).cpu().tolist()} "
        f"root_rotation_mae_rad={float(q_error[:, 3:6].mean()):.9g} "
        f"finger_mae_rad={float(q_error[:, 6:].mean()):.9g} "
        f"object_joint_mae_mean_rad={float(object_error.mean()):.9g} "
        f"object_joint_mae_max_rad={float(object_error.max()):.9g} "
        f"object_position_error_mean_m={float(object_position_error.mean()):.9g} "
        f"object_position_error_final_m={float(object_position_error[-1]):.9g} "
        f"object_position_error_max_m={float(object_position_error.max()):.9g} "
        f"object_rotation_error_mean_rad={float(object_rotation_error.mean()):.9g} "
        f"object_rotation_error_final_rad={float(object_rotation_error[-1]):.9g} "
        f"mean_all_force_n={float(all_force.mean()):.9g} "
        f"p95_all_force_n={float(torch.quantile(all_force, 0.95)):.9g} "
        f"p99_all_force_n={float(torch.quantile(all_force, 0.99)):.9g} "
        f"max_all_force_n={float(all_force.max()):.9g} "
        f"force_over_100n_steps={int((all_force > 100.0).sum())} "
        f"force_over_500n_steps={int((all_force > 500.0).sum())} "
        f"pinch_steps={int(pinch.sum())} contact_steps={int((all_force > 0).sum())} "
        f"top_phase_force={top_pairs} worst_joint_mae={worst_joints}",
        f" top_link_force_max_mean_steps={link_force_summary[:8]}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
