#!/usr/bin/env python3
"""Search a low-variance constant finger residual for the ARCTIC pickup phase."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--steps", type=int, default=110)
parser.add_argument("--start-phase", type=int, default=52)
parser.add_argument("--ramp-steps", type=int, default=10)
parser.add_argument("--replicas", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def warm_up_to_phase(raw, env, phase: int) -> None:
    """Reach ``phase`` through physics instead of teleporting to its reference state.

    Teleporting made later-stage residuals look feasible in isolation even when
    the preceding trajectory reached a different object pose and velocity.  A
    residual found from that fictitious state generally dropped the object as
    soon as it was spliced into a full rollout.
    """
    env.reset()
    zero_actions = torch.zeros(
        (args.num_envs, raw.cfg.action_space.shape[0]), device=raw.device
    )
    while int(raw.phase_buf[0].item()) < phase:
        env.step(zero_actions)


def main() -> None:
    cfg = parse_env_cfg(
        "DexCoDesign-MANO-Residual-Direct-Play-v0",
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=False,
    )
    cfg.articulate_mode = True
    cfg.articulated_object_fix_root_link = False
    cfg.disable_object_failure_termination = True
    cfg.require_cross_link_pinch = True
    cfg.contact_force_threshold = 0.5
    cfg.residual_root_position_scale = 0.0
    cfg.residual_root_rotation_scale = 0.0
    cfg.residual_finger_scale = 0.5
    env = gym.make("DexCoDesign-MANO-Residual-Direct-Play-v0", cfg=cfg)
    raw = env.unwrapped

    finger_indices = raw._finger_action_indices
    dimension = int(finger_indices.numel())
    mean = torch.zeros(dimension, device=raw.device)
    std = torch.full((dimension,), 0.35, device=raw.device)
    generator = torch.Generator(device=raw.device).manual_seed(20260921)
    if args.num_envs % args.replicas != 0:
        raise ValueError("num-envs must be divisible by replicas")
    candidate_count = args.num_envs // args.replicas
    best_score = float("inf")
    best_action = None
    best_metrics = None

    for round_index in range(args.rounds):
        candidate_samples = mean + std * torch.randn(
            (candidate_count, dimension), device=raw.device, generator=generator
        )
        candidate_samples = candidate_samples.clamp(-1.0, 1.0)
        samples = candidate_samples.repeat_interleave(args.replicas, dim=0)
        actions = torch.zeros(
            (args.num_envs, raw.cfg.action_space.shape[0]), device=raw.device
        )
        actions[:, finger_indices] = samples
        warm_up_to_phase(raw, env, args.start_phase)

        error_sum = torch.zeros(args.num_envs, device=raw.device)
        rotation_error_sum = torch.zeros_like(error_sum)
        articulation_error_sum = torch.zeros_like(error_sum)
        pinch_count = torch.zeros_like(error_sum)
        initial_z = raw.object.data.root_pos_w[:, 2].clone()
        maximum_z = initial_z.clone()
        end_phase = min(args.start_phase + args.steps, raw._reference_length - 1)
        reference_segment = raw.reference_object_pose[
            args.start_phase : end_phase + 1, 2
        ]
        reference_lift = float(
            (reference_segment.max() - reference_segment[0]).item()
        )
        for step_index in range(args.steps):
            blend = min(1.0, float(step_index + 1) / max(1, args.ramp_steps))
            env.step(actions * blend)
            error_sum += raw._object_position_error
            rotation_error_sum += raw._object_rotation_error
            articulation_error_sum += raw._object_articulation_error
            pinch_count += raw._last_pinch_contact.to(torch.float32)
            maximum_z = torch.maximum(maximum_z, raw.object.data.root_pos_w[:, 2])

        mean_error = error_sum / args.steps
        mean_rotation_error = rotation_error_sum / args.steps
        mean_articulation_error = articulation_error_sum / args.steps
        pinch_fraction = pinch_count / args.steps
        lift = maximum_z - initial_z
        # Prefer low tracking error and sustained grasp. Rewarding raw lift
        # alone selects candidates that throw the tool above the reference.
        lift_error = (lift - reference_lift).abs()
        per_env_score = (
            mean_error
            + 0.01 * mean_rotation_error
            + 0.01 * mean_articulation_error
            - 0.03 * pinch_fraction
            + 0.5 * lift_error
        )
        replica_score = per_env_score.reshape(candidate_count, args.replicas)
        # Mean performance alone still favors contact-chaos lottery tickets.
        # Penalize variance and the worst replica so the chosen residual also
        # transfers to a clean one-environment rollout.
        score = (
            replica_score.mean(dim=1)
            + 0.5 * replica_score.std(dim=1)
            + 0.5 * replica_score.max(dim=1).values
        )
        elite_count = max(8, candidate_count // 16)
        elite_indices = torch.topk(score, elite_count, largest=False).indices
        elites = candidate_samples[elite_indices]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(0.03, 0.45)

        index = int(torch.argmin(score).item())
        candidate_score = float(score[index].item())
        if candidate_score < best_score:
            best_score = candidate_score
            best_action = candidate_samples[index].detach().cpu()
            replica_slice = slice(
                index * args.replicas, (index + 1) * args.replicas
            )
            best_metrics = {
                "round": round_index,
                "replicas": args.replicas,
                "mean_position_error_m": float(
                    mean_error[replica_slice].mean().item()
                ),
                "mean_rotation_error_rad": float(
                    mean_rotation_error[replica_slice].mean().item()
                ),
                "mean_articulation_error_rad": float(
                    mean_articulation_error[replica_slice].mean().item()
                ),
                "pinch_fraction": float(
                    pinch_fraction[replica_slice].mean().item()
                ),
                "maximum_lift_m": float(lift[replica_slice].mean().item()),
                "reference_lift_m": reference_lift,
            }
        replica_slice = slice(index * args.replicas, (index + 1) * args.replicas)
        print(
            "ARCTIC_GRASP_SEARCH_ROUND "
            f"round={round_index} best_score={candidate_score:.9f} "
            f"mean_position_error_m={float(mean_error[replica_slice].mean()):.9f} "
            f"pinch_fraction={float(pinch_fraction[replica_slice].mean()):.9f} "
            f"maximum_lift_m={float(lift[replica_slice].mean()):.9f}",
            flush=True,
        )

    assert best_action is not None and best_metrics is not None
    result = {
        **best_metrics,
        "start_phase": args.start_phase,
        "steps": args.steps,
        "finger_action_names": [raw._action_joint_names[i] for i in finger_indices.tolist()],
        "finger_actions": best_action.tolist(),
    }
    print("ARCTIC_GRASP_SEARCH_RESULT " + json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
