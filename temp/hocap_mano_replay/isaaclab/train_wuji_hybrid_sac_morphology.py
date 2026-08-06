#!/usr/bin/env python3
"""Persistent-Isaac hybrid SAC search over WUJI morphology."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--prototype-bank-root", type=Path, required=True)
parser.add_argument("--seed-trajectory", type=Path, required=True)
parser.add_argument("--population", type=int, default=4096)
parser.add_argument(
    "--physics-batch-size",
    type=int,
    default=4096,
    help=(
        "exact PhysX candidates per scene; the default evaluates the complete "
        "population in one scene to avoid rebuilding fixed support/object assets"
    ),
)
parser.add_argument("--generations", type=int, default=20)
parser.add_argument(
    "--continue-after-success",
    action="store_true",
    help="run every requested generation even when a 445/445 candidate exists",
)
parser.add_argument("--sac-updates", type=int, default=400)
parser.add_argument("--sac-batch-size", type=int, default=1024)
parser.add_argument("--uniform-fraction", type=float, default=0.15)
parser.add_argument("--elite-mutation-fraction", type=float, default=0.30)
parser.add_argument("--elite-replay-fraction", type=float, default=0.05)
parser.add_argument("--elite-mutation-sigma", type=float, default=0.12)
parser.add_argument("--reward-scale", type=float, default=0.01)
parser.add_argument(
    "--optimizer-backend",
    choices=("skrl", "custom"),
    default="skrl",
    help="mature SKRL SAC or the preserved custom experimental baseline",
)
parser.add_argument("--retarget-iterations", type=int, default=4)
parser.add_argument(
    "--shared-ppo-iterations",
    type=int,
    default=0,
    help=(
        "official SKRL PPO iterations trained on the complete morphology "
        "population before each outer SAC update; zero preserves morphology-only search"
    ),
)
parser.add_argument(
    "--morphology-replicas",
    type=int,
    default=1,
    help="independent PPO environments per morphology on the global DDP job",
)
parser.add_argument(
    "--fixed-reference",
    type=Path,
    help=(
        "one canonical WUJI reference reused unchanged by every morphology and "
        "every outer generation; required implicitly by shared PPO"
    ),
)
parser.add_argument(
    "--ppo-checkpoint",
    type=Path,
    help="optional initial shared PPO checkpoint; subsequent generations resume automatically",
)
parser.add_argument(
    "--target-reward",
    type=float,
    default=float("inf"),
    help="optional best mean morphology reward that ends the outer loop",
)
parser.add_argument("--seed", type=int, default=20260805)
parser.add_argument("--task", default="DexCoDesign-Hand-Residual-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.output_root = args_cli.output_root.resolve()
args_cli.prototype_bank_root = args_cli.prototype_bank_root.resolve()
args_cli.seed_trajectory = args_cli.seed_trajectory.resolve()
if args_cli.physics_batch_size < 1:
    parser.error("--physics-batch-size must be positive")
if args_cli.shared_ppo_iterations < 0:
    parser.error("--shared-ppo-iterations must be non-negative")
if args_cli.morphology_replicas < 1:
    parser.error("--morphology-replicas must be positive")
if args_cli.fixed_reference is not None:
    args_cli.fixed_reference = args_cli.fixed_reference.expanduser().resolve()
if args_cli.ppo_checkpoint is not None:
    args_cli.ppo_checkpoint = args_cli.ppo_checkpoint.expanduser().resolve()
bank_manifest_path = (
    args_cli.prototype_bank_root / "prepared/physx_batch_manifest.json"
)
os.environ["DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"] = str(bank_manifest_path)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import skrl
import torch
from skrl.utils.runner.torch import Runner

import isaaclab.sim as sim_utils
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.mano_residual import mano_residual_env as env_module
from isaaclab_tasks.utils.hydra import hydra_task_config

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_wuji_morphology import vector_to_graph  # noqa: E402
from gpu_wuji_retarget import (  # noqa: E402
    WujiBatchKinematics,
    joint_names_from_seed,
    quat_xyzw_matrix,
)
from wuji_morphology_space import resolve_design_vectors  # noqa: E402
from wuji_parametric_usd import attach_manifest  # noqa: E402
from skrl_sac_wuji_morphology import (  # noqa: E402
    ProposalBatch,
    SkrlConditionalMorphologySAC,
)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def retarget(
    vectors_path: Path,
    output: Path,
    kinematics: WujiBatchKinematics,
    seed_q: torch.Tensor,
    seed_arrays: dict[str, np.ndarray],
    iterations: int,
) -> dict[str, float | int]:
    start = time.perf_counter()
    vectors_np = np.load(vectors_path).astype(np.float32)
    resolved = resolve_design_vectors(vectors_np).astype(np.float32)
    vectors = torch.from_numpy(resolved).to(kinematics.device)
    q, wrist_delta, benchmark = kinematics.solve(
        vectors, seed_q, iterations=iterations, candidate_chunk=128
    )
    wrist_position = seed_arrays["wrist_position"]
    wrist_quaternion = seed_arrays["wrist_quaternion_xyzw"]
    rotation = quat_xyzw_matrix(torch.from_numpy(wrist_quaternion).to(kinematics.device))
    wrist_position_all = (
        torch.from_numpy(wrist_position).unsqueeze(0)
        + torch.einsum("tij,ktj->kti", rotation.cpu(), wrist_delta)
    ).numpy()
    count = len(vectors_np)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vectors=vectors_np,
        joint_names=np.asarray(kinematics.joint_names),
        frame_ids=seed_arrays["frame_ids"],
        qpos_ids=seed_arrays["qpos_ids"],
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=wrist_quaternion,
        qpos=q.numpy(),
        wrist_delta_local=wrist_delta.numpy(),
        wrist_position_all=wrist_position_all,
        wrist_quaternion_xyzw_all=np.broadcast_to(
            wrist_quaternion[None], (count, *wrist_quaternion.shape)
        ).copy(),
        metadata_json=np.asarray(json.dumps({"benchmark": benchmark})),
    )
    return {
        "seconds": time.perf_counter() - start,
        "candidates": count,
        "solver_seconds": float(benchmark["seconds"]),
    }


def prepare_assets(
    vectors_path: Path,
    retarget_path: Path | None,
    generation_root: Path,
    bank_manifest: dict,
    fixed_reference: Path | None = None,
) -> tuple[dict, dict[str, float]]:
    timings: dict[str, float] = {}
    vectors = np.load(vectors_path)
    graphs = {
        "schema_version": 1,
        "hands": [
            vector_to_graph(vector.astype(np.float64), f"wuji_physx_{i:06d}")
            for i, vector in enumerate(vectors)
        ],
    }
    graphs_path = generation_root / "graphs.json"
    graphs_path.write_text(json.dumps(graphs, indent=2) + "\n")
    ir_root = generation_root / "hand_ir"
    generator_env = os.environ.copy()
    generator_env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(REPO_ROOT / "source/dexcodesign"),
                generator_env.get("PYTHONPATH", ""),
            ],
        )
    )
    generator_env["HAND_GRAPH_SPEC_PATH"] = str(graphs_path)
    generator_env["HAND_GENERATION_ROOT"] = str(ir_root)
    generator_env["HAND_GENERATION_SEED"] = "0"
    start = time.perf_counter()
    run(
        [sys.executable, "-m", "dexcodesign.morphology.generate"],
        env=generator_env,
    )
    timings["graph_ir_seconds"] = time.perf_counter() - start

    prepared = generation_root / "prepared"
    bank_compiled = args_cli.prototype_bank_root / "prepared/compiled/compiled_hands.json"
    template_usd = Path(bank_manifest["hand_usd_paths"][0])
    template_reference = Path(bank_manifest["reference_paths"][0])
    start = time.perf_counter()
    prepare_command = [
        sys.executable,
        str(SCRIPT_ROOT / "prepare_wuji_parametric_usd_smoke.py"),
        str(vectors_path if fixed_reference is not None else retarget_path),
        str(ir_root / "hand_ir.json"),
            "--template-usd",
            str(template_usd),
            "--template-reference",
            str(template_reference),
            "--prototype-bank-manifest",
            str(bank_manifest_path),
            "--prototype-bank-compiled",
            str(bank_compiled),
            "--output-root",
            str(prepared),
            "--limit",
            str(len(vectors)),
    ]
    if fixed_reference is not None:
        prepare_command.extend(("--fixed-reference", str(fixed_reference)))
    elif retarget_path is None:
        raise ValueError("retarget_path is required without fixed_reference")
    run(prepare_command)
    timings["asset_overlay_prepare_seconds"] = time.perf_counter() - start
    manifest_path = prepared / "physx_batch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    start = time.perf_counter()
    timings["usd_attach_seconds"] = (
        0.0
        if manifest.get("runtime_parametric_overlays", False)
        else attach_manifest(manifest)
    )
    timings["usd_attach_total_seconds"] = time.perf_counter() - start
    manifest["parametric_template_usd"] = None
    manifest["parametric_template_usd_paths"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, timings


def configure_batch(cfg, manifest: dict) -> None:
    usd_paths = [Path(value).resolve() for value in manifest["hand_usd_paths"]]
    reference_paths = [
        Path(value).resolve() for value in manifest["reference_paths"]
    ]
    env_module.MORPHOLOGY_BATCH_MANIFEST = manifest
    env_module._batch_usd_paths = usd_paths
    env_module._batch_reference_paths = reference_paths
    env_module.REFERENCE_PATH = reference_paths[0]
    cfg.hand_cfg.spawn.usd_path = [str(path) for path in usd_paths]
    cfg.scene.num_envs = len(usd_paths)
    cfg.scene.replicate_physics = False
    cfg.scene.clone_in_fabric = False
    cfg.randomize_start_phase = False
    cfg.episode_length_s = 15.0


def evaluate_batch(env, manifest: dict, global_offset: int) -> tuple[list[dict], float]:
    raw = env.unwrapped
    env.reset()
    count = len(manifest["vectors"])
    actions = torch.zeros((count, raw.action_dim), device=raw.device)
    active = torch.ones(count, dtype=torch.bool, device=raw.device)
    pose = torch.zeros(count, device=raw.device)
    contact = torch.zeros(count, device=raw.device)
    phase = torch.full((count,), -1, dtype=torch.long, device=raw.device)
    position = torch.full((count,), float("nan"), device=raw.device)
    orientation = torch.full((count,), float("nan"), device=raw.device)
    pinch = torch.zeros(count, dtype=torch.long, device=raw.device)
    thumb_force = torch.zeros(count, device=raw.device)
    other_force = torch.zeros(count, device=raw.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(raw._reference_length + 2):
            _, _, terminated, truncated, _ = env.step(actions)
            pose[active] += raw._last_pose_tracking_reward[active]
            contact[active] += raw._last_contact_reward[active]
            pinch[active] += raw._last_pinch_contact[active].long()
            thumb_force[active] = torch.maximum(
                thumb_force[active], raw._last_thumb_contact_force[active]
            )
            other_force[active] = torch.maximum(
                other_force[active], raw._last_other_finger_contact_force[active]
            )
            finished = active & (
                torch.as_tensor(terminated, device=raw.device)
                | torch.as_tensor(truncated, device=raw.device)
            )
            if finished.any():
                phase[finished] = raw._last_evaluated_phase[finished]
                position[finished] = raw._object_position_error[finished]
                orientation[finished] = raw._object_rotation_error[finished]
                active[finished] = False
            if not active.any():
                break
    seconds = time.perf_counter() - start
    if active.any():
        phase[active] = raw._last_evaluated_phase[active]
        position[active] = raw._object_position_error[active]
        orientation[active] = raw._object_rotation_error[active]
    total = pose + contact
    rows = []
    for i in range(count):
        rows.append(
            {
                "candidate_index": global_offset + i,
                "candidate_id": manifest["candidate_ids"][i],
                "vector": manifest["vectors"][i],
                "total_reward": float(total[i].item()),
                "pose_reward": float(pose[i].item()),
                "contact_reward": float(contact[i].item()),
                "pinch_contact_steps": int(pinch[i].item()),
                "phase": int(phase[i].item()),
                "success": bool(phase[i].item() >= raw._reference_length - 1),
                "position_error_m": float(position[i].item()),
                "orientation_error_rad": float(orientation[i].item()),
                "max_thumb_contact_force_n": float(thumb_force[i].item()),
                "max_other_finger_contact_force_n": float(other_force[i].item()),
            }
        )
    return rows, seconds


def slice_manifest(manifest: dict, begin: int, end: int) -> dict:
    count = len(manifest["vectors"])
    result = {}
    for key, value in manifest.items():
        if isinstance(value, list) and len(value) == count:
            result[key] = value[begin:end]
        else:
            result[key] = value
    return result


def replicate_manifest(
    manifest: dict,
    replicas: int,
    global_morphology_indices: list[int],
) -> dict:
    """Repeat every morphology without changing its frozen reference."""

    morphology_count = len(manifest["vectors"])
    if morphology_count != len(global_morphology_indices):
        raise ValueError("global morphology index count does not match manifest")
    result: dict = {}
    for key, value in manifest.items():
        if isinstance(value, list) and len(value) == morphology_count:
            result[key] = [item for item in value for _ in range(replicas)]
        else:
            result[key] = value
    result["candidate_ids"] = [
        f"wuji_physx_{global_index:06d}_replica_{replica:03d}"
        for global_index in global_morphology_indices
        for replica in range(replicas)
    ]
    result["morphology_indices"] = [
        global_index
        for global_index in global_morphology_indices
        for _ in range(replicas)
    ]
    result["replica_indices"] = [
        replica
        for _ in global_morphology_indices
        for replica in range(replicas)
    ]
    result["morphology_replicas"] = replicas
    result["fixed_reference_shared_across_morphologies"] = True
    return result


def distributed_barrier() -> None:
    if skrl.config.torch.is_distributed:
        torch.distributed.barrier()


def evaluate_shared_policy(
    env,
    raw_env,
    runner: Runner,
    manifest: dict,
) -> tuple[list[dict], float]:
    """Evaluate one deterministic shared policy and aggregate replicas."""

    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)
    observations, _ = env.reset()
    agent_act_requires_states = "states" in inspect.signature(
        runner.agent.act
    ).parameters
    count = raw_env.num_envs
    active = torch.ones(count, dtype=torch.bool, device=raw_env.device)
    pose = torch.zeros(count, device=raw_env.device)
    contact = torch.zeros(count, device=raw_env.device)
    phase = torch.full((count,), -1, dtype=torch.long, device=raw_env.device)
    position = torch.full((count,), float("nan"), device=raw_env.device)
    orientation = torch.full((count,), float("nan"), device=raw_env.device)
    pinch = torch.zeros(count, dtype=torch.long, device=raw_env.device)
    successful = torch.zeros(count, dtype=torch.bool, device=raw_env.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(raw_env._reference_length + 2):
            if agent_act_requires_states:
                outputs = runner.agent.act(
                    observations, None, timestep=0, timesteps=0
                )
            else:
                outputs = runner.agent.act(
                    observations, timestep=0, timesteps=0
                )
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = env.step(actions)
            pose[active] += raw_env._last_pose_tracking_reward[active]
            contact[active] += raw_env._last_contact_reward[active]
            pinch[active] += raw_env._last_pinch_contact[active].long()
            terminated_tensor = torch.as_tensor(
                terminated, device=raw_env.device
            ).reshape(-1)
            truncated_tensor = torch.as_tensor(
                truncated, device=raw_env.device
            ).reshape(-1)
            finished = active & (terminated_tensor | truncated_tensor)
            if finished.any():
                phase[finished] = raw_env._last_evaluated_phase[finished]
                position[finished] = raw_env._object_position_error[finished]
                orientation[finished] = raw_env._object_rotation_error[finished]
                successful[finished] = (
                    truncated_tensor[finished]
                    & ~terminated_tensor[finished]
                    & (
                        phase[finished]
                        >= raw_env._reference_length - 1
                    )
                )
                active[finished] = False
            if not active.any():
                break
    seconds = time.perf_counter() - start
    if active.any():
        phase[active] = raw_env._last_evaluated_phase[active]
        position[active] = raw_env._object_position_error[active]
        orientation[active] = raw_env._object_rotation_error[active]
        successful[active] = phase[active] >= raw_env._reference_length - 1

    total = pose + contact
    morphology_indices = torch.as_tensor(
        manifest["morphology_indices"],
        device=raw_env.device,
        dtype=torch.long,
    )
    vectors = np.asarray(manifest["vectors"], dtype=np.float32)
    rows: list[dict] = []
    for morphology_index in sorted(set(manifest["morphology_indices"])):
        ids = (morphology_indices == morphology_index).nonzero(
            as_tuple=False
        ).flatten()
        first = int(ids[0].item())
        group_total = total[ids]
        group_phase = phase[ids]
        group_success = successful[ids]
        rows.append(
            {
                "candidate_index": int(morphology_index),
                "candidate_id": f"wuji_physx_{morphology_index:06d}",
                "vector": vectors[first].tolist(),
                "total_reward": float(group_total.mean().item()),
                "reward_std": float(group_total.std(unbiased=False).item()),
                "pose_reward": float(pose[ids].mean().item()),
                "contact_reward": float(contact[ids].mean().item()),
                "pinch_contact_steps": float(pinch[ids].float().mean().item()),
                "phase": int(group_phase.max().item()),
                "mean_phase": float(group_phase.float().mean().item()),
                "success": bool(group_success.any().item()),
                "success_count": int(group_success.sum().item()),
                "replicas": int(len(ids)),
                "position_error_m": float(position[ids].nanmean().item()),
                "orientation_error_rad": float(orientation[ids].nanmean().item()),
            }
        )
    return rows, seconds


def train_and_evaluate_shared_ppo(
    env_cfg,
    agent_cfg: dict,
    manifest: dict,
    generation: int,
    output: Path,
    checkpoint_to_load: Path | None,
    rank: int,
    local_rank: int,
) -> tuple[list[dict], dict[str, float | int | str]]:
    """Train the official shared SKRL PPO, then evaluate it deterministically."""

    cfg = copy.deepcopy(env_cfg)
    configure_batch(cfg, manifest)
    cfg.sim.device = f"cuda:{local_rank}"
    cfg.seed = args_cli.seed + rank
    ppo_cfg = copy.deepcopy(agent_cfg)
    ppo_cfg["seed"] = args_cli.seed
    rollouts = int(ppo_cfg["agent"]["rollouts"])
    ppo_cfg["trainer"]["timesteps"] = (
        args_cli.shared_ppo_iterations * rollouts
    )
    ppo_cfg["trainer"]["close_environment_at_exit"] = False
    experiment = ppo_cfg["agent"]["experiment"]
    experiment["directory"] = str(output / "ppo_logs" / f"rank_{rank:02d}")
    experiment["experiment_name"] = f"generation_{generation:03d}"
    experiment["checkpoint_interval"] = 0

    start = time.perf_counter()
    raw_gym_env = gym.make(args_cli.task, cfg=cfg)
    initialization_seconds = time.perf_counter() - start
    raw_env = raw_gym_env.unwrapped
    env = SkrlVecEnvWrapper(raw_gym_env, ml_framework="torch")
    runner = Runner(env, ppo_cfg)
    if checkpoint_to_load is not None:
        runner.agent.load(str(checkpoint_to_load))
    start = time.perf_counter()
    runner.run()
    training_seconds = time.perf_counter() - start

    checkpoint = output / "ppo_checkpoints" / f"generation_{generation:03d}.pt"
    if rank == 0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        runner.agent.save(str(checkpoint))
        runner.agent.save(str(output / "shared_ppo_latest.pt"))
    distributed_barrier()
    rows, evaluation_seconds = evaluate_shared_policy(
        env, raw_env, runner, manifest
    )
    env.close()
    return rows, {
        "local_envs": len(manifest["vectors"]),
        "ppo_iterations": args_cli.shared_ppo_iterations,
        "rollout_steps": rollouts,
        "initialization_seconds": initialization_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint": str(checkpoint),
    }


def shared_ppo_outer_search(
    env_cfg,
    agent_cfg: dict,
    output: Path,
    bank_manifest: dict,
) -> None:
    """Alternate several shared-policy PPO updates with one outer SAC update."""

    rank = int(skrl.config.torch.rank)
    local_rank = int(skrl.config.torch.local_rank)
    world_size = int(skrl.config.torch.world_size)
    if args_cli.optimizer_backend != "skrl":
        raise ValueError("shared PPO outer search requires --optimizer-backend skrl")
    if args_cli.population % world_size:
        raise ValueError(
            f"population {args_cli.population} must be divisible by world size "
            f"{world_size}"
        )
    fixed_reference = args_cli.fixed_reference or Path(
        bank_manifest["reference_paths"][0]
    ).resolve()
    if not fixed_reference.is_file():
        raise FileNotFoundError(f"fixed WUJI reference not found: {fixed_reference}")
    if args_cli.ppo_checkpoint is not None and not args_cli.ppo_checkpoint.is_file():
        raise FileNotFoundError(
            f"initial shared PPO checkpoint not found: {args_cli.ppo_checkpoint}"
        )

    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "fixed_reference_contract.json").write_text(
            json.dumps(
                {
                    "reference": str(fixed_reference),
                    "retarget_per_generation": False,
                    "policy_reference_is_immutable": True,
                    "population": args_cli.population,
                    "morphology_replicas": args_cli.morphology_replicas,
                    "global_envs": (
                        args_cli.population * args_cli.morphology_replicas
                    ),
                    "world_size": world_size,
                    "envs_per_rank": (
                        args_cli.population
                        // world_size
                        * args_cli.morphology_replicas
                    ),
                    "ppo_iterations_per_sac_generation": (
                        args_cli.shared_ppo_iterations
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    distributed_barrier()

    optimizer = None
    if rank == 0:
        optimizer = SkrlConditionalMorphologySAC(
            population=args_cli.population,
            generations=args_cli.generations,
            gradient_steps=args_cli.sac_updates,
            batch_size=args_cli.sac_batch_size,
            uniform_fraction=args_cli.uniform_fraction,
            elite_mutation_fraction=args_cli.elite_mutation_fraction,
            elite_replay_fraction=args_cli.elite_replay_fraction,
            elite_mutation_sigma=args_cli.elite_mutation_sigma,
            reward_scale=args_cli.reward_scale,
            seed=args_cli.seed,
            output_root=output,
            device=f"cuda:{local_rank}",
        )

    history: list[dict] = []
    for generation in range(args_cli.generations):
        generation_root = output / f"generation_{generation:03d}"
        vectors_path = generation_root / "vectors.npy"
        proposal: ProposalBatch | None = None
        if rank == 0:
            generation_root.mkdir(parents=True, exist_ok=True)
            assert optimizer is not None
            proposal = optimizer.propose(generation)
            np.save(vectors_path, proposal.vectors)
            vectors_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "backend": "skrl-2.1-sac",
                        "generation": generation,
                        "sample_source_counts": {
                            source: proposal.sample_sources.count(source)
                            for source in sorted(set(proposal.sample_sources))
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
        distributed_barrier()
        global_vectors = np.load(vectors_path).astype(np.float32)
        per_rank = args_cli.population // world_size
        begin = rank * per_rank
        end = begin + per_rank
        global_indices = list(range(begin, end))
        rank_root = generation_root / f"rank_{rank:02d}"
        rank_root.mkdir(parents=True, exist_ok=True)
        local_vectors_path = rank_root / "vectors.npy"
        np.save(local_vectors_path, global_vectors[begin:end])
        local_manifest, prepare_timings = prepare_assets(
            local_vectors_path,
            None,
            rank_root,
            bank_manifest,
            fixed_reference=fixed_reference,
        )
        expanded_manifest = replicate_manifest(
            local_manifest,
            args_cli.morphology_replicas,
            global_indices,
        )
        expanded_manifest_path = rank_root / "prepared/ppo_batch_manifest.json"
        expanded_manifest_path.write_text(
            json.dumps(expanded_manifest, indent=2) + "\n"
        )

        if generation:
            sim_utils.create_new_stage()
        checkpoint_to_load = (
            args_cli.ppo_checkpoint
            if generation == 0
            else output / "shared_ppo_latest.pt"
        )
        if checkpoint_to_load is not None and not checkpoint_to_load.is_file():
            raise FileNotFoundError(
                f"shared PPO checkpoint unavailable at generation {generation}: "
                f"{checkpoint_to_load}"
            )
        local_rows, ppo_timings = train_and_evaluate_shared_ppo(
            env_cfg,
            agent_cfg,
            expanded_manifest,
            generation,
            output,
            checkpoint_to_load,
            rank,
            local_rank,
        )
        if skrl.config.torch.is_distributed:
            gathered: list[list[dict] | None] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_rows)
            all_rows = [
                row
                for rank_rows in gathered
                if rank_rows is not None
                for row in rank_rows
            ]
            timing_gather: list[dict | None] = [None] * world_size
            torch.distributed.all_gather_object(timing_gather, ppo_timings)
        else:
            all_rows = local_rows
            timing_gather = [ppo_timings]

        should_stop = False
        if rank == 0:
            assert optimizer is not None
            assert proposal is not None
            all_rows.sort(key=lambda row: row["total_reward"], reverse=True)
            row_by_index = {row["candidate_index"]: row for row in all_rows}
            if len(row_by_index) != args_cli.population:
                raise RuntimeError(
                    f"expected {args_cli.population} morphology results, got "
                    f"{len(row_by_index)}"
                )
            rewards = np.asarray(
                [row_by_index[index]["total_reward"] for index in range(args_cli.population)],
                dtype=np.float32,
            )
            optimizer_status = optimizer.observe(generation, proposal, rewards)
            for row in all_rows:
                index = row["candidate_index"]
                row["sample_source"] = proposal.sample_sources[index]
                row["palm_prototype_index"] = int(proposal.palm_indices[index])
                row["requested_palm_expansion"] = float(
                    proposal.semantic_vectors[index, 0]
                )
                row["semantic_vector"] = proposal.semantic_vectors[index].tolist()
            source_metrics = {}
            for source in sorted(set(proposal.sample_sources)):
                values = np.asarray(
                    [
                        row["total_reward"]
                        for row in all_rows
                        if row["sample_source"] == source
                    ],
                    dtype=np.float64,
                )
                source_metrics[source] = {
                    "count": int(len(values)),
                    "reward_mean": float(values.mean()),
                    "reward_max": float(values.max()),
                }
            optimizer_status["sample_source_metrics"] = source_metrics
            summary = {
                "schema_version": 1,
                "algorithm": "fixed_reference_shared_ppo_outer_skrl_sac",
                "retarget_performed": False,
                "fixed_reference": str(fixed_reference),
                "population": args_cli.population,
                "morphology_replicas": args_cli.morphology_replicas,
                "global_envs": args_cli.population * args_cli.morphology_replicas,
                "world_size": world_size,
                "ppo_iterations": args_cli.shared_ppo_iterations,
                "best": all_rows[0],
                "results": all_rows,
                "prepare_timings_rank0": prepare_timings,
                "ppo_timings_by_rank": timing_gather,
                "optimizer_status": optimizer_status,
            }
            (generation_root / "shared_ppo_results.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            history.append(
                {
                    "generation": generation,
                    "best_reward": all_rows[0]["total_reward"],
                    "mean_reward": float(rewards.mean()),
                    "best_phase": all_rows[0]["phase"],
                    "success_count": sum(row["success"] for row in all_rows),
                    "ppo_iterations_total": (
                        (generation + 1) * args_cli.shared_ppo_iterations
                    ),
                }
            )
            (output / "shared_ppo_training_history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            print(
                "WUJI_SHARED_PPO_SAC_GENERATION "
                f"generation={generation} "
                f"best_reward={all_rows[0]['total_reward']:.9f} "
                f"mean_reward={rewards.mean():.9f} "
                f"best_phase={all_rows[0]['phase']}/445 "
                f"successes={sum(row['success'] for row in all_rows)} "
                f"ppo_iterations_total="
                f"{(generation + 1) * args_cli.shared_ppo_iterations}",
                flush=True,
            )
            should_stop = all_rows[0]["total_reward"] >= args_cli.target_reward
        if skrl.config.torch.is_distributed:
            stop_tensor = torch.tensor(
                int(should_stop), device=f"cuda:{local_rank}", dtype=torch.int32
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            if rank == 0:
                print(
                    f"WUJI_SHARED_PPO_TARGET_REACHED target={args_cli.target_reward}",
                    flush=True,
                )
            break
        distributed_barrier()

    if rank == 0:
        print("WUJI_SHARED_PPO_SAC_COMPLETE", flush=True)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    output = args_cli.output_root
    output.mkdir(parents=True, exist_ok=True)
    bank_manifest = json.loads(bank_manifest_path.read_text())
    if args_cli.shared_ppo_iterations:
        shared_ppo_outer_search(env_cfg, agent_cfg, output, bank_manifest)
        return
    if skrl.config.torch.is_distributed:
        raise RuntimeError(
            "distributed execution is only supported with "
            "--shared-ppo-iterations > 0"
        )
    with np.load(args_cli.seed_trajectory) as seed:
        joint_names = joint_names_from_seed(seed)
        seed_arrays = {
            "qpos": seed["qpos"].astype(np.float32),
            "wrist_position": seed["wrist_position"].astype(np.float32),
            "wrist_quaternion_xyzw": seed["wrist_quaternion_xyzw"].astype(
                np.float32
            ),
            "frame_ids": seed["frame_ids"].astype(np.int64),
            "qpos_ids": seed["qpos_ids"].astype(np.int64),
        }
    kinematics = WujiBatchKinematics(joint_names, torch.device("cuda"))
    seed_q = torch.from_numpy(seed_arrays["qpos"]).cuda()
    state_path = output / "hybrid_sac_state.pt"
    replay_path = output / "hybrid_sac_replay.npz"
    history = []
    previous_summary: Path | None = None
    stage_created = False
    skrl_optimizer = None
    if args_cli.optimizer_backend == "skrl":
        skrl_optimizer = SkrlConditionalMorphologySAC(
            population=args_cli.population,
            generations=args_cli.generations,
            gradient_steps=args_cli.sac_updates,
            batch_size=args_cli.sac_batch_size,
            uniform_fraction=args_cli.uniform_fraction,
            elite_mutation_fraction=args_cli.elite_mutation_fraction,
            elite_replay_fraction=args_cli.elite_replay_fraction,
            elite_mutation_sigma=args_cli.elite_mutation_sigma,
            reward_scale=args_cli.reward_scale,
            seed=args_cli.seed,
            output_root=output,
            device="cuda",
        )
    for generation in range(args_cli.generations):
        generation_root = output / f"generation_{generation:03d}"
        generation_root.mkdir(parents=True, exist_ok=True)
        vectors_path = generation_root / "vectors.npy"
        sac_start = time.perf_counter()
        proposal: ProposalBatch | None = None
        if skrl_optimizer is not None:
            proposal = skrl_optimizer.propose(generation)
            np.save(vectors_path, proposal.vectors)
            proposal_status = {
                "backend": "skrl-2.1-sac",
                "generation": generation,
                "palm_representation": (
                    "continuous_action_quantized_at_physx_boundary"
                ),
                "sample_source_counts": {
                    source: proposal.sample_sources.count(source)
                    for source in sorted(set(proposal.sample_sources))
                },
            }
            vectors_path.with_suffix(".json").write_text(
                json.dumps(proposal_status, indent=2) + "\n"
            )
        else:
            command = [
                sys.executable,
                str(SCRIPT_ROOT / "hybrid_sac_wuji_morphology.py"),
                "--state",
                str(state_path),
                "--replay",
                str(replay_path),
                "--output-vectors",
                str(vectors_path),
                "--population",
                str(args_cli.population),
                "--updates",
                str(args_cli.sac_updates),
                "--batch-size",
                str(args_cli.sac_batch_size),
                "--uniform-fraction",
                str(args_cli.uniform_fraction),
                "--reward-scale",
                str(args_cli.reward_scale),
                "--seed",
                str(args_cli.seed),
                "--device",
                "cuda",
            ]
            if previous_summary is not None:
                command.extend(("--exact-summary", str(previous_summary)))
            run(command)
        timings = {"sac_update_and_sample_seconds": time.perf_counter() - sac_start}
        retarget_path = generation_root / "gpu_retarget_all.npz"
        timings["retarget"] = retarget(
            vectors_path,
            retarget_path,
            kinematics,
            seed_q,
            seed_arrays,
            args_cli.retarget_iterations,
        )
        manifest, prepare_timings = prepare_assets(
            vectors_path, retarget_path, generation_root, bank_manifest
        )
        timings.update(prepare_timings)
        rows = []
        initialization_seconds = rollout_seconds = 0.0
        batch_records = []
        for begin in range(0, args_cli.population, args_cli.physics_batch_size):
            end = min(begin + args_cli.physics_batch_size, args_cli.population)
            if stage_created:
                sim_utils.create_new_stage()
            stage_created = True
            batch = slice_manifest(manifest, begin, end)
            cfg = copy.deepcopy(env_cfg)
            configure_batch(cfg, batch)
            start = time.perf_counter()
            env = gym.make(args_cli.task, cfg=cfg)
            initialization = time.perf_counter() - start
            batch_rows, rollout_time = evaluate_batch(env, batch, begin)
            env.close()
            rows.extend(batch_rows)
            initialization_seconds += initialization
            rollout_seconds += rollout_time
            batch_record = {
                "begin": begin,
                "end": end,
                "initialization_seconds": initialization,
                "rollout_seconds": rollout_time,
                "best_phase": max(row["phase"] for row in batch_rows),
                "best_reward": max(row["total_reward"] for row in batch_rows),
            }
            batch_records.append(batch_record)
            print("WUJI_SAC_PHYSX_BATCH " + json.dumps(batch_record), flush=True)
        rows.sort(key=lambda row: row["total_reward"], reverse=True)
        optimizer_status = None
        if skrl_optimizer is not None:
            assert proposal is not None
            row_by_index = {row["candidate_index"]: row for row in rows}
            rewards = np.asarray(
                [row_by_index[index]["total_reward"] for index in range(args_cli.population)],
                dtype=np.float32,
            )
            update_start = time.perf_counter()
            optimizer_status = skrl_optimizer.observe(generation, proposal, rewards)
            timings["optimizer_update_seconds"] = time.perf_counter() - update_start
            for row in rows:
                index = row["candidate_index"]
                row["sample_source"] = proposal.sample_sources[index]
                row["palm_prototype_index"] = int(proposal.palm_indices[index])
                row["requested_palm_expansion"] = float(
                    proposal.semantic_vectors[index, 0]
                )
                row["semantic_vector"] = proposal.semantic_vectors[index].tolist()
            source_metrics = {}
            for source in sorted(set(proposal.sample_sources)):
                source_rows = [row for row in rows if row["sample_source"] == source]
                source_rewards = np.asarray(
                    [row["total_reward"] for row in source_rows], dtype=np.float64
                )
                source_metrics[source] = {
                    "count": len(source_rows),
                    "reward_mean": float(source_rewards.mean()),
                    "reward_median": float(np.median(source_rewards)),
                    "reward_p90": float(np.quantile(source_rewards, 0.90)),
                    "reward_max": float(source_rewards.max()),
                    "success_count": sum(row["success"] for row in source_rows),
                }
            optimizer_status["sample_source_metrics"] = source_metrics
        timings["physx_initialization_seconds"] = initialization_seconds
        timings["physx_rollout_seconds"] = rollout_seconds
        summary = {
            "schema_version": 1,
            "backend": "persistent_isaaclab_physx_gpu",
            "algorithm": (
                "skrl_sac_with_continuous_palm_action"
                if skrl_optimizer is not None
                else "episode_level_hybrid_sac"
            ),
            "all_candidates_physically_evaluated": True,
            "proxy_used": False,
            "top_k_prefilter_used": False,
            "candidate_count": args_cli.population,
            "completed": len(rows),
            "best": rows[0],
            "results": rows,
            "timings": timings,
            "batches": batch_records,
            "optimizer_status": optimizer_status,
        }
        summary_path = generation_root / "physx_results.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        history.append(
            {
                "generation": generation,
                "best_reward": rows[0]["total_reward"],
                "best_phase": rows[0]["phase"],
                "success_count": sum(row["success"] for row in rows),
                "timings": timings,
            }
        )
        (output / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(
            "WUJI_HYBRID_SAC_GENERATION "
            f"generation={generation} candidates={len(rows)}/{args_cli.population} "
            f"best_reward={rows[0]['total_reward']:.9f} "
            f"best_phase={rows[0]['phase']}/445 "
            f"successes={sum(row['success'] for row in rows)} "
            f"timings={json.dumps(timings, sort_keys=True)}",
            flush=True,
        )
        previous_summary = summary_path
        if any(row["success"] for row in rows):
            print("WUJI_HYBRID_SAC_SUCCESS", flush=True)
            if not args_cli.continue_after_success:
                break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
