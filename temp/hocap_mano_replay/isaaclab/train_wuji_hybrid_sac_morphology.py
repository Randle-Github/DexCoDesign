#!/usr/bin/env python3
"""Persistent-Isaac hybrid SAC search over WUJI morphology."""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--prototype-bank-root", type=Path, help="Required except with --original-source-hand")
parser.add_argument(
    "--original-source-hand", action="store_true",
    help="One fixed original WUJI USD, fresh Torch retargeting, 385D PPO, no asset generation or SAC updates",
)
parser.add_argument("--seed-trajectory", type=Path, help="Legacy morphology-only retargeting seed; ignored by palm-geometry shared PPO")
parser.add_argument(
    "--num-morphologies", "--population", dest="population", type=int, default=4096,
    help="number of distinct hand designs proposed per morphology generation",
)
parser.add_argument(
    "--physics-batch-size",
    type=int,
    default=4096,
    help=(
        "exact PhysX candidates per scene; the default evaluates the complete "
        "population in one scene to avoid rebuilding fixed support/object assets"
    ),
)
parser.add_argument(
    "--rollouts-per-proposal",
    type=int,
    default=1,
    help=(
        "independent zero-residual PhysX rollouts per proposed morphology; "
        "their mean reward is used for the SAC update"
    ),
)
parser.add_argument("--generations", type=int, default=20)
parser.add_argument(
    "--isolate-physics-per-generation",
    action="store_true",
    help=(
        "evaluate each generation in a fresh Isaac Sim child process so the OS "
        "reclaims PhysX/USD native allocations when that generation finishes"
    ),
)
parser.add_argument("--physics-worker-manifest", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--physics-worker-output", type=Path, help=argparse.SUPPRESS)
parser.add_argument(
    "--continue-after-success",
    action="store_true",
    help="run every requested generation even when a 445/445 candidate exists",
)
parser.add_argument("--sac-updates", type=int, default=400)
parser.add_argument("--sac-batch-size", type=int, default=1024)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="log SKRL SAC/PPO metrics to Weights & Biases",
)
parser.add_argument(
    "--ppo-episode-log-window-steps", type=int, default=1600,
    help="Training-step window for episode-weighted W&B return/length means; persists across generations",
)
parser.add_argument("--wandb-project", default="DexCoDesign")
parser.add_argument("--wandb-group", default="wuji-hybrid-sac")
parser.add_argument("--wandb-run-name", default="conditional_morphology_sac")
parser.add_argument(
    "--video", action="store_true",
    help="record an existing zero-residual training rollout on scheduled generations; upload with --wandb",
)
parser.add_argument("--video-interval", type=int, default=10, help="record generations 0, N, 2N, ...")
parser.add_argument(
    "--video-candidate-index", type=int, default=0,
    help="proposal index to record, replica 0; this is not the best-candidate selection",
)
parser.add_argument(
    "--video-length", type=int, default=445,
    help="maximum control steps recorded, stopping when this rollout terminates",
)
parser.add_argument(
    "--video-stride", type=int, default=2,
    help="record every N control steps; MP4 fps follows the simulation timestep",
)
parser.add_argument("--video-width", type=int, default=640)
parser.add_argument("--video-height", type=int, default=480)
parser.add_argument(
    "--fixed-palm-prototype",
    type=int,
    choices=range(32),
    metavar="{0..31}",
    help=(
        "freeze the palm at one precompiled prototype and remove it from the "
        "SAC action; prototype 0 is the exact source WUJI palm"
    ),
)
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
parser.add_argument("--retarget-iterations", type=int, default=14)
parser.add_argument(
    "--mano-rollout", type=Path,
    default=Path(__file__).resolve().parents[3] / "artifacts/isaaclab_mano_residual/refined_mano/successful_rollout.npz",
    help="Captured MANO hand states and actual object motion for direct palm-geometry co-training targets",
)
parser.add_argument(
    "--ppo-cycles-per-generation", "--shared-ppo-iterations",
    dest="shared_ppo_iterations",
    type=int,
    default=0,
    help=(
        "PPO collection/update cycles trained on the complete morphology "
        "population before each outer SAC update; zero preserves morphology-only search"
    ),
)
parser.add_argument(
    "--ppo-observation-mode",
    choices=("legacy", "palm_geometry"),
    default="legacy",
    help=(
        "shared-PPO observation: palm_geometry uses 60 candidate-specific "
        "collision-surface landmarks (385 values) and retargets every proposal"
    ),
)
parser.add_argument(
    "--geometry-inward-direction-mode",
    choices=("reference_object", "kinematic_normal", "negative_kinematic_normal"),
    default="kinematic_normal",
    help="surface side used by the 60-point palm-geometry observation",
)
parser.add_argument(
    "--train-envs-per-morphology", "--morphology-replicas",
    dest="morphology_replicas",
    type=int,
    default=1,
    help="parallel PPO training environments per morphology across all ranks",
)
parser.add_argument(
    "--eval-envs-per-morphology",
    type=int,
    default=None,
    help=(
        "environments per morphology scored during deterministic shared-PPO evaluation; "
        "defaults to all training environments, must not exceed the training count; "
        "reuses the first K replicas and still steps the full training scene"
    ),
)
parser.add_argument(
    "--ppo-rollout-multiplier",
    type=int,
    default=1,
    help=(
        "temporal sample multiplier per physical morphology environment; "
        "PPO collects base_rollouts * multiplier steps before each update"
    ),
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
    "--grouped-zero-action-vectors",
    type=Path,
    help=(
        "debug one fixed vector population in the grouped replicated scene "
        "with exactly zero residual actions and no PPO/SAC updates"
    ),
)
parser.add_argument(
    "--force-source-morphology",
    action="store_true",
    help=(
        "PPO isolation control: replace every proposed morphology by the exact "
        "unchanged source vector and disable morphology optimizer updates"
    ),
)
parser.add_argument(
    "--morphology-context",
    action="store_true",
    help="append the normalized morphology design vector to every PPO observation",
)
parser.add_argument(
    "--fixed-ppo-vectors",
    type=Path,
    help="fixed heterogeneous vector population for a PPO-only isolation run",
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
original_cli_argv = list(sys.argv[1:])
args_cli, hydra_args = parser.parse_known_args()
args_cli.output_root = args_cli.output_root.resolve()
if args_cli.original_source_hand:
    if args_cli.population != 1 or args_cli.shared_ppo_iterations < 1:
        parser.error("--original-source-hand requires --num-morphologies 1 and positive --ppo-cycles-per-generation")
    if args_cli.ppo_observation_mode != "palm_geometry":
        parser.error("--original-source-hand requires --ppo-observation-mode palm_geometry")
    if args_cli.fixed_ppo_vectors is not None or args_cli.grouped_zero_action_vectors is not None:
        parser.error("--original-source-hand cannot use fixed-vector or grouped-zero-action modes")
    if args_cli.fixed_palm_prototype not in (None, 0):
        parser.error("--original-source-hand cannot select a different palm prototype")
    args_cli.force_source_morphology = True
    args_cli.fixed_palm_prototype = 0
elif args_cli.prototype_bank_root is None:
    parser.error("--prototype-bank-root is required without --original-source-hand")
if args_cli.prototype_bank_root is not None:
    args_cli.prototype_bank_root = args_cli.prototype_bank_root.resolve()
if args_cli.seed_trajectory is not None:
    args_cli.seed_trajectory = args_cli.seed_trajectory.resolve()
if not args_cli.original_source_hand:
    bank_signature_path = args_cli.prototype_bank_root / "vectors.schema.json"
    if not bank_signature_path.is_file():
        parser.error(
            "prototype bank has no grammar signature; rebuild it with "
            "build_wuji_palm_prototype_bank.sbatch instead of reusing the legacy bank: "
            f"{bank_signature_path}"
        )
    bank_signature = json.loads(bank_signature_path.read_text(encoding="utf-8"))
    if bank_signature.get("palm_collision_partition") != "source_base_and_palm_v1":
        print(
            "[WARNING] This cached prototype bank uses the legacy combined palm/base "
            "collision hull. Use prepare_wuji_split_palm_bank.py to create a corrected "
            "copy and pass its directory with --prototype-bank-root.", flush=True,
        )
    expected_bank_signature = {
        "grammar_id": "general-simulation-hand-v3",
        "source_hand": "wuji_hand_2",
        "vector_dimension": 23,
        "palm_layout_mode": "source_star_fusion",
        "palm_prototype_count": 32,
        "palm_expansion_range": [0.0, 0.70],
        "zero_prototype_is_exact_source": True,
    }
    signature_mismatch = {
        key: {"expected": expected, "actual": bank_signature.get(key)}
        for key, expected in expected_bank_signature.items()
        if bank_signature.get(key) != expected
    }
    if signature_mismatch:
        parser.error(
            "prototype bank belongs to a different morphology grammar: "
            + json.dumps(signature_mismatch, sort_keys=True)
        )
if args_cli.physics_batch_size < 1:
    parser.error("--physics-batch-size must be positive")
if args_cli.rollouts_per_proposal < 1:
    parser.error("--rollouts-per-proposal must be positive")
if (
    args_cli.fixed_palm_prototype is not None
    and args_cli.optimizer_backend != "skrl"
):
    parser.error("--fixed-palm-prototype currently requires --optimizer-backend skrl")
if args_cli.shared_ppo_iterations < 0:
    parser.error("--ppo-cycles-per-generation must be non-negative")
if args_cli.morphology_replicas < 1:
    parser.error("--train-envs-per-morphology must be positive")
if args_cli.eval_envs_per_morphology is not None:
    if args_cli.shared_ppo_iterations < 1:
        parser.error("--eval-envs-per-morphology requires shared PPO training")
    if not 1 <= args_cli.eval_envs_per_morphology <= args_cli.morphology_replicas:
        parser.error("--eval-envs-per-morphology must be between 1 and --train-envs-per-morphology")
else:
    args_cli.eval_envs_per_morphology = args_cli.morphology_replicas
if args_cli.ppo_episode_log_window_steps < 1:
    parser.error("--ppo-episode-log-window-steps must be positive")
if args_cli.ppo_rollout_multiplier < 1:
    parser.error("--ppo-rollout-multiplier must be positive")
if args_cli.ppo_observation_mode == "palm_geometry":
    if not args_cli.shared_ppo_iterations:
        parser.error("--ppo-observation-mode palm_geometry requires shared PPO")
    if args_cli.morphology_context:
        parser.error(
            "palm_geometry is already morphology-conditioned and must remain "
            "385-dimensional; remove --morphology-context"
        )
    if args_cli.fixed_reference is not None:
        parser.error(
            "palm_geometry co-training retargets every proposal; remove "
            "--fixed-reference"
        )
if args_cli.video:
    if args_cli.shared_ppo_iterations or args_cli.grouped_zero_action_vectors is not None:
        parser.error("--video currently supports the morphology-only zero-residual rollout workflow")
    if min(
        args_cli.video_interval, args_cli.video_length, args_cli.video_stride,
        args_cli.video_width, args_cli.video_height,
    ) < 1:
        parser.error("video interval, length, stride and resolution must be positive")
    if args_cli.video_width % 2 or args_cli.video_height % 2:
        parser.error("MP4 video width and height must be even")
    if not 0 <= args_cli.video_candidate_index < args_cli.population:
        parser.error("--video-candidate-index must be within the population")
    if shutil.which("ffmpeg") is None:
        parser.error("--video requires ffmpeg on PATH")
if args_cli.fixed_reference is not None:
    args_cli.fixed_reference = args_cli.fixed_reference.expanduser().resolve()
if args_cli.grouped_zero_action_vectors is not None:
    args_cli.grouped_zero_action_vectors = (
        args_cli.grouped_zero_action_vectors.expanduser().resolve()
    )
if args_cli.fixed_ppo_vectors is not None:
    args_cli.fixed_ppo_vectors = args_cli.fixed_ppo_vectors.expanduser().resolve()
if args_cli.ppo_checkpoint is not None:
    args_cli.ppo_checkpoint = args_cli.ppo_checkpoint.expanduser().resolve()
if args_cli.physics_worker_manifest is not None:
    args_cli.physics_worker_manifest = args_cli.physics_worker_manifest.resolve()
if args_cli.physics_worker_output is not None:
    args_cli.physics_worker_output = args_cli.physics_worker_output.resolve()
if args_cli.original_source_hand:
    bank_manifest_path = None
    original_root = Path(__file__).resolve().parents[3] / "artifacts/isaaclab_all_hands_residual"
    for original_path in (
        original_root / "assets/wuji_hand_2/hand.usd",
        original_root / "prepared/wuji_hand_2/hand_rl.urdf",
        original_root / "prepared/wuji_hand_2/reference.npz",
    ):
        if not original_path.is_file():
            parser.error(f"Missing original WUJI asset: {original_path}")
    os.environ.pop("DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST", None)
    os.environ["DEXCODESIGN_HAND_ID"] = "wuji_hand_2"
    os.environ["DEXCODESIGN_REFERENCE_PATH"] = str(original_root / "prepared/wuji_hand_2/reference.npz")
else:
    bank_manifest_path = args_cli.prototype_bank_root / "prepared/physx_batch_manifest.json"
    os.environ["DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"] = str(bank_manifest_path)
sys.argv = [sys.argv[0]] + hydra_args
# Keep the isolated coordinator and non-recording generations non-rendering.
# A recording worker enables offscreen cameras, while remaining --headless.
if args_cli.video:
    if args_cli.physics_worker_manifest is not None:
        recording_worker = bool(
            json.loads(args_cli.physics_worker_manifest.read_text()).get("training_video")
        )
        args_cli.enable_cameras = recording_worker or args_cli.enable_cameras
    elif not args_cli.isolate_physics_per_generation:
        args_cli.enable_cameras = True
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
from wuji_ppo_logging import CompletedEpisodeWindow, enable_ppo_logging, track_completed_episodes  # noqa: E402
from gpu_wuji_retarget import (  # noqa: E402
    WujiBatchKinematics,
    joint_names_from_seed,
    quat_xyzw_matrix,
)
from wuji_general_space import (  # noqa: E402
    LOWER_BOUNDS,
    SOURCE_VECTOR,
    UPPER_BOUNDS,
    VECTOR_NAMES,
    graph_from_search_vector,
    resolve_design_vectors,
)
from wuji_parametric_usd import (  # noqa: E402
    attach_manifest,
    build_hand_super_environment,
)
from skrl_sac_wuji_morphology import (  # noqa: E402
    ProposalBatch,
    SkrlConditionalMorphologySAC,
)
from wuji_rollout_video import (  # noqa: E402
    StreamedRolloutVideo,
    close_rgb_render_product,
    generation_video_spec,
    video_wandb_metrics,
)


REFERENCE_PHASE_COUNT = 445


def log_generation_to_wandb(
    generation: int,
    rows: list[dict],
    optimizer_status: dict | None,
    cumulative_environment_steps: int,
    video_records: list[dict] | None = None,
) -> None:
    """Log population summaries and one queryable candidate table to W&B."""
    if not args_cli.wandb:
        return
    try:
        import wandb

        rewards = np.asarray(
            [row["total_reward"] for row in rows], dtype=np.float64
        )
        phases = np.asarray([row["phase"] for row in rows], dtype=np.float64)
        success = np.asarray(
            [row["success"] for row in rows], dtype=np.float64
        )
        best_row = max(rows, key=lambda row: row["total_reward"])
        table = wandb.Table(
            columns=[
                "generation",
                "candidate_index",
                "candidate_id",
                "total_reward",
                "reward_std",
                "reward_min",
                "reward_median",
                "reward_max",
                "pose_reward",
                "contact_reward",
                "phase",
                "phase_std",
                "phase_min",
                "phase_mean",
                "phase_median",
                "phase_max",
                "environment_steps",
                "survival",
                "survival_ratio",
                "success",
                "success_count",
                "success_rate",
                "rollouts",
                "position_error_m",
                "orientation_error_rad",
                "sample_source",
                "palm_prototype_index",
                "requested_palm_expansion",
                "design_vector",
            ]
        )
        for row in sorted(rows, key=lambda item: item["candidate_index"]):
            phase = int(row["phase"])
            table.add_data(
                generation,
                int(row["candidate_index"]),
                row["candidate_id"],
                float(row["total_reward"]),
                float(row.get("reward_std", 0.0)),
                float(row.get("reward_min", row["total_reward"])),
                float(row.get("reward_median", row["total_reward"])),
                float(row.get("reward_max", row["total_reward"])),
                float(row["pose_reward"]),
                float(row["contact_reward"]),
                phase,
                float(row.get("phase_std", 0.0)),
                int(row.get("phase_min", phase)),
                float(row.get("phase_mean", phase)),
                float(row.get("phase_median", phase)),
                int(row.get("phase_max", phase)),
                int(row.get("environment_steps", max(phase, 0))),
                f"{phase}/{REFERENCE_PHASE_COUNT}",
                phase / REFERENCE_PHASE_COUNT,
                bool(row["success"]),
                int(row.get("success_count", int(bool(row["success"])))),
                float(row.get("success_rate", float(bool(row["success"])))),
                int(row.get("rollouts", 1)),
                float(row["position_error_m"]),
                float(row["orientation_error_rad"]),
                row.get("sample_source", "unknown"),
                row.get("palm_prototype_index"),
                row.get("requested_palm_expansion"),
                row.get("semantic_vector", row.get("vector")),
            )
        metrics: dict[str, object] = {
            "Progress / Environment steps": cumulative_environment_steps,
            "Reward / Instantaneous reward (max)": float(rewards.max()),
            "Reward / Instantaneous reward (mean)": float(rewards.mean()),
            "Reward / Instantaneous reward (min)": float(rewards.min()),
            "Survival / Phase (max)": float(phases.max()),
            "Survival / Phase (mean)": float(phases.mean()),
            "Survival / Phase (min)": float(phases.min()),
            "Survival / Ratio (max)": float(
                phases.max() / REFERENCE_PHASE_COUNT
            ),
            "Survival / Ratio (mean)": float(
                phases.mean() / REFERENCE_PHASE_COUNT
            ),
            "Survival / Ratio (min)": float(
                phases.min() / REFERENCE_PHASE_COUNT
            ),
            "Survival / Success ratio": float(success.mean()),
            "Robustness / Rollouts per proposal": int(
                rows[0].get("rollouts", 1)
            ),
            "Robustness / Mean success rate": float(
                np.mean(
                    [
                        row.get("success_rate", float(bool(row["success"])))
                        for row in rows
                    ]
                )
            ),
            "Robustness / Mean reward std": float(
                np.mean([row.get("reward_std", 0.0) for row in rows])
            ),
            "Robustness / Best candidate reward std": float(
                best_row.get("reward_std", 0.0)
            ),
            "Robustness / Best candidate success rate": float(
                best_row.get(
                    "success_rate", float(bool(best_row["success"]))
                )
            ),
            "Robustness / Best candidate rollout success ratio": float(
                best_row.get(
                    "success_rate", float(bool(best_row["success"]))
                )
            ),
            "Robustness / Best candidate successful rollouts": int(
                best_row.get("success_count", int(bool(best_row["success"])))
            ),
            "Robustness / Best candidate phase min": float(
                best_row.get("phase_min", best_row["phase"])
            ),
            "Robustness / Best candidate phase mean": float(
                best_row.get("phase_mean", best_row["phase"])
            ),
            "Robustness / Best candidate phase median": float(
                best_row.get("phase_median", best_row["phase"])
            ),
            "Robustness / Best candidate phase max": float(
                best_row.get("phase_max", best_row["phase"])
            ),
            "Robustness / Best candidate survival ratio mean": float(
                best_row.get("phase_mean", best_row["phase"])
                / REFERENCE_PHASE_COUNT
            ),
            "Robustness / Best candidate survival ratio median": float(
                best_row.get("phase_median", best_row["phase"])
                / REFERENCE_PHASE_COUNT
            ),
            "Survival / Max (out of 445)": (
                f"{int(phases.max())}/{REFERENCE_PHASE_COUNT}"
            ),
            "Survival / Mean (out of 445)": (
                f"{phases.mean():.2f}/{REFERENCE_PHASE_COUNT}"
            ),
            "Survival / Min (out of 445)": (
                f"{int(phases.min())}/{REFERENCE_PHASE_COUNT}"
            ),
            "Morphology / Candidate performance": table,
            "Environment steps / Reward (max)": float(rewards.max()),
            "Environment steps / Reward (mean)": float(rewards.mean()),
            "Environment steps / Reward (min)": float(rewards.min()),
            "Environment steps / Survival ratio (max)": float(
                phases.max() / REFERENCE_PHASE_COUNT
            ),
            "Environment steps / Survival ratio (mean)": float(
                phases.mean() / REFERENCE_PHASE_COUNT
            ),
            "Environment steps / Survival ratio (min)": float(
                phases.min() / REFERENCE_PHASE_COUNT
            ),
        }
        if optimizer_status is not None:
            metrics["Morphology / Replay size"] = int(
                optimizer_status.get("replay_size", 0)
            )
            if "elite_archive_best_reward" in optimizer_status:
                metrics["Morphology / Elite archive best reward"] = float(
                    optimizer_status["elite_archive_best_reward"]
                )
        if generation == 0:
            wandb.define_metric("Progress / Environment steps")
            wandb.define_metric(
                "Environment steps/*",
                step_metric="Progress / Environment steps",
            )
        try:
            metrics.update(video_wandb_metrics(video_records or [], wandb))
        except Exception as exc:
            print(f"WARNING: failed to upload rollout video: {exc}", flush=True)
        if args_cli.shared_ppo_iterations:
            if generation == 0:
                wandb.define_metric("Morphology/generation")
                for name in metrics:
                    if not name.startswith("Environment steps/"):
                        wandb.define_metric(name, step_metric="Morphology/generation")
                wandb.define_metric("Evaluation/*", step_metric="Morphology/generation")
            metrics["Morphology/generation"] = generation
            metrics["Evaluation/episode_return_mean"] = float(rewards.mean())
            metrics["Evaluation/episode_return_max"] = float(rewards.max())
            episode_count = sum(row["replicas"] for row in rows)
            metrics["Evaluation/episode_steps_mean"] = (
                sum(row["environment_steps"] for row in rows) / episode_count
            )
            metrics["Evaluation/episode_steps_min"] = min(row["episode_steps_min"] for row in rows)
            metrics["Evaluation/episode_steps_max"] = max(row["episode_steps_max"] for row in rows)
            metrics["Evaluation/success_rate"] = sum(row["success_count"] for row in rows) / episode_count
            metrics["Evaluation/start_phase"] = 0
            for row in rows:
                prefix = f"Evaluation/hand_{row['candidate_index']}"
                metrics[f"{prefix}/episode_steps_mean"] = row["environment_steps"] / row["replicas"]
                metrics[f"{prefix}/episode_return_mean"] = row["total_reward"]
                metrics[f"{prefix}/success_rate"] = row["success_count"] / row["replicas"]
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=generation)
    except Exception as exc:
        print(
            f"WARNING: failed to log generation {generation} to W&B: {exc}",
            flush=True,
        )


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def process_rss_gib() -> float | None:
    """Return this process's current resident memory on Linux."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0**2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def close_physics_batch(env) -> dict[str, object]:
    """Close one batch's environment and USD stage."""

    rss_before = process_rss_gib()
    close_error = None
    stage_close_error = None
    stage_closed = False
    try:
        env.close()
    except Exception as exc:  # preserve cleanup attempts after a partial close
        close_error = repr(exc)

    # Kit and PhysX perform part of stage destruction asynchronously. Pumping
    # the app before and after close_stage lets those deferred releases run.
    try:
        for _ in range(2):
            simulation_app.update()
        stage_closed = bool(sim_utils.close_stage())
        for _ in range(2):
            simulation_app.update()
    except Exception as exc:  # do not hide the original rollout exception
        stage_close_error = repr(exc)

    result: dict[str, object] = {
        "rss_before_gib": rss_before,
        "stage_closed": stage_closed,
    }
    if close_error is not None:
        result["environment_close_error"] = close_error
    if stage_close_error is not None:
        result["stage_close_error"] = stage_close_error
    return result


def release_process_memory() -> dict[str, object]:
    """Release caches after the caller has dropped its environment reference."""

    collected = gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Native Kit/PhysX allocations use the process heap rather than Python's
    # allocator. Return any now-free glibc arenas to Linux when supported.
    native_heap_trimmed = False
    if sys.platform.startswith("linux"):
        try:
            native_heap_trimmed = bool(
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            )
        except (AttributeError, OSError):
            pass

    return {
        "rss_after_gib": process_rss_gib(),
        "python_objects_collected": collected,
        "native_heap_trimmed": native_heap_trimmed,
    }


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


def retarget_mano(vectors_path, output, kinematics, task_targets, iterations):
    """Retarget each candidate from neutral to the captured MANO sequence."""
    from wuji_sequential_retarget import solve_mano_sequence

    start = time.perf_counter()
    vectors = np.load(vectors_path).astype(np.float32)
    solved, benchmark = solve_mano_sequence(
        kinematics, resolve_design_vectors(vectors).astype(np.float32), task_targets, iterations
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, vectors=vectors, joint_names=np.asarray(kinematics.joint_names),
        frame_ids=task_targets["frame_ids"], **solved,
        source_mano_rollout=task_targets["source_mano_rollout"],
        source_mano_rollout_sha256=task_targets["source_mano_rollout_sha256"],
        metadata_json=np.asarray(json.dumps({"benchmark": benchmark})),
    )
    return {"seconds": time.perf_counter()-start, "candidates": len(vectors),
            "solver_seconds": benchmark["seconds"]}


def prepare_assets(
    vectors_path: Path,
    retarget_path: Path | None,
    generation_root: Path,
    bank_manifest: dict,
    fixed_reference: Path | None = None,
    task_targets_path: Path | None = None,
) -> tuple[dict, dict[str, float]]:
    timings: dict[str, float] = {}
    vectors = np.load(vectors_path)
    graphs = {
        "schema_version": 1,
        "hands": [
            graph_from_search_vector(
                vector.astype(np.float64), f"wuji_physx_{i:06d}"
            )
            for i, vector in enumerate(vectors)
        ],
    }
    graphs_path = generation_root / "graphs.json"
    graphs_path.write_text(json.dumps(graphs, indent=2) + "\n")
    ir_root = generation_root / "hand_ir"
    generator_env = os.environ.copy()
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
        str(SCRIPT_ROOT / "prepare_wuji_parametric_training_assets.py"),
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
    if task_targets_path is not None:
        prepare_command.extend(("--task-targets", str(task_targets_path)))
    if fixed_reference is not None:
        prepare_command.extend(("--fixed-reference", str(fixed_reference)))
    elif retarget_path is None:
        raise ValueError("retarget_path is required without fixed_reference")
    grouped_replication = (
        args_cli.morphology_replicas > 1
        and (
            args_cli.shared_ppo_iterations > 0
            or args_cli.grouped_zero_action_vectors is not None
        )
    )
    if grouped_replication:
        prepare_command.append("--materialize-candidate-assets")
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
    if grouped_replication:
        super_usd, local_origins = build_hand_super_environment(
            manifest,
            prepared / "super_environment" / "hands.usd",
            spacing=0.65,
        )
        manifest["hand_super_environment_usd"] = str(super_usd)
        manifest["hand_super_environment_origins"] = local_origins.tolist()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, timings


def configure_batch(cfg, manifest: dict) -> None:
    if manifest.get("original_source_hand", False):
        # Keep the standalone WUJI topology and ordinary physics cloning.
        env_module.MORPHOLOGY_BATCH_MANIFEST = None
        env_module.REFERENCE_PATH = Path(manifest["reference_paths"][0])
        cfg.hand_cfg.spawn.usd_path = manifest["hand_usd_paths"][0]
        cfg.geometry_urdf_path = manifest["hand_urdf_paths"][0]
        cfg.scene.num_envs = len(manifest["vectors"])
        cfg.scene.replicate_physics = True
        cfg.scene.clone_in_fabric = False
        cfg.randomize_start_phase = False
        cfg.episode_length_s = 15.0
        cfg.observation_mode = "palm_geometry"
        cfg.observation_space = env_module.PALM_GEOMETRY_OBSERVATION_DIM
        cfg.morphology_context_dim = 0
        return
    usd_paths = [Path(value).resolve() for value in manifest["hand_usd_paths"]]
    reference_paths = [
        Path(value).resolve() for value in manifest["reference_paths"]
    ]
    env_module.MORPHOLOGY_BATCH_MANIFEST = manifest
    env_module._batch_usd_paths = usd_paths
    env_module._batch_reference_paths = reference_paths
    env_module.REFERENCE_PATH = reference_paths[0]
    cfg.hand_cfg.spawn.usd_path = [str(path) for path in usd_paths]
    if manifest.get("grouped_physics_replication", False):
        cfg.hand_cfg.prim_path = (
            "/World/envs/env_.*/SuperEnvironment/morph_.*/Hand"
        )
        cfg.object_cfg.prim_path = (
            "/World/envs/env_.*/SuperEnvironment/morph_.*/Object"
        )
        cfg.scene.num_envs = int(manifest["morphology_replicas"])
        origins = np.asarray(
            manifest["hand_super_environment_origins"][
                : int(manifest["unique_morphology_count"])
            ],
            dtype=np.float32,
        )
        extent = origins.max(axis=0) - origins.min(axis=0)
        cfg.scene.env_spacing = float(max(extent[:2]) + 1.3)
        cfg.scene.replicate_physics = True
    else:
        cfg.scene.num_envs = len(usd_paths)
        cfg.scene.replicate_physics = False
    cfg.scene.clone_in_fabric = False
    cfg.randomize_start_phase = False
    cfg.episode_length_s = 15.0
    context = manifest.get("policy_morphology_context")
    cfg.morphology_context_dim = 0 if context is None else len(context[0])
    if cfg.observation_mode == "palm_geometry":
        if context is not None:
            raise ValueError(
                "palm_geometry observations cannot append a design-vector context"
            )
        cfg.observation_space = env_module.PALM_GEOMETRY_OBSERVATION_DIM
    else:
        cfg.observation_space = env_module.OBSERVATION_DIM + cfg.morphology_context_dim


def normalized_morphology_context(vectors: np.ndarray) -> np.ndarray:
    """Preserve ordered design semantics while scaling every value to [-1, 1]."""

    values = np.asarray(vectors, dtype=np.float32)
    span = (UPPER_BOUNDS - LOWER_BOUNDS).astype(np.float32)
    return (2.0 * (values - LOWER_BOUNDS) / span - 1.0).astype(np.float32)


def evaluate_batch(
    env, manifest: dict, global_offset: int, video: StreamedRolloutVideo | None = None,
) -> tuple[list[dict], float]:
    raw = env.unwrapped
    env.reset()
    if video is not None:
        video.warm_up(env)
    count = len(manifest["vectors"])
    actions = torch.zeros((count, raw.action_dim), device=raw.device)
    active = torch.ones(count, dtype=torch.bool, device=raw.device)
    environment_steps = torch.zeros(
        count, dtype=torch.long, device=raw.device
    )
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
        for step in range(raw._reference_length + 2):
            # Capture the state before stepping: env.step auto-resets finished
            # replicas, so post-step RGB could otherwise show a second episode.
            if video is not None and active[video.env_index].item():
                video.capture(raw.render, step)
            environment_steps[active] += 1
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
        candidate_index = (
            int(manifest["morphology_indices"][i])
            if "morphology_indices" in manifest
            else global_offset + i
        )
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": manifest["candidate_ids"][i],
                "replica_index": (
                    int(manifest["replica_indices"][i])
                    if "replica_indices" in manifest
                    else 0
                ),
                "vector": manifest["vectors"][i],
                "total_reward": float(total[i].item()),
                "pose_reward": float(pose[i].item()),
                "contact_reward": float(contact[i].item()),
                "pinch_contact_steps": int(pinch[i].item()),
                "phase": int(phase[i].item()),
                "environment_steps": int(environment_steps[i].item()),
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


def repeat_manifest_for_rollouts(
    manifest: dict,
    replicas: int,
    global_morphology_indices: list[int],
) -> dict:
    """Expand a normal heterogeneous batch with independent rollout replicas."""

    morphology_count = len(manifest["vectors"])
    if morphology_count != len(global_morphology_indices):
        raise ValueError("global morphology index count does not match manifest")
    result: dict = {}
    # Candidate-major ordering keeps all replicas of one proposal adjacent.
    for key, value in manifest.items():
        if isinstance(value, list) and len(value) == morphology_count:
            result[key] = [item for item in value for _ in range(replicas)]
        else:
            result[key] = value
    result["candidate_ids"] = [
        f"wuji_physx_{global_index:06d}_rollout_{replica:03d}"
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
    result["rollouts_per_proposal"] = replicas
    result["unique_morphology_count"] = morphology_count
    result["grouped_physics_replication"] = False
    return result


def aggregate_rollout_rows(
    raw_rows: list[dict],
    original_manifest: dict,
    global_morphology_indices: list[int],
) -> list[dict]:
    """Aggregate replicated rollout outcomes into one SAC reward per proposal."""

    grouped: dict[int, list[dict]] = {
        index: [] for index in global_morphology_indices
    }
    for row in raw_rows:
        grouped[int(row["candidate_index"])].append(row)

    rows: list[dict] = []
    for local_index, candidate_index in enumerate(global_morphology_indices):
        replicas = grouped[candidate_index]
        if not replicas:
            raise RuntimeError(f"proposal {candidate_index} has no rollout results")

        def values(key: str, dtype=np.float64) -> np.ndarray:
            return np.asarray([row[key] for row in replicas], dtype=dtype)

        rewards = values("total_reward")
        poses = values("pose_reward")
        contacts = values("contact_reward")
        phases = values("phase")
        environment_steps = values("environment_steps", dtype=np.int64)
        successes = values("success", dtype=bool)
        positions = values("position_error_m")
        orientations = values("orientation_error_rad")
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": original_manifest["candidate_ids"][local_index],
                "vector": original_manifest["vectors"][local_index],
                # SAC observes the Monte Carlo mean, not one lucky rollout.
                "total_reward": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_min": float(rewards.min()),
                "reward_median": float(np.median(rewards)),
                "reward_max": float(rewards.max()),
                "pose_reward": float(poses.mean()),
                "pose_reward_std": float(poses.std()),
                "contact_reward": float(contacts.mean()),
                "contact_reward_std": float(contacts.std()),
                "pinch_contact_steps": float(
                    values("pinch_contact_steps").mean()
                ),
                "phase": int(round(float(phases.mean()))),
                "phase_mean": float(phases.mean()),
                "phase_median": float(np.median(phases)),
                "phase_std": float(phases.std()),
                "phase_min": int(phases.min()),
                "phase_max": int(phases.max()),
                "survival_ratio_mean": float(
                    phases.mean() / REFERENCE_PHASE_COUNT
                ),
                "survival_ratio_median": float(
                    np.median(phases) / REFERENCE_PHASE_COUNT
                ),
                # A proposal is robustly successful only when every rollout succeeds.
                "success": bool(successes.all()),
                "success_count": int(successes.sum()),
                "success_rate": float(successes.mean()),
                "rollouts": len(replicas),
                "environment_steps": int(environment_steps.sum()),
                "environment_steps_mean": float(environment_steps.mean()),
                "position_error_m": float(positions.mean()),
                "position_error_std_m": float(positions.std()),
                "orientation_error_rad": float(orientations.mean()),
                "orientation_error_std_rad": float(orientations.std()),
                "max_thumb_contact_force_n": float(
                    values("max_thumb_contact_force_n").max()
                ),
                "max_other_finger_contact_force_n": float(
                    values("max_other_finger_contact_force_n").max()
                ),
            }
        )
    return rows


def evaluate_manifest_in_process(env_cfg, manifest: dict) -> dict[str, object]:
    """Evaluate one generation's manifest in the current Isaac Sim process."""

    rows: list[dict] = []
    rollout_rows: list[dict] = []
    initialization_seconds = 0.0
    rollout_seconds = 0.0
    batch_records: list[dict] = []
    video_records: list[dict] = []
    video_spec = manifest.get("training_video")
    stage_created = False
    population = len(manifest["vectors"])
    for begin in range(0, population, args_cli.physics_batch_size):
        end = min(begin + args_cli.physics_batch_size, population)
        # The preceding batch closes its stage explicitly. On the first batch
        # of a later generation there is therefore no current stage even
        # though this helper's local stage_created flag starts as False.
        if stage_created or sim_utils.get_current_stage() is None:
            sim_utils.create_new_stage()
        stage_created = True
        batch = slice_manifest(manifest, begin, end)
        morphology_indices = list(range(begin, end))
        rollout_batch = repeat_manifest_for_rollouts(
            batch,
            args_cli.rollouts_per_proposal,
            morphology_indices,
        )
        cfg = copy.deepcopy(env_cfg)
        configure_batch(cfg, rollout_batch)
        record_this_batch = video_spec is not None and begin <= video_spec["candidate_index"] < end
        video_env_index = 0
        if record_this_batch:
            video_env_index = (video_spec["candidate_index"] - begin) * args_cli.rollouts_per_proposal
            cfg.viewer.env_index = video_env_index
            cfg.viewer.resolution = (video_spec["width"], video_spec["height"])
        start = time.perf_counter()
        env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if record_this_batch else None)
        initialization = time.perf_counter() - start
        video = None
        try:
            if record_this_batch:
                video = StreamedRolloutVideo(video_spec, env.unwrapped.step_dt, video_env_index)
            batch_rollout_rows, rollout_time = evaluate_batch(
                env, rollout_batch, begin, video
            )
        finally:
            if video is not None:
                try:
                    video_record = video.finish()
                except Exception as exc:
                    video_record = {**video_spec, "path": None, "error": repr(exc)}
                video_record["render_cleanup_errors"] = close_rgb_render_product(env)
                video_records.append(video_record)
                print("WUJI_SAC_ROLLOUT_VIDEO " + json.dumps(video_record), flush=True)
            cleanup_status = close_physics_batch(env)
            del env
            del cfg
            del rollout_batch
            cleanup_status.update(release_process_memory())
        print(
            "WUJI_SAC_PHYSX_CLEANUP "
            + json.dumps(cleanup_status, sort_keys=True),
            flush=True,
        )
        batch_rows = aggregate_rollout_rows(
            batch_rollout_rows,
            batch,
            morphology_indices,
        )
        rows.extend(batch_rows)
        rollout_rows.extend(batch_rollout_rows)
        initialization_seconds += initialization
        rollout_seconds += rollout_time
        batch_record = {
            "begin": begin,
            "end": end,
            "proposals": end - begin,
            "rollouts_per_proposal": args_cli.rollouts_per_proposal,
            "physical_environments": len(batch_rollout_rows),
            "initialization_seconds": initialization,
            "rollout_seconds": rollout_time,
            "cleanup": cleanup_status,
            "best_phase": max(row["phase"] for row in batch_rows),
            "best_reward": max(row["total_reward"] for row in batch_rows),
        }
        batch_records.append(batch_record)
        print("WUJI_SAC_PHYSX_BATCH " + json.dumps(batch_record), flush=True)
    return {
        "rows": rows,
        "rollout_rows": rollout_rows,
        "initialization_seconds": initialization_seconds,
        "rollout_seconds": rollout_seconds,
        "batch_records": batch_records,
        "video_records": video_records,
    }


def run_physics_worker(env_cfg) -> None:
    """Run one generation in a disposable process and serialize its results."""

    if args_cli.physics_worker_manifest is None or args_cli.physics_worker_output is None:
        raise ValueError("physics worker requires both hidden path arguments")
    manifest = json.loads(args_cli.physics_worker_manifest.read_text())
    result = evaluate_manifest_in_process(env_cfg, manifest)
    args_cli.physics_worker_output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.physics_worker_output.write_text(json.dumps(result) + "\n")
    print(
        "WUJI_SAC_PHYSX_WORKER_COMPLETE "
        f"pid={os.getpid()} rss_gib={process_rss_gib()}",
        flush=True,
    )


def evaluate_manifest_isolated(env_cfg, manifest_path: Path, output_path: Path) -> dict:
    """Evaluate a generation in a child process that exits after PhysX use."""

    del env_cfg  # configuration is parsed independently inside the worker
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *original_cli_argv,
        "--physics-worker-manifest",
        str(manifest_path),
        "--physics-worker-output",
        str(output_path),
    ]
    print(
        "WUJI_SAC_PHYSX_WORKER_START "
        f"coordinator_pid={os.getpid()} manifest={manifest_path}",
        flush=True,
    )
    run(command)
    result = json.loads(output_path.read_text())
    print(
        "WUJI_SAC_PHYSX_WORKER_REAPED "
        f"coordinator_pid={os.getpid()} rss_gib={process_rss_gib()}",
        flush=True,
    )
    return result


def replicate_manifest(
    manifest: dict,
    replicas: int,
    global_morphology_indices: list[int],
) -> dict:
    """Repeat every morphology together with its own frozen reference."""

    morphology_count = len(manifest["vectors"])
    if morphology_count != len(global_morphology_indices):
        raise ValueError("global morphology index count does not match manifest")
    result: dict = {}
    # Replica-major ordering matches one heterogeneous super-environment
    # containing every morphology. The complete super-environment is then
    # cloned, with one official PhysX replication registration, N times.
    for key, value in manifest.items():
        if isinstance(value, list) and len(value) == morphology_count:
            result[key] = [item for _ in range(replicas) for item in value]
        else:
            result[key] = value
    result["candidate_ids"] = [
        f"wuji_physx_{global_index:06d}_replica_{replica:03d}"
        for replica in range(replicas)
        for global_index in global_morphology_indices
    ]
    result["morphology_indices"] = [
        global_index
        for _ in range(replicas)
        for global_index in global_morphology_indices
    ]
    result["replica_indices"] = [
        replica
        for replica in range(replicas)
        for _ in global_morphology_indices
    ]
    result["morphology_replicas"] = replicas
    result["unique_morphology_count"] = morphology_count
    result["grouped_physics_replication"] = replicas > 1 and not manifest.get("original_source_hand", False)
    result["fixed_reference_shared_across_morphologies"] = bool(
        manifest.get("fixed_reference")
    )
    return result


def distributed_barrier() -> None:
    if skrl.config.torch.is_distributed:
        torch.distributed.barrier()


def grouped_zero_action_debug(env_cfg, output: Path, bank_manifest: dict) -> None:
    """Evaluate known morphologies in the replicated super-env with zero action."""

    if args_cli.fixed_reference is None:
        raise ValueError("grouped zero-action debug requires --fixed-reference")
    if args_cli.morphology_replicas <= 1:
        raise ValueError("grouped zero-action debug requires replicas > 1")
    assert args_cli.grouped_zero_action_vectors is not None
    if not args_cli.grouped_zero_action_vectors.is_file():
        raise FileNotFoundError(
            f"debug vectors not found: {args_cli.grouped_zero_action_vectors}"
        )
    vectors = np.load(args_cli.grouped_zero_action_vectors).astype(np.float32)
    if len(vectors) != args_cli.population:
        raise ValueError(
            f"debug vector count {len(vectors)} != population {args_cli.population}"
        )
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / "vectors.npy"
    np.save(vectors_path, vectors)
    manifest, prepare_timings = prepare_assets(
        vectors_path,
        None,
        output,
        bank_manifest,
        fixed_reference=args_cli.fixed_reference,
    )
    expanded = replicate_manifest(
        manifest,
        args_cli.morphology_replicas,
        list(range(args_cli.population)),
    )
    expanded_path = output / "prepared/grouped_zero_action_manifest.json"
    expanded_path.write_text(json.dumps(expanded, indent=2) + "\n")
    cfg = copy.deepcopy(env_cfg)
    configure_batch(cfg, expanded)
    start = time.perf_counter()
    env = gym.make(args_cli.task, cfg=cfg)
    initialization_seconds = time.perf_counter() - start
    raw_rows, rollout_seconds = evaluate_batch(env, expanded, 0)
    env.close()

    morphology_indices = np.asarray(expanded["morphology_indices"], dtype=np.int64)
    rows: list[dict] = []
    for morphology_index in range(args_cli.population):
        ids = np.flatnonzero(morphology_indices == morphology_index)
        replicas = [raw_rows[int(index)] for index in ids]
        rewards = np.asarray(
            [row["total_reward"] for row in replicas], dtype=np.float64
        )
        phases = np.asarray([row["phase"] for row in replicas], dtype=np.int64)
        successes = np.asarray([row["success"] for row in replicas], dtype=bool)
        rows.append(
            {
                "candidate_index": morphology_index,
                "vector": vectors[morphology_index].tolist(),
                "total_reward": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "phase": int(phases.max()),
                "mean_phase": float(phases.mean()),
                "success": bool(successes.any()),
                "success_count": int(successes.sum()),
                "replicas": int(len(ids)),
            }
        )
    rows.sort(key=lambda row: row["total_reward"], reverse=True)
    farthest = max(rows, key=lambda row: row["phase"])
    summary = {
        "schema_version": 1,
        "algorithm": "grouped_superenv_zero_action_debug",
        "ppo_enabled": False,
        "sac_updates_performed": 0,
        "retarget_performed": False,
        "fixed_reference": str(args_cli.fixed_reference),
        "source_vectors": str(args_cli.grouped_zero_action_vectors),
        "population": args_cli.population,
        "morphology_replicas": args_cli.morphology_replicas,
        "global_envs": args_cli.population * args_cli.morphology_replicas,
        "best": rows[0],
        "farthest": farthest,
        "successful_morphologies": sum(row["success"] for row in rows),
        "successful_replicas": sum(row["success_count"] for row in rows),
        "prepare_timings": prepare_timings,
        "initialization_seconds": initialization_seconds,
        "rollout_seconds": rollout_seconds,
        "results": rows,
    }
    (output / "grouped_zero_action_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        "WUJI_GROUPED_ZERO_ACTION_COMPLETE "
        f"best_reward={rows[0]['total_reward']:.9f} "
        f"best_phase={rows[0]['phase']}/445 "
        f"max_phase={farthest['phase']}/445 "
        f"successful_morphologies={summary['successful_morphologies']} "
        f"successful_replicas={summary['successful_replicas']}",
        flush=True,
    )


def evaluate_shared_policy(
    env,
    raw_env,
    runner: Runner,
    manifest: dict,
    eval_envs_per_morphology: int | None = None,
) -> tuple[list[dict], float]:
    """Score the first K replicas per morphology; the full scene still steps."""

    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)
    # SKRL's IsaacLab wrapper caches reset() after the first call. Re-arm it
    # so evaluation starts a new episode and refreshes wrapped observations.
    if hasattr(env, "_reset_once"):
        env._reset_once = True
    with torch.inference_mode():
        observations, _ = env.reset()
    if hasattr(raw_env, "phase_buf") and torch.any(raw_env.phase_buf != 0):
        raise RuntimeError("Shared PPO evaluation must start at reference phase zero")
    agent_act_requires_states = "states" in inspect.signature(
        runner.agent.act
    ).parameters
    count = raw_env.num_envs
    if len(manifest["morphology_indices"]) != count:
        raise ValueError("evaluation morphology mapping does not match environment count")
    groups: dict[int, list[int]] = {}
    for env_index, morphology_index in enumerate(manifest["morphology_indices"]):
        groups.setdefault(morphology_index, []).append(env_index)
    selected = torch.zeros(count, dtype=torch.bool, device=raw_env.device)
    for ids in groups.values():
        requested = len(ids) if eval_envs_per_morphology is None else eval_envs_per_morphology
        if not 1 <= requested <= len(ids):
            raise ValueError("evaluation count must be positive and not exceed replicas per morphology")
        selected[ids[:requested]] = True
    active = selected.clone()
    environment_steps = torch.zeros(
        count, dtype=torch.long, device=raw_env.device
    )
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
            environment_steps[active] += 1
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
        ids = ((morphology_indices == morphology_index) & selected).nonzero(
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
                "environment_steps": int(environment_steps[ids].sum().item()),
                "episode_steps_min": int(environment_steps[ids].min().item()),
                "episode_steps_max": int(environment_steps[ids].max().item()),
                "episode_steps_mean": float(environment_steps[ids].float().mean().item()),
                "evaluation_start_phase": 0,
                "evaluation_action_mode": "deterministic",
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
    episode_window: CompletedEpisodeWindow | None = None,
) -> tuple[list[dict], dict[str, float | int | str]]:
    """Train the official shared SKRL PPO, then evaluate it deterministically."""

    cfg = copy.deepcopy(env_cfg)
    configure_batch(cfg, manifest)
    cfg.sim.device = f"cuda:{local_rank}"
    cfg.seed = args_cli.seed + rank
    ppo_cfg = copy.deepcopy(agent_cfg)
    ppo_cfg["seed"] = args_cli.seed
    base_rollouts = int(ppo_cfg["agent"]["rollouts"])
    rollouts = base_rollouts * args_cli.ppo_rollout_multiplier
    ppo_cfg["agent"]["rollouts"] = rollouts
    ppo_cfg["trainer"]["timesteps"] = (
        args_cli.shared_ppo_iterations * rollouts
    )
    ppo_cfg["trainer"]["close_environment_at_exit"] = False
    experiment = ppo_cfg["agent"]["experiment"]
    experiment["directory"] = str(output / "ppo_logs" / f"rank_{rank:02d}")
    experiment["experiment_name"] = (
        f"{args_cli.wandb_run_name}_ppo_generation_{generation:03d}"
    )
    experiment["checkpoint_interval"] = 0
    # SAC initializes the persistent outer run; PPO must not initialize another.
    experiment["wandb"] = False
    if args_cli.wandb:
        experiment["write_interval"] = rollouts

    start = time.perf_counter()
    raw_gym_env = gym.make(args_cli.task, cfg=cfg)
    initialization_seconds = time.perf_counter() - start
    raw_env = raw_gym_env.unwrapped
    env = SkrlVecEnvWrapper(raw_gym_env, ml_framework="torch")
    runner = Runner(env, ppo_cfg)
    if episode_window is not None:
        track_completed_episodes(runner.agent, episode_window)
    if args_cli.wandb and rank == 0:
        import wandb
        enable_ppo_logging(
            runner.agent, wandb.run, generation=generation,
            steps_per_generation=ppo_cfg["trainer"]["timesteps"],
            global_envs=args_cli.population * args_cli.morphology_replicas,
        )
    if checkpoint_to_load is not None:
        runner.agent.load(str(checkpoint_to_load))
    start = time.perf_counter()
    runner.run()
    training_seconds = time.perf_counter() - start
    if episode_window is not None:
        episode_window.finish_generation()

    checkpoint = output / "ppo_checkpoints" / f"generation_{generation:03d}.pt"
    if rank == 0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        runner.agent.save(str(checkpoint))
        runner.agent.save(str(output / "shared_ppo_latest.pt"))
    distributed_barrier()
    rows, evaluation_seconds = evaluate_shared_policy(
        env, raw_env, runner, manifest, args_cli.eval_envs_per_morphology
    )
    env.close()
    return rows, {
        "local_envs": len(manifest["vectors"]),
        "ppo_iterations": args_cli.shared_ppo_iterations,
        "base_rollout_steps": base_rollouts,
        "ppo_episode_log_window_steps": args_cli.ppo_episode_log_window_steps,
        "ppo_rollout_multiplier": args_cli.ppo_rollout_multiplier,
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
    geometry_observations = args_cli.ppo_observation_mode == "palm_geometry"
    task_targets = None
    task_targets_path = None
    if geometry_observations:
        env_cfg.observation_mode = "palm_geometry"
        env_cfg.observation_space = env_module.PALM_GEOMETRY_OBSERVATION_DIM
        env_cfg.morphology_context_dim = 0
        env_cfg.geometry_inward_direction_mode = (
            args_cli.geometry_inward_direction_mode
        )
        from gpu_wuji_retarget import SOURCE_URDF, parse_urdf
        retarget_joint_names = list(parse_urdf(SOURCE_URDF)[1])
        if args_cli.seed_trajectory is not None:
            print("[MANO_RETARGET] --seed-trajectory is ignored; using neutral candidate initialization", flush=True)
        from wuji_mano_task_targets import build_mano_task_targets
        task_targets_path = output / "mano_task_targets.npz"
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            task_targets = build_mano_task_targets(args_cli.mano_rollout)
            np.savez_compressed(task_targets_path, **task_targets)
        distributed_barrier()
        with np.load(task_targets_path, allow_pickle=False) as data:
            task_targets = {key: data[key] for key in data.files}
        retarget_kinematics = WujiBatchKinematics(
            retarget_joint_names, torch.device(f"cuda:{local_rank}")
        )
        fixed_reference = None
    else:
        retarget_kinematics = None
        fixed_reference = args_cli.fixed_reference or Path(
            bank_manifest["reference_paths"][0]
        ).resolve()
        if not fixed_reference.is_file():
            raise FileNotFoundError(
                f"fixed WUJI reference not found: {fixed_reference}"
            )
    if args_cli.ppo_checkpoint is not None and not args_cli.ppo_checkpoint.is_file():
        raise FileNotFoundError(
            f"initial shared PPO checkpoint not found: {args_cli.ppo_checkpoint}"
        )
    if args_cli.fixed_ppo_vectors is not None:
        if not args_cli.fixed_ppo_vectors.is_file():
            raise FileNotFoundError(
                f"fixed PPO vectors not found: {args_cli.fixed_ppo_vectors}"
            )
        fixed_ppo_vectors = np.load(args_cli.fixed_ppo_vectors).astype(np.float32)
        if fixed_ppo_vectors.shape != (args_cli.population, len(VECTOR_NAMES)):
            raise ValueError(
                f"fixed PPO vectors have shape {fixed_ppo_vectors.shape}, expected "
                f"({args_cli.population}, {len(VECTOR_NAMES)})"
            )
    else:
        fixed_ppo_vectors = None

    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "co_training_contract.json").write_text(
            json.dumps(
                {
                    "observation_mode": args_cli.ppo_observation_mode,
                    "observation_dimension": int(env_cfg.observation_space),
                    "reference": (
                        None if fixed_reference is None else str(fixed_reference)
                    ),
                    "mano_task_targets": None if task_targets_path is None else str(task_targets_path),
                    "mano_rollout": None if task_targets is None else str(task_targets["source_mano_rollout"]),
                    "mano_rollout_sha256": None if task_targets is None else str(task_targets["source_mano_rollout_sha256"]),
                    "retarget_target_source": "captured_mano_fk" if geometry_observations else "fixed_reference",
                    "retarget_initialization": "neutral_then_previous_candidate_frame" if geometry_observations else None,
                    "retarget_wrist_orientation": "mano_with_candidate_neutral_alignment" if geometry_observations else None,
                    "seed_trajectory_used": False,
                    "retarget_per_generation": geometry_observations,
                    "retarget_per_proposal": geometry_observations,
                    "candidate_collision_surface_landmarks": geometry_observations,
                    "original_source_hand": args_cli.original_source_hand,
                    "asset_generation_skipped": args_cli.original_source_hand,
                    "geometry_inward_direction_mode": (
                        args_cli.geometry_inward_direction_mode
                        if geometry_observations else None
                    ),
                    "population": args_cli.population,
                    "morphology_replicas": args_cli.morphology_replicas,
                    "eval_envs_per_morphology": args_cli.eval_envs_per_morphology,
                    "global_eval_envs": args_cli.population * args_cli.eval_envs_per_morphology,
                    "global_envs": (
                        args_cli.population * args_cli.morphology_replicas
                    ),
                    "physical_replicas_per_morphology": (
                        args_cli.morphology_replicas
                    ),
                    "ppo_rollout_multiplier": args_cli.ppo_rollout_multiplier,
                    "effective_env_equivalent": (
                        args_cli.population
                        * args_cli.morphology_replicas
                        * args_cli.ppo_rollout_multiplier
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
            wandb=args_cli.wandb,
            wandb_project=args_cli.wandb_project,
            wandb_group=args_cli.wandb_group,
            wandb_run_name=f"{args_cli.wandb_run_name}_sac",
            fixed_palm_prototype=args_cli.fixed_palm_prototype,
        )

    episode_window = None
    if args_cli.wandb:
        import wandb
        if rank == 0:
            wandb.define_metric("PPO/train_steps")
            wandb.define_metric("PPO/Completed episodes / *", step_metric="PPO/train_steps")
        episode_window = CompletedEpisodeWindow(
            window_steps=args_cli.ppo_episode_log_window_steps,
            global_envs=args_cli.population * args_cli.morphology_replicas,
            log=wandb.run.log if rank == 0 else None,
            distributed=skrl.config.torch.is_distributed,
        )
    history: list[dict] = []
    cumulative_environment_steps = 0
    for generation in range(args_cli.generations):
        generation_root = output / f"generation_{generation:03d}"
        vectors_path = generation_root / "vectors.npy"
        proposal: ProposalBatch | None = None
        if rank == 0:
            generation_root.mkdir(parents=True, exist_ok=True)
            assert optimizer is not None
            proposal = optimizer.propose(generation)
            if fixed_ppo_vectors is not None:
                proposal = ProposalBatch(
                    vectors=fixed_ppo_vectors.copy(),
                    semantic_vectors=resolve_design_vectors(fixed_ppo_vectors).astype(
                        np.float32
                    ),
                    observations=proposal.observations,
                    actions=proposal.actions,
                    palm_indices=np.rint(fixed_ppo_vectors[:, 0]).astype(np.int64),
                    sample_sources=tuple(
                        "fixed_ppo_population" for _ in range(args_cli.population)
                    ),
                )
            elif args_cli.force_source_morphology:
                source_vectors = np.repeat(
                    SOURCE_VECTOR[None, :], args_cli.population, axis=0
                ).astype(np.float32)
                proposal = ProposalBatch(
                    vectors=source_vectors,
                    semantic_vectors=resolve_design_vectors(source_vectors).astype(
                        np.float32
                    ),
                    observations=proposal.observations,
                    actions=proposal.actions,
                    palm_indices=np.zeros(args_cli.population, dtype=np.int64),
                    sample_sources=tuple(
                        "forced_source_identity" for _ in range(args_cli.population)
                    ),
                )
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
        local_retarget_path = rank_root / "gpu_retarget_all.npz"
        if geometry_observations:
            assert retarget_kinematics is not None
            assert task_targets is not None
            retarget_timings = retarget_mano(
                local_vectors_path, local_retarget_path, retarget_kinematics,
                task_targets, args_cli.retarget_iterations,
            )
        else:
            retarget_timings = {
                "seconds": 0.0,
                "candidates": len(global_indices),
                "solver_seconds": 0.0,
                "skipped": True,
                "fixed_reference": str(fixed_reference),
            }
        if args_cli.original_source_hand:
            from wuji_original_source_assets import prepare_original_source_hand
            local_manifest, prepare_timings = prepare_original_source_hand(
                local_retarget_path, task_targets_path, rank_root / "prepared", REPO_ROOT,
            )
        else:
            local_manifest, prepare_timings = prepare_assets(
                local_vectors_path,
                local_retarget_path if geometry_observations else None,
                rank_root,
                bank_manifest,
                fixed_reference=fixed_reference,
                task_targets_path=task_targets_path,
            )
        prepare_timings["retarget_seconds"] = float(retarget_timings["seconds"])
        prepare_timings["retarget_solver_seconds"] = float(
            retarget_timings["solver_seconds"]
        )
        expanded_manifest = replicate_manifest(
            local_manifest,
            args_cli.morphology_replicas,
            global_indices,
        )
        if args_cli.morphology_context:
            expanded_manifest["policy_morphology_context"] = (
                normalized_morphology_context(
                    np.asarray(expanded_manifest["vectors"], dtype=np.float32)
                ).tolist()
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
            episode_window=episode_window,
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
            optimizer_status = (
                {
                    "optimizer_update_skipped": True,
                    "reason": "forced_source_morphology_ppo_isolation",
                }
                if args_cli.force_source_morphology or fixed_ppo_vectors is not None
                else optimizer.observe(generation, proposal, rewards)
            )
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
            cumulative_environment_steps += sum(
                int(row["environment_steps"]) for row in all_rows
            )
            log_generation_to_wandb(
                generation,
                all_rows,
                optimizer_status,
                cumulative_environment_steps,
            )
            summary = {
                "schema_version": 1,
                "algorithm": (
                    "retargeted_palm_geometry_shared_ppo_outer_skrl_sac"
                    if geometry_observations
                    else "fixed_reference_shared_ppo_outer_skrl_sac"
                ),
                "force_source_morphology": args_cli.force_source_morphology,
                "original_source_hand": args_cli.original_source_hand,
                "morphology_context": args_cli.morphology_context,
                "morphology_context_dim": len(VECTOR_NAMES) if args_cli.morphology_context else 0,
                "fixed_ppo_vectors": (
                    None if args_cli.fixed_ppo_vectors is None else str(args_cli.fixed_ppo_vectors)
                ),
                "observation_mode": args_cli.ppo_observation_mode,
                "observation_dimension": int(env_cfg.observation_space),
                "retarget_performed": geometry_observations,
                "retarget_per_proposal": geometry_observations,
                "fixed_reference": (
                    None if fixed_reference is None else str(fixed_reference)
                ),
                "population": args_cli.population,
                "morphology_replicas": args_cli.morphology_replicas,
                "eval_envs_per_morphology": args_cli.eval_envs_per_morphology,
                "global_eval_envs": args_cli.population * args_cli.eval_envs_per_morphology,
                "ppo_rollout_multiplier": args_cli.ppo_rollout_multiplier,
                "effective_env_equivalent": (
                    args_cli.population
                    * args_cli.morphology_replicas
                    * args_cli.ppo_rollout_multiplier
                ),
                "global_envs": args_cli.population * args_cli.morphology_replicas,
                "cumulative_environment_steps": cumulative_environment_steps,
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
                    "cumulative_environment_steps": cumulative_environment_steps,
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

    if episode_window is not None:
        episode_window.flush()
    if rank == 0:
        print("WUJI_SHARED_PPO_SAC_COMPLETE", flush=True)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    if (args_cli.physics_worker_manifest is None) != (
        args_cli.physics_worker_output is None
    ):
        raise ValueError(
            "--physics-worker-manifest and --physics-worker-output must be used together"
        )
    if args_cli.physics_worker_manifest is not None:
        run_physics_worker(env_cfg)
        return
    output = args_cli.output_root
    output.mkdir(parents=True, exist_ok=True)
    bank_manifest = {} if bank_manifest_path is None else json.loads(bank_manifest_path.read_text())
    if args_cli.grouped_zero_action_vectors is not None:
        grouped_zero_action_debug(env_cfg, output, bank_manifest)
        return
    if args_cli.shared_ppo_iterations:
        shared_ppo_outer_search(env_cfg, agent_cfg, output, bank_manifest)
        return
    if skrl.config.torch.is_distributed:
        raise RuntimeError(
            "distributed execution is only supported with "
            "--ppo-cycles-per-generation > 0"
        )
    fixed_reference = args_cli.fixed_reference
    kinematics: WujiBatchKinematics | None = None
    seed_q: torch.Tensor | None = None
    seed_arrays: dict[str, np.ndarray] | None = None
    if fixed_reference is None:
        if args_cli.seed_trajectory is None:
            raise ValueError("Legacy morphology-only mode requires --seed-trajectory or --fixed-reference")
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
    if fixed_reference is not None:
        if not fixed_reference.is_file():
            raise FileNotFoundError(
                f"fixed WUJI reference not found: {fixed_reference}"
            )
        (output / "fixed_reference_contract.json").write_text(
            json.dumps(
                {
                    "reference": str(fixed_reference),
                    "retarget_per_generation": False,
                    "shared_ppo_iterations": 0,
                    "zero_residual_physics_evaluation": True,
                    "only_experimental_difference": (
                        "reuse_first_retarget_for_all_morphologies"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    state_path = output / "hybrid_sac_state.pt"
    replay_path = output / "hybrid_sac_replay.npz"
    history = []
    previous_summary: Path | None = None
    cumulative_environment_steps = 0
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
            wandb=args_cli.wandb,
            wandb_project=args_cli.wandb_project,
            wandb_group=args_cli.wandb_group,
            wandb_run_name=args_cli.wandb_run_name,
            fixed_palm_prototype=args_cli.fixed_palm_prototype,
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
                    if args_cli.fixed_palm_prototype is None
                    else "fixed_precompiled_prototype"
                ),
                "fixed_palm_prototype": args_cli.fixed_palm_prototype,
                "learned_action_dimension": int(proposal.actions.shape[1]),
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
        if fixed_reference is None:
            assert kinematics is not None
            assert seed_q is not None
            assert seed_arrays is not None
            timings["retarget"] = retarget(
                vectors_path,
                retarget_path,
                kinematics,
                seed_q,
                seed_arrays,
                args_cli.retarget_iterations,
            )
        else:
            timings["retarget"] = {
                "seconds": 0.0,
                "candidates": args_cli.population,
                "solver_seconds": 0.0,
                "skipped": True,
                "fixed_reference": str(fixed_reference),
            }
        manifest, prepare_timings = prepare_assets(
            vectors_path,
            None if fixed_reference is not None else retarget_path,
            generation_root,
            bank_manifest,
            fixed_reference=fixed_reference,
        )
        timings.update(prepare_timings)
        manifest["training_video"] = generation_video_spec(args_cli.output_root, generation, args_cli)
        if args_cli.isolate_physics_per_generation:
            worker_manifest_path = generation_root / "physics_worker_manifest.json"
            worker_output_path = generation_root / "physics_worker_results.json"
            worker_manifest_path.write_text(json.dumps(manifest) + "\n")
            evaluation = evaluate_manifest_isolated(
                env_cfg,
                worker_manifest_path,
                worker_output_path,
            )
        else:
            evaluation = evaluate_manifest_in_process(env_cfg, manifest)
        rows = evaluation["rows"]
        rollout_rows = evaluation["rollout_rows"]
        initialization_seconds = float(evaluation["initialization_seconds"])
        rollout_seconds = float(evaluation["rollout_seconds"])
        batch_records = evaluation["batch_records"]
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
            for row in rollout_rows:
                index = row["candidate_index"]
                row["sample_source"] = proposal.sample_sources[index]
                row["palm_prototype_index"] = int(
                    proposal.palm_indices[index]
                )
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
        cumulative_environment_steps += sum(
            int(row["environment_steps"]) for row in rows
        )
        log_generation_to_wandb(
            generation,
            rows,
            optimizer_status,
            cumulative_environment_steps,
            evaluation.get("video_records", []),
        )
        timings["physx_initialization_seconds"] = initialization_seconds
        timings["physx_rollout_seconds"] = rollout_seconds
        summary = {
            "schema_version": 1,
            "backend": "persistent_isaaclab_physx_gpu",
            "algorithm": (
                (
                    "skrl_sac_fixed_reference_no_ppo"
                    if fixed_reference is not None
                    else "skrl_sac_per_proposal_retarget_no_ppo"
                )
                if skrl_optimizer is not None
                else "episode_level_hybrid_sac"
            ),
            "retarget_performed": fixed_reference is None,
            "retarget_per_proposal": fixed_reference is None,
            "fixed_reference": (
                str(fixed_reference) if fixed_reference is not None else None
            ),
            "fixed_palm_prototype": args_cli.fixed_palm_prototype,
            "rollouts_per_proposal": args_cli.rollouts_per_proposal,
            "shared_ppo_iterations": 0,
            "all_candidates_physically_evaluated": True,
            "proxy_used": False,
            "top_k_prefilter_used": False,
            "candidate_count": args_cli.population,
            "completed": len(rows),
            "completed_rollouts": len(rollout_rows),
            "video_records": evaluation.get("video_records", []),
            "cumulative_environment_steps": cumulative_environment_steps,
            "best": rows[0],
            "results": rows,
            "rollout_results": rollout_rows,
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
                "best_phase_mean": rows[0]["phase_mean"],
                "best_phase_median": rows[0]["phase_median"],
                "best_success_rate": rows[0]["success_rate"],
                "success_count": sum(row["success"] for row in rows),
                "successful_rollouts": sum(
                    row["success_count"] for row in rows
                ),
                "rollouts_per_proposal": args_cli.rollouts_per_proposal,
                "cumulative_environment_steps": cumulative_environment_steps,
                "timings": timings,
            }
        )
        (output / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(
            "WUJI_HYBRID_SAC_GENERATION "
            f"generation={generation} candidates={len(rows)}/{args_cli.population} "
            f"rollouts={len(rollout_rows)} "
            f"best_reward={rows[0]['total_reward']:.9f} "
            f"best_mean_phase={rows[0]['phase_mean']:.2f}/445 "
            f"best_median_phase={rows[0]['phase_median']:.2f}/445 "
            f"best_success_ratio={rows[0]['success_count']}/"
            f"{rows[0]['rollouts']} "
            f"robust_successes={sum(row['success'] for row in rows)} "
            f"successful_rollouts={sum(row['success_count'] for row in rows)} "
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
