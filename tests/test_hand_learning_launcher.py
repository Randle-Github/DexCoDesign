from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/dexcodesign/launch_hand_learning.sh"
HYBRID_RUNNER = (
    ROOT / "temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py"
)
HYBRID_SBATCH = (
    ROOT / "temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.sbatch"
)
COMMON = [
    "--reference",
    "/tmp/reference.npz",
    "--object-usd",
    "/tmp/object.usd",
    "--output",
    "/tmp/output",
    "--dry-run",
]


def run_launcher(mode: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), mode, *COMMON, *extra],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ppo_only_dry_run() -> None:
    result = run_launcher("ppo", "--num-envs", "4096", "--max-iterations", "1200")
    assert result.returncode == 0, result.stderr
    assert "train_all_hands.sbatch" in result.stdout
    assert "NUM_ENVS_PER_HAND=4096" in result.stdout


def test_pure_sac_dry_run() -> None:
    result = run_launcher(
        "sac", "--population", "128", "--physics-batch-size", "64"
    )
    assert result.returncode == 0, result.stderr
    assert "SHARED_PPO_ITERATIONS=0" in result.stdout
    assert "PHYSICS_BATCH_SIZE=64" in result.stdout
    assert "GPU_COUNT=1" in result.stdout


def test_sac_ppo_multi_gpu_requests_and_shards_allocation() -> None:
    result = run_launcher(
        "sac-ppo",
        "--population",
        "128",
        "--gpus",
        "4",
        "--morphology-replicas",
        "32",
        "--rollout-multiplier",
        "3",
    )
    assert result.returncode == 0, result.stderr
    assert "--gpus=a40:4" in result.stdout
    assert "GPU_COUNT=4" in result.stdout
    assert "PHYSICS_BATCH_SIZE=32" in result.stdout
    assert "MORPHOLOGY_REPLICAS=32" in result.stdout
    assert "PPO_ROLLOUT_MULTIPLIER=3" in result.stdout
    assert "MORPHOLOGY_CONTEXT=1" in result.stdout


def test_sac_ppo_rejects_non_divisible_population() -> None:
    result = run_launcher("sac-ppo", "--population", "130", "--gpus", "4")
    assert result.returncode == 2
    assert "must be divisible" in result.stderr


def test_pure_sac_rejects_multi_gpu() -> None:
    result = run_launcher("sac", "--gpus", "2")
    assert result.returncode == 2
    assert "persistent Isaac process" in result.stderr


def test_hybrid_runner_keeps_rank_local_sampling_and_global_gather() -> None:
    runner = HYBRID_RUNNER.read_text(encoding="utf-8")
    sbatch = HYBRID_SBATCH.read_text(encoding="utf-8")
    assert "per_rank = args_cli.population // world_size" in runner
    assert "global_vectors[begin:end]" in runner
    assert "torch.distributed.all_gather_object" in runner
    assert "normalized_morphology_context" in runner
    assert '--nproc_per_node="${GPU_COUNT}"' in sbatch
    assert '--morphology-replicas "${MORPHOLOGY_REPLICAS}"' in sbatch
    assert '--ppo-rollout-multiplier "${PPO_ROLLOUT_MULTIPLIER}"' in sbatch
