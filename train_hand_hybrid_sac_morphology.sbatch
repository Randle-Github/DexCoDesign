#!/usr/bin/env bash
#SBATCH --job-name=wuji-hybrid-sac
#SBATCH --partition=ravichandar-lab
#SBATCH --account=ravichandar-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-wuji-hybrid-sac-%j.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_sac/cluster_run_${SLURM_JOB_ID:-manual}}"
POPULATION="${POPULATION:-1024}"
PHYSICS_BATCH_SIZE="${PHYSICS_BATCH_SIZE:-16}"
GENERATIONS="${GENERATIONS:-100}"
SAC_UPDATES="${SAC_UPDATES:-400}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-1024}"
UNIFORM_FRACTION="${UNIFORM_FRACTION:-0.15}"
ELITE_MUTATION_FRACTION="${ELITE_MUTATION_FRACTION:-0.30}"
ELITE_REPLAY_FRACTION="${ELITE_REPLAY_FRACTION:-0.05}"
ELITE_MUTATION_SIGMA="${ELITE_MUTATION_SIGMA:-0.12}"
REWARD_SCALE="${REWARD_SCALE:-0.01}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-skrl}"
RETARGET_ITERATIONS="${RETARGET_ITERATIONS:-4}"
SHARED_PPO_ITERATIONS="${SHARED_PPO_ITERATIONS:-0}"
MORPHOLOGY_REPLICAS="${MORPHOLOGY_REPLICAS:-1}"
PPO_ROLLOUT_MULTIPLIER="${PPO_ROLLOUT_MULTIPLIER:-1}"
GPU_COUNT="${GPU_COUNT:-1}"
GROUPED_ZERO_ACTION_VECTORS="${GROUPED_ZERO_ACTION_VECTORS:-}"
FORCE_SOURCE_MORPHOLOGY="${FORCE_SOURCE_MORPHOLOGY:-0}"
MORPHOLOGY_CONTEXT="${MORPHOLOGY_CONTEXT:-1}"
FIXED_PPO_VECTORS="${FIXED_PPO_VECTORS:-}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-}"
TARGET_REWARD="${TARGET_REWARD:-inf}"
KIT_ARGS="${KIT_ARGS:---/log/level=error}"
CONTINUE_AFTER_SUCCESS="${CONTINUE_AFTER_SUCCESS:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-njyunhai-georgia-tech}"
WANDB_PROJECT="${WANDB_PROJECT:-DexCoDesign}"
WANDB_GROUP="${WANDB_GROUP:-wuji-hybrid-sac}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-wuji_hand_2_g04_1_refined_pop${POPULATION}_gen${GENERATIONS}_${SLURM_JOB_ID:-manual}}"
BANK_ROOT="${BANK_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/all_hands_success_action_retarget/wuji_hand_2/retargeted_trajectory.npz}"
FIXED_REFERENCE="${FIXED_REFERENCE:-${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/reference.npz}"
OBJECT_USD="${DEXCODESIGN_OBJECT_USD_PATH:-${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/g04_1.usd}"

CONDA_SH="${CONDA_SH:-/coc/flash5/yhan389/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-DexCoDesign_sim4_5_lab_2_3_2}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

require_file "${CONDA_SH}"
require_file "${REPO_ROOT}/isaaclab.sh"
require_file "${BANK_ROOT}/vectors.schema.json"
require_file "${BANK_ROOT}/prepared/physx_batch_manifest.json"
require_file "${SEED_TRAJECTORY}"
require_file "${FIXED_REFERENCE}"
require_file "${OBJECT_USD}"
require_file "${REPO_ROOT}/artifacts/hand_morphology/reference_graphs.json"
require_file "${REPO_ROOT}/artifacts/hand_morphology/mechanism_bundles.json"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD}"
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_ENTITY
else
  export WANDB_MODE="disabled"
fi

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Host: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Population: ${POPULATION}; PhysX batch: ${PHYSICS_BATCH_SIZE}; generations: ${GENERATIONS}"
echo "Weights & Biases mode: ${WANDB_MODE}"
echo "Weights & Biases run: ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  EXTRA_ARGS+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-run-name "${WANDB_RUN_NAME}"
  )
fi
if [[ "${CONTINUE_AFTER_SUCCESS}" == "1" ]]; then
  EXTRA_ARGS+=(--continue-after-success)
fi
if [[ -n "${FIXED_REFERENCE}" ]]; then
  EXTRA_ARGS+=(--fixed-reference "${FIXED_REFERENCE}")
fi
if [[ -n "${GROUPED_ZERO_ACTION_VECTORS}" ]]; then
  EXTRA_ARGS+=(--grouped-zero-action-vectors "${GROUPED_ZERO_ACTION_VECTORS}")
fi
if [[ "${FORCE_SOURCE_MORPHOLOGY}" == "1" ]]; then
  EXTRA_ARGS+=(--force-source-morphology)
fi
if [[ "${MORPHOLOGY_CONTEXT}" == "1" ]]; then
  EXTRA_ARGS+=(--morphology-context)
fi
if [[ -n "${FIXED_PPO_VECTORS}" ]]; then
  EXTRA_ARGS+=(--fixed-ppo-vectors "${FIXED_PPO_VECTORS}")
fi
if [[ -n "${PPO_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--ppo-checkpoint "${PPO_CHECKPOINT}")
fi
PROGRAM_ARGS=(
  temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py
  --output-root "${OUTPUT_ROOT}" \
  --prototype-bank-root "${BANK_ROOT}" \
  --seed-trajectory "${SEED_TRAJECTORY}" \
  --population "${POPULATION}" \
  --physics-batch-size "${PHYSICS_BATCH_SIZE}" \
  --generations "${GENERATIONS}" \
  --sac-updates "${SAC_UPDATES}" \
  --sac-batch-size "${SAC_BATCH_SIZE}" \
  --uniform-fraction "${UNIFORM_FRACTION}" \
  --elite-mutation-fraction "${ELITE_MUTATION_FRACTION}" \
  --elite-replay-fraction "${ELITE_REPLAY_FRACTION}" \
  --elite-mutation-sigma "${ELITE_MUTATION_SIGMA}" \
  --reward-scale "${REWARD_SCALE}" \
  --optimizer-backend "${OPTIMIZER_BACKEND}" \
  --retarget-iterations "${RETARGET_ITERATIONS}" \
  --shared-ppo-iterations "${SHARED_PPO_ITERATIONS}" \
  --morphology-replicas "${MORPHOLOGY_REPLICAS}" \
  --ppo-rollout-multiplier "${PPO_ROLLOUT_MULTIPLIER}" \
  --target-reward "${TARGET_REWARD}" \
  "${EXTRA_ARGS[@]}" \
  --kit_args="${KIT_ARGS}" \
  --headless
)
if (( GPU_COUNT == 1 )); then
  ./isaaclab.sh -p "${PROGRAM_ARGS[@]}"
else
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  ./isaaclab.sh -p -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPU_COUNT}" \
    "${PROGRAM_ARGS[@]}" \
    --distributed
fi
