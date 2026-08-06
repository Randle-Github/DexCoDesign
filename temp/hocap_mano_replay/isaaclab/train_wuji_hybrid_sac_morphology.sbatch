#!/usr/bin/env bash
#SBATCH --job-name=wuji-hybrid-sac
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_morphology_hybrid_sac/slurm-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/run}"
POPULATION="${POPULATION:-4096}"
PHYSICS_BATCH_SIZE="${PHYSICS_BATCH_SIZE:-4096}"
GENERATIONS="${GENERATIONS:-20}"
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
FIXED_REFERENCE="${FIXED_REFERENCE:-}"
GROUPED_ZERO_ACTION_VECTORS="${GROUPED_ZERO_ACTION_VECTORS:-}"
FORCE_SOURCE_MORPHOLOGY="${FORCE_SOURCE_MORPHOLOGY:-0}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-}"
TARGET_REWARD="${TARGET_REWARD:-inf}"
KIT_ARGS="${KIT_ARGS:---/log/level=error}"
CONTINUE_AFTER_SUCCESS="${CONTINUE_AFTER_SUCCESS:-0}"
BANK_ROOT="${BANK_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/palm_prototype_bank_32_expanded_0p70}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
EXTRA_ARGS=()
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
