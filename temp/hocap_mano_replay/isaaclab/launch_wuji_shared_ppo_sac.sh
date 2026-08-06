#!/usr/bin/env bash
# Submit fixed-reference shared-PPO morphology co-design on 1-4 GPUs.

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
GPU_COUNT="${GPU_COUNT:-1}"
POPULATION="${POPULATION:-128}"
MORPHOLOGY_REPLICAS="${MORPHOLOGY_REPLICAS:-32}"
SHARED_PPO_ITERATIONS="${SHARED_PPO_ITERATIONS:-20}"
GENERATIONS="${GENERATIONS:-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/shared_ppo_sac}"

for value_name in GPU_COUNT POPULATION MORPHOLOGY_REPLICAS SHARED_PPO_ITERATIONS GENERATIONS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s.\n' "${value_name}" "${value}" >&2
    exit 2
  fi
done
if (( GPU_COUNT > 4 )); then
  printf 'GPU_COUNT=%s exceeds the validated 1-4 GPU configuration.\n' "${GPU_COUNT}" >&2
  exit 2
fi
if (( POPULATION % GPU_COUNT != 0 )); then
  printf 'POPULATION=%s must be divisible by GPU_COUNT=%s.\n' \
    "${POPULATION}" "${GPU_COUNT}" >&2
  exit 2
fi

GLOBAL_ENVS=$((POPULATION * MORPHOLOGY_REPLICAS))
ENVS_PER_RANK=$((GLOBAL_ENVS / GPU_COUNT))
printf 'Submitting shared policy: GPUs=%s morphologies=%s replicas=%s global_envs=%s envs_per_rank=%s PPO/SAC=%s generations=%s\n' \
  "${GPU_COUNT}" "${POPULATION}" "${MORPHOLOGY_REPLICAS}" \
  "${GLOBAL_ENVS}" "${ENVS_PER_RANK}" "${SHARED_PPO_ITERATIONS}" \
  "${GENERATIONS}"

exec /opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch \
  --gpus="a40:${GPU_COUNT}" \
  --export="ALL,GPU_COUNT=${GPU_COUNT},POPULATION=${POPULATION},PHYSICS_BATCH_SIZE=${POPULATION},MORPHOLOGY_REPLICAS=${MORPHOLOGY_REPLICAS},SHARED_PPO_ITERATIONS=${SHARED_PPO_ITERATIONS},GENERATIONS=${GENERATIONS},CONTINUE_AFTER_SUCCESS=1,OPTIMIZER_BACKEND=skrl,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.sbatch"
