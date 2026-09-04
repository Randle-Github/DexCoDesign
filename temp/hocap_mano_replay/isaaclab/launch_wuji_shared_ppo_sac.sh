#!/usr/bin/env bash
# Submit fixed-reference shared-PPO morphology co-design on 1-4 GPUs.

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
GPU_COUNT="${GPU_COUNT:-1}"
POPULATION="${POPULATION:-128}"
MORPHOLOGY_REPLICAS="${MORPHOLOGY_REPLICAS:-32}"
PPO_ROLLOUT_MULTIPLIER="${PPO_ROLLOUT_MULTIPLIER:-1}"
SHARED_PPO_ITERATIONS="${SHARED_PPO_ITERATIONS:-20}"
GENERATIONS="${GENERATIONS:-12}"
MORPHOLOGY_CONTEXT="${MORPHOLOGY_CONTEXT:-1}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/shared_ppo_sac}"

for value_name in GPU_COUNT POPULATION MORPHOLOGY_REPLICAS PPO_ROLLOUT_MULTIPLIER SHARED_PPO_ITERATIONS GENERATIONS; do
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
EFFECTIVE_ENV_EQUIVALENT=$((GLOBAL_ENVS * PPO_ROLLOUT_MULTIPLIER))
printf 'Submitting shared policy: GPUs=%s morphologies=%s physical_replicas=%s physical_envs=%s envs_per_rank=%s rollout_multiplier=%s effective_env_equivalent=%s PPO/SAC=%s generations=%s\n' \
  "${GPU_COUNT}" "${POPULATION}" "${MORPHOLOGY_REPLICAS}" \
  "${GLOBAL_ENVS}" "${ENVS_PER_RANK}" "${PPO_ROLLOUT_MULTIPLIER}" \
  "${EFFECTIVE_ENV_EQUIVALENT}" "${SHARED_PPO_ITERATIONS}" "${GENERATIONS}"

# This launcher is intentionally the real joint co-design path. Explicitly
# override isolation variables so a caller's debug environment cannot silently
# freeze the SAC population or replace it with the source morphology.
if [[ "${MORPHOLOGY_CONTEXT}" != "1" ]]; then
  printf 'MORPHOLOGY_CONTEXT must remain 1 for shared-policy joint co-design.\n' >&2
  exit 2
fi

exec /opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch \
  --gpus="a40:${GPU_COUNT}" \
  --export="ALL,GPU_COUNT=${GPU_COUNT},POPULATION=${POPULATION},PHYSICS_BATCH_SIZE=${POPULATION},MORPHOLOGY_REPLICAS=${MORPHOLOGY_REPLICAS},PPO_ROLLOUT_MULTIPLIER=${PPO_ROLLOUT_MULTIPLIER},SHARED_PPO_ITERATIONS=${SHARED_PPO_ITERATIONS},GENERATIONS=${GENERATIONS},CONTINUE_AFTER_SUCCESS=1,OPTIMIZER_BACKEND=skrl,MORPHOLOGY_CONTEXT=1,FORCE_SOURCE_MORPHOLOGY=0,FIXED_PPO_VECTORS=,PPO_CHECKPOINT=${PPO_CHECKPOINT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.sbatch"
