#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/all_hands_independent.conf}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'Configuration not found: %s\nCopy and edit %s first.\n' \
    "${CONFIG_PATH}" "${SCRIPT_DIR}/all_hands_independent.conf.example" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

: "${GPU_COUNT:?Set GPU_COUNT to 1, 2, 4, or 8}"
: "${NUM_ENVS_PER_HAND:?Set NUM_ENVS_PER_HAND}"
: "${HAND_ARRAY_SPEC:?Set HAND_ARRAY_SPEC}"
: "${RESULT_SET:?Set RESULT_SET}"
: "${MAX_ITERATIONS:?Set MAX_ITERATIONS}"

case "${GPU_COUNT}" in
  1|2|4|8) ;;
  *)
    printf 'GPU_COUNT must be one of 1, 2, 4, 8; got %s.\n' "${GPU_COUNT}" >&2
    exit 2
    ;;
esac
if [[ ! "${NUM_ENVS_PER_HAND}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_ENVS_PER_HAND must be a positive integer, got %s.\n' \
    "${NUM_ENVS_PER_HAND}" >&2
  exit 2
fi
if [[ ! "${HAND_ARRAY_SPEC}" =~ ^[0-9,-]+$ ]]; then
  printf 'HAND_ARRAY_SPEC contains unsupported characters: %s.\n' \
    "${HAND_ARRAY_SPEC}" >&2
  exit 2
fi
if [[ ! "${RESULT_SET}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'RESULT_SET contains unsupported characters: %s.\n' "${RESULT_SET}" >&2
  exit 2
fi

TOTAL_ITERATIONS_BUDGET="${TOTAL_ITERATIONS_BUDGET:-${MAX_ITERATIONS}}"
RESUME_FROM_RESULT="${RESUME_FROM_RESULT:-0}"
RUN_SEED="${RUN_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNSAFE_HIGH_ENVS="${ALLOW_UNSAFE_HIGH_ENVS:-0}"

if (( NUM_ENVS_PER_HAND > 8192 )) && [[ "${ALLOW_UNSAFE_HIGH_ENVS}" != "1" ]]; then
  printf 'NUM_ENVS_PER_HAND=%s exceeds the validated 8192-env PhysX limit.\n' \
    "${NUM_ENVS_PER_HAND}" >&2
  exit 2
fi

printf 'Submitting independent all-hand PPO: concurrency=%s, envs_per_hand=%s, hands=%s\n' \
  "${GPU_COUNT}" "${NUM_ENVS_PER_HAND}" "${HAND_ARRAY_SPEC}"
printf 'Each Slurm array task requests exactly one GPU and trains exactly one hand; DDP is not used.\n'

SBATCH_ARGS=(
  --array="${HAND_ARRAY_SPEC}%${GPU_COUNT}"
  --export="ALL,RESULT_SET=${RESULT_SET},NUM_ENVS_PER_HAND=${NUM_ENVS_PER_HAND},MAX_ITERATIONS=${MAX_ITERATIONS},TOTAL_ITERATIONS_BUDGET=${TOTAL_ITERATIONS_BUDGET},RESUME_FROM_RESULT=${RESUME_FROM_RESULT},RUN_SEED=${RUN_SEED},GPU_CONCURRENCY=${GPU_COUNT},ALLOW_UNSAFE_HIGH_ENVS=${ALLOW_UNSAFE_HIGH_ENVS}"
  "${SCRIPT_DIR}/train_all_hands.sbatch"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf '\n'
else
  sbatch "${SBATCH_ARGS[@]}"
fi
