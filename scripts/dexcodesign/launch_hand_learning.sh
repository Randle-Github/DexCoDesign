#!/usr/bin/env bash
# One explicit launcher for PPO-only, morphology-SAC-only, and SAC+PPO.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_hand_learning.sh MODE --reference FILE --object-usd FILE --output DIR [options]

MODE:
  ppo       fixed morphology, residual PPO only
  sac       general-grammar morphology SAC, zero residual PPO updates
  sac-ppo   general-grammar morphology SAC with shared residual PPO updates

Options:
  --hand ID                 hand ID; SAC modes currently require wuji_hand_2
  --task-id ID              result label (default: custom_task)
  --population N            morphology candidates per SAC generation (default: 128)
  --generations N           SAC generations (default: 12)
  --ppo-iterations N        PPO updates per SAC generation (sac-ppo only; default: 20)
  --num-envs N              PPO environments (default: 4096)
  --max-iterations N        PPO-only iterations (default: 1200)
  --gpus N                  GPUs for SAC-PPO sample sharding (default: 1)
  --morphology-replicas N   physical replicas per morphology (default: 1)
  --rollout-multiplier N    PPO rollout reuse multiplier (default: 1)
  --physics-batch-size N    pure-SAC candidates per persistent scene batch
  --dry-run                 print the resolved submission command
EOF
}

if (( $# < 1 )); then
  usage >&2
  exit 2
fi
MODE="$1"
shift
HAND_ID="wuji_hand_2"
TASK_ID="custom_task"
REFERENCE=""
OBJECT_USD=""
OUTPUT=""
POPULATION=128
GENERATIONS=12
PPO_ITERATIONS=20
NUM_ENVS=4096
MAX_ITERATIONS=1200
GPU_COUNT=1
MORPHOLOGY_REPLICAS=1
PPO_ROLLOUT_MULTIPLIER=1
PHYSICS_BATCH_SIZE=""
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --hand) HAND_ID="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --object-usd) OBJECT_USD="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --population) POPULATION="$2"; shift 2 ;;
    --generations) GENERATIONS="$2"; shift 2 ;;
    --ppo-iterations) PPO_ITERATIONS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
    --gpus) GPU_COUNT="$2"; shift 2 ;;
    --morphology-replicas) MORPHOLOGY_REPLICAS="$2"; shift 2 ;;
    --rollout-multiplier) PPO_ROLLOUT_MULTIPLIER="$2"; shift 2 ;;
    --physics-batch-size) PHYSICS_BATCH_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %s.\n' "${label}" "${value}" >&2
    exit 2
  fi
}

if [[ "${MODE}" != "ppo" && "${MODE}" != "sac" && "${MODE}" != "sac-ppo" ]]; then
  printf 'MODE must be ppo, sac, or sac-ppo; got %s\n' "${MODE}" >&2
  exit 2
fi
for required in REFERENCE OBJECT_USD OUTPUT; do
  if [[ -z "${!required}" ]]; then
    printf '%s is required.\n' "${required}" >&2
    exit 2
  fi
done
require_positive_integer --population "${POPULATION}"
require_positive_integer --generations "${GENERATIONS}"
require_positive_integer --num-envs "${NUM_ENVS}"
require_positive_integer --max-iterations "${MAX_ITERATIONS}"
require_positive_integer --gpus "${GPU_COUNT}"
require_positive_integer --morphology-replicas "${MORPHOLOGY_REPLICAS}"
require_positive_integer --rollout-multiplier "${PPO_ROLLOUT_MULTIPLIER}"
if [[ -n "${PHYSICS_BATCH_SIZE}" ]]; then
  require_positive_integer --physics-batch-size "${PHYSICS_BATCH_SIZE}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_BIN="${SBATCH_BIN:-/opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch}"
CONDA_ENV="${CONDA_ENV:-codesign}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    printf 'Activate conda first, or export CONDA_SH=/path/to/conda.sh.\n' >&2
    exit 2
  fi
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  printf 'Conda activation script does not exist: %s\n' "${CONDA_SH}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}/slurm"
COMMON_EXPORTS="REPO_ROOT_OVERRIDE=${REPO_ROOT},CONDA_SH=${CONDA_SH},CONDA_ENV=${CONDA_ENV}"
if [[ "${MODE}" == "ppo" ]]; then
  if (( GPU_COUNT != 1 )); then
    printf 'ppo mode runs one hand per GPU; submit independent hands instead of --gpus %s.\n' "${GPU_COUNT}" >&2
    exit 2
  fi
  EXPORTS="ALL,${COMMON_EXPORTS},TASK_ID=${TASK_ID},HAND_ID_OVERRIDE=${HAND_ID},DEXCODESIGN_REFERENCE_PATH=${REFERENCE},DEXCODESIGN_OBJECT_USD_PATH=${OBJECT_USD},OUTPUT_ROOT=${OUTPUT},RESULT_SET=${TASK_ID},NUM_ENVS_PER_HAND=${NUM_ENVS},MAX_ITERATIONS=${MAX_ITERATIONS},TOTAL_ITERATIONS_BUDGET=${MAX_ITERATIONS}"
  TARGET="${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_all_hands.sbatch"
else
  if [[ "${HAND_ID}" != "wuji_hand_2" ]]; then
    printf '%s currently supports wuji_hand_2 only; got %s.\n' "${MODE}" "${HAND_ID}" >&2
    exit 2
  fi
  if [[ "${MODE}" == "sac" ]]; then
    PPO_ITERATIONS=0
    if (( GPU_COUNT != 1 )); then
      printf 'pure morphology SAC uses one persistent Isaac process; --gpus must be 1.\n' >&2
      exit 2
    fi
  elif (( PPO_ITERATIONS < 1 )); then
    printf 'sac-ppo requires --ppo-iterations >= 1.\n' >&2
    exit 2
  fi
  if [[ "${MODE}" == "sac-ppo" ]] && (( POPULATION % GPU_COUNT != 0 )); then
    printf 'SAC-PPO population %s must be divisible by GPU count %s.\n' "${POPULATION}" "${GPU_COUNT}" >&2
    exit 2
  fi
  if [[ -z "${PHYSICS_BATCH_SIZE}" ]]; then
    if [[ "${MODE}" == "sac-ppo" ]]; then
      PHYSICS_BATCH_SIZE=$(( POPULATION / GPU_COUNT ))
    else
      PHYSICS_BATCH_SIZE="${POPULATION}"
    fi
  fi
  EXPORTS="ALL,${COMMON_EXPORTS},DEXCODESIGN_OBJECT_USD_PATH=${OBJECT_USD},FIXED_REFERENCE=${REFERENCE},OUTPUT_ROOT=${OUTPUT},POPULATION=${POPULATION},PHYSICS_BATCH_SIZE=${PHYSICS_BATCH_SIZE},GENERATIONS=${GENERATIONS},SHARED_PPO_ITERATIONS=${PPO_ITERATIONS},CONTINUE_AFTER_SUCCESS=1,OPTIMIZER_BACKEND=skrl,MORPHOLOGY_CONTEXT=1,GPU_COUNT=${GPU_COUNT},MORPHOLOGY_REPLICAS=${MORPHOLOGY_REPLICAS},PPO_ROLLOUT_MULTIPLIER=${PPO_ROLLOUT_MULTIPLIER}"
  TARGET="${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.sbatch"
fi

COMMAND=("${SBATCH_BIN}")
COMMAND+=("--output=${OUTPUT}/slurm/%x-%j.out")
if [[ "${MODE}" == "sac-ppo" ]] && (( GPU_COUNT > 1 )); then
  COMMAND+=("--gpus=a40:${GPU_COUNT}")
fi
COMMAND+=("--export=${EXPORTS}" "${TARGET}")
printf 'DEXCODESIGN_MODE=%s hand=%s task=%s\n' "${MODE}" "${HAND_ID}" "${TASK_ID}"
if (( DRY_RUN )); then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  exec "${COMMAND[@]}"
fi
