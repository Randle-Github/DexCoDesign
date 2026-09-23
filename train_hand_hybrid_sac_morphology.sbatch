#!/bin/bash
#SBATCH --job-name=wuji-hybrid-sac
#SBATCH --partition=ravichandar-lab
#SBATCH --account=ravichandar-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a40:1
# September 22 probe: bishop's allocated L40S reported uncorrectable ECC.
# Remove this exclusion after cluster admins restore/verify GPU health.
#SBATCH --exclude=bishop
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/slurm-wuji-hybrid-sac-%j.out

# Submit from the repository (create slurm_logs before sbatch). GPU type alone
# does not validate a node: see docs/wuji_skynet_runtime.md for tested nodes.
set -eo pipefail

echo "WUJI batch script started: job=${SLURM_JOB_ID:-none} host=$(hostname)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd -P)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_sac/cluster_run_${SLURM_JOB_ID:-manual}}"
POPULATION="${NUM_MORPHOLOGIES:-${POPULATION:-64}}"
PHYSICS_BATCH_SIZE="${PHYSICS_BATCH_SIZE:-4}"
ROLLOUTS_PER_PROPOSAL="${ROLLOUTS_PER_PROPOSAL:-128}"
GENERATIONS="${GENERATIONS:-100}"
SAC_UPDATES="${SAC_UPDATES:-64}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-64}"
UNIFORM_FRACTION="${UNIFORM_FRACTION:-0.15}"
ELITE_MUTATION_FRACTION="${ELITE_MUTATION_FRACTION:-0.30}"
ELITE_REPLAY_FRACTION="${ELITE_REPLAY_FRACTION:-0.05}"
ELITE_MUTATION_SIGMA="${ELITE_MUTATION_SIGMA:-0.12}"
REWARD_SCALE="${REWARD_SCALE:-0.01}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-skrl}"
RETARGET_ITERATIONS="${RETARGET_ITERATIONS:-14}"
SHARED_PPO_ITERATIONS="${PPO_CYCLES_PER_GENERATION:-${SHARED_PPO_ITERATIONS:-0}}"
PPO_OBSERVATION_MODE="${PPO_OBSERVATION_MODE:-legacy}"
GEOMETRY_INWARD_DIRECTION_MODE="${GEOMETRY_INWARD_DIRECTION_MODE:-kinematic_normal}"
MORPHOLOGY_REPLICAS="${TRAIN_ENVS_PER_MORPHOLOGY:-${MORPHOLOGY_REPLICAS:-1}}"
PPO_ROLLOUT_MULTIPLIER="${PPO_ROLLOUT_MULTIPLIER:-1}"
GPU_COUNT="${GPU_COUNT:-1}"
GROUPED_ZERO_ACTION_VECTORS="${GROUPED_ZERO_ACTION_VECTORS:-}"
FORCE_SOURCE_MORPHOLOGY="${FORCE_SOURCE_MORPHOLOGY:-0}"
MORPHOLOGY_CONTEXT="${MORPHOLOGY_CONTEXT:-1}"
FIXED_PPO_VECTORS="${FIXED_PPO_VECTORS:-}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-}"
TARGET_REWARD="${TARGET_REWARD:-inf}"
KIT_ARGS="${KIT_ARGS:---/log/level=error}"
# Skynet's shared UJITSO datastore has returned missing cooked convex blocks
# while the job still exits successfully. Cook meshes directly without that
# cache; this does not change collision approximations or physics parameters.
PHYSX_DISABLE_MESH_CACHE="${PHYSX_DISABLE_MESH_CACHE:-1}"
CONTINUE_AFTER_SUCCESS="${CONTINUE_AFTER_SUCCESS:-1}"
ISOLATE_PHYSICS_PER_GENERATION="${ISOLATE_PHYSICS_PER_GENERATION:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
VIDEO_ENABLED="${VIDEO_ENABLED:-0}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-10}"
VIDEO_CANDIDATE_INDEX="${VIDEO_CANDIDATE_INDEX:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-445}"
VIDEO_STRIDE="${VIDEO_STRIDE:-2}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
WANDB_ENTITY="${WANDB_ENTITY:-njyunhai-georgia-tech}"
WANDB_PROJECT="${WANDB_PROJECT:-DexCoDesign}"
WANDB_GROUP="${WANDB_GROUP:-wuji-hybrid-sac}"
FIXED_PALM_PROTOTYPE="${FIXED_PALM_PROTOTYPE-0}"
RETARGET_PER_PROPOSAL="${RETARGET_PER_PROPOSAL:-1}"
if [[ "${PPO_OBSERVATION_MODE}" == "palm_geometry" ]]; then
  if (( SHARED_PPO_ITERATIONS < 1 )); then
    echo "ERROR: palm_geometry requires SHARED_PPO_ITERATIONS > 0" >&2
    exit 1
  fi
  if [[ "${RETARGET_PER_PROPOSAL}" != "1" ]]; then
    echo "ERROR: palm_geometry co-training requires RETARGET_PER_PROPOSAL=1" >&2
    exit 1
  fi
  # The 60 points already expose morphology and the trained checkpoint has a
  # fixed 385-value input. Do not append the legacy 23-value design context.
  MORPHOLOGY_CONTEXT=0
fi
if [[ -n "${FIXED_PALM_PROTOTYPE}" ]]; then
  PALM_RUN_TAG="palm${FIXED_PALM_PROTOTYPE}"
else
  PALM_RUN_TAG="palmlearned"
fi
if [[ "${RETARGET_PER_PROPOSAL}" == "1" ]]; then
  RETARGET_RUN_TAG="retarget"
else
  RETARGET_RUN_TAG="fixedref"
fi
WANDB_RUN_NAME="${WANDB_RUN_NAME:-wuji_hand_2_g04_1_refined_${PALM_RUN_TAG}_${RETARGET_RUN_TAG}_rollouts${ROLLOUTS_PER_PROPOSAL}_pop${POPULATION}_physxbs${PHYSICS_BATCH_SIZE}_gen${GENERATIONS}_${SLURM_JOB_ID:-manual}}"
BANK_ROOT="${BANK_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/all_hands_success_action_retarget/wuji_hand_2/retargeted_trajectory.npz}"
FIXED_REFERENCE="${FIXED_REFERENCE-}"
case "${RETARGET_PER_PROPOSAL}" in
  1)
    if [[ -n "${FIXED_REFERENCE}" ]]; then
      echo "ERROR: FIXED_REFERENCE conflicts with RETARGET_PER_PROPOSAL=1" >&2
      exit 1
    fi
    ;;
  0)
    if [[ -z "${FIXED_REFERENCE}" ]]; then
      FIXED_REFERENCE="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/reference.npz"
    fi
    ;;
  *)
    echo "ERROR: RETARGET_PER_PROPOSAL must be 0 or 1" >&2
    exit 1
    ;;
esac
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
if [[ "${PPO_OBSERVATION_MODE}" != "palm_geometry" ]]; then
  require_file "${SEED_TRAJECTORY}"
fi
if [[ -n "${FIXED_REFERENCE}" ]]; then
  require_file "${FIXED_REFERENCE}"
fi
require_file "${OBJECT_USD}"
require_file "${REPO_ROOT}/artifacts/hand_morphology/reference_graphs.json"
require_file "${REPO_ROOT}/artifacts/hand_morphology/mechanism_bundles.json"

if ! source "${CONDA_SH}"; then
  echo "ERROR: failed to initialize conda from ${CONDA_SH}" >&2
  exit 1
fi
if ! conda activate "${CONDA_ENV}"; then
  echo "ERROR: failed to activate conda environment ${CONDA_ENV}" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${CONDA_PREFIX}/lib"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac"
# Keep per-job Kit preferences separate from interactive/shared settings.
RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wuji-${SLURM_JOB_ID:-manual}.XXXXXX")"
KIT_ARGS+=" --/app/userConfigPath=${RUNTIME_DIR}/user.config.json"
if [[ "${PHYSX_DISABLE_MESH_CACHE}" == "1" ]]; then
  KIT_ARGS+=" --/physics/cooking/ujitsoCollisionCooking=false --/persistent/physics/useLocalMeshCache=false"
fi
# A native crash must not fill shared storage with multi-GB core dumps.
ulimit -c 0
NOFILE_HARD="$(ulimit -Hn)"
if [[ "${NOFILE_HARD}" == unlimited ]] || (( NOFILE_HARD >= 65536 )); then
  ulimit -Sn 65536
else
  ulimit -Sn "${NOFILE_HARD}"
fi
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
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
echo "Git commit: $(git rev-parse --short HEAD); Conda: ${CONDA_PREFIX}"
echo "Kit runtime directory: ${RUNTIME_DIR}; bypass shared collision cache: ${PHYSX_DISABLE_MESH_CACHE}"
echo "Output: ${OUTPUT_ROOT}"
echo "Population: ${POPULATION}; PhysX batch: ${PHYSICS_BATCH_SIZE}; rollouts/proposal: ${ROLLOUTS_PER_PROPOSAL}; generations: ${GENERATIONS}"
echo "Fixed palm prototype: ${FIXED_PALM_PROTOTYPE:-learned}; per-proposal retargeting: ${RETARGET_PER_PROPOSAL}"
echo "PPO observation: ${PPO_OBSERVATION_MODE}; inward direction: ${GEOMETRY_INWARD_DIRECTION_MODE}"
echo "Process-isolated PhysX per generation: ${ISOLATE_PHYSICS_PER_GENERATION}"
echo "Weights & Biases mode: ${WANDB_MODE}"
echo "Rollout video recording: ${VIDEO_ENABLED}; interval: ${VIDEO_INTERVAL}; candidate: ${VIDEO_CANDIDATE_INDEX}; replica: 0"
echo "Weights & Biases run: ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
nvidia-smi -L
# Fail clearly on a node whose graphics runtime cannot load the GLU dependency.
python - <<'PY'
import ctypes
import sys
from importlib import metadata

ctypes.CDLL("libGLU.so.1")
print("Python:", sys.version.split()[0])
for package in ("torch", "skrl", "gymnasium", "numpy", "wandb"):
    print(f"{package}: {metadata.version(package)}")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("No usable CUDA GPU in this Slurm allocation")
sample = torch.ones((32, 32), device="cuda:0")
assert torch.isfinite(sample @ sample).all().item()
torch.cuda.synchronize()
print("CUDA preflight passed:", torch.cuda.get_device_name(0))
PY

EXTRA_ARGS=()
if [[ -n "${EVAL_ENVS_PER_MORPHOLOGY:-}" ]]; then
  EXTRA_ARGS+=(--eval-envs-per-morphology "${EVAL_ENVS_PER_MORPHOLOGY}")
fi
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
if [[ "${VIDEO_ENABLED}" == "1" ]]; then
  EXTRA_ARGS+=(
    --video
    --video-interval "${VIDEO_INTERVAL}"
    --video-candidate-index "${VIDEO_CANDIDATE_INDEX}"
    --video-length "${VIDEO_LENGTH}"
    --video-stride "${VIDEO_STRIDE}"
    --video-width "${VIDEO_WIDTH}"
    --video-height "${VIDEO_HEIGHT}"
  )
fi
if [[ "${ISOLATE_PHYSICS_PER_GENERATION}" == "1" ]]; then
  EXTRA_ARGS+=(--isolate-physics-per-generation)
fi
if [[ -n "${FIXED_REFERENCE}" ]]; then
  EXTRA_ARGS+=(--fixed-reference "${FIXED_REFERENCE}")
fi
if [[ -n "${FIXED_PALM_PROTOTYPE}" ]]; then
  EXTRA_ARGS+=(--fixed-palm-prototype "${FIXED_PALM_PROTOTYPE}")
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
  --num-morphologies "${POPULATION}" \
  --physics-batch-size "${PHYSICS_BATCH_SIZE}" \
  --rollouts-per-proposal "${ROLLOUTS_PER_PROPOSAL}" \
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
  --ppo-cycles-per-generation "${SHARED_PPO_ITERATIONS}" \
  --ppo-observation-mode "${PPO_OBSERVATION_MODE}" \
  --geometry-inward-direction-mode "${GEOMETRY_INWARD_DIRECTION_MODE}" \
  --train-envs-per-morphology "${MORPHOLOGY_REPLICAS}" \
  --ppo-rollout-multiplier "${PPO_ROLLOUT_MULTIPLIER}" \
  --target-reward "${TARGET_REWARD}" \
  "${EXTRA_ARGS[@]}" \
  --kit_args="${KIT_ARGS}" \
  --headless
  "$@"
)
# Kit's native fast exit bypasses Python atexit and can truncate the W&B file.
# Finish logging before SimulationApp.close(), without editing the trainer or
# disabling fast shutdown (normal plugin teardown also crashed in node tests).
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  PROGRAM_ARGS=(scripts/tools/run_isaac_with_wandb.py "${PROGRAM_ARGS[@]}")
fi
RESOURCE_WRAPPER=()
if [[ -x /usr/bin/time ]]; then
  RESOURCE_WRAPPER=(/usr/bin/time -v)
fi
if (( GPU_COUNT == 1 )); then
  "${RESOURCE_WRAPPER[@]}" ./isaaclab.sh -p "${PROGRAM_ARGS[@]}"
else
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  "${RESOURCE_WRAPPER[@]}" ./isaaclab.sh -p -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPU_COUNT}" \
    "${PROGRAM_ARGS[@]}" \
    --distributed
fi
