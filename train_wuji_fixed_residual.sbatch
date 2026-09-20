#!/usr/bin/env bash
#SBATCH --job-name=wuji-fixed-residual
#SBATCH --partition=ravichandar-lab
#SBATCH --account=ravichandar-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-wuji-fixed-residual-%j.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
OBSERVATION_MODE="${OBSERVATION_MODE:-palm_geometry}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
RUN_SEED="${RUN_SEED:-42}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_fixed_residual/${OBSERVATION_MODE}/seed${RUN_SEED}_${SLURM_JOB_ID:-manual}}"
REFERENCE_PATH="${REFERENCE_PATH:-${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/reference.npz}"
GEOMETRY_URDF="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/hand_rl.urdf"
OBJECT_USD="${DEXCODESIGN_OBJECT_USD_PATH:-${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/g04_1.usd}"
CONDA_SH="${CONDA_SH:-/coc/flash5/yhan389/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-DexCoDesign_sim4_5_lab_2_3_2}"

case "${OBSERVATION_MODE}" in
  palm_geometry) TASK="DexCoDesign-WUJI-PalmGeometry-Residual-Direct-v0" ;;
  legacy) TASK="DexCoDesign-Hand-Residual-Direct-v0" ;;
  *) echo "OBSERVATION_MODE must be palm_geometry or legacy" >&2; exit 2 ;;
esac
for value in "${NUM_ENVS}" "${MAX_ITERATIONS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS and MAX_ITERATIONS must be positive integers" >&2
    exit 2
  fi
done
case "${WANDB_ENABLED}" in
  1) WANDB_BOOL=True ;;
  0) WANDB_BOOL=False ;;
  *) echo "WANDB_ENABLED must be 0 or 1" >&2; exit 2 ;;
esac
TRAIN_ARGS=(
  scripts/reinforcement_learning/skrl/train.py
  --task "${TASK}" --algorithm PPO --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERATIONS}" --seed "${RUN_SEED}" --headless
  "env.geometry_urdf_path=${GEOMETRY_URDF}"
  "agent.agent.experiment.directory=${OUTPUT_ROOT}/ppo_logs"
  "agent.agent.experiment.experiment_name=wuji_original_${OBSERVATION_MODE}_seed${RUN_SEED}"
  "agent.agent.experiment.wandb=${WANDB_BOOL}"
  "agent.agent.experiment.wandb_kwargs.group=wuji-fixed-residual"
  "agent.agent.experiment.wandb_kwargs.tags=[WUJI,PPO,${OBSERVATION_MODE}]"
)
if [[ -n "${PPO_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--checkpoint "${PPO_CHECKPOINT}")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Original WUJI; observation=%s; output=%s\n' "${OBSERVATION_MODE}" "${OUTPUT_ROOT}"
  printf 'DEXCODESIGN_HAND_ID=wuji_hand_2 DEXCODESIGN_REFERENCE_PATH=%q DEXCODESIGN_OBJECT_USD_PATH=%q ' "${REFERENCE_PATH}" "${OBJECT_USD}"
  printf '%q ' "${REPO_ROOT}/isaaclab.sh" -p "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

for path in "${CONDA_SH}" "${REFERENCE_PATH}" "${GEOMETRY_URDF}" "${OBJECT_USD}" \
  "${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/assets/wuji_hand_2/hand.usd"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Missing required file: %s\n' "${path}" >&2
    exit 1
  fi
done
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_HAND_ID=wuji_hand_2
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH}"
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD}"
unset DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST
export HAND_BEST_ROLLOUT_PATH="${OUTPUT_ROOT}/farthest_rollout.npz"
export HAND_SUCCESS_TRAJECTORY_PATH="${OUTPUT_ROOT}/successful_rollout.npz"
export HAND_EXIT_AFTER_SUCCESS_CAPTURE=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [[ "${WANDB_ENABLED}" == 0 ]]; then
  export WANDB_MODE=disabled
fi
./isaaclab.sh -p "${TRAIN_ARGS[@]}"
