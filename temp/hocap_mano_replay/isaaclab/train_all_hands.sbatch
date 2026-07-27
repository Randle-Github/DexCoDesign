#!/usr/bin/env bash
#SBATCH --job-name=all-hand-ppo
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_all_hands_residual/slurm/train-%A_%a.out

set -euo pipefail

HAND_IDS=(
  ability_hand
  schunk_svh
  wuji_hand_2
  sharpa_wave_01
  tesollo_dg5f
  unitree_dex5_1
  robotera_xhand1
  orca_hand_v2
  shadow_hand_e
  allegro_hand_v5
  midas_hand
  ruka_v2
  inspire_rh56dfx
)
HAND_ID="${HAND_IDS[${SLURM_ARRAY_TASK_ID}]}"
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual"
RESULT_SET="${RESULT_SET:-results}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_SET}/${HAND_ID}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300}"
RUN_SEED="${RUN_SEED:-42}"
RUN_LABEL="${HAND_ID}_max${MAX_ITERATIONS}_seed${RUN_SEED}"
SUCCESS_TRAJECTORY="${RESULT_DIR}/successful_rollout.npz"
BEST_TRAJECTORY="${RESULT_DIR}/farthest_rollout.npz"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_HAND_ID="${HAND_ID}"
export HAND_SUCCESS_TRAJECTORY_PATH="${SUCCESS_TRAJECTORY}"
export HAND_BEST_ROLLOUT_PATH="${BEST_TRAJECTORY}"
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${RESULT_DIR}" "${OUTPUT_ROOT}/slurm"

if [[ ! -f "${OUTPUT_ROOT}/assets/.all_hands_original_human_v2" ]]; then
  printf 'All-hand assets are not ready: %s\n' "${OUTPUT_ROOT}/assets" >&2
  exit 1
fi

set +e
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-Hand-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 1024 \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed "${RUN_SEED}" \
  --headless \
  "agent.agent.experiment.experiment_name=${RUN_LABEL}"
TRAIN_EXIT=$?
set -e

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
if [[ -n "${RUN_DIR}" ]]; then
  printf '%s\n' "${RUN_DIR}" > "${RESULT_DIR}/run_dir.txt"
fi
if [[ ! -f "${SUCCESS_TRAJECTORY}" && ! -f "${BEST_TRAJECTORY}" ]]; then
  printf 'Training produced no captured rollout for %s (exit=%s).\n' \
    "${HAND_ID}" "${TRAIN_EXIT}" >&2
  exit "${TRAIN_EXIT:-1}"
fi

python temp/hocap_mano_replay/isaaclab/summarize_hand_training.py \
  --hand-id "${HAND_ID}" \
  --success "${SUCCESS_TRAJECTORY}" \
  --best "${BEST_TRAJECTORY}" \
  --run-dir "${RUN_DIR}" \
  --iterations "${MAX_ITERATIONS}" \
  --output "${RESULT_DIR}/training_status.json"

printf 'HAND_TRAINING_COMPLETE hand_id=%s exit=%s status=%s\n' \
  "${HAND_ID}" "${TRAIN_EXIT}" "${RESULT_DIR}/training_status.json"
