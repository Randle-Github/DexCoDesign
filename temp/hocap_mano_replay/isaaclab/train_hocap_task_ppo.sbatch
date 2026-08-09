#!/usr/bin/env bash
#SBATCH --job-name=hocap-ppo
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/ppo-%A_%a.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
TASK_IDS=(g04_pick_place g15_vertical_local g06_rotation_heavy g19_mixed_reorientation g02_horizontal_reorientation)
INDEX="${SLURM_ARRAY_TASK_ID:?missing array index}"
TASK_ID="${TASK_IDS[INDEX]}"
HAND_ID_OVERRIDE="${HAND_ID_OVERRIDE:-mano}"
export TASK_ID HAND_ID_OVERRIDE
export OUTPUT_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/ppo"
export RESULT_SET="${TASK_ID}"
export DEXCODESIGN_OBJECT_USD_PATH="${REPO_ROOT}/artifacts/hocap_multitask/assets/${TASK_ID}/object.usd"
if [[ "${HAND_ID_OVERRIDE}" == "mano" ]]; then
  export DEXCODESIGN_REFERENCE_PATH="${REPO_ROOT}/temp/hocap_mano_replay/data/tasks/${TASK_ID}/isaaclab_reference.npz"
else
  export DEXCODESIGN_REFERENCE_PATH="${REPO_ROOT}/artifacts/hocap_multitask/prepared/${TASK_ID}/${HAND_ID_OVERRIDE}/reference.npz"
fi
export MAX_ITERATIONS="${MAX_ITERATIONS:-1200}"
export TOTAL_ITERATIONS_BUDGET="${TOTAL_ITERATIONS_BUDGET:-${MAX_ITERATIONS}}"
export NUM_ENVS_PER_HAND="${NUM_ENVS_PER_HAND:-4096}"
export STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
bash "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_all_hands.sbatch"
