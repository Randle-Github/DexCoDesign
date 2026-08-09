#!/usr/bin/env bash
#SBATCH --job-name=hocap-wuji-sac
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/sac-%A_%a.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
TASK_IDS=(g04_pick_place g15_vertical_local g06_rotation_heavy g19_mixed_reorientation g02_horizontal_reorientation g18_container_lift g18_variant_lift g01_large_reorientation)
TASK_ID="${TASK_IDS[${SLURM_ARRAY_TASK_ID:?missing array index}]}"
if [[ "${REQUIRE_MANO_SUCCESS:-1}" == "1" ]]; then
  MANO_STATUS="${REPO_ROOT}/artifacts/hocap_multitask/ppo/${TASK_ID}/mano/training_status.json"
  python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["success"] else 3)' "${MANO_STATUS}" || {
    printf 'HOCAP_SKIPPED_MANO_FAILED task=%s status=%s\n' "${TASK_ID}" "${MANO_STATUS}"
    exit 0
  }
fi
export DEXCODESIGN_OBJECT_USD_PATH="${REPO_ROOT}/artifacts/hocap_multitask/assets/${TASK_ID}/object.usd"
export OUTPUT_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/wuji_sac/${TASK_ID}"
export FIXED_REFERENCE="${REPO_ROOT}/artifacts/hocap_multitask/prepared/${TASK_ID}/wuji_hand_2/reference.npz"
export POPULATION="${POPULATION:-4096}"
export PHYSICS_BATCH_SIZE="${PHYSICS_BATCH_SIZE:-4096}"
export GENERATIONS="${GENERATIONS:-30}"
export SHARED_PPO_ITERATIONS=0
export CONTINUE_AFTER_SUCCESS=1
bash "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.sbatch"
