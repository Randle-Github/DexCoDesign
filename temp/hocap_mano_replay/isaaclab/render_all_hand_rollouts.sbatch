#!/usr/bin/env bash
#SBATCH --job-name=all-hand-render
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_all_hands_residual/slurm/render-%A_%a.out

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
RESULT_DIR="${OUTPUT_ROOT}/results/${HAND_ID}"
STATUS="${RESULT_DIR}/training_status.json"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_HAND_ID="${HAND_ID}"
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

TRAJECTORY="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_rollout"])' \
  "${STATUS}")"

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/replay_hand_rollout.py \
  --trajectory "${TRAJECTORY}" \
  --output-dir "${RESULT_DIR}" \
  --enable_cameras \
  --headless
