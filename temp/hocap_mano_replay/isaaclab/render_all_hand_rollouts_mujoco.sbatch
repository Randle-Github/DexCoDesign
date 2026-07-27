#!/usr/bin/env bash
#SBATCH --job-name=all-hand-mj-render
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_all_hands_residual/slurm/mj-render-%A_%a.out

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
RESULT_SET="${RESULT_SET:-results_active}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_SET}/${HAND_ID}"
STATUS="${RESULT_DIR}/training_status.json"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${OUTPUT_ROOT}/reference_python_deps:${PYTHONPATH:-}"

TRAJECTORY="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_rollout"])' \
  "${STATUS}")"

python temp/hocap_mano_replay/scripts/render_captured_hand_rollout_mujoco.py \
  --hand-id "${HAND_ID}" \
  --trajectory "${TRAJECTORY}" \
  --output-dir "${RESULT_DIR}"
