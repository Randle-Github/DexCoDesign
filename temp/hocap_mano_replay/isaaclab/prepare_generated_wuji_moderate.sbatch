#!/usr/bin/env bash
#SBATCH --job-name=wuji-morph-assets
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_all_hands_residual/slurm/wuji-morph-assets-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
HAND_ID="wuji_moderate_demo"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual"
RUNTIME_ROOT="${REPO_ROOT}/temp/hocap_mano_replay/generated_wuji_moderate/runtime"
TRAJECTORY="${REPO_ROOT}/temp/hocap_mano_replay/generated_wuji_moderate/retargeted_trajectory.npz"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUTPUT_ROOT}/slurm" "${OUTPUT_ROOT}/assets/${HAND_ID}"

python temp/hocap_mano_replay/scripts/prepare_generated_hand_rl_reference.py \
  "${RUNTIME_ROOT}" \
  "${TRAJECTORY}" \
  --output-dir "${OUTPUT_ROOT}/prepared"

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "${OUTPUT_ROOT}/prepared/${HAND_ID}/hand_rl.urdf" \
  "${OUTPUT_ROOT}/assets/${HAND_ID}/hand.usd" \
  --fix-base \
  --joint-stiffness 300 \
  --joint-damping 34.6410162 \
  --joint-target-type position \
  --headless

PHYSICS_USD="${OUTPUT_ROOT}/assets/${HAND_ID}/configuration/hand_physics.usd"
if [[ ! -f "${PHYSICS_USD}" ]] || [[ "$(wc -c < "${PHYSICS_USD}")" -le 1024 ]]; then
  printf 'GENERATED_HAND_ASSET_FAILED hand_id=%s physics_usd=%s\n' \
    "${HAND_ID}" "${PHYSICS_USD}" >&2
  exit 1
fi
printf 'GENERATED_HAND_ASSET_READY hand_id=%s usd=%s\n' \
  "${HAND_ID}" "${OUTPUT_ROOT}/assets/${HAND_ID}/hand.usd"
