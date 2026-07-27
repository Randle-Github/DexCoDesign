#!/usr/bin/env bash
#SBATCH --job-name=mano-zero
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/zero-residual-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
RESULT_PATH="${OUTPUT_ROOT}/diagnostics/zero_residual_official_joints.json"
FULL_REFERENCE="${FULL_REFERENCE:-0}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES

RUNTIME_LIB_DIR="${OUTPUT_ROOT}/runtime_libs"
mkdir -p "${RUNTIME_LIB_DIR}" "$(dirname "${RESULT_PATH}")"
ln -sfn "${CONDA_PREFIX}/lib/libGLU.so.1" "${RUNTIME_LIB_DIR}/libGLU.so.1"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${OUTPUT_ROOT}/assets/.egoengine_actuators_v3" ]]; then
  bash temp/hocap_mano_replay/isaaclab/prepare_assets.sh
fi

EXTRA_ARGS=()
if [[ "${FULL_REFERENCE}" == "1" ]]; then
  RESULT_PATH="${OUTPUT_ROOT}/diagnostics/zero_residual_official_joints_full.json"
  EXTRA_ARGS+=(--ignore-early-termination)
fi

./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/diagnose_zero_residual.py \
  --output "${RESULT_PATH}" \
  "${EXTRA_ARGS[@]}" \
  --headless
