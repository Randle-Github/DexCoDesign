#!/usr/bin/env bash
#SBATCH --job-name=mano-success-replay
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/success-replay-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
RESULT_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/success_capture_fixed_clamp_reset0"
TRAJECTORY="${RESULT_DIR}/successful_rollout.npz"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
mkdir -p "${RUNTIME_LIB_DIR}" "${RESULT_DIR}"
ln -sfn "${CONDA_PREFIX}/lib/libGLU.so.1" "${RUNTIME_LIB_DIR}/libGLU.so.1"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/replay_successful_trajectory.py \
  --trajectory "${TRAJECTORY}" \
  --output-dir "${RESULT_DIR}" \
  --enable_cameras \
  --headless
