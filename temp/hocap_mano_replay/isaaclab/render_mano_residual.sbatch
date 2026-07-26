#!/usr/bin/env bash
#SBATCH --job-name=mano-residual-render
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/render-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${OUTPUT_ROOT}/runtime_libs"
mkdir -p "${RUNTIME_LIB_DIR}"
ln -sfn "${CONDA_PREFIX}/lib/libGLU.so.1" "${RUNTIME_LIB_DIR}/libGLU.so.1"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${OUTPUT_ROOT}/assets/mano_visuals.json" ]]; then
  bash temp/hocap_mano_replay/isaaclab/prepare_assets.sh
fi

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name '*.pt' -print | sort -V | tail -n 1)"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --disable_fabric \
  --headless

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_ROOT}/results"
