#!/usr/bin/env bash
#SBATCH --job-name=mano-residual
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/slurm-%j.out

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

mkdir -p "${OUTPUT_ROOT}"
if ! python -c "from importlib.metadata import version; version('isaacsim')" >/dev/null 2>&1; then
  python -m pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
fi
if ! python -c "from importlib.metadata import version; [version(p) for p in ('isaaclab', 'isaaclab_tasks', 'isaaclab_rl', 'skrl')]" >/dev/null 2>&1; then
  python -m pip install \
    -e source/isaaclab \
    -e source/isaaclab_assets \
    -e source/isaaclab_tasks \
    -e "source/isaaclab_rl[skrl]"
  python -m pip install typing_extensions==4.12.2
fi

if [[ ! -f "${OUTPUT_ROOT}/assets/mano_left.usd" \
   || ! -f "${OUTPUT_ROOT}/assets/mano_visuals.json" \
   || ! -f "${OUTPUT_ROOT}/assets/g04_1.usd" \
   || ! -f "${OUTPUT_ROOT}/assets/.egoengine_mass_scale_v2" ]]; then
  bash temp/hocap_mano_replay/isaaclab/prepare_assets.sh
fi

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 1024 \
  --max_iterations 300 \
  --headless

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name '*.pt' -print | sort -V | tail -n 1)"
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --headless

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_ROOT}/results"
