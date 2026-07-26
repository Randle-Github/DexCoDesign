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

mkdir -p "${OUTPUT_ROOT}"
if ! python -c "import isaacsim, isaaclab, skrl" >/dev/null 2>&1; then
  python -m pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
  ./isaaclab.sh --install skrl
fi

if [[ ! -f "${OUTPUT_ROOT}/assets/mano_left.usd" || ! -f "${OUTPUT_ROOT}/assets/g04_1.usd" ]]; then
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
