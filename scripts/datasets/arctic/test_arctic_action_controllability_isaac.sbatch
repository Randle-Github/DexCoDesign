#!/usr/bin/env bash
#SBATCH --job-name=scissors-control-test
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/controllability-%j.out
set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_MANO_SIDE=right
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH:?Set REFERENCE_PATH}"
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD_PATH:-$ROOT/artifacts/arctic_mano_preview/assets/objects/scissors.usd}"
export LD_LIBRARY_PATH="$ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"
./isaaclab.sh -p scripts/datasets/arctic/test_arctic_action_controllability_isaac.py \
  --headless --device cuda:0 \
  --phase "${PHASE:-369}" --steps "${STEPS:-45}" \
  --num-envs "${NUM_ENVS:-1024}" --seed "${SEED:-0}"
