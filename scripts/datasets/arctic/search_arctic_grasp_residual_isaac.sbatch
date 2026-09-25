#!/usr/bin/env bash
#SBATCH --job-name=scissors-grasp-search
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/grasp-search-%j.out
set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_MANO_SIDE=right
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH:-$ROOT/artifacts/arctic_mano_dynamic/00_scissors/isaaclab_reference_right_fixedtip_dynamic_tabletop_30hz.npz}"
export DEXCODESIGN_OBJECT_USD_PATH="$ROOT/artifacts/arctic_mano_preview/assets/objects/scissors_vhacd.usd"
export LD_LIBRARY_PATH="$ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"
./isaaclab.sh -p scripts/datasets/arctic/search_arctic_grasp_residual_isaac.py \
  --headless --device cuda:0 --num-envs "${NUM_ENVS:-512}" \
  --rounds "${ROUNDS:-6}" --steps "${STEPS:-110}" --start-phase "${START_PHASE:-52}" \
  --replicas "${REPLICAS:-4}"
