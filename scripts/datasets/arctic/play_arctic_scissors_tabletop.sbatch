#!/usr/bin/env bash
#SBATCH --job-name=scissors-video
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --exclude=gundam,baymax
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/slurm-%j.out

set -euo pipefail

ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_scissors_rollout

export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=disabled
export DEXCODESIGN_MANO_SIDE=right
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH:-$ROOT/artifacts/arctic_mano_table_fixed/00_scissors/isaaclab_reference_right_ik_fixedtip_contactcore_30hz.npz}"
export DEXCODESIGN_OBJECT_USD_PATH="$ROOT/artifacts/arctic_mano_preview/assets/objects/scissors.usd"
export LD_LIBRARY_PATH="$ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT:?Set CHECKPOINT to the trained policy path}" \
  --video \
  --video_length "${VIDEO_LENGTH:-511}" \
  --headless \
  env.articulate_mode=true \
  env.log_rollout_diagnostics=true
