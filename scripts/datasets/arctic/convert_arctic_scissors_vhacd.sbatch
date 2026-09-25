#!/usr/bin/env bash
#SBATCH --job-name=scissors-vhacd
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/convert-vhacd-%j.out
set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  datasets/arctic_v1/assets/object_urdf/scissors.urdf \
  artifacts/arctic_mano_preview/assets/objects/scissors_vhacd.usd \
  --collider-type convex_decomposition \
  --joint-stiffness 0 --joint-damping 0 --headless
