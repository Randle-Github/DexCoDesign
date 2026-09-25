#!/usr/bin/env bash
#SBATCH --job-name=arctic-assets
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_rl_100/logs/assets-%j.out

set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_rl_100/logs artifacts/arctic_mano_preview/assets/objects
export OMNI_KIT_ACCEPT_EULA=YES
for urdf in datasets/arctic_v1/assets/object_urdf/*.urdf; do
  object_id="$(basename "$urdf" .urdf)"
  output="artifacts/arctic_mano_preview/assets/objects/${object_id}_vhacd.usd"
  if [[ -f "$output" ]]; then
    continue
  fi
  ./isaaclab.sh -p scripts/tools/convert_urdf.py "$urdf" "$output" \
    --collider-type convex_decomposition \
    --joint-stiffness 0 --joint-damping 0 --headless
done
echo ARCTIC_OBJECT_ASSETS_READY
