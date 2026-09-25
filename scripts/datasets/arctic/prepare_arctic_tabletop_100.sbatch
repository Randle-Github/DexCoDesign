#!/usr/bin/env bash
#SBATCH --job-name=arctic-tabletop100
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_rl_100/logs/tabletop-%j.out

set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_rl_100/logs
python scripts/datasets/arctic/prepare_arctic_tabletop.py \
  --root datasets/arctic_v1 --seconds 1.0 --write-trajectories --allow-unstable
