#!/usr/bin/env bash
#SBATCH --job-name=arctic-prepare
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_rl_100/logs/prepare-%A_%a.out

set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_rl_100/logs
export OMP_NUM_THREADS=1
python scripts/datasets/arctic/prepare_arctic_batch_item.py \
  --index "${SLURM_ARRAY_TASK_ID}" --iterations "${IK_ITERATIONS:-24}"
