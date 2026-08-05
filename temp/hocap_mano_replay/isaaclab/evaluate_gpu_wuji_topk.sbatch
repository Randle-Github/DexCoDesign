#!/usr/bin/env bash
#SBATCH --job-name=wuji-exact-topk
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_gpu_search/exact-topk-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
PROPOSAL="${PROPOSAL:?set PROPOSAL to the GPU proposal npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
TOP_K="${TOP_K:-32}"
WORKERS="${WORKERS:-16}"
EXACT_IK_ITERATIONS="${EXACT_IK_ITERATIONS:-4}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1

python temp/hocap_mano_replay/scripts/evaluate_gpu_wuji_topk.py \
  "${PROPOSAL}" \
  --output-root "${OUTPUT_ROOT}" \
  --top-k "${TOP_K}" \
  --workers "${WORKERS}" \
  --exact-ik-iterations "${EXACT_IK_ITERATIONS}"

printf 'WUJI_GPU_TOPK_JOB_COMPLETE top_k=%s output=%s\n' "${TOP_K}" "${OUTPUT_ROOT}"
