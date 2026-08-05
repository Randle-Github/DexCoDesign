#!/usr/bin/env bash
#SBATCH --job-name=wuji-mjx-batch
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_gpu_search/slurm-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
CANDIDATES="${CANDIDATES:-1024}"
STEPS="${STEPS:-64}"
MODEL="${MODEL:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/physics_cache/wuji_source_physical.xml}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/artifacts/wuji_gpu_search/mjx_${CANDIDATES}.json}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign-mjx
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/artifacts/wuji_gpu_search"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python temp/hocap_mano_replay/scripts/benchmark_mjx_morphology_batch.py \
  "${MODEL}" --candidates "${CANDIDATES}" --steps "${STEPS}" --output "${OUTPUT}"
