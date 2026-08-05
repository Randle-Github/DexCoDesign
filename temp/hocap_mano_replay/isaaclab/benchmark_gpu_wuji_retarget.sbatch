#!/usr/bin/env bash
#SBATCH --job-name=wuji-gpu-ik
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
CANDIDATES="${CANDIDATES:-4096}"
ITERATIONS="${ITERATIONS:-4}"
CANDIDATE_CHUNK="${CANDIDATE_CHUNK:-128}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/artifacts/wuji_gpu_search/gpu_retarget_${CANDIDATES}.npz}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/artifacts/wuji_gpu_search"
export OMP_NUM_THREADS=4

python temp/hocap_mano_replay/scripts/gpu_wuji_retarget.py \
  --seed-trajectory "${SEED_TRAJECTORY}" \
  --output "${OUTPUT}" \
  --candidates "${CANDIDATES}" \
  --iterations "${ITERATIONS}" \
  --candidate-chunk "${CANDIDATE_CHUNK}" \
  --device cuda

printf 'WUJI_GPU_RETARGET_COMPLETE candidates=%s output=%s\n' "${CANDIDATES}" "${OUTPUT}"
