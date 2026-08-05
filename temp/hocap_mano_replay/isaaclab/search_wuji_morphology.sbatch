#!/usr/bin/env bash
#SBATCH --job-name=wuji-morph-cem
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_morphology_search/slurm-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
mkdir -p "${REPO_ROOT}/artifacts/wuji_morphology_search"

python temp/hocap_mano_replay/scripts/search_wuji_morphology.py \
  --output-root "${RESULT_ROOT}" \
  --generations 5 \
  --population 8 \
  --workers 4 \
  --iterations 18 \
  --seed 20260805

printf 'WUJI_MORPHOLOGY_SEARCH_COMPLETE result=%s\n' "${RESULT_ROOT}"
