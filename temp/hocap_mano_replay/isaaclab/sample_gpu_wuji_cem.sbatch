#!/usr/bin/env bash
#SBATCH --job-name=wuji-cem-sample
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_gpu_search/cem-sample-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
STATE="${STATE:?set STATE}"
VECTORS="${VECTORS:?set VECTORS}"
POPULATION="${POPULATION:-4096}"
EXACT_SUMMARY="${EXACT_SUMMARY:-}"
GENERATION_SEED="${GENERATION_SEED:-20260805}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
command=(python temp/hocap_mano_replay/scripts/update_gpu_wuji_cem.py
  --state "${STATE}" --output-vectors "${VECTORS}"
  --population "${POPULATION}" --seed "${GENERATION_SEED}")
if [[ -n "${EXACT_SUMMARY}" ]]; then
  command+=(--exact-summary "${EXACT_SUMMARY}")
fi
"${command[@]}"
