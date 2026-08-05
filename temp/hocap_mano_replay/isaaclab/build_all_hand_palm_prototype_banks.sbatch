#!/usr/bin/env bash
#SBATCH --job-name=all-palm-banks
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hand_morphology/palm_banks_32/slurm-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/hand_morphology/palm_banks_32}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
python temp/hocap_mano_replay/scripts/write_all_hand_palm_prototype_graphs.py \
  "${OUTPUT_ROOT}/graphs.json"
python scripts/dexcodesign/compile_hand_graph_batch.py \
  "${OUTPUT_ROOT}/graphs.json" \
  --output-dir "${OUTPUT_ROOT}/compiled" \
  --workers 32 --palm-only --seed 0
python temp/hocap_mano_replay/scripts/audit_all_hand_palm_prototype_bank.py \
  "${OUTPUT_ROOT}/compiled/compiled_hands.json" \
  --output "${OUTPUT_ROOT}/bank_manifest.json"
printf 'ALL_HAND_PALM_BANKS_COMPLETE output=%s\n' "${OUTPUT_ROOT}"
