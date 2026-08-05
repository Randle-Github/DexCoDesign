#!/usr/bin/env bash
#SBATCH --job-name=wuji-param-usd
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/param-usd-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
LIMIT="${LIMIT:-1}"
RETARGET_BATCH="${RETARGET_BATCH:?set RETARGET_BATCH}"
COMPILED_HANDS="${COMPILED_HANDS:?set COMPILED_HANDS}"
TEMPLATE_USD="${TEMPLATE_USD:?set TEMPLATE_USD}"
TEMPLATE_REFERENCE="${TEMPLATE_REFERENCE:?set TEMPLATE_REFERENCE}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES

python temp/hocap_mano_replay/scripts/prepare_wuji_parametric_usd_smoke.py \
  "${RETARGET_BATCH}" "${COMPILED_HANDS}" \
  --template-usd "${TEMPLATE_USD}" \
  --template-reference "${TEMPLATE_REFERENCE}" \
  --output-root "${OUTPUT_ROOT}/prepared" \
  --limit "${LIMIT}"

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/prepared/physx_batch_manifest.json" \
  --output "${OUTPUT_ROOT}/physx_results.json" \
  --convert-urdf --headless

test -s "${OUTPUT_ROOT}/physx_results.json"
printf 'WUJI_PARAMETRIC_USD_SMOKE_COMPLETE count=%s output=%s\n' \
  "${LIMIT}" "${OUTPUT_ROOT}/physx_results.json"
