#!/usr/bin/env bash
#SBATCH --job-name=wuji-palm-bank
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/palm-bank-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/palm_prototype_bank_32}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES

python temp/hocap_mano_replay/scripts/write_wuji_palm_prototype_vectors.py \
  "${OUTPUT_ROOT}/vectors.npy"
python temp/hocap_mano_replay/scripts/gpu_wuji_retarget.py \
  --seed-trajectory "${SEED_TRAJECTORY}" \
  --vectors "${OUTPUT_ROOT}/vectors.npy" \
  --output "${OUTPUT_ROOT}/gpu_retarget_all.npz" \
  --iterations 4 --candidate-chunk 32 --device cuda \
  --save-trajectories --skip-proxy
python temp/hocap_mano_replay/scripts/prepare_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/gpu_retarget_all.npz" \
  --output-root "${OUTPUT_ROOT}/prepared" \
  --workers 32 --limit 32
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/convert_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/prepared/physx_batch_manifest.json" --headless
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/prepared/physx_batch_manifest.json" \
  --output "${OUTPUT_ROOT}/direct_physx_results.json" --headless

test -s "${OUTPUT_ROOT}/direct_physx_results.json"
printf 'WUJI_PALM_PROTOTYPE_BANK_COMPLETE levels=32 output=%s\n' "${OUTPUT_ROOT}"
