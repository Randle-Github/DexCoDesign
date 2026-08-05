#!/usr/bin/env bash
#SBATCH --job-name=wuji-physx-batch
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/slurm-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/smoke}"
VECTORS="${VECTORS:?set VECTORS to a morphology vectors.npy}"
LIMIT="${LIMIT:-4}"
WORKERS="${WORKERS:-${LIMIT}}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES

python temp/hocap_mano_replay/scripts/gpu_wuji_retarget.py \
  --seed-trajectory "${SEED_TRAJECTORY}" \
  --vectors "${VECTORS}" \
  --output "${OUTPUT_ROOT}/gpu_retarget_all.npz" \
  --iterations 4 \
  --candidate-chunk 128 \
  --device cuda \
  --save-trajectories \
  --skip-proxy

python temp/hocap_mano_replay/scripts/prepare_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/gpu_retarget_all.npz" \
  --output-root "${OUTPUT_ROOT}/prepared" \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
  "${OUTPUT_ROOT}/prepared/physx_batch_manifest.json" \
  --output "${OUTPUT_ROOT}/physx_results.json" \
  --convert-urdf \
  --headless

printf 'WUJI_PHYSX_SMOKE_COMPLETE candidates=%s output=%s\n' \
  "${LIMIT}" "${OUTPUT_ROOT}/physx_results.json"
