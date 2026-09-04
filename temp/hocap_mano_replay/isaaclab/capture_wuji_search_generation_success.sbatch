#!/usr/bin/env bash
#SBATCH --job-name=wuji-gen-capture
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/capture-generation-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
SEARCH_ROOT="${SEARCH_ROOT:?missing SEARCH_ROOT}"
GENERATION="${GENERATION:?missing GENERATION}"
OUTPUT_ROOT="${OUTPUT_ROOT:?missing OUTPUT_ROOT}"
GENERATION_DIR="$(printf 'generation_%03d' "${GENERATION}")"
MANIFEST="${SEARCH_ROOT}/${GENERATION_DIR}/prepared/physx_batch_manifest.json"
CAPTURE="${OUTPUT_ROOT}/successful_rollout.npz"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1
export DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST="${MANIFEST}"
export HAND_SUCCESS_TRAJECTORY_PATH="${CAPTURE}"
export HAND_BEST_ROLLOUT_PATH="${OUTPUT_ROOT}/farthest_rollout.npz"
export HAND_CAPTURE_BEST_SUCCESS=1

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
  "${MANIFEST}" \
  --output "${OUTPUT_ROOT}/generation_capture_evaluation.json" \
  --no-seed \
  --headless

test -s "${CAPTURE}"
python -c 'import json,numpy as np,sys; d=np.load(sys.argv[1]); m=json.loads(str(d["metadata_json"])); assert m["success"] and m["final_phase"] == m["reference_last_phase"]; print("WUJI_GENERATION_SUCCESS_CAPTURE_COMPLETE", json.dumps(m, sort_keys=True))' "${CAPTURE}"
