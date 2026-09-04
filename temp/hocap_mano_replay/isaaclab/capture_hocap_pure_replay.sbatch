#!/usr/bin/env bash
#SBATCH --job-name=hocap-pure-replay
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/pure-replay-%A_%a.out

set -euo pipefail

TASK_IDS=(
  g04_pick_place
  g06_rotation_heavy
  g18_container_lift
  g18_variant_lift
  g01_early_reorientation
)
TASK_ID="${TASK_IDS[${SLURM_ARRAY_TASK_ID:?missing array index}]}"
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/pure_replay/${TASK_ID}/wuji_hand_2"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1
export DEXCODESIGN_HAND_ID=wuji_hand_2
export DEXCODESIGN_REFERENCE_PATH="${REPO_ROOT}/artifacts/hocap_multitask/prepared/${TASK_ID}/wuji_hand_2/reference.npz"
export DEXCODESIGN_OBJECT_USD_PATH="${REPO_ROOT}/artifacts/hocap_multitask/assets/${TASK_ID}/object.usd"
export HAND_BEST_ROLLOUT_PATH="${OUTPUT_ROOT}/farthest_rollout.npz"
unset HAND_SUCCESS_TRAJECTORY_PATH MANO_SUCCESS_TRAJECTORY_PATH

./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/diagnose_zero_residual.py \
  --task DexCoDesign-Hand-Residual-Direct-v0 \
  --output "${OUTPUT_ROOT}/diagnostics.json" \
  --headless

test -s "${OUTPUT_ROOT}/farthest_rollout.npz"
printf '{"selected_rollout":"%s","status":"pure_replay"}\n' \
  "${OUTPUT_ROOT}/farthest_rollout.npz" > "${OUTPUT_ROOT}/training_status.json"
python -c 'import json,numpy as np,sys; d=np.load(sys.argv[1]); print("HOCAP_PURE_REPLAY_CAPTURED", json.dumps(json.loads(str(d["metadata_json"])), sort_keys=True))' "${OUTPUT_ROOT}/farthest_rollout.npz"
