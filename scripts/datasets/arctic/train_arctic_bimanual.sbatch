#!/usr/bin/env bash
#SBATCH --job-name=arctic-bimanual
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --exclude=gundam,baymax
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_bimanual/logs/train-%j.out

set -euo pipefail

ROOT="${DEXCODESIGN_REPO_ROOT:-/coc/flash7/yliu3735/workspace/DexCoDesign}"
SAMPLE_ID="${SAMPLE_ID:?SAMPLE_ID is required}"
OBJECT_ID="${OBJECT_ID:?OBJECT_ID is required}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT}/artifacts/arctic_bimanual/${SAMPLE_ID}}"

mkdir -p "${ROOT}/artifacts/arctic_bimanual/logs" "${ROOT}/artifacts/arctic_bimanual/rollouts"
export BIMANUAL_MODE=true
export MANO_SIDE=right
export REFERENCE_PATH="${REFERENCE_ROOT}/isaaclab_reference_right.npz"
export AUXILIARY_REFERENCE_PATH="${REFERENCE_ROOT}/isaaclab_reference_left.npz"
export OBJECT_USD_PATH="${ROOT}/artifacts/arctic_mano_preview/assets/objects/${OBJECT_ID}_vhacd.usd"
export RUN_NAME="arctic_bimanual_${SAMPLE_ID}"
export RUN_SEED="${RUN_SEED:-2026}"
export TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-300}"
export NUM_ENVS="${NUM_ENVS:-256}"
export OBJECT_DISABLE_GRAVITY=false
export OBJECT_Z_REWARD_MULTIPLIER="${OBJECT_Z_REWARD_MULTIPLIER:-1.0}"
export OBJECT_AIRBORNE_REWARD_WEIGHT=0.0
export REQUIRE_CROSS_LINK_PINCH=false
export SCISSORS_HANDLE_CONTACT_GATE=false
export OBJECT_FAILURE_DISTANCE="${OBJECT_FAILURE_DISTANCE:-0.05}"
export OBJECT_FAILURE_ORIENTATION="${OBJECT_FAILURE_ORIENTATION:-1.5}"
export OBJECT_FAILURE_ARTICULATION="${OBJECT_FAILURE_ARTICULATION:-1.5}"
export HAND_BEST_ROLLOUT_PATH="${ROOT}/artifacts/arctic_bimanual/rollouts/${SAMPLE_ID}.npz"

cd "${ROOT}"
bash scripts/datasets/arctic/train_arctic_scissors.sbatch
