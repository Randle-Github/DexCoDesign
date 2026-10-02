#!/usr/bin/env bash
#SBATCH --job-name=arctic-rl100
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --exclude=gundam,baymax
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_rl_100/logs/train-%A_%a.out

set -euo pipefail
ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
cd "$ROOT"
mkdir -p artifacts/arctic_rl_100/logs artifacts/arctic_rl_100/rollouts

mapfile -t RECORDS < <(find artifacts/arctic_rl_100 -mindepth 2 -maxdepth 2 \
  -name record.json -print | sort)
record="${RECORDS[${SLURM_ARRAY_TASK_ID}]}"
sample_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sample_id"])' "$record")"
object_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["object_id"])' "$record")"
reference="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference"])' "$record")"

export REFERENCE_PATH="$reference"
export OBJECT_USD_PATH="$ROOT/artifacts/arctic_mano_preview/assets/objects/${object_id}_vhacd.usd"
export RUN_NAME="arctic100_${sample_id}"
export RUN_SEED="$((1000 + SLURM_ARRAY_TASK_ID))"
export TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-300}"
export NUM_ENVS="${NUM_ENVS:-256}"
export HAND_BEST_ROLLOUT_PATH="$ROOT/artifacts/arctic_rl_100/rollouts/${sample_id}.npz"

bash scripts/datasets/arctic/train_arctic_ppo.sbatch
echo "ARCTIC_BATCH_TRAIN_COMPLETE index=${SLURM_ARRAY_TASK_ID} sample=${sample_id} object=${object_id}"
