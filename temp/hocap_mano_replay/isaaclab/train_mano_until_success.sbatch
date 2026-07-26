#!/usr/bin/env bash
#SBATCH --job-name=mano-until-success
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/until_success/slurm-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
RESULT_DIR="${OUTPUT_ROOT}/until_success"
RUN_SEED="${RUN_SEED:-42}"
CHUNK_ITERATIONS="${CHUNK_ITERATIONS:-100}"
CONTACT_WEIGHT="${CONTACT_WEIGHT:-6.0}"
TOTAL_ITERATIONS="${INITIAL_ITERATIONS:-0}"
CHECKPOINT="${INITIAL_CHECKPOINT:-}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${OUTPUT_ROOT}/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${RESULT_DIR}/evaluations"

while true; do
  TOTAL_ITERATIONS=$((TOTAL_ITERATIONS + CHUNK_ITERATIONS))
  RUN_LABEL="until_success_iter${TOTAL_ITERATIONS}_seed${RUN_SEED}"
  TRAIN_ARGS=(
    --task DexCoDesign-MANO-Residual-Direct-v0
    --algorithm PPO
    --num_envs 1024
    --max_iterations "${CHUNK_ITERATIONS}"
    --seed "${RUN_SEED}"
    --headless
  )
  if [[ -n "${CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--checkpoint "${CHECKPOINT}")
  fi

  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    "${TRAIN_ARGS[@]}" \
    "env.contact_reward_weight=${CONTACT_WEIGHT}" \
    "agent.agent.experiment.experiment_name=${RUN_LABEL}"

  RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
  CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name 'agent_*.pt' -print | sort -V | tail -n 1)"
  EVAL_JSON="${RESULT_DIR}/evaluations/iter_${TOTAL_ITERATIONS}.json"

  set +e
  ./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/evaluate_mano_checkpoint.py \
    --task DexCoDesign-MANO-Residual-Direct-Eval-v0 \
    --algorithm PPO \
    --checkpoint "${CHECKPOINT}" \
    --output "${EVAL_JSON}" \
    --seed "${RUN_SEED}" \
    --headless \
    "env.contact_reward_weight=${CONTACT_WEIGHT}"
  EVAL_EXIT=$?
  set -e

  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "Evaluation did not write ${EVAL_JSON}." >&2
    exit "${EVAL_EXIT:-1}"
  fi
  if [[ "${EVAL_EXIT}" -ne 0 && "${EVAL_EXIT}" -ne 2 ]]; then
    echo "Evaluation crashed with exit code ${EVAL_EXIT}." >&2
    exit "${EVAL_EXIT}"
  fi

  if python -c \
    'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["success"] else 1)' \
    "${EVAL_JSON}"; then
    echo "Successful complete-reference rollout at ${TOTAL_ITERATIONS} iterations."
    break
  else
    echo "Evaluation failed at ${TOTAL_ITERATIONS} iterations; continuing."
  fi
done

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --headless \
  "env.contact_reward_weight=${CONTACT_WEIGHT}"

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RESULT_DIR}/results" \
  --contact-weight "${CONTACT_WEIGHT}"
cp "${EVAL_JSON}" "${RESULT_DIR}/results/success_evaluation.json"
