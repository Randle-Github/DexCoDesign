#!/usr/bin/env bash
#SBATCH --job-name=mano-egoengine-obs
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/egoengine_observation/slurm-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
RESULT_NAME="${RESULT_NAME:-egoengine_observation}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_NAME}"
RUN_SEED="${RUN_SEED:-42}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-1500}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${OUTPUT_ROOT}/runtime_libs"
mkdir -p "${RUNTIME_LIB_DIR}" "${RESULT_DIR}/results"
ln -sfn "${CONDA_PREFIX}/lib/libGLU.so.1" "${RUNTIME_LIB_DIR}/libGLU.so.1"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${OUTPUT_ROOT}/assets/.egoengine_actuators_v3" ]]; then
  bash temp/hocap_mano_replay/isaaclab/prepare_assets.sh
fi

RUN_LABEL="${RESULT_NAME}_${TRAIN_ITERATIONS}iter_seed${RUN_SEED}"
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 1024 \
  --max_iterations "${TRAIN_ITERATIONS}" \
  --seed "${RUN_SEED}" \
  --headless \
  "agent.agent.experiment.experiment_name=${RUN_LABEL}"

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
CHECKPOINT="${RUN_DIR}/checkpoints/best_agent.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name 'agent_*.pt' -print | sort -V | tail -n 1)"
fi

set +e
./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/evaluate_mano_checkpoint.py \
  --task DexCoDesign-MANO-Residual-Direct-Eval-v0 \
  --algorithm PPO \
  --checkpoint "${CHECKPOINT}" \
  --output "${RESULT_DIR}/evaluation.json" \
  --seed "${RUN_SEED}" \
  --headless
EVAL_EXIT=$?
set -e
if [[ "${EVAL_EXIT}" -ne 0 && "${EVAL_EXIT}" -ne 2 ]]; then
  exit "${EVAL_EXIT}"
fi

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --headless

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RESULT_DIR}/results" \
  --contact-weight 2.0
cp "${RESULT_DIR}/evaluation.json" "${RESULT_DIR}/results/evaluation.json"
