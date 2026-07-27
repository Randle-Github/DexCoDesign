#!/usr/bin/env bash
#SBATCH --job-name=mano-residual-render
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/render-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
RESULT_NAME="${RESULT_NAME:-rendered_checkpoint}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_NAME}"
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

if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="${RUN_DIR}/checkpoints/best_agent.pt"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name 'agent_*.pt' -print | sort -V | tail -n 1)"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  set +e
  ./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/evaluate_mano_checkpoint.py \
    --task DexCoDesign-MANO-Residual-Direct-Eval-v0 \
    --algorithm PPO \
    --checkpoint "${CHECKPOINT}" \
    --output "${RESULT_DIR}/evaluation_diagnostics.json" \
    --seed 42 \
    --headless
  EVAL_EXIT=$?
  set -e
  if [[ "${EVAL_EXIT}" -ne 0 && "${EVAL_EXIT}" -ne 2 ]]; then
    exit "${EVAL_EXIT}"
  fi
fi

PLAY_OVERRIDES=()
if [[ "${LEGACY_ACTION_SCALES:-0}" == "1" ]]; then
  PLAY_OVERRIDES+=(
    "env.residual_root_position_scale=0.04"
    "env.residual_root_rotation_scale=0.025"
    "env.residual_finger_scale=0.15"
  )
fi

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --headless \
  "${PLAY_OVERRIDES[@]}"

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RESULT_DIR}/results" \
  --contact-weight 2.0
if [[ -f "${RESULT_DIR}/evaluation_diagnostics.json" ]]; then
  cp "${RESULT_DIR}/evaluation_diagnostics.json" "${RESULT_DIR}/results/evaluation_diagnostics.json"
fi
