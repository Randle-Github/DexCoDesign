#!/usr/bin/env bash
#SBATCH --job-name=arctic-ppo
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_training/train-%j.out

set -euo pipefail

ROOT="${DEXCODESIGN_REPO_ROOT:-/coc/flash7/yliu3735/workspace/DexCoDesign}"
REFERENCE_PATH="${REFERENCE_PATH:?REFERENCE_PATH is required}"
OBJECT_USD_PATH="${OBJECT_USD_PATH:?OBJECT_USD_PATH is required}"
MANO_SIDE="${MANO_SIDE:-right}"
BIMANUAL_MODE="${BIMANUAL_MODE:-false}"
if [[ "${BIMANUAL_MODE}" == "true" ]]; then
  BIMANUAL_REFERENCE_PATH="${BIMANUAL_REFERENCE_PATH:?BIMANUAL_REFERENCE_PATH is required for bimanual training}"
  export DEXCODESIGN_BIMANUAL_REFERENCE_PATH="${BIMANUAL_REFERENCE_PATH}"
fi

source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "${ROOT}"
mkdir -p artifacts/arctic_training

export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${ROOT}/source/dexcodesign:${ROOT}/source/isaaclab_tasks:${ROOT}/source/isaaclab_assets:${ROOT}/source/isaaclab:${PYTHONPATH:-}"
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH}"
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD_PATH}"
export DEXCODESIGN_MANO_SIDE="${MANO_SIDE}"
export LD_LIBRARY_PATH="${ROOT}/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"

for side in "${MANO_SIDE}"; do
  usd="artifacts/isaaclab_mano_residual/assets/mano_${side}.usd"
  if [[ ! -f "${usd}" ]]; then
    ./isaaclab.sh -p scripts/tools/convert_urdf.py \
      "assets/robot_hands/direct_motor/mano/${side}/hand.urdf" "${usd}" \
      --joint-stiffness 0 --joint-damping 0 --headless
  fi
done
if [[ "${BIMANUAL_MODE}" == "true" ]]; then
  other_side=left
  if [[ "${MANO_SIDE}" == "left" ]]; then other_side=right; fi
  other_usd="artifacts/isaaclab_mano_residual/assets/mano_${other_side}.usd"
  if [[ ! -f "${other_usd}" ]]; then
    ./isaaclab.sh -p scripts/tools/convert_urdf.py \
      "assets/robot_hands/direct_motor/mano/${other_side}/hand.urdf" "${other_usd}" \
      --joint-stiffness 0 --joint-damping 0 --headless
  fi
fi

checkpoint_args=()
if [[ -n "${CHECKPOINT:-}" ]]; then checkpoint_args=(--checkpoint "${CHECKPOINT}"); fi
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 --algorithm PPO \
  --num_envs "${NUM_ENVS:-256}" --max_iterations "${TRAIN_ITERATIONS:-300}" \
  --seed "${RUN_SEED:-42}" "${checkpoint_args[@]}" --headless \
  env.articulate_mode=true env.bimanual_mode="${BIMANUAL_MODE}" \
  env.articulated_object_fix_root_link=false \
  env.require_free_object_root=true env.require_dynamic_object_root=true \
  "agent.agent.experiment.experiment_name=${RUN_NAME:-arctic_ppo}"
