#!/usr/bin/env bash
#SBATCH --job-name=arctic-scissors
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --exclude=gundam,baymax
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_training/slurm-%j.out

set -euo pipefail

REPO_ROOT="${DEXCODESIGN_REPO_ROOT:-/coc/flash7/yliu3735/workspace/DexCoDesign}"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/arctic_scissors_training"
NUM_ENVS="${NUM_ENVS:-512}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-300}"
RUN_SEED="${RUN_SEED:-42}"
RUN_NAME="${RUN_NAME:-arctic_scissors_articulated}"
CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=disabled
# Prefer the explicitly selected checkout over any editable install that may
# point at another branch.  This keeps isolated validation worktrees honest.
export PYTHONPATH="${REPO_ROOT}/source/dexcodesign:${REPO_ROOT}/source/isaaclab_tasks:${REPO_ROOT}/source/isaaclab_assets:${REPO_ROOT}/source/isaaclab:${PYTHONPATH:-}"
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH:-${REPO_ROOT}/artifacts/arctic_mano_dynamic/00_scissors/isaaclab_reference_right_fixedtip_dynamic_tabletop_30hz.npz}"
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD_PATH:-${REPO_ROOT}/artifacts/arctic_mano_preview/assets/objects/scissors_vhacd.usd}"
export DEXCODESIGN_MANO_SIDE="${MANO_SIDE:-right}"
if [[ "${BIMANUAL_MODE:-false}" == "true" ]]; then
  if [[ -z "${AUXILIARY_REFERENCE_PATH:-}" ]]; then
    echo "BIMANUAL_MODE=true requires AUXILIARY_REFERENCE_PATH" >&2
    exit 2
  fi
  export DEXCODESIGN_AUXILIARY_REFERENCE_PATH="${AUXILIARY_REFERENCE_PATH}"
fi

RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

RUN_LABEL="${RUN_NAME}_${TRAIN_ITERATIONS}iter_seed${RUN_SEED}"
if [[ "${DEXCODESIGN_MANO_SIDE}" == "right" && ! -f "${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/mano_right.usd" ]]; then
  ./isaaclab.sh -p scripts/tools/convert_urdf.py \
    assets/robot_hands/direct_motor/mano/right/hand.urdf \
    artifacts/isaaclab_mano_residual/assets/mano_right.usd \
    --joint-stiffness 0 --joint-damping 0 --headless
fi
if [[ "${BIMANUAL_MODE:-false}" == "true" && ! -f "${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/mano_left.usd" ]]; then
  ./isaaclab.sh -p scripts/tools/convert_urdf.py \
    assets/robot_hands/direct_motor/mano/left/hand.urdf \
    artifacts/isaaclab_mano_residual/assets/mano_left.usd \
    --joint-stiffness 0 --joint-damping 0 --headless
fi
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${TRAIN_ITERATIONS}" \
  --seed "${RUN_SEED}" \
  "${CHECKPOINT_ARGS[@]}" \
  --headless \
  env.articulate_mode=true \
  env.bimanual_mode="${BIMANUAL_MODE:-false}" \
  env.articulated_object_fix_root_link=false \
  env.object_disable_gravity="${OBJECT_DISABLE_GRAVITY:-false}" \
  env.require_free_object_root=true \
  env.require_dynamic_object_root=true \
  env.residual_root_position_scale="${ROOT_POSITION_SCALE:-0.03}" \
  env.residual_root_rotation_scale="${ROOT_ROTATION_SCALE:-0.10}" \
  env.residual_finger_scale="${FINGER_SCALE:-0.30}" \
  env.object_position_reward_weight="${OBJECT_POSITION_WEIGHT:-12.0}" \
  env.object_z_reward_multiplier="${OBJECT_Z_REWARD_MULTIPLIER:-3.0}" \
  env.object_rotation_reward_weight="${OBJECT_ROTATION_WEIGHT:-2.0}" \
  env.object_articulation_reward_weight="${OBJECT_ARTICULATION_WEIGHT:-2.0}" \
  env.exponential_pose_reward=true \
  env.pose_tracking_reward_scale="${POSE_REWARD_SCALE:-4.0}" \
  env.contact_reward_weight="${CONTACT_WEIGHT:-2.0}" \
  env.object_airborne_reward_weight="${OBJECT_AIRBORNE_REWARD_WEIGHT:-2.0}" \
  env.object_airborne_clearance="${OBJECT_AIRBORNE_CLEARANCE:-0.01}" \
  env.require_cross_link_pinch="${REQUIRE_CROSS_LINK_PINCH:-true}" \
  env.scissors_handle_contact_gate="${SCISSORS_HANDLE_CONTACT_GATE:-true}" \
  env.contact_tracking_sigma="${CONTACT_TRACKING_SIGMA:-0.04}" \
  env.contact_force_threshold="${CONTACT_FORCE_THRESHOLD:-0.5}" \
  env.contact_force_safe_threshold="${CONTACT_FORCE_SAFE_THRESHOLD:-20.0}" \
  env.contact_force_penalty_weight="${CONTACT_FORCE_PENALTY_WEIGHT:-0.02}" \
  env.residual_action_penalty_weight="${RESIDUAL_ACTION_PENALTY_WEIGHT:-0.01}" \
  env.randomize_start_phase="${RANDOMIZE_START_PHASE:-false}" \
  env.randomize_start_phase_fraction="${RANDOMIZE_START_PHASE_FRACTION:-1.0}" \
  env.random_start_phase_min="${RANDOM_START_PHASE_MIN:-0}" \
  env.random_start_phase_max="${RANDOM_START_PHASE_MAX:--1}" \
  env.random_start_episode_length="${RANDOM_START_EPISODE_LENGTH:-0}" \
  env.object_failure_distance="${OBJECT_FAILURE_DISTANCE:-0.25}" \
  env.object_failure_orientation="${OBJECT_FAILURE_ORIENTATION:-3.14}" \
  env.object_failure_articulation="${OBJECT_FAILURE_ARTICULATION:-3.14}" \
  env.disable_object_failure_termination="${DISABLE_OBJECT_FAILURE_TERMINATION:-false}" \
  agent.models.policy.min_log_std="${MIN_LOG_STD:--5.0}" \
  agent.models.policy.max_log_std="${MAX_LOG_STD:--1.0}" \
  agent.models.policy.initial_log_std="${INITIAL_LOG_STD:--3.5}" \
  agent.agent.entropy_loss_scale="${ENTROPY_LOSS_SCALE:-0.001}" \
  agent.agent.learning_rate="${LEARNING_RATE:-1.0e-4}" \
  "agent.agent.experiment.experiment_name=${RUN_LABEL}"

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
CHECKPOINT="${RUN_DIR}/checkpoints/best_agent.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name 'agent_*.pt' -print | sort -V | tail -n 1)"
fi
printf 'ARCTIC_SCISSORS_TRAINING_COMPLETE run_dir=%s checkpoint=%s\n' "${RUN_DIR}" "${CHECKPOINT}"
