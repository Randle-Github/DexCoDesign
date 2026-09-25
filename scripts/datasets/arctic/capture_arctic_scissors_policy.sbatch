#!/usr/bin/env bash
#SBATCH --job-name=scissors-capture
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --exclude=gundam,baymax,qt-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/capture-%j.out

set -euo pipefail

ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_scissors_rollout

export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE=disabled
export DEXCODESIGN_MANO_SIDE=right
export DEXCODESIGN_REFERENCE_PATH="${REFERENCE_PATH:-$ROOT/artifacts/arctic_mano_dynamic/00_scissors/isaaclab_reference_right_fixedtip_dynamic_tabletop_30hz.npz}"
export DEXCODESIGN_OBJECT_USD_PATH="${OBJECT_USD_PATH:-$ROOT/artifacts/arctic_mano_preview/assets/objects/scissors_vhacd.usd}"
export LD_LIBRARY_PATH="$ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"
export HAND_BEST_ROLLOUT_PATH="${ROLLOUT_PATH:-$ROOT/artifacts/arctic_scissors_rollout/right_contactcore_ppo_rollout.npz}"
rm -f "$HAND_BEST_ROLLOUT_PATH"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT:?Set CHECKPOINT to the trained policy path}" \
  --max_steps "${MAX_STEPS:-700}" \
  --headless \
  env.articulate_mode=true \
  env.exponential_pose_reward=true \
  env.pose_tracking_reward_scale="${POSE_REWARD_SCALE:-4.0}" \
  env.articulated_object_fix_root_link=false \
  env.object_disable_gravity="${OBJECT_DISABLE_GRAVITY:-false}" \
  env.require_free_object_root=true \
  env.require_dynamic_object_root=true \
  env.require_cross_link_pinch="${REQUIRE_CROSS_LINK_PINCH:-true}" \
  env.contact_tracking_sigma="${CONTACT_TRACKING_SIGMA:-0.04}" \
  env.contact_force_threshold="${CONTACT_FORCE_THRESHOLD:-0.5}" \
  env.residual_root_position_scale="${ROOT_POSITION_SCALE:-0.03}" \
  env.residual_root_rotation_scale="${ROOT_ROTATION_SCALE:-0.10}" \
  env.residual_finger_scale="${FINGER_SCALE:-0.30}" \
  env.randomize_start_phase=false \
  env.log_rollout_diagnostics=true

test -s "$HAND_BEST_ROLLOUT_PATH"
printf 'ARCTIC_SCISSORS_CAPTURE_COMPLETE rollout=%s\n' "$HAND_BEST_ROLLOUT_PATH"
