#!/usr/bin/env bash
#SBATCH --job-name=scissors-render
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny,gundam,baymax,qt-1,megazord
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_rollout/render-%j.out

set -euo pipefail

ROOT="${DEXCODESIGN_REPO_ROOT:-/coc/flash7/yliu3735/workspace/DexCoDesign}"
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p artifacts/arctic_scissors_rollout
export TMPDIR="$ROOT/artifacts/isaac_tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"
extra_args=()
if [[ -n "${KIT_ARGS:-}" ]]; then
  extra_args+=("--kit_args=$KIT_ARGS")
fi
if [[ -n "${EXPERIENCE:-}" ]]; then
  extra_args+=("--experience" "$EXPERIENCE")
fi

render_args=(
  -p scripts/datasets/arctic/render_arctic_policy_rollout_isaac.py
  --rollout "${ROLLOUT_PATH:-$ROOT/artifacts/arctic_scissors_rollout/right_contactcore_ppo_rollout.npz}"
  --hand-usd "$ROOT/artifacts/isaaclab_mano_residual/assets/mano_right.usd"
  --object-usd "${OBJECT_USD_PATH:-$ROOT/artifacts/arctic_mano_preview/assets/objects/scissors.usd}"
  --output "${VIDEO_PATH:-$ROOT/artifacts/arctic_scissors_rollout/right_contactcore_ppo_rollout.mp4}"
  --fps 30
  --enable_cameras
  "${extra_args[@]}"
)
if [[ -n "${SECOND_HAND_USD_PATH:-}" ]]; then
  render_args+=(--second-hand-usd "${SECOND_HAND_USD_PATH}")
fi
if [[ "${GUI_RENDER:-0}" == "1" ]]; then
  xvfb-run -a -s "-screen 0 1280x960x24" ./isaaclab.sh "${render_args[@]}"
else
  ./isaaclab.sh "${render_args[@]}" --headless
fi
