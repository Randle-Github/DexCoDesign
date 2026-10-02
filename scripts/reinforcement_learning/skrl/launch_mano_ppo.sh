#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit one MANO residual-PPO run to SkyNet.

Usage:
  bash scripts/reinforcement_learning/skrl/launch_mano_ppo.sh \
    --mode single --reference REF.npz --object-usd OBJECT.usd [options]

Modes:
  single                 One hand, rigid object (default)
  bimanual               Two hands, rigid object
  articulated-single     One hand, articulated object
  articulated-bimanual   Two hands, articulated object

Required:
  --reference PATH          Primary hand's Isaac Lab reference NPZ
  --object-usd PATH         Object USD
  --second-reference PATH   Required for either bimanual mode

Options:
  --side left|right         Primary hand (default: left)
  --job-name NAME           Slurm job name (default: mano-ppo)
  --run-name NAME           PPO experiment name (default: job name)
  --output-dir DIR          Log and best-rollout directory (default: artifacts/rl_runs/NAME)
  --partition NAME          Slurm partition (default: overcap)
  --account NAME            Slurm account (default: rl2-lab)
  --qos NAME                Optional Slurm QoS
  --gpu-type NAME           One GPU of this type, or any (default: a40)
  --cpus N                  CPUs per job (default: 12)
  --mem SIZE                Slurm memory request (default: 64G)
  --time HH:MM:SS           Slurm time limit (default: 02:00:00)
  --num-envs N              Parallel Isaac environments (default: 256)
  --iterations N            PPO updates (default: 300)
  --seed N                  Random seed (default: 42)
  --checkpoint PATH         Resume from a checkpoint
  --dry-run                 Validate inputs and print the submission command

Environment overrides: SBATCH_BIN, CONDA_SH, CONDA_ENV (default: codesign).
The worker uses this same script, so Slurm runs Bash directly; no --wrap shell.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$PWD" "$1"; fi
}
positive_integer() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"; }

MODE=single
ORIGINAL_ARGS=("$@")
REFERENCE=""
SECOND_REFERENCE=""
OBJECT_USD=""
SIDE=left
JOB_NAME=mano-ppo
RUN_NAME=""
OUTPUT_DIR=""
PARTITION=overcap
ACCOUNT=rl2-lab
QOS=""
GPU_TYPE=a40
CPUS=12
MEM=64G
TIME_LIMIT=02:00:00
NUM_ENVS=256
ITERATIONS=300
SEED=42
CHECKPOINT=""
DRY_RUN=0

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --mode|--reference|--second-reference|--object-usd|--side|--job-name|--run-name|--output-dir|--partition|--account|--qos|--gpu-type|--cpus|--mem|--time|--num-envs|--iterations|--seed|--checkpoint)
      (( $# >= 2 )) || die "$1 needs a value"
      case "$1" in
        --mode) MODE="$2" ;;
        --reference) REFERENCE="$2" ;;
        --second-reference) SECOND_REFERENCE="$2" ;;
        --object-usd) OBJECT_USD="$2" ;;
        --side) SIDE="$2" ;;
        --job-name) JOB_NAME="$2" ;;
        --run-name) RUN_NAME="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --partition) PARTITION="$2" ;;
        --account) ACCOUNT="$2" ;;
        --qos) QOS="$2" ;;
        --gpu-type) GPU_TYPE="$2" ;;
        --cpus) CPUS="$2" ;;
        --mem) MEM="$2" ;;
        --time) TIME_LIMIT="$2" ;;
        --num-envs) NUM_ENVS="$2" ;;
        --iterations) ITERATIONS="$2" ;;
        --seed) SEED="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
      esac
      shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$MODE" in
  single|bimanual|articulated-single|articulated-bimanual) ;;
  *) die "unknown mode: $MODE" ;;
esac
[[ "$SIDE" == left || "$SIDE" == right ]] || die "--side must be left or right"
[[ -n "$REFERENCE" ]] || die "--reference is required"
[[ -n "$OBJECT_USD" ]] || die "--object-usd is required"
[[ "$JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid --job-name: $JOB_NAME"
RUN_NAME="${RUN_NAME:-$JOB_NAME}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid --run-name: $RUN_NAME"
[[ "$GPU_TYPE" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid --gpu-type: $GPU_TYPE"
positive_integer --cpus "$CPUS"
positive_integer --num-envs "$NUM_ENVS"
positive_integer --iterations "$ITERATIONS"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a nonnegative integer"

BIMANUAL=false
ARTICULATED=false
case "$MODE" in
  bimanual|articulated-bimanual) BIMANUAL=true ;;
esac
case "$MODE" in
  articulated-single|articulated-bimanual) ARTICULATED=true ;;
esac
if [[ "$BIMANUAL" == true ]]; then
  [[ -n "$SECOND_REFERENCE" ]] || die "--second-reference is required for $MODE"
elif [[ -n "$SECOND_REFERENCE" ]]; then
  die "--second-reference is only valid in a bimanual mode"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/scripts/reinforcement_learning/skrl/launch_mano_ppo.sh"
REFERENCE="$(absolute_path "$REFERENCE")"
OBJECT_USD="$(absolute_path "$OBJECT_USD")"
if [[ -n "$SECOND_REFERENCE" ]]; then SECOND_REFERENCE="$(absolute_path "$SECOND_REFERENCE")"; fi
if [[ -n "$CHECKPOINT" ]]; then CHECKPOINT="$(absolute_path "$CHECKPOINT")"; fi
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR:-$REPO_ROOT/artifacts/rl_runs/$RUN_NAME}")"

[[ -s "$REFERENCE" ]] || die "reference is missing or empty: $REFERENCE"
[[ -s "$OBJECT_USD" ]] || die "object USD is missing or empty: $OBJECT_USD"
if [[ "$BIMANUAL" == true ]]; then
  [[ -s "$SECOND_REFERENCE" ]] || die "second reference is missing or empty: $SECOND_REFERENCE"
fi
if [[ -n "$CHECKPOINT" ]]; then [[ -s "$CHECKPOINT" ]] || die "checkpoint is missing: $CHECKPOINT"; fi
[[ -s "$REPO_ROOT/artifacts/isaaclab_mano_residual/assets/mano_$SIDE.usd" ]] || die "MANO $SIDE USD is missing"
if [[ "$BIMANUAL" == true ]]; then
  OTHER_SIDE=left
  if [[ "$SIDE" == left ]]; then OTHER_SIDE=right; fi
  [[ -s "$REPO_ROOT/artifacts/isaaclab_mano_residual/assets/mano_$OTHER_SIDE.usd" ]] || die "MANO $OTHER_SIDE USD is missing"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SBATCH_BIN="${SBATCH_BIN:-}"
  if [[ -z "$SBATCH_BIN" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
      SBATCH_BIN="$(command -v sbatch)"
    elif [[ -x /opt/slurm/Ubuntu-20.04/current/bin/sbatch ]]; then
      SBATCH_BIN=/opt/slurm/Ubuntu-20.04/current/bin/sbatch
    else
      SBATCH_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin/sbatch
    fi
  fi
  command=(
    "$SBATCH_BIN" --parsable --job-name="$JOB_NAME" --partition="$PARTITION"
    --account="$ACCOUNT" --cpus-per-task="$CPUS" --mem="$MEM"
    --time="$TIME_LIMIT" --output="$OUTPUT_DIR/slurm-%j.out"
    --chdir="$PWD" --export=ALL
  )
  if [[ "$GPU_TYPE" == any ]]; then command+=(--gpus=1); else command+=(--gpus="$GPU_TYPE:1"); fi
  if [[ -n "$QOS" ]]; then command+=(--qos="$QOS"); fi
  submit_args=()
  for arg in "${ORIGINAL_ARGS[@]}"; do
    if [[ "$arg" != --dry-run ]]; then submit_args+=("$arg"); fi
  done
  command+=("$SCRIPT_PATH" "${submit_args[@]}")
  if (( DRY_RUN )); then
    printf 'mode=%s side=%s reference=%s object=%s output=%s\n' "$MODE" "$SIDE" "$REFERENCE" "$OBJECT_USD" "$OUTPUT_DIR"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
  fi
  [[ -x "$SBATCH_BIN" ]] || die "sbatch is unavailable: $SBATCH_BIN"
  mkdir -p "$OUTPUT_DIR"
  exec "${command[@]}"
fi

(( ! DRY_RUN )) || die "--dry-run cannot be used inside a Slurm job"
CONDA_SH="${CONDA_SH:-/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || die "Conda activation script is missing: $CONDA_SH"
# Slurm executes this file through its Bash shebang, never via /bin/sh --wrap.
source "$CONDA_SH"
conda activate "${CONDA_ENV:-codesign}"
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE="${WANDB_MODE:-disabled}"
export DEXCODESIGN_REFERENCE_PATH="$REFERENCE"
export DEXCODESIGN_OBJECT_USD_PATH="$OBJECT_USD"
export DEXCODESIGN_MANO_SIDE="$SIDE"
export HAND_BEST_ROLLOUT_PATH="$OUTPUT_DIR/best_rollout.npz"
if [[ "$BIMANUAL" == true ]]; then
  export DEXCODESIGN_BIMANUAL_REFERENCE_PATH="$SECOND_REFERENCE"
else
  unset DEXCODESIGN_BIMANUAL_REFERENCE_PATH || true
fi
export PYTHONPATH="$REPO_ROOT/source/dexcodesign:$REPO_ROOT/source/isaaclab_tasks:$REPO_ROOT/source/isaaclab_assets:$REPO_ROOT/source/isaaclab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$REPO_ROOT/artifacts/isaaclab_mano_residual/runtime_libs:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
printf 'job=%s commit=%s mode=%s side=%s reference=%s object=%s\n' \
  "$SLURM_JOB_ID" "$(git rev-parse --short HEAD)" "$MODE" "$SIDE" "$REFERENCE" "$OBJECT_USD"
train_command=(
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py
  --task DexCoDesign-MANO-Residual-Direct-v0 --algorithm PPO
  --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" --seed "$SEED"
)
if [[ -n "$CHECKPOINT" ]]; then train_command+=(--checkpoint "$CHECKPOINT"); fi
train_command+=(
  --headless "env.articulate_mode=$ARTICULATED" "env.bimanual_mode=$BIMANUAL"
  "agent.agent.experiment.experiment_name=$RUN_NAME"
)
if [[ "$ARTICULATED" == true ]]; then
  train_command+=(
    env.articulated_object_fix_root_link=false
    env.require_free_object_root=true
    env.require_dynamic_object_root=true
  )
fi
exec "${train_command[@]}"
