#!/usr/bin/env bash
#SBATCH --job-name=all-hand-ppo
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_all_hands_residual/slurm/train-%A_%a.out

set -euo pipefail

HAND_IDS=(
  ability_hand
  schunk_svh
  wuji_hand_2
  sharpa_wave_01
  tesollo_dg5f
  unitree_dex5_1
  robotera_xhand1
  orca_hand_v2
  shadow_hand_e
  allegro_hand_v5
  midas_hand
  ruka_v2
  inspire_rh56dfx
  mano
)
HAND_ID="${HAND_ID_OVERRIDE:-${HAND_IDS[${SLURM_ARRAY_TASK_ID}]}}"
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual"
RESULT_SET="${RESULT_SET:-results}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_SET}/${HAND_ID}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300}"
TOTAL_ITERATIONS_BUDGET="${TOTAL_ITERATIONS_BUDGET:-${MAX_ITERATIONS}}"
RESUME_FROM_RESULT="${RESUME_FROM_RESULT:-0}"
RUN_SEED="${RUN_SEED:-42}"
NUM_ENVS_PER_HAND="${NUM_ENVS_PER_HAND:-4096}"
ALLOW_UNSAFE_HIGH_ENVS="${ALLOW_UNSAFE_HIGH_ENVS:-0}"
RUN_LABEL="${HAND_ID}_max${TOTAL_ITERATIONS_BUDGET}_seed${RUN_SEED}"
SUCCESS_TRAJECTORY="${RESULT_DIR}/successful_rollout.npz"
BEST_TRAJECTORY="${RESULT_DIR}/farthest_rollout.npz"

if [[ ! "${NUM_ENVS_PER_HAND}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_ENVS_PER_HAND must be a positive integer, got %s.\n' \
    "${NUM_ENVS_PER_HAND}" >&2
  exit 2
fi
if (( NUM_ENVS_PER_HAND > 8192 )) && [[ "${ALLOW_UNSAFE_HIGH_ENVS}" != "1" ]]; then
  printf 'Refusing NUM_ENVS_PER_HAND=%s: MANO at 16384 overflowed the PhysX GPU patch buffer. ' \
    "${NUM_ENVS_PER_HAND}" >&2
  printf 'Increase and validate the PhysX contact buffers first, or explicitly set ALLOW_UNSAFE_HIGH_ENVS=1.\n' >&2
  exit 2
fi

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_HAND_ID="${HAND_ID}"
export HAND_SUCCESS_TRAJECTORY_PATH="${SUCCESS_TRAJECTORY}"
export HAND_BEST_ROLLOUT_PATH="${BEST_TRAJECTORY}"
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${RESULT_DIR}" "${OUTPUT_ROOT}/slurm"
printf '{"hand_id":"%s","num_envs":%s,"max_iterations":%s,"total_iterations_budget":%s,"seed":%s,"gpu_concurrency":%s,"slurm_job_id":"%s","slurm_array_task_id":"%s"}\n' \
  "${HAND_ID}" "${NUM_ENVS_PER_HAND}" "${MAX_ITERATIONS}" \
  "${TOTAL_ITERATIONS_BUDGET}" "${RUN_SEED}" "${GPU_CONCURRENCY:-1}" \
  "${SLURM_JOB_ID:-}" \
  "${SLURM_ARRAY_TASK_ID}" > "${RESULT_DIR}/training_config.json"

if [[ ! -f "${OUTPUT_ROOT}/assets/.all_hands_original_human_v2" ]]; then
  printf 'All-hand assets are not ready: %s\n' "${OUTPUT_ROOT}/assets" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/.object_convex_decomposition_v1" ]]; then
  printf 'Convex-decomposition object asset is not ready. Run prepare_assets.sh first.\n' >&2
  exit 1
fi

TRAIN_ARGS=()
if [[ "${RESUME_FROM_RESULT}" == "1" ]]; then
  PREVIOUS_RUN_DIR="$(<"${RESULT_DIR}/run_dir.txt")"
  INITIAL_CHECKPOINT="${PREVIOUS_RUN_DIR}/checkpoints/best_agent.pt"
  if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
    INITIAL_CHECKPOINT="$(
      find "${PREVIOUS_RUN_DIR}/checkpoints" -type f -name 'agent_*.pt' -print \
        | sort -V | tail -n 1
    )"
  fi
  if [[ -z "${INITIAL_CHECKPOINT}" || ! -f "${INITIAL_CHECKPOINT}" ]]; then
    printf 'No checkpoint available to resume %s from %s.\n' \
      "${HAND_ID}" "${PREVIOUS_RUN_DIR}" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--checkpoint "${INITIAL_CHECKPOINT}")
  RUN_LABEL="${HAND_ID}_continued_max${TOTAL_ITERATIONS_BUDGET}_seed${RUN_SEED}"
  printf 'HAND_RESUME hand_id=%s checkpoint=%s additional_iterations=%s total_budget=%s\n' \
    "${HAND_ID}" "${INITIAL_CHECKPOINT}" "${MAX_ITERATIONS}" \
    "${TOTAL_ITERATIONS_BUDGET}"
fi

set +e
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-Hand-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs "${NUM_ENVS_PER_HAND}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed "${RUN_SEED}" \
  --headless \
  "${TRAIN_ARGS[@]}" \
  "agent.agent.experiment.experiment_name=${RUN_LABEL}"
TRAIN_EXIT=$?
set -e

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
if [[ -n "${RUN_DIR}" ]]; then
  printf '%s\n' "${RUN_DIR}" > "${RESULT_DIR}/run_dir.txt"
fi
if [[ ! -f "${SUCCESS_TRAJECTORY}" && ! -f "${BEST_TRAJECTORY}" ]]; then
  printf 'Training produced no captured rollout for %s (exit=%s).\n' \
    "${HAND_ID}" "${TRAIN_EXIT}" >&2
  exit "${TRAIN_EXIT:-1}"
fi

python temp/hocap_mano_replay/isaaclab/summarize_hand_training.py \
  --hand-id "${HAND_ID}" \
  --success "${SUCCESS_TRAJECTORY}" \
  --best "${BEST_TRAJECTORY}" \
  --run-dir "${RUN_DIR}" \
  --iterations "${TOTAL_ITERATIONS_BUDGET}" \
  --output "${RESULT_DIR}/training_status.json"

printf 'HAND_TRAINING_COMPLETE hand_id=%s exit=%s status=%s\n' \
  "${HAND_ID}" "${TRAIN_EXIT}" "${RESULT_DIR}/training_status.json"
