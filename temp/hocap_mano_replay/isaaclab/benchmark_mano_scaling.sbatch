#!/usr/bin/env bash
#SBATCH --job-name=mano-scale
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_scaling/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
RESULT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_scaling"
WORLD_SIZE="${WORLD_SIZE:-1}"
ENVS_PER_RANK="${ENVS_PER_RANK:?Set ENVS_PER_RANK}"
MAX_ITERATIONS="${MAX_ITERATIONS:?Set MAX_ITERATIONS}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"
RESULT_DIR="${RESULT_ROOT}/${RUN_LABEL}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export DEXCODESIGN_HAND_ID=mano
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${RESULT_DIR}" "${RESULT_ROOT}/slurm"

printf '{"world_size":%s,"envs_per_rank":%s,"global_envs":%s,"rollout":16,"iterations":%s,"seed":42}\n' \
  "${WORLD_SIZE}" "${ENVS_PER_RANK}" "$((WORLD_SIZE * ENVS_PER_RANK))" \
  "${MAX_ITERATIONS}" > "${RESULT_DIR}/benchmark_config.json"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 2 > "${RESULT_DIR}/gpu_samples.csv" &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true' EXIT

START_SECONDS=$(date +%s)
set +e
if [[ "${WORLD_SIZE}" == "1" ]]; then
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task DexCoDesign-MANO-Residual-Direct-v0 \
    --algorithm PPO \
    --num_envs "${ENVS_PER_RANK}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --seed 42 \
    --headless \
    "agent.agent.experiment.experiment_name=${RUN_LABEL}"
  TRAIN_EXIT=$?
else
  ./isaaclab.sh -p -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${WORLD_SIZE}" \
    scripts/reinforcement_learning/skrl/train.py \
    --distributed \
    --task DexCoDesign-MANO-Residual-Direct-v0 \
    --algorithm PPO \
    --num_envs "${ENVS_PER_RANK}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --seed 42 \
    --headless \
    "agent.agent.experiment.experiment_name=${RUN_LABEL}"
  TRAIN_EXIT=$?
fi
set -e
END_SECONDS=$(date +%s)
kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
trap - EXIT

ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))
GLOBAL_ENVS=$((WORLD_SIZE * ENVS_PER_RANK))
TRANSITIONS=$((GLOBAL_ENVS * 16 * MAX_ITERATIONS))
RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 \
  -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
printf '%s\n' "${RUN_DIR}" > "${RESULT_DIR}/run_dir.txt"
printf '{"exit_code":%s,"elapsed_seconds":%s,"transitions":%s,"samples_per_second":%.6f,"run_dir":"%s"}\n' \
  "${TRAIN_EXIT}" "${ELAPSED_SECONDS}" "${TRANSITIONS}" \
  "$(awk -v n="${TRANSITIONS}" -v t="${ELAPSED_SECONDS}" 'BEGIN {print n/t}')" \
  "${RUN_DIR}" > "${RESULT_DIR}/timing.json"

printf 'MANO_SCALING_COMPLETE label=%s world_size=%s envs_per_rank=%s global_envs=%s iterations=%s transitions=%s elapsed_seconds=%s exit=%s run_dir=%s\n' \
  "${RUN_LABEL}" "${WORLD_SIZE}" "${ENVS_PER_RANK}" "${GLOBAL_ENVS}" \
  "${MAX_ITERATIONS}" "${TRANSITIONS}" "${ELAPSED_SECONDS}" \
  "${TRAIN_EXIT}" "${RUN_DIR}"
exit "${TRAIN_EXIT}"
