#!/usr/bin/env bash
#SBATCH --job-name=wuji-hybrid-sac
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_morphology_hybrid_sac/slurm-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/run}"
POPULATION="${POPULATION:-4096}"
PHYSICS_BATCH_SIZE="${PHYSICS_BATCH_SIZE:-4096}"
GENERATIONS="${GENERATIONS:-20}"
SAC_UPDATES="${SAC_UPDATES:-400}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-1024}"
UNIFORM_FRACTION="${UNIFORM_FRACTION:-0.15}"
ELITE_MUTATION_FRACTION="${ELITE_MUTATION_FRACTION:-0.30}"
ELITE_REPLAY_FRACTION="${ELITE_REPLAY_FRACTION:-0.05}"
ELITE_MUTATION_SIGMA="${ELITE_MUTATION_SIGMA:-0.12}"
REWARD_SCALE="${REWARD_SCALE:-0.01}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-skrl}"
RETARGET_ITERATIONS="${RETARGET_ITERATIONS:-4}"
KIT_ARGS="${KIT_ARGS:---/log/level=error}"
CONTINUE_AFTER_SUCCESS="${CONTINUE_AFTER_SUCCESS:-0}"
BANK_ROOT="${BANK_ROOT:-${REPO_ROOT}/artifacts/wuji_physx_search/palm_prototype_bank_32_expanded_0p70}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
EXTRA_ARGS=()
if [[ "${CONTINUE_AFTER_SUCCESS}" == "1" ]]; then
  EXTRA_ARGS+=(--continue-after-success)
fi
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py \
  --output-root "${OUTPUT_ROOT}" \
  --prototype-bank-root "${BANK_ROOT}" \
  --seed-trajectory "${SEED_TRAJECTORY}" \
  --population "${POPULATION}" \
  --physics-batch-size "${PHYSICS_BATCH_SIZE}" \
  --generations "${GENERATIONS}" \
  --sac-updates "${SAC_UPDATES}" \
  --sac-batch-size "${SAC_BATCH_SIZE}" \
  --uniform-fraction "${UNIFORM_FRACTION}" \
  --elite-mutation-fraction "${ELITE_MUTATION_FRACTION}" \
  --elite-replay-fraction "${ELITE_REPLAY_FRACTION}" \
  --elite-mutation-sigma "${ELITE_MUTATION_SIGMA}" \
  --reward-scale "${REWARD_SCALE}" \
  --optimizer-backend "${OPTIMIZER_BACKEND}" \
  --retarget-iterations "${RETARGET_ITERATIONS}" \
  "${EXTRA_ARGS[@]}" \
  --kit_args="${KIT_ARGS}" \
  --headless
