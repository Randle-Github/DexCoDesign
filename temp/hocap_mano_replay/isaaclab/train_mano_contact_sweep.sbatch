#!/usr/bin/env bash
#SBATCH --job-name=mano-contact
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/isaaclab_mano_residual/sweeps/slurm-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual"
CONTACT_WEIGHT="${CONTACT_WEIGHT:?CONTACT_WEIGHT must be set}"
RUN_SEED="${RUN_SEED:-42}"
WEIGHT_LABEL="${CONTACT_WEIGHT//./p}"
RUN_LABEL="contact_w${WEIGHT_LABEL}_seed${RUN_SEED}"
RESULT_DIR="${OUTPUT_ROOT}/sweeps/${RUN_LABEL}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${OUTPUT_ROOT}/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 1024 \
  --max_iterations 300 \
  --seed "${RUN_SEED}" \
  --headless \
  "env.contact_reward_weight=${CONTACT_WEIGHT}" \
  "agent.agent.experiment.experiment_name=${RUN_LABEL}"

RUN_DIR="$(find logs/skrl/mano_residual -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_LABEL}" -print | sort | tail -n 1)"
CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -type f -name '*.pt' -print | sort -V | tail -n 1)"
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --video \
  --video_length 445 \
  --headless \
  "env.contact_reward_weight=${CONTACT_WEIGHT}"

python temp/hocap_mano_replay/isaaclab/export_training_artifacts.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RESULT_DIR}" \
  --contact-weight "${CONTACT_WEIGHT}"
