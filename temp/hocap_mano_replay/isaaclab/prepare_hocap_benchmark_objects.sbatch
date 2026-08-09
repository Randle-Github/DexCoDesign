#!/usr/bin/env bash
#SBATCH --job-name=hocap-objects
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/objects-%A_%a.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
TASK_IDS=(g04_pick_place g15_vertical_local g06_rotation_heavy g19_mixed_reorientation g02_horizontal_reorientation g18_container_lift g18_variant_lift g01_large_reorientation)
OBJECT_IDS=(G04_1 G15_2 G06_3 G19_3 G02_1 G18_1 G18_4 G01_3)
INDEX="${SLURM_ARRAY_TASK_ID:?missing array index}"
TASK_ID="${TASK_IDS[INDEX]}"
OBJECT_ID="${OBJECT_IDS[INDEX]}"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/assets/${TASK_ID}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
RUNTIME_LIB_DIR="${REPO_ROOT}/artifacts/isaaclab_mano_residual/runtime_libs"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
mkdir -p "${OUTPUT_ROOT}"
./isaaclab.sh -p scripts/tools/convert_mesh.py \
  "temp/hocap_mano_replay/data/tasks/models/${OBJECT_ID}/cleaned_mesh_2000.obj" \
  "${OUTPUT_ROOT}/object.usd" \
  --collision-approximation convexDecomposition --mass 0.015 --headless
test -s "${OUTPUT_ROOT}/object.usd"
touch "${OUTPUT_ROOT}/.ready"
printf 'HOCAP_OBJECT_READY task=%s object=%s usd=%s\n' "${TASK_ID}" "${OBJECT_ID}" "${OUTPUT_ROOT}/object.usd"
