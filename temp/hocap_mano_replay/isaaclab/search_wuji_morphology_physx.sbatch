#!/usr/bin/env bash
#SBATCH --job-name=wuji-physx-search
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/search-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
POPULATION="${POPULATION:-256}"
GENERATIONS="${GENERATIONS:-5}"
WORKERS="${WORKERS:-32}"
GENERATION_SEED="${GENERATION_SEED:-20260805}"
SEED_TRAJECTORY="${SEED_TRAJECTORY:-${REPO_ROOT}/artifacts/wuji_morphology_search/wuji_v2_cem_rl_reward/source/retarget_physics/ik/cache/wuji_source_ik.npz}"
STATE="${OUTPUT_ROOT}/cem_state.json"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES

previous_results=""
for ((generation=0; generation<GENERATIONS; generation++)); do
  generation_root="${OUTPUT_ROOT}/generation_$(printf '%02d' "${generation}")"
  vectors="${generation_root}/vectors.npy"
  retarget="${generation_root}/gpu_retarget_all.npz"
  prepared="${generation_root}/prepared"
  results="${generation_root}/physx_results.json"
  mkdir -p "${generation_root}"

  sample_command=(python temp/hocap_mano_replay/scripts/update_gpu_wuji_cem.py
    --state "${STATE}" --output-vectors "${vectors}"
    --population "${POPULATION}" --seed "${GENERATION_SEED}")
  if [[ -n "${previous_results}" ]]; then
    sample_command+=(--exact-summary "${previous_results}")
  fi
  "${sample_command[@]}"

  python temp/hocap_mano_replay/scripts/gpu_wuji_retarget.py \
    --seed-trajectory "${SEED_TRAJECTORY}" \
    --vectors "${vectors}" \
    --output "${retarget}" \
    --iterations 4 \
    --candidate-chunk 128 \
    --device cuda \
    --save-trajectories \
    --skip-proxy

  python temp/hocap_mano_replay/scripts/prepare_wuji_morphology_physx_batch.py \
    "${retarget}" \
    --output-root "${prepared}" \
    --workers "${WORKERS}"

  ./isaaclab.sh -p \
    temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
    "${prepared}/physx_batch_manifest.json" \
    --output "${results}" \
    --convert-urdf \
    --headless

  python - "${results}" "${POPULATION}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
assert result["backend"] == "isaaclab_physx_gpu"
assert result["all_candidates_physically_evaluated"] is True
assert result["proxy_used"] is False
assert result["top_k_prefilter_used"] is False
assert result["candidate_count"] == expected
assert result["completed"] == expected
print(
    "WUJI_PHYSX_GENERATION_COMPLETE "
    f"candidates={expected}/{expected} "
    f"best_reward={result['best']['total_reward']:.9f} "
    f"best_phase={result['best']['phase']}/445",
    flush=True,
)
PY
  previous_results="${results}"
done

printf 'WUJI_PHYSX_SEARCH_COMPLETE generations=%s population=%s output=%s\n' \
  "${GENERATIONS}" "${POPULATION}" "${OUTPUT_ROOT}"
