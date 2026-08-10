#!/usr/bin/env bash
#SBATCH --job-name=wuji-success-video
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_morphology_hybrid_sac/success-video-%j.out

set -euo pipefail

REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
SEARCH_ROOT="${SEARCH_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/hybrid_sac_expanded_exact_4096x20_seed20260806_retry1}"
GENERATION="${GENERATION:-0}"
GENERATION_DIR="$(printf 'generation_%03d' "${GENERATION}")"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-3527}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/wuji_morphology_hybrid_sac/success_candidate_003527}"
PREPARED="${OUTPUT_ROOT}/prepared"
MANIFEST="${PREPARED}/physx_batch_manifest.json"
SEARCH_MANIFEST="${SEARCH_ROOT}/${GENERATION_DIR}/prepared/physx_batch_manifest.json"
CAPTURE_MANIFEST="${OUTPUT_ROOT}/capture_replicas_manifest.json"
CAPTURE="${OUTPUT_ROOT}/successful_rollout.npz"
CAPTURE_REPLICAS="${CAPTURE_REPLICAS:-1024}"

source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=1

if [[ "${RENDER_ONLY:-0}" != "1" ]]; then
  RETARGET_BATCH="${SEARCH_ROOT}/${GENERATION_DIR}/gpu_retarget_all.npz"
  if [[ -s "${RETARGET_BATCH}" ]]; then
    python temp/hocap_mano_replay/scripts/prepare_wuji_morphology_physx_batch.py \
      "${RETARGET_BATCH}" \
      --output-root "${PREPARED}" \
      --offset "${CANDIDATE_INDEX}" \
      --limit 1 \
      --workers 1 \
      --include-visuals
  else
    # Persistent-scene searches intentionally store a fixed retarget once and
    # only materialize collision overlays per candidate. Reconstruct the one
    # selected visual mesh from its vector without running retarget again.
    python temp/hocap_mano_replay/scripts/prepare_wuji_selected_visual_manifest.py \
      "${SEARCH_MANIFEST}" \
      --index "${CANDIDATE_INDEX}" \
      --output-root "${PREPARED}"
  fi

  ./isaaclab.sh -p \
    temp/hocap_mano_replay/isaaclab/convert_wuji_morphology_physx_batch.py \
    "${MANIFEST}" --headless

  python temp/hocap_mano_replay/scripts/repeat_wuji_morphology_manifest.py \
    "${MANIFEST}" "${CAPTURE_MANIFEST}" \
    --index 0 \
    --repeats "${CAPTURE_REPLICAS}"

  # Capture the selected candidate's fully materialized mesh.  It is generated
  # from the exact saved vector and avoids mutating/reusing a shared prototype
  # USD across replicas. Replicas expose GPU PhysX contact nondeterminism.
  export DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST="${CAPTURE_MANIFEST}"
  export HAND_SUCCESS_TRAJECTORY_PATH="${CAPTURE}"
  export HAND_BEST_ROLLOUT_PATH="${OUTPUT_ROOT}/farthest_rollout.npz"
  ./isaaclab.sh -p \
    temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
    "${CAPTURE_MANIFEST}" \
    --output "${OUTPUT_ROOT}/capture_evaluation.json" \
    --no-seed \
    --headless
fi
test -s "${CAPTURE}"

unset HAND_SUCCESS_TRAJECTORY_PATH HAND_BEST_ROLLOUT_PATH
export DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST="${MANIFEST}"
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/replay_hand_rollout.py \
  --trajectory "${CAPTURE}" \
  --output-dir "${OUTPUT_ROOT}" \
  --enable_cameras \
  --headless

test -s "${OUTPUT_ROOT}/farthest_rollout.mp4"
mv "${OUTPUT_ROOT}/farthest_rollout.mp4" "${OUTPUT_ROOT}/successful_rollout.mp4"
printf 'WUJI_MORPHOLOGY_SUCCESS_VIDEO_COMPLETE candidate=%s output=%s\n' \
  "${CANDIDATE_INDEX}" "${OUTPUT_ROOT}/successful_rollout.mp4"
