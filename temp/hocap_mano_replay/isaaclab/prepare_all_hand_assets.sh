#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual"
PREPARED_ROOT="${OUTPUT_ROOT}/prepared"
ASSET_ROOT="${OUTPUT_ROOT}/assets"
CAPTURE="${REPO_ROOT}/artifacts/isaaclab_mano_residual/success_capture_fixed_clamp_reset0/successful_rollout.npz"
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
)

if [[ ! -f "${CAPTURE}" ]]; then
  printf 'Missing captured human trajectory: %s\n' "${CAPTURE}" >&2
  exit 1
fi

mkdir -p "${PREPARED_ROOT}" "${ASSET_ROOT}"
bash "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/prepare_assets.sh"

python "${REPO_ROOT}/temp/hocap_mano_replay/scripts/retarget_captured_success_all_hands.py" \
  --capture "${CAPTURE}" \
  --max-frames 446 \
  --solve-only

python "${REPO_ROOT}/temp/hocap_mano_replay/scripts/prepare_all_hand_rl_references.py" \
  --capture "${CAPTURE}" \
  --output-dir "${PREPARED_ROOT}"

for hand_id in "${HAND_IDS[@]}"; do
  hand_root="${ASSET_ROOT}/${hand_id}"
  mkdir -p "${hand_root}"
  "${REPO_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/tools/convert_urdf.py" \
    "${PREPARED_ROOT}/${hand_id}/hand_rl.urdf" \
    "${hand_root}/hand.usd" \
    --fix-base \
    --joint-stiffness 300 \
    --joint-damping 34.6410162 \
    --joint-target-type position \
    --headless
  printf 'HAND_ASSET_READY hand_id=%s usd=%s\n' \
    "${hand_id}" "${hand_root}/hand.usd"
done

touch "${ASSET_ROOT}/.all_hands_residual_v1"
printf 'ALL_HAND_ASSETS_READY count=%s\n' "${#HAND_IDS[@]}"
