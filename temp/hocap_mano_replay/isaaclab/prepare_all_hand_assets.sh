#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual"
PREPARED_ROOT="${OUTPUT_ROOT}/prepared"
ASSET_ROOT="${OUTPUT_ROOT}/assets"
CAPTURE="${REPO_ROOT}/temp/hocap_mano_replay/data/subset/subject_7/20231022_192832/isaaclab_reference.npz"
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
if [[ -n "${HAND_IDS_OVERRIDE:-}" ]]; then
  read -r -a HAND_IDS <<< "${HAND_IDS_OVERRIDE}"
fi

if [[ ! -f "${CAPTURE}" ]]; then
  printf 'Missing original Human/MANO reference: %s\n' "${CAPTURE}" >&2
  exit 1
fi

mkdir -p "${PREPARED_ROOT}" "${ASSET_ROOT}"
if [[ ! -f "${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/.egoengine_actuators_v3" \
   || ! -f "${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets/.object_convex_decomposition_v1" ]]; then
  bash "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/prepare_assets.sh"
fi

# Keep the MuJoCo-only reference builder isolated from the user's conda
# environment. Isaac Lab itself does not depend on this directory.
REFERENCE_PYTHON_DEPS="${OUTPUT_ROOT}/reference_python_deps"
mkdir -p "${REFERENCE_PYTHON_DEPS}"
ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${REFERENCE_PYTHON_DEPS}:${PYTHONPATH:-}"
if ! python -c 'import mujoco' >/dev/null 2>&1; then
  python -m pip install \
    --disable-pip-version-check \
    --target "${REFERENCE_PYTHON_DEPS}" \
    "mujoco==3.3.7"
fi

python "${REPO_ROOT}/temp/hocap_mano_replay/scripts/retarget_captured_success_all_hands.py" \
  --capture "${CAPTURE}" \
  --max-frames 446 \
  --solve-only

python "${REPO_ROOT}/temp/hocap_mano_replay/scripts/prepare_all_hand_rl_references.py" \
  --capture "${CAPTURE}" \
  --output-dir "${PREPARED_ROOT}"

python "${REPO_ROOT}/temp/hocap_mano_replay/scripts/audit_collision_coverage.py" \
  --prepared-root "${PREPARED_ROOT}" \
  --object-mesh "${REPO_ROOT}/temp/hocap_mano_replay/data/subset/models/G04_1/cleaned_mesh_2000.obj" \
  --output "${OUTPUT_ROOT}/collision_coverage_audit.json" \
  --require-complete-prepared

# Do not expose the isolated reference-builder packages (notably its NumPy)
# to Isaac Sim's tightly pinned Python runtime.
export PYTHONPATH="${ORIGINAL_PYTHONPATH}"

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
  physics_usd="${hand_root}/configuration/hand_physics.usd"
  if [[ ! -f "${physics_usd}" ]] || [[ "$(wc -c < "${physics_usd}")" -le 1024 ]]; then
    printf 'HAND_ASSET_FAILED hand_id=%s physics_usd=%s\n' \
      "${hand_id}" "${physics_usd}" >&2
    exit 1
  fi
  printf 'HAND_ASSET_READY hand_id=%s usd=%s\n' \
    "${hand_id}" "${hand_root}/hand.usd"
done

if [[ -z "${HAND_IDS_OVERRIDE:-}" ]]; then
  touch "${ASSET_ROOT}/.all_hands_original_human_v2"
  printf 'ALL_HAND_ASSETS_READY count=%s\n' "${#HAND_IDS[@]}"
else
  printf 'SELECTED_HAND_ASSETS_READY count=%s\n' "${#HAND_IDS[@]}"
fi
