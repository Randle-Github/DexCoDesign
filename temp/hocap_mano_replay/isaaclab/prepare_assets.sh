#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSET_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets"
mkdir -p "${ASSET_ROOT}"

"${REPO_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/tools/convert_urdf.py" \
  "${REPO_ROOT}/assets/robot_hands/direct_motor/mano/left/hand.urdf" \
  "${ASSET_ROOT}/mano_left.usd" \
  --fix-base \
  --joint-stiffness 300 \
  --joint-damping 34.6410162 \
  --joint-target-type position \
  --headless

"${REPO_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/temp/hocap_mano_replay/isaaclab/prepare_mano_visuals.py" \
  --urdf "${REPO_ROOT}/assets/robot_hands/direct_motor/mano/left/hand.urdf" \
  --output-dir "${ASSET_ROOT}/mano_visuals" \
  --headless

"${REPO_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/tools/convert_mesh.py" \
  "${REPO_ROOT}/temp/hocap_mano_replay/data/subset/models/G04_1/cleaned_mesh_2000.obj" \
  "${ASSET_ROOT}/g04_1.usd" \
  --collision-approximation convexHull \
  --mass 0.015 \
  --headless

touch "${ASSET_ROOT}/.egoengine_actuators_v3"
