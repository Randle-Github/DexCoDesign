#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSET_ROOT="${REPO_ROOT}/artifacts/isaaclab_mano_residual/assets"
mkdir -p "${ASSET_ROOT}"

"${REPO_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/tools/convert_urdf.py" \
  "${REPO_ROOT}/assets/robot_hands/direct_motor/mano/left/hand.urdf" \
  "${ASSET_ROOT}/mano_left.usd" \
  --joint-stiffness 120 \
  --joint-damping 8 \
  --joint-target-type position \
  --headless

"${REPO_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/tools/convert_mesh.py" \
  "${REPO_ROOT}/temp/hocap_mano_replay/data/subset/models/G04_1/cleaned_mesh_2000.obj" \
  "${ASSET_ROOT}/g04_1.usd" \
  --make-instanceable \
  --collision-approximation convexHull \
  --mass 0.15 \
  --headless
