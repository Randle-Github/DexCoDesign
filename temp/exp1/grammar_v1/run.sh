#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
cd "$ROOT"

PALM_GENERATION_MODE="${PALM_GENERATION_MODE:-hybrid_source_topology}"
PALM_TRANSVERSE_ARCH="${PALM_TRANSVERSE_ARCH:-0.0}"
PALM_LONGITUDINAL_ARCH="${PALM_LONGITUDINAL_ARCH:-0.0}"
PALM_CENTRAL_CUP="${PALM_CENTRAL_CUP:-0.0}"

temp/.venv/bin/python temp/exp1/grammar_v1/build_bundle_library.py
temp/.venv/bin/python temp/exp1/grammar_v1/generate_hands.py
temp/.venv/bin/python temp/exp1/grammar_v1/compile_meshes.py \
  --palm-generation-mode "$PALM_GENERATION_MODE" \
  --transverse-arch "$PALM_TRANSVERSE_ARCH" \
  --longitudinal-arch "$PALM_LONGITUDINAL_ARCH" \
  --central-cup "$PALM_CENTRAL_CUP"
temp/.venv/bin/mjpython temp/exp1/grammar_v1/render_hands.py
