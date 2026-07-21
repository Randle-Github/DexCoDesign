#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
cd "$ROOT"

temp/.venv/bin/python temp/exp1/grammar_v1/build_bundle_library.py
temp/.venv/bin/python temp/exp1/grammar_v1/generate_hands.py
temp/.venv/bin/python temp/exp1/grammar_v1/compile_meshes.py
temp/.venv/bin/mjpython temp/exp1/grammar_v1/render_hands.py

