#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"
temp/.venv/bin/python temp/exp1/prepare_dataset.py
/opt/anaconda3/envs/dexretarget/bin/python temp/exp1/train.py
/opt/anaconda3/envs/dexretarget/bin/python temp/exp1/generate.py --count 100
temp/.venv/bin/python temp/exp1/build_mesh_library.py
temp/.venv/bin/python temp/exp1/decompose_geometry.py
/opt/anaconda3/envs/dexretarget/bin/python temp/exp1/view_generated.py --check

echo
echo "Headless pipeline passed. Open the interactive gallery with:"
echo "/opt/anaconda3/envs/dexretarget/bin/mjpython temp/exp1/view_generated.py --page 1"
