#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
cd "$ROOT"

temp/.venv/bin/python scripts/dexcodesign/build_reference_demo.py
temp/.venv/bin/mjpython scripts/dexcodesign/render_reference_demo.py
