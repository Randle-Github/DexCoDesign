# DexCoDesign morphology pipeline

The production pipeline lives in `dexcodesign.morphology` and has no
dependency on `temp/`.

```text
source URDF/MJCF/USD companion
        ↓
bilateral standard URDF
        ↓
active direct joint or passive mimic joint
        ↓
fixed-joint-contracted rigid-link graph
        ↓
source-local motor/link grammar
        ↓
graph morphology edits
        ↓
source-topology palm deformation + rigid-link mesh deformation
        ↓
compiled 100-hand library
```

Run from the repository root:

```bash
.venv-morphology/bin/python scripts/dexcodesign/generate_hand_morphologies.py
```

Add `--render` for the MuJoCo contact sheet, `--resume` after an interrupted
mesh build, or `--rebuild-reference` after changing a source URDF or the
semantic scaffold.

Outputs are placed in `artifacts/hand_morphology/`.
