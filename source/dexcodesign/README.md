# DexCoDesign grammar/compiler core

This package is the model-free foundation of the current HandIR method.  It
contains:

- versioned HandIR and `MechanismBundle` schemas;
- a migration importer for the audited 14-hand canonical graph/rigid-part library;
- strict graph and source motor/link candidate validation;
- deterministic source-bound mesh compilation;
- twenty WUJI Hand 2 grammar-edit examples.

It intentionally contains no SAC, PPO, graph encoder, decoder, or learned mesh
model yet.

## Current hardware boundary

The migration importer can preserve the audited joint-to-child-link ownership,
but the legacy canonical graph does not contain complete vendor motor,
transmission, tendon, mimic, or equality metadata.  Such bundles are marked
`joint_bound_pending_native_hardware_parse`; they must not be presented as
resolved physical motor specifications.

The native URDF/MJCF/USD importer will replace this bridge and populate actual
source actuator/transmission bindings.  The HandIR, grammar, and compiler APIs
do not depend on the bridge.

## Reference demo

```bash
cd /Users/liuyangcen/workspace/DexCoDesign
temp/.venv/bin/python scripts/dexcodesign/build_reference_demo.py
temp/.venv/bin/mjpython scripts/dexcodesign/render_reference_demo.py
```

Outputs are written under `temp/design_grammar/outputs`.
