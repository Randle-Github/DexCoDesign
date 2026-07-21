# Grammar v1: structure and visual validation

This directory is the first executable prototype of the Design Grammar in
[`assets/representation.md`](../../../assets/representation.md). It replaces the
legacy free crossover/regression logic in `strict_v2` for graph and visual
validation.

## Hard rules

- A hand has at most five active fingers.
- `ADD_FINGER` is currently a strict 4-to-5 operation on a real four-finger
  palm seed.
- Increasing DoF replaces a complete source-valid finger mechanism bundle; it
  never inserts an arbitrary joint or donor link.
- Every link candidate is owned by the selected mechanism bundle.
- A repeated source-local motor type may select only candidates it originally
  drove under the same semantic/depth interface.
- No cross-vendor motor equivalence is inferred from public URDF joint names or
  appearance.
- The same palm affine transform is applied to the palm visual and every
  finger-root connector.
- Every generated graph must be connected and acyclic.

Public robot descriptions generally do not expose trustworthy physical motor
part numbers or complete transmission data. The current motor classes are
therefore conservative source-local identifiers, not claims that two physical
motors are equivalent. A hardware BOM should replace these identifiers before
manufacturability or torque constraints are evaluated.

## Pipeline

```text
strict_v2 canonical source graphs
  -> source-local mechanism bundle database
  -> grammar-constrained HandIR samples
  -> source-owned compound-part mesh assembly
  -> MuJoCo tiled render
  -> one 100-hand image
```

Run the complete deterministic pipeline from the repository root:

```bash
temp/exp1/grammar_v1/run.sh
```

The renderer requires the local MuJoCo environment in `temp/.venv`. On macOS,
run the rendering stage through `mjpython` so Metal can create an OpenGL
context.

## Main outputs

- `outputs/mechanism_bundle_library.json`: source bundles, candidates, motor
  equivalence policy, and rejected contaminated source blocks.
- `outputs/generated_hand_ir.json`: 100 generated graph/transform records.
- `outputs/grammar_audit.json`: topology, DoF, ownership, and connector audits.
- `outputs/compiled_hands.json`: exact visual-part assembly manifest.
- `outputs/grammar_100_hands.png`: all 100 hands in one MuJoCo-rendered image.

Latest deterministic audit:

```text
finger count                 4-5
DoF                          11-22
4-to-5 additions             50
increased-DoF designs        89
connected / acyclic          100 / 100
invalid motor-link binding   0
invalid bundle ownership     0
palm-slot connector error    0
```

## Scope boundary

This is a graph and visual-assembly prototype. It does not yet prove valid
contact physics or manufacturability. Collision recipes, inertia, physical
motor/BOM verification, joint-sweep clearance, wrist/base integration, and
Isaac Lab export/reload validation belong in the deterministic compiler stage.

