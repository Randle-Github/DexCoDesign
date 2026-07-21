# Grammar v1: structure and visual validation

This directory is the first executable prototype of the Design Grammar in
[`assets/representation.md`](../../../assets/representation.md). It replaces the
legacy free crossover/regression logic in `strict_v2` for graph and visual
validation.

## Hard rules

- A hand has at most five active fingers.
- Finger count, topology, semantic role, and cyclic thumb/finger order are
  copied from the selected source hand. No finger is added, removed, or crossed
  past another finger. Attachment poses may move with an order-preserving palm edit.
- A generated hand has exactly one `seed_source`. Its palm and every finger,
  motor, link, tip, and visual mesh must come from that source hand.
- Cross-hand bundle replacement, cross-hand mesh crossover, and same-source
  finger permutation are all forbidden.
- Finger mesh edits are limited to bounded axial link length and transverse
  link radius scales. Palm shape and finger-base poses remain editable, but
  must preserve source cyclic order and all graph connectors.
- Every link candidate is owned by the selected mechanism bundle.
- A repeated source-local motor type may select only candidates it originally
  drove under the same semantic/depth interface.
- No cross-vendor motor equivalence is inferred from public URDF joint names or
  appearance.
- The generated HandIR graph is the only source of finger-root attachment
  translation, rotation, and joint axis.
- The current batch retains anthropomorphic, symmetric, and asymmetric palm
  edits, but maps target slots to the original cyclic finger order.
- Every palm starts from the selected source palm CAD mesh. Large radial edits
  use the graph-conditioned House convex hull only as a target deformation
  cage; they do not replace the visual palm with a newly extruded slab.
- Radial graph layouts interpolate the source finger-root arrangement with the
  House target before compiling either fingers or palm. The default House
  contribution is 52–66%; it increases only when required for motor clearance.
- Source-topology palm deformation is rejected if any planar vertex moves more
  than 30% of the source palm's maximum planar extent.
- Palm thickness coordinates and the wrist/root mount region are immutable.
  Non-anthropomorphic layouts reserve a mount exclusion sector so no generated
  finger slot competes with that mechanical interface.
- Non-digit base/transmission branches are copied into HandIR and compiled with
  identity transforms. For tendon/underactuated sources, palm/root and every
  proximal transmission housing are additionally marked geometry-locked.
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
  -> graph-conditioned palm + source-owned finger-part mesh assembly
  -> MuJoCo tiled render
  -> one 100-hand image
```

`generated_hand_ir.json` contains the authoritative attachment data in each
`finger_slot`:

```text
attachment_translation + attachment_rotation
                    ↓ (copied, never modified)
             AttachmentPatch
                    ↓
         palm outline / extrusion
```

The current canonical coordinate convention is XZ for the palm plane and Y
for thickness. It is explicit in `PalmGeometryParams`, rather than inferred
from a mesh bounding box.

## Palm generation modes

- `hybrid_source_topology` is the pipeline default. Tendon/underactuated hands
  are restricted to bounded anthropomorphic palm edits; mechanically direct
  hands may also use order-preserving symmetric/asymmetric layouts.
- `source_topology_house` computes the official-style motor-footprint hull as
  a target cage, then globally maps the source palm footprint to that cage.
  It keeps the original visual vertex/face topology and every local thickness
  coordinate. A disk around the source wrist/root mount is copied exactly,
  with a smooth transition annulus into the deformable palm body.
  The attachment graph and cage share the same source/House interpolation, so
  reducing shape change cannot detach fingers from the palm.
- `house_hull` follows the official House of Dextra generator: graph motor
  frames produce tangential grip footprints; a center grid prevents collapse;
  the 2D convex hull is extruded into a closed palm. Radius and center-grid size
  are calibrated from the selected original palm bounds, and thickness is
  copied from that palm exactly.
- `template_deform` preserves the original source palm vertices, faces, shell
  details, and thickness coordinates. It is restricted to small local
  anthropomorphic attachment changes. Large circular layouts are deliberately
  rejected from this path because they create folded or spiky shells.
- `fixed_template` is the strict backward-compatible mode with no local palm
  shape deformation.
- `attachment_hull` projects all graph-defined finger and wrist patches,
  computes their convex hull, adds a boundary margin, and extrudes a watertight
  palm. It is retained as a baseline, not used by the current 100-hand result.
- `parametric_2_5d` adds locked-patch-aware corner rounding and optional
  transverse arch, longitudinal arch, and central cup deformation.

All non-fixed modes export a separate low-face, watertight collision prism
using the same outline and coordinate system. In the extrusion modes,
attachment and wrist patches are locked against arch/cup deformation. In
`template_deform` and `source_topology_house`, the complete local thickness
coordinate is copied unchanged and the root mount lock has exactly zero
vertex displacement.

The implementation was checked against the official
[House of Dextra generator](https://github.com/An-Axolotl/House-Of-Dextra-XEmbodied-CoDesign/blob/main/Generation/converters/generate_palm_mesh.py)
and the [paper](https://arxiv.org/abs/2512.03743). The official implementation
uses two grip points per motor plus a center grid. This repository adds the
inner two footprint corners because its candidate motors come from multiple
vendors and have different root housings; the resulting palm is still one 2D
convex hull with fixed-thickness extrusion.

Run the complete deterministic pipeline from the repository root:

```bash
temp/exp1/grammar_v1/run.sh
```

The default command uses `hybrid_source_topology`. Select another mode explicitly
when needed:

```bash
temp/exp1/grammar_v1/run.sh

PALM_GENERATION_MODE=fixed_template temp/exp1/grammar_v1/run.sh

PALM_GENERATION_MODE=house_hull temp/exp1/grammar_v1/run.sh

PALM_GENERATION_MODE=parametric_2_5d \
PALM_TRANSVERSE_ARCH=0.015 \
PALM_LONGITUDINAL_ARCH=0.010 \
PALM_CENTRAL_CUP=0.012 \
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
- `outputs/palm_demo/palm_layouts_debug.png`: 3/4/5-finger palm meshes,
  attachment patches, and graph-frame XYZ axes.

## Tests and demo

Run the dependency-free unittest suite:

```bash
temp/.venv/bin/python -m unittest \
  temp/exp1/grammar_v1/test_palm_generator.py -v
```

Generate the 3/4/5-finger meshes, JSON metadata, and debug MJCF; add `--render`
when running through `mjpython`:

```bash
temp/.venv/bin/mjpython \
  temp/exp1/grammar_v1/demo_palm_generator.py --render
```

The tests cover frame invariance, patch coverage, coordinate consistency,
watertight collision geometry, degenerate faces, 3/4/5-finger layouts, source
surface-topology preservation, and exact thickness invariance.

Latest deterministic audit:

```text
finger count                         4-5
DoF                                  10-20 (includes retained platform DoF)
synthetic 4-to-5 additions           0 (disabled)
complete-bundle finger removals      0
increased-DoF designs                0 (disabled by source ownership)
layout modes                         anthropomorphic 66 / symmetric 17 / asymmetric 17
connected / acyclic                  100 / 100
invalid motor-link binding           0
invalid bundle ownership             0
cross-source mesh assignments        0
hands with mixed source meshes       0
duplicate source bundle assignments  0
finger role/order permutations       0
cyclic finger-order violations        0
protected platform nodes             24
protected transmission hands         49
protected hardware changes           0
graph palm-slot connector error      0
template attachment-frame error      0
source visual topology preserved      100 / 100 palms
exact root mount lock                 100 / 100 palms (max error 0)
source-palm thickness-coordinate error 0
edited palms                           100 / 100
maximum planar palm displacement       21.45% of source palm extent
protected tendon-base lock failures    0
finger-root visual overlap            476 / 476
watertight collision palms           100 / 100
```

## Scope boundary

This remains a graph and visual-assembly prototype. `render_hands.py` writes a
static MuJoCo inspection scene; the repository does not yet contain a complete
URDF/MJCF articulation exporter. The compiler now persists the exact graph
attachment matrices and joint-frame invariance result so that a future
exporter can consume them without mesh-derived estimation. Inertia, motor/BOM
verification, joint-sweep clearance, full wrist integration, and Isaac Lab
export/reload validation remain separate compiler-stage work.
