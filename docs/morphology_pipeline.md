# Hand Morphology Generation Pipeline

## Directory layout

```text
assets/robot_hands/
  registry.json                       original source registry
  direct_motor/                       bilateral normalized source assets
  morphology/canonical_scaffold.json  semantic roles and upright convention

source/dexcodesign/dexcodesign/morphology/
  source_graph.py     direct URDF → rigid-link reference graph
  bundles.py          source-local motor/link grammar
  generate.py         graph and morphology sampling
  palm_geometry.py    source-topology palm deformation
  mesh_compiler.py    graph → complete per-link meshes
  render.py           optional MuJoCo contact sheet

artifacts/hand_morphology/
  reference_graphs.json
  reference_rigid_parts/
  mechanism_bundles.json
  generated_100/
    hand_ir.json
    generation_summary.json
    compiled_hands.json
    meshes/
    hands_100.png
```

`artifacts/` is reproducible and intentionally not committed.

## 1. Source asset normalization

`scripts/assets/build_direct_motor_hands.py` reads every bilateral entry in
`assets/robot_hands/registry.json` and writes self-contained URDF assets to
`assets/robot_hands/direct_motor/`.

Every movable joint becomes exactly one of:

- **active direct**: independent joint with a URDF transmission;
- **passive mimic**: affine follower of one active joint.

Existing URDF mimic chains are flattened. MJCF equality relations become
affine mimics. A tendon group is approximated by one active master and
range-normalized mimic followers. Any remaining movable joint is promoted to
active; MANO is therefore fully active. Link trees, joint frames, axes, ranges,
visual meshes, collision meshes, inertials, and available effort limits are
preserved.

## 2. Reference graph preprocessing

`source_graph.py` reads the normalized right-hand URDFs.

1. Evaluate every link frame at zero joint position.
2. Contract links separated only by fixed joints into one rigid part.
3. Merge all visual meshes belonging to that rigid part.
4. Apply the source-specific similarity transform into the shared upright
   canonical frame.
5. Attach the semantic role from `canonical_scaffold.json`.

A graph node is therefore one rigid body between movable joints, not one CAD
file. A geometryless joint frame remains a graph node; missing visual geometry
never deletes a DoF.

The result is cached because this is source preprocessing, not a morphology
search action.

## 3. Design grammar

`bundles.py` creates complete source-local finger mechanism bundles.

- A motor/joint can only select link candidates originally paired with that
  motor type and interface class in the same source hand.
- Meshes never cross between source hands.
- Thumb/index/middle/ring/pinky order is preserved.
- A source finger bundle is instantiated at most once per generated hand.
- At most five fingers are allowed.
- Four-finger sources do not synthesize a fifth finger.
- Base and palm-internal DoFs are copied unchanged.

The general simulation grammar is source-local: different source hands have
different vector dimensions, and absent fingers or phalanges are not padded
into the vector. MANO remains the immutable human reference and is not a robot
morphology seed. Its current production search variables are:

- independent proximal/middle/distal longitudinal scale for every physical
  finger present in the source (`0.87–1.17` in preview generation; explicit
  graph requests may use `0.55–1.60`);
- four shared radial controls: normal-finger body, normal-finger distal,
  thumb body, and thumb distal (`0.60–1.45` explicit); a control is omitted
  when the corresponding source geometry does not exist;
- palm X/Z in-plane scale and a bounded in-plane yaw for sources without a
  protected transmission platform;
- palm layout mode and continuous `palm_expansion`;
- small edge-tangent finger-root displacement in anthropomorphic mode.

Finger mechanisms are complete source-owned bundles. The current production
generator does not add, remove, splice, or cross-source fingers, even though
older exploratory helpers for bundle removal remain in the module. Joint axes,
ranges, active/mimic relations, palm thickness, and protected hardware are not
search variables.

## 4. Graph generation

`generate.py` produces 100 typed rigid-link trees. It edits:

- source-bound finger link length and radial scale;
- palm in-plane shape;
- finger-root locations along the palm boundary;
- anthropomorphic, symmetric, or asymmetric palm layouts when compatible.

`palm_expansion` is the continuous source-to-layout interpolation parameter.
`0` preserves the source finger-root placement, while `1` reaches the selected
House/star layout. Intermediate values move the palm interfaces and complete
finger roots together. Explicit values that violate motor-footprint clearance
fail instead of being silently changed.

Palm thickness is fixed. The wrist/mount region is fixed. Protected
under-palm hardware is fixed. Active and mimic classification, joint range,
joint axis, and base/palm DoFs are preserved.

For physics search, `palm_expansion` selects one of 32 ordered prototypes.
Each source hand owns one prototype bank and all morphology candidates derived
from that source reuse it. The network sees the ordered scalar and prototype
index, never a one-hot vector. Exact visual meshes are generated only for
inspection/final output and may be discarded immediately afterward.

Only the invariants required for compilation are checked. The old exploratory
`grammar_audit.json` and dozens of diagnostic fields have been removed.

## 5. Mesh generation

`mesh_compiler.py` fills the graph from the root outward.

- Finger links use the selected source-owned rigid-part mesh.
- Length and radial deformation are applied in the link's semantic frame.
- The palm keeps the original source vertices and faces.
- Palm deformation occurs only in its local plane; the thickness coordinate
  is unchanged.
- The root mount region is locked.
- Palm-side interface vertices follow the same graph transform as each
  finger-root joint.
- Visual and collision geometry remain separate.

The compiler writes one resumable manifest per generated hand. An interrupted
large run can continue with `--resume`.

## 6. Required fail-fast conditions

The production pipeline stops only for conditions that would make the result
structurally invalid:

- missing registered source hand;
- disconnected or cyclic graph;
- duplicate node or finger bundle;
- cross-source mesh assignment;
- invalid motor/link ownership;
- changed finger order;
- more than five fingers;
- changed source DoF or active/mimic partition;
- changed palm thickness or protected base;
- disconnected finger-root bounds;
- invalid collision mesh.

Quantities such as deformation fraction are metadata, not arbitrary rejection
thresholds.

## 7. Running

Create the environment once:

```bash
python3.12 -m venv .venv-morphology
.venv-morphology/bin/pip install -r scripts/dexcodesign/morphology_requirements.txt
```

Generate 100 hands:

```bash
.venv-morphology/bin/python scripts/dexcodesign/generate_hand_morphologies.py
```

Compile one explicit graph/source request without an LLM:

```bash
.venv-morphology/bin/python scripts/dexcodesign/compile_hand_graph.py \
  assets/robot_hands/morphology/hand_graph.example.json
```

The output directory is content-addressed. Repeating an identical request is a
cache hit and does not rerun mesh generation. The graph schema accepts a source
hand, palm layout/expansion/scale/yaw, and per-finger length/radius scales.

## Runtime certification boundary

`HandIR + compiled OBJ` is geometry-complete but is not by itself a
retarget/RL-ready robot asset. A generated hand may only be marked
`runtime_ready` after a deterministic runtime exporter and the following gates
all succeed:

1. convert canonical coordinates back to the source hand's physical units and
   export the required handedness;
2. emit a complete URDF with visual, collision, inertial, joint-limit,
   transmission, and affine mimic records;
3. register semantic palm and fingertip links without a hard-coded hand list;
4. retarget all 446 frames using active joints only, then verify every mimic
   relation on every frame;
5. import the URDF into Isaac Lab, verify collision coverage and hand/object
   contact while hand/support collision remains disabled;
6. pass one-environment reset/step and the configured vectorized-environment
   smoke test.

Until that exporter and certification command are present, compiled morphology
artifacts must not be fed directly to the existing all-hands RL launcher.

Useful options:

```text
--seed N                 deterministic morphology sample
--rebuild-direct-motor   rebuild standard bilateral URDF assets
--rebuild-reference      rebuild cached rigid reference parts
--resume                 reuse completed per-hand mesh manifests
--render                 create the 100-hand MuJoCo contact sheet
--palm-expansion X       source-to-House/star palm expansion in [0, 1]
```
