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

## 4. Graph generation

`generate.py` produces 100 typed rigid-link trees. It edits:

- source-bound finger link length and radial scale;
- palm in-plane shape;
- finger-root locations along the palm boundary;
- anthropomorphic, symmetric, or asymmetric palm layouts when compatible.

Palm thickness is fixed. The wrist/mount region is fixed. Protected
under-palm hardware is fixed. Active and mimic classification, joint range,
joint axis, and base/palm DoFs are preserved.

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

Useful options:

```text
--seed N                 deterministic morphology sample
--rebuild-direct-motor   rebuild standard bilateral URDF assets
--rebuild-reference      rebuild cached rigid reference parts
--resume                 reuse completed per-hand mesh manifests
--render                 create the 100-hand MuJoCo contact sheet
```
