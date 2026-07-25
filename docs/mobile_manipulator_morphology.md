# Mobile-manipulator morphology grammar

The executable pipeline lives in
`source/dexcodesign/dexcodesign/mobile_morphology` and follows the same staged
contract as the hand pipeline:

1. parse each EEF-free URDF as a link/joint graph and fill link nodes with
   source-bound visual meshes;
2. build a default-deny, source-local grammar;
3. precompute content-addressed, source-bound link variants;
4. sample graph rewrites into `RobotIR`;
5. compile each graph back to a complete URDF using cached variants;
6. independently validate topology, mesh paths, frozen-base semantics,
   bilateral arm edits, connector interfaces, and frozen joint orientations;
7. render the compiled population with MuJoCo.

Run the full 32-robot pipeline with:

```bash
.venv-morphology/bin/python \
  scripts/dexcodesign/generate_mobile_manipulator_morphologies.py --render
```

Outputs are written to
`artifacts/mobile_manipulator_morphology/generated_32/`.

The first run, or an intentional source refresh, builds the shared cache:

```bash
.venv-morphology/bin/python \
  scripts/dexcodesign/generate_mobile_manipulator_morphologies.py \
  --rebuild-preprocess
```

## Grammar productions

- `vertical_link_length`: changes length by `{0.5, 1.5}` only along world Z
  in the original upright display pose. The display pose matters because the
  Dexmate and Galaxea URDF zero configurations do not have every arm segment
  hanging vertically. The world-Z direction is converted once into each source
  link's local frame. A precomputed connector-span rig locks the actual
  proximal mesh cap, moves the distal cap by exactly the same displacement as
  the distal graph edge, and deforms only the middle. Terminal wrist/tool
  segments remain eligible for `1.5×` extension, but not `0.5×` contraction:
  their source CAD does not contain enough material between both interfaces to
  contract by half without inversion. This is a construction-time grammar
  constraint, not a render-time disconnection filter. The other two
  original-pose world directions are unchanged.
- `shoulder_width`: is the only lateral deformation. It changes width by
  `{0.75, 1.25}` along original-pose world Y on the nearest meshed common
  shoulder carrier: `torso_l3` for Dexmate, `link_torso_5` for RB-Y1, and
  `torso_link4` for Galaxea. Every fixed attachment edge between that mesh and
  both shoulder joints receives the same world-Y rule, so the carrier mesh and
  the complete left/right shoulder paths remain connected and symmetric.

All arm productions are atomic left/right pairs. The downloaded Dexmate source
URDF is never edited; its native left/right intermediate frames are preserved.
Display joint values remain exactly at the source visualization pose and are
not morphology/search variables.

The grammar is default-deny. Root/base, wheels, steering links, base sensors,
and every link or joint not explicitly present in a production group are
immutable. Joint origin RPY and joint axes are immutable for every production.
Both the generator and compiler audit this constraint.

Each generated robot receives one production so its visible change has one
unambiguous cause. The renderer performs no corrective IK. Length variants
update mass, center of mass, and the inertia tensor consistently with the
selected longitudinal change.

## Precomputed search cache

`preprocess.py` compiles every discrete source/link/factor combination once
into `artifacts/mobile_manipulator_morphology/precomputed_link_variants/`.
The manifest `precomputed_link_variants.json` records the source content hash,
owner frame, connector-span rig, longitudinal map, and cached mesh path. A
repeated run checks the content address and reuses the mesh. The exact current
variant and mesh-owner counts are written into the manifest summary.

Population search and per-robot URDF compilation never deform or copy a mesh.
They only sample grammar records, update small joint/inertial fields, and
reference the shared cache. The generated robot directories therefore contain
URDFs only.

For a large cached search, use:

```bash
.venv-morphology/bin/python \
  scripts/dexcodesign/generate_mobile_manipulator_morphologies.py \
  --count 1000 --seed 7 --fast-search
```

This writes to `generated_1000/`, reuses the graph/grammar/mesh cache, compiles
the URDFs, and skips only the redundant mesh-loading validation pass. Action
validation and compiler immutability audits still run for every robot. Omit
`--fast-search` for the full independent validation report.

The population renderer creates four MuJoCo panels. Each panel contains eight
robots in a two-row by four-column arrangement.
