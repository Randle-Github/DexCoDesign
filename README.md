# DexCoDesign

DexCoDesign studies robot-hand morphology and learning/control algorithms jointly from human demonstrations. This repository starts from a focused snapshot of the official [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) codebase.

## Baseline

- Isaac Lab: `v2.3.2`
- Upstream commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Compatible Isaac Sim releases: 4.5, 5.0, and 5.1
- Primary RL integration: SKRL; RL-Games is retained as a second reference implementation

The upstream release supports PPO and SAC through its RL integrations. Its bundled Isaac Lab task configurations are primarily PPO configurations; DexCoDesign-specific SAC agent configurations still need to be added for the selected continuous-control hand tasks.

## Dexterous-hand assets

The repository includes 14 hand families with explicit left and right entry points under [`assets/robot_hands`](assets/robot_hands). Exact upstream commits, licenses, preferred Isaac-compatible formats, derivation flags, and load paths are recorded in [`registry.json`](assets/robot_hands/registry.json). LinkerHand is intentionally excluded.

Validate all 28 registered entries with:

```bash
python scripts/assets/validate_robot_hands.py
```

## Co-design method

The active method is specified in [`assets/representation.md`](assets/representation.md).
DexCoDesign does not regress complete meshes or freely pair actuators with
unrelated link geometry.  It compiles a versioned `HandIR` through a strict
Design Grammar:

- one morphology node is one maximal rigid functional link between adjacent
  movable or explicitly retained semantic joint frames;
- fixed CAD/helper links are folded into that rigid cluster, and all owned
  visuals are transformed into its local frame and integrated as one compound
  visual part;
- a source motor, its transmission, its driven link contract, and its candidate
  link meshes form one indivisible `MechanismBundle`;
- selecting a motor therefore permits only its original source-bound candidate
  meshes and their bounded connector-preserving deformations;
- visual and collision geometry are generated separately;
- morphology-specific retargeting and residual PPO evaluate each compiled hand,
  while grammar-constrained Hybrid SAC proposes local morphology edits.

The executable production pipeline is
[`source/dexcodesign/dexcodesign/morphology`](source/dexcodesign/dexcodesign/morphology);
its complete data flow is documented in
[`docs/morphology_pipeline.md`](docs/morphology_pipeline.md). It has no
dependency on `temp/`. Generated reference parts and morphology libraries are
written to `artifacts/hand_morphology/`.

The two supported hand grammars and the boundary between reusable morphology
code and the experimental HO-Cap/WUJI training layer are recorded in
[`docs/project_architecture_audit.md`](docs/project_architecture_audit.md).
WUJI morphology SAC consumes the same general simulation-hand grammar and palm
prototype bank as direct morphology generation; it does not define a third
grammar.

Launch the three learning modes through one interface:

```bash
scripts/dexcodesign/launch_hand_learning.sh ppo     --reference REF.npz --object-usd OBJECT.usd --output OUT
scripts/dexcodesign/launch_hand_learning.sh sac     --reference REF.npz --object-usd OBJECT.usd --output OUT
scripts/dexcodesign/launch_hand_learning.sh sac-ppo --reference REF.npz --object-usd OBJECT.usd --output OUT
```

Generate 100 source-bound hands with:

```bash
.venv-morphology/bin/python scripts/dexcodesign/generate_hand_morphologies.py
```

The analogous default-deny grammar for the EEF-free Dexmate Vega, RB-Y1, and
Galaxea R1 mobile manipulators is documented in
[`docs/mobile_manipulator_morphology.md`](docs/mobile_manipulator_morphology.md).
It freezes each mobile base, enforces bilateral arm edits, and generates 32
source-mesh-bound robot URDFs plus one MuJoCo contact sheet.

## Retained scope

- Isaac Lab simulation, environment, sensor, controller, and asset APIs
- Isaac Lab RL wrappers
- Isaac Lab Mimic, teleoperation, and imitation-learning scripts
- Allegro Hand and Shadow Hand in-hand manipulation environments
- Manager-based manipulation tasks, including reach, lift, stack, cabinet, and DexSuite
- Factory, FORGE, AutoMate, and Franka manipulation references
- SKRL and RL-Games training/play entry points
- Core demos, tutorials, environment utilities, and asset/data conversion tools

CI configuration, Docker files, documentation-site sources, benchmarks, distributed Ray tooling, tests, and unrelated locomotion/navigation/drone task suites were intentionally omitted. See [UPSTREAM.md](UPSTREAM.md) for provenance and update instructions.

## Installation

Isaac Lab requires a supported NVIDIA GPU environment. Install a compatible Isaac Sim release first, then from this repository run:

```bash
./isaaclab.sh --install skrl
```

List the retained environments:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

Run the official Allegro Hand PPO example:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Repose-Cube-Allegro-Direct-v0 \
  --algorithm PPO \
  --headless
```

The next project layer should add DexCoDesign-owned task definitions, common evaluation metrics, and explicit PPO/SAC configurations without modifying the vendored Isaac Lab core unnecessarily.
