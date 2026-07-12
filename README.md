# DexCoDesign

DexCoDesign studies robot-hand morphology and learning/control algorithms jointly from human demonstrations. This repository starts from a focused snapshot of the official [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) codebase.

## Baseline

- Isaac Lab: `v2.3.2`
- Upstream commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Compatible Isaac Sim releases: 4.5, 5.0, and 5.1
- Primary RL integration: SKRL; RL-Games is retained as a second reference implementation

The upstream release supports PPO and SAC through its RL integrations. Its bundled Isaac Lab task configurations are primarily PPO configurations; DexCoDesign-specific SAC agent configurations still need to be added for the selected continuous-control hand tasks.

## Dexterous-hand assets

The repository includes 15 hand families with explicit left and right entry points under [`assets/robot_hands`](assets/robot_hands). Exact upstream commits, licenses, preferred Isaac-compatible formats, derivation flags, and load paths are recorded in [`registry.json`](assets/robot_hands/registry.json). LinkerHand is intentionally excluded.

Validate all 30 registered entries with:

```bash
python scripts/assets/validate_robot_hands.py
```

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
