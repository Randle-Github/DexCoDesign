# Experiment 1: right-hand graph encoder and MuJoCo generation

> **Deprecated:** the original generator below used a nearest-template graph
> and is retained only for comparison. The current connected, noise-decoded,
> cross-hand mesh model is in [`strict_v2/`](strict_v2/README.md).

This is a CPU-only proof of concept for the representation proposed in
`assets/representation.md`. It trains on exactly the 14 registered **right**
hands. Left-hand assets are not included in the dataset or checkpoint.

## What this version learns

Each source hand is converted to a padded kinematic graph with:

- functional node roles: palm, base/wrist, five digit roles, sensor, other;
- incoming joint type, geometry count and graph depth;
- the real parent/child adjacency from the audited URDF/MJCF/USD structure;
- a continuous 12-D procedural morphology target measured from the canonical
  source geometry and graph statistics.

`model.py` contains a three-layer dense GCN. Masked mean/max graph pooling
produces a 16-D latent. Three regression heads reconstruct:

1. the 12-D continuous morphology vector;
2. canonically ordered node features;
3. graph adjacency logits.

All 14 samples are used for reconstruction/regression training. The reported
numbers are therefore training reconstruction metrics, not a claim of
generalization from a held-out test set.

## Generation and legal graph mutation

Fourteen examples are not enough to learn a trustworthy unconstrained graph
grammar. The generator therefore separates two operations:

1. the network interpolates graph embeddings and regresses continuous
   morphology/mesh deformation parameters;
2. a bounded graph-grammar executor mutates a balanced real source graph by
   adding/removing digit branches, adding/removing a terminal link, toggling
   fixed/hinge edges, and changing base-joint count.

The realized set contains exactly 25 each of 3-, 4-, 5-, and 6-digit hands.
Actual compiled DOF ranges from 7 to 31 and node count from 11 to 50.

This tests the encoder/latent/regression/grammar/compiler loop without
pretending that a 14-sample dataset learned arbitrary raw adjacency output.

## Mesh and geometry layers

Visual geometry is extracted per link from the original URDF/MJCF/USD assets.
MANO and already-light meshes retain their source face count. Complex meshes
use connectivity-preserving QEM when it passes bounds checks; otherwise the
source link mesh is retained. The full 100-hand MJCF contains roughly 13M
source-derived visual faces.

`decompose_geometry.py` separately fits the searchable/collision layer. It
decomposes 260 visual links into 443 local compound components, with at most
six OBB/capsule/ellipsoid parts per link. This decomposition does not replace
or modify the visual mesh.

## Environments

No new conda environment is required:

- `temp/.venv`: geometry extraction, including USD through `pxr`;
- `/opt/anaconda3/envs/dexretarget`: PyTorch CPU training and MuJoCo.

Run the complete non-interactive pipeline:

```bash
cd /Users/liuyangcen/workspace/DexCoDesign
temp/exp1/run_exp1.sh
```

Open one interactive 20-hand page on macOS:

```bash
cd /Users/liuyangcen/workspace/DexCoDesign
/opt/anaconda3/envs/dexretarget/bin/mjpython temp/exp1/view_generated.py --page 1
```

Use `--page 1` through `--page 5`. The complete 100-hand model still compiles,
but the MacBook graphics context is more stable when displaying 20 of the 13M
faces at a time. `render_snapshot.py` renders five pages and stitches the final
100-hand overview without reducing mesh faces.

Controls: left-drag rotates, right-drag pans, mouse wheel zooms, and Escape
closes the window.

## Outputs

- `outputs/dataset.json`: 14 right-hand padded graphs and targets;
- `outputs/graph_autoencoder.pt`: trained CPU checkpoint;
- `outputs/training_metrics.json`: reconstruction losses and errors;
- `outputs/generated_hands.json`: 100 decoded design records and provenance;
- `outputs/realized_graphs.json`: actual emitted nodes, digit count and DOF;
- `outputs/mesh_library.json`: link-level source-derived visual mesh metadata;
- `outputs/geometry_decomposition.json`: searchable compound geometry;
- `outputs/generated_100_hands.xml`: full 100-hand, ~13M-face MuJoCo scene;
- `outputs/render_pages/`: five 20-hand MJCF scenes and renders;
- `outputs/generated_100_hands.png`: stitched 100-hand overview.
