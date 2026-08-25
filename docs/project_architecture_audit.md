# Hand co-design architecture

This is the current executable boundary. Hand morphology has exactly **two**
grammars. Optimizers and training modes do not own a third morphology schema.

## The two grammars

| Grammar | Purpose | Canonical module | Vector semantics |
|---|---|---|---|
| MiDas manufacturing | MiDas designs constrained by its mechanical rules | `dexcodesign.morphology.midas_grammar` | Fixed 15-D, source is the zero vector |
| General simulation hand | Every non-MANO simulation morphology, including WUJI SAC | `dexcodesign.morphology.general_grammar` | Source-dependent dimension, source is the zero vector |

MANO is a task/reference hand and is never morphed. The MiDas source may still
be bound to the general simulation grammar for an unconstrained simulation
ablation; that does not create a third grammar.

The general grammar owns all of the following semantics:

- an ordered bank of 32 real palm prototypes;
- prototype 0 is the exact source palm;
- prototypes 1–31 interpolate the complete finger-root position and orientation
  from the source layout toward the radial/star layout over expansion 0–0.70;
- each editable main-chain phalanx has an independent length variable;
- normal-finger body/distal widths are shared, and thumb body/distal widths are
  shared separately;
- motor housings, connector caps, joint axes/ranges, active/mimic relations and
  protected transmission hardware are invariant.

The graph compiler generates both a real visual palm mesh and a real collision
palm mesh for every selected prototype. Palm and finger roots use the same
interface transform, so the mesh is not repaired by a post-processing check.

## One morphology path for WUJI SAC

`temp/hocap_mano_replay/scripts/wuji_general_space.py` is an optimizer adapter,
not a grammar. It binds `wuji_hand_2` to the general grammar and converts only
between two representations:

```text
SAC search vector
  [integer palm prototype 0..31, canonical latent coordinates]
        |
        v
general_grammar.decode_vector / graph_spec_from_vector
        |
        v
shared HandIR + visual/collision mesh compiler
```

For WUJI this is 23 dimensions: one palm prototype, three palm affine values,
15 per-phalanx length values and four shared width values. The old independent
14-D `wuji_morphology_space.py` has been removed. SAC, CEM utilities, GPU
retarget support and selected-result evaluation all import the same adapter and
therefore the same general decoder.

The 32 palm meshes are compiled/cached before large PhysX batches. Runtime
candidates select a prototype and apply the remaining continuous overlays; the
optimizer does not rebuild the palm mesh inside every simulation step.
The cache is named `palm_prototype_bank_general_v3_source_star_0p70` and carries
a grammar signature. Training refuses the legacy unsigned/anthropomorphic bank
instead of silently accepting a numerically similar `0.70` cache.

## Learning entry points

Use `scripts/dexcodesign/launch_hand_learning.sh` as the user-facing launcher:

```text
ppo      fixed morphology + residual PPO
sac      general-grammar morphology SAC, no PPO updates
sac-ppo  the same morphology SAC + residual PPO updates
```

All three modes take the same task inputs: reference NPZ, object USD and output
directory. `ppo` supports the registered robot hands. The current morphology
SAC runtime is intentionally WUJI-only, but its morphology definition is the
general grammar rather than a WUJI-only grammar.

`sac-ppo --gpus N` requests all `N` GPUs in one Slurm allocation and launches
one distributed worker per GPU. Rank 0 samples the global morphology
population, every rank receives a disjoint contiguous `population / N` slice,
and only the scalar rollout results are gathered for the outer SAC update.
`--morphology-replicas` controls physical replicas of each local morphology;
`--rollout-multiplier` controls additional PPO rollout reuse. The launcher
rejects a population that is not divisible by the GPU count. Single-GPU
SAC-PPO is the same code path with world size one. Pure morphology SAC keeps a
single persistent Isaac process and batches candidates with
`--physics-batch-size`; independent fixed-hand PPO jobs are scheduled one hand
per GPU.

The Isaac/HO-Cap implementation remains under `temp/hocap_mano_replay` because
it is cluster integration code. Its location does not change the morphology
contract above.

## Verified WUJI preview

`scripts/dexcodesign/generate_wuji_sac_preview.py` creates exactly 32 WUJI
candidates through the SAC/general-grammar path. Candidate 0 is the source;
every palm prototype is used once. The verification run checks:

- 32 distinct visual palm meshes and 32 distinct collision palm meshes;
- strictly ordered expansion from 0.0 to 0.70;
- all 160 palm/finger interfaces meshed;
- maximum palm interface-frame error 0.0;
- source DoF, joint axes and transmission semantics unchanged.

Generated preview meshes are deleted after rendering. The vector, graph,
compiler metadata, audit JSON and final image are retained and can reproduce
them.

## Retention policy

Keep source assets, reference graphs/rigid parts, mechanism bundles, task data,
selected rollouts, final videos/curves and small manifests. Generated debug
populations, temporary mesh directories, failed comparison outputs and Python
caches are reproducible and should be deleted rather than committed.
