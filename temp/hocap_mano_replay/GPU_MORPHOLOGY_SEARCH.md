# GPU morphology search

The production WUJI morphology search evaluates every sampled design with the
same Isaac Lab/PhysX rollout semantics as residual RL. There is no kinematic or
contact proxy and no top-k prefilter. The maintained outer optimizer is SKRL
2.1 SAC; the earlier custom hybrid-SAC implementation remains available only as
an experimental baseline through `--optimizer-backend custom`.

## Search space

The WUJI reshape vector has one discrete and thirteen continuous variables:

- `palm_expansion_index`: one of 32 precompiled, physically valid palm
  prototypes spanning expansion `[0.00, 0.70]`.
- palm scale X/Z and palm yaw.
- independent length and radius scales for the five fingers.

The palm index is intentionally not represented as a continuous SAC action and
rounded afterward. Rounding would map many actions to the same physical hand,
which makes the actor gradient ambiguous near prototype boundaries. A custom
Gumbel-softmax or hybrid categorical SAC would reintroduce an unmaintained RL
core.

Instead, every generation uses exact stratified enumeration:

1. The 32 palm prototypes are scheduled uniformly and shuffled.
2. The palm index is encoded as a one-hot observation/context for SKRL SAC.
3. SKRL SAC outputs only the thirteen continuous morphology variables,
   conditioned on that palm context.
4. With the default population of 4,096, every palm prototype receives exactly
   128 candidates per generation.
5. Candidate zero is always the exact unmodified source hand, so pure replay is
   measured in every generation.

This is the selected production treatment for the discrete palm variable. It
guarantees coverage, avoids aliased actions, keeps the SAC actor/critics inside
the mature SKRL implementation, and allows the final selection to compare all
32 real palm collision geometries. If future graph grammars contain too many
discrete choices for exact stratification, retain a nonzero quota for every
choice and allocate only the remaining candidates with a bandit scheduler; do
not round continuous SAC actions into categorical structures.

## Per-generation pipeline

`train_wuji_hybrid_sac_morphology.py` owns one Isaac application and one Slurm
allocation for the complete search:

1. `SkrlConditionalMorphologySAC.propose` samples continuous actions for the
   balanced palm contexts and reserves 15% uniform exploration.
2. GPU batched IK retargets the full 446-frame Human/MANO command trajectory to
   all candidates, warm-started from the source-hand trajectory.
3. Graph IR and lightweight runtime overlays are generated for every candidate.
   Palm collision geometry is selected from the 32-prototype bank; unchanged
   mechanism assets are shared.
4. One PhysX scene contains all 4,096 distinct morphology environments. Each
   receives the normal residual-RL reward/contact/termination implementation:
   C-error pose reward, binary thumb-versus-other-finger pinch contact, dynamic
   object physics, and 5 cm / 1.5 rad early termination.
5. Every measured rollout is recorded as a terminal one-step transition. For
   morphology selection the Bellman target therefore reduces to the exact
   rollout return; SKRL performs the actor/critic/entropy updates.
6. The SKRL checkpoint, exact candidate vectors, retarget trajectories, PhysX
   results, and per-generation summary are saved before the next generation.

The fixed table/object state is reused conceptually, but a new heterogeneous
PhysX scene must still be authored and cooked when all 4,096 morphology assets
change. On an A40 the 446-step rollout takes about 34 seconds, while scene
creation and simulation startup take about 525--540 seconds. Retargeting takes
about 13 seconds and graph/overlay preparation about 88 seconds. Scene
initialization is therefore the dominant remaining bottleneck.

## Verified integration

The SKRL path was exercised end-to-end with 64 candidates for two generations:

- all 64 candidates received exact PhysX rollouts in both generations;
- all 32 palm prototypes appeared exactly twice per generation;
- each stored palm index matched the physical candidate manifest;
- `proxy_used=false` and `top_k_prefilter_used=false`;
- the checkpoint and 128 replay transitions were saved successfully.

The fixed-budget 4,096-candidate custom-baseline run completed 20 generations
and provides the migration reference. Population mean reward increased from
35.72 to 329.79, P90 from 55.48 to 810.87, average reached phase from 66.44 to
198.10, and successes from 2 to 145 per generation. Its global best was 841.81
at generation 10. These results confirm that exact physical morphology search
learns a substantially better distribution; future production runs should use
the default `--optimizer-backend skrl` path above.
