# GPU morphology search

The production WUJI morphology search evaluates every sampled design with the
same Isaac Lab/PhysX rollout semantics as residual RL. There is no kinematic or
contact proxy and no top-k prefilter. The maintained outer optimizer is SKRL
2.1 SAC; the earlier custom hybrid-SAC implementation remains available only as
an experimental baseline through `--optimizer-backend custom`.

## Search space

The WUJI semantic reshape vector has fourteen continuous variables:

- `palm_expansion`: an ordered scalar spanning `[0.00, 0.70]`.
- palm scale X/Z and palm yaw.
- independent length and radius scales for the five fingers.

SKRL SAC outputs all fourteen ordered continuous values. Only the PhysX
execution boundary maps palm expansion to the nearest of 32 precompiled
collision prototypes. Replay retains the requested continuous expansion, the
realized prototype index, and the measured physical reward. Thus the critic can
learn a smooth ordered value landscape without treating neighboring expansion
values as unrelated one-hot categories. Candidate zero remains the exact
unmodified source hand. Finalists can be recompiled with their exact continuous
palm expansion rather than the search-time collision approximation.

## Per-generation pipeline

`train_wuji_hybrid_sac_morphology.py` owns one Isaac application and one Slurm
allocation for the complete search:

1. `SkrlConditionalMorphologySAC.propose` samples the complete continuous
   morphology action and reserves 15% uniform exploration.
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
