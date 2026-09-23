# WUJI 385-D SAC + PPO co-training handoff

## Joint-space observation baseline with matched retargeting (2026-09-22)

To compare legacy joint-space inputs against the 385D palm-geometry inputs on
the same fixed population, pass both:

```bash
--ppo-observation-mode legacy --retarget-per-generation
```

This runs the same candidate-specific sequential MANO retargeting, captured
object references, per-generation generated USD preparation, scene recreation,
and checkpoint continuation. Legacy WUJI observations have 86 values: current
joint positions (26), current thumb/index positions (6), object pose (7),
demonstration thumb/index goal poses (14), reference joint positions (26), and
reference object pose (7). The shared actor and critic use this input; residual
actions and reward definitions are unchanged. Omit `--morphology-context` for
this baseline. Train a fresh policy rather than loading a 385D checkpoint.

Without `--retarget-per-generation`, legacy mode preserves its historical
shared-fixed-reference behavior, which is not an observation-only comparison.
The new flag requires shared PPO and cannot be combined with `--fixed-reference`.
Palm-geometry mode still retargets automatically. Use the same
`--fixed-ppo-vectors` file, corrected prototype bank, environment counts, PPO
cycles, and seed for paired experiments; fixed vectors disable SAC updates.

For changes since `fd73d0c`, including physics, evaluation, retargeting, collision
geometry, logging, playback, and remaining limitations, see
[the September 22 update note](../source/README_WUJI_UPDATES.md).

## Canonical working tree

Use only:

```text
/home/yhan389/Desktop/DexCoDesign
```

`DexCoDesign-all-hands` is an older development clone. The relevant changes
are already present in `DexCoDesign`, whose training driver is a functional
superset (W&B, fixed-palm search, repeated rollouts, video, and the existing
memory-isolation workflow). Do not launch this experiment from the all-hands
clone and do not copy its older driver over the main driver.

These changes are currently uncommitted in a pre-existing dirty worktree.
Preserve them and all unrelated user changes: do not run `git reset --hard`,
`git checkout -- <path>`, or bulk-clean untracked files. Review and commit only
the explicitly relevant files after the smoke test succeeds.

## Intended algorithm

Each outer generation performs the following sequence:

1. The morphology SAC proposes a population of design vectors.
2. Every proposed morphology is retargeted independently from the fixed MANO
   command trajectory. A fixed WUJI reference is not reused in 385-D mode.
3. Candidate assets are prepared with their own joint origins, link transforms,
   connector-preserving mesh deformation, collision geometry and reference
   trajectory.
4. Each candidate gets 60 collision-surface landmarks: three ordered points on
   each of the 20 moving finger links.
5. FK of that candidate's retargeted trajectory produces its reference
   landmark trajectory.
6. One shared PPO policy is trained across the complete heterogeneous
   morphology population.
7. The trained policy is evaluated deterministically. Per-morphology rewards
   are returned to the morphology SAC for its update.
8. PPO is saved to `shared_ppo_latest.pt` and resumed in the next outer
   generation.

Morphology is not appended as a separate design vector. It is exposed through
the candidate-specific current and reference surface landmarks.

## 385-dimensional PPO observation

All geometry and poses are expressed relative to the current palm frame:

```text
current 60 surface points       60 x 3 = 180
reference 60 surface points     60 x 3 = 180
reference palm pose                       7
current object pose                        7
reference object pose                      7
gravity in palm frame                      3
normalized phase                           1
                                      -------
                                          385
```

The three points on each moving link are `pad_center`, `distal_left`, and
`distal_right`. They are intersected with the collision surface used by the
configured PhysX approximation. `kinematic_normal` is the object-independent
hand-only inward mode selected for this experiment.

## Main files

- `temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py`
  - Adds `--ppo-observation-mode palm_geometry`.
  - Retargets every proposal in shared-PPO mode.
  - Trains one shared 385-D PPO and feeds deterministic evaluation rewards to
    SAC.
  - Writes `co_training_contract.json`, per-generation summaries, PPO
    checkpoints and W&B metrics.
- `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/mano_residual_env.py`
  - Accepts 385-D observations for both fixed WUJI and morphology batches.
  - Builds candidate-specific geometry offsets and reference landmark
    trajectories.
  - Applies runtime joint, affine and connector-preserving mesh overlays.
  - Keeps the original fixed-hand single-agent PPO task operational.
- `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/palm_geometry_observation.py`
  - Implements collision landmarks, candidate-aware FK and the 385-D feature
    construction.
- `source/dexcodesign/dexcodesign/morphology/parametric_mesh.py`
  - Shared connector-preserving deformation used by both landmark generation
    and runtime USD geometry.
- `temp/hocap_mano_replay/scripts/prepare_wuji_parametric_training_assets.py`
  - Stores candidate URDFs, transforms, mesh deformations, joint origins and
    reference paths in the PhysX manifest.
- `temp/hocap_mano_replay/isaaclab/wuji_parametric_usd.py`
  - Applies the candidate geometry to USD assets.
- `tests/test_palm_geometry_observation.py`
  - Covers observation dimensions, frame invariance, collision landmarks and
    morphology overrides.

## Compatibility contract

- `--ppo-observation-mode` defaults to `legacy`; existing morphology-only runs
  are unchanged.
- Fixed single-agent PPO continues to use task
  `DexCoDesign-WUJI-PalmGeometry-Residual-Direct-v0` and remains 385-D.
- `palm_geometry` rejects `--morphology-context`; adding the 23-D design vector
  would change the input dimension and break the intended policy contract.
- `palm_geometry` rejects `--fixed-reference`; each morphology must use its own
  retargeted reference.
- WUJI joint topology, action ordering and action dimension remain unchanged.

## Random initialization

For a genuinely new run, do **not** pass `--ppo-checkpoint`. Generation 0 then
creates both SAC and PPO from random network parameters. Generation 1 and later
correctly resume PPO from the same run's `shared_ppo_latest.pt`.

Always use a new output directory. Reusing an existing output root can mix
results and checkpoints from an earlier experiment.

## Recommended launch command

```bash
cd /home/yhan389/Desktop/DexCoDesign
conda activate DexCoDesign_sim4_5_lab_2_3_2

BANK="$PWD/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70"
SEED="$PWD/artifacts/all_hands_success_action_retarget/wuji_hand_2/retargeted_trajectory.npz"
OBJECT_USD="$PWD/artifacts/isaaclab_mano_residual/assets/g04_1.usd"
RUN_NAME="wuji_385_real_cotrain_random_$(date +%Y%m%d_%H%M%S)"

DEXCODESIGN_OBJECT_USD_PATH="$OBJECT_USD" \
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py \
  --prototype-bank-root "$BANK" \
  --seed-trajectory "$SEED" \
  --fixed-palm-prototype 0 \
  --num-morphologies 64 \
  --physics-batch-size 64 \
  --generations 100 \
  --sac-updates 64 \
  --sac-batch-size 64 \
  --retarget-iterations 4 \
  --ppo-cycles-per-generation 4 \
  --train-envs-per-morphology 1 \
  --ppo-rollout-multiplier 1 \
  --ppo-observation-mode palm_geometry \
  --geometry-inward-direction-mode kinematic_normal \
  --optimizer-backend skrl \
  --continue-after-success \
  --wandb \
  --wandb-project DexCoDesign \
  --wandb-group wuji-hybrid-sac-cotrain \
  --wandb-run-name "$RUN_NAME" \
  --headless \
  --output-root "$PWD/artifacts/wuji_sac/$RUN_NAME"
```

There is intentionally no `--ppo-checkpoint` in this command.

In shared-PPO co-training, `--train-envs-per-morphology` controls parallel PPO training
environments per morphology. Deterministic evaluation reuses a selectable subset of
these environments and averages their episode returns per morphology. `--rollouts-per-proposal` belongs to the
morphology-only zero-residual evaluation workflow and is not the robustness
control for this shared-PPO path. Start with one morphology replica. Raising it
to four creates `64 x 4 = 256` physical policy environments.

`--ppo-rollout-multiplier` is different again: it multiplies the temporal PPO
rollout horizon before each PPO update; it does not create independent physics
replicas.

## Preflight and smoke tests

Static validation completed in the main tree:

```text
Python compilation: passed
tests/test_palm_geometry_observation.py: 9 passed
```

Before the 64 x 100 run, launch a new two-candidate/two-generation GPU smoke by
changing only:

```bash
--num-morphologies 2
--physics-batch-size 2
--generations 2
--sac-updates 1
--sac-batch-size 2
--retarget-iterations 1
--ppo-cycles-per-generation 1
--wandb-run-name wuji_385_cotrain_smoke
--output-root "$PWD/artifacts/wuji_sac/wuji_385_cotrain_smoke"
```

Verify all of the following before scaling up:

1. `co_training_contract.json` reports `observation_mode: palm_geometry`,
   `observation_dimension: 385`, and `retarget_per_proposal: true`.
2. Both generations finish and generation 1 loads the saved shared PPO.
3. Each rank manifest contains candidate-specific `reference_paths`,
   `hand_urdf_paths`, `parametric_mesh_deformations`, and joint origins.
4. Candidate landmark offsets/reference trajectories are not identical for two
   different design vectors.
5. No `WUJI morphology URDF/simulator FK mismatch` is raised.

## Resource notes

- `population=64`, one morphology replica and a heterogeneous morphology scene
  is materially heavier than the two-candidate smoke.
- Watch host RAM and GPU memory during the first several generations.
- Do not enable the morphology-only `--video` path together with shared PPO;
  the current driver explicitly rejects that combination.
- The old all-hands clone lacks the complete artifacts tree and must not be used
  as the launch directory for this experiment.

## Clear CLI names and compatibility

Use `--num-morphologies`, `--train-envs-per-morphology`, and
`--ppo-cycles-per-generation`. The previous names `--population`,
`--morphology-replicas`, and `--shared-ppo-iterations` remain accepted as aliases
with identical defaults and behavior. Internal configuration destinations and
saved JSON field names remain compatible with existing tools.

For 32 designs, 128 training environments per design, and 64 PPO cycles:

```bash
--num-morphologies 32 --train-envs-per-morphology 128 --ppo-cycles-per-generation 64
```

The batch scripts also accept `NUM_MORPHOLOGIES`, `TRAIN_ENVS_PER_MORPHOLOGY`,
and `PPO_CYCLES_PER_GENERATION`; these take precedence over the old environment
variable names when both are set.

`--eval-envs-per-morphology 32` scores 32 environments per morphology using
mean deterministic episode return (failure counts as an episode end). It defaults
to the training count and must be positive and no larger than that count.
The evaluator selects the first K replicas of each morphology in manifest order,
resets the scene, and stops when all selected replicas have finished their first
episode. Unselected replicas and auto-reset episodes do not contribute to scores.
The whole training scene is still stepped; this option does not reduce scene
memory or create a smaller evaluation scene. Identical deterministic replicas
may yield identical returns; no initial-state randomization is added.

The batch-script equivalent is `EVAL_ENVS_PER_MORPHOLOGY=32`.
PPO cycles each collect the YAML rollout horizon times
`--ppo-rollout-multiplier`, then update PPO before collecting again.

## Shared reference banks for replicas

Grouped morphology replicas now store reference joint/control/object trajectories,
collision landmark offsets, and reference landmark/palm trajectories once per
unique morphology on CPU and GPU. Each environment holds a bank index and its
own phase; lookup gathers only the current target for that environment. Geometry
construction and reference FK run once per morphology. Startup FK validation
still checks every physical replica, batching those checks by morphology.
Replica metadata must agree on reference paths, assets and geometry overrides;
inconsistent replicas are rejected before sharing data. Fixed-hand reference
storage is unchanged, and ungrouped batches retain independent bank entries.

For 32 designs and 128 replicas each, the float32 reference landmark tensor with
446 frames and 60 XYZ landmarks shrinks from 1,254.4 MiB to 9.8 MiB per copy.
This does not reduce independent simulator state, contact buffers, PPO rollout
storage, or the expanded manifest; total memory savings must be measured.

## PPO reward logging in co-training

With `--wandb`, rank zero mirrors SKRL PPO tracking aggregates directly to the
persistent outer W&B run once per rollout/update cycle. PPO no longer initializes
an extra W&B run per generation. TensorBoard output is preserved.

- `PPO/Reward / Episode return (mean)` (plus min/max) mirrors SKRL's recent
  completed-episode return statistics, averaged over the logging interval. It
  includes failures and appears after completed episodes become available.
- `PPO/Reward / Instantaneous reward (mean)` reports training step rewards.
- Other SKRL metrics, including losses and episode lengths, use the `PPO/` prefix.
- `PPO/train_steps` is cumulative vector steps across generations;
  `PPO/train_transitions` multiplies that by the global training environment count.
- `Evaluation/episode_return_mean` and `Evaluation/episode_return_max` explicitly
  name the deterministic morphology evaluation scores. Legacy evaluation metric
  names are retained for existing dashboards.

PPO charts use `PPO/train_steps`; evaluation charts use `Morphology/generation`.
W&B's internal history step advances automatically so interleaved PPO and
morphology logs do not move backward. This requires restarting training to load
the code change; already-running processes keep their existing logging behavior.

## Evaluation reset correction (2026-09-18)

SKRL IsaacLabWrapper.reset() only resets physics on its first call; later
calls return cached observations. The shared PPO evaluator previously called
reset() after training without rearming this behavior, so its evaluation
continued in-progress training episodes. Historical final phases therefore
are not survival lengths from phase zero, and historical morphology rewards
are partial-episode returns. This affects fixed-population evaluation and
SAC reward collection, not the PPO update calculation itself.

The evaluator now rearms `_reset_once`, resets inside inference mode, and
asserts all reference phases are zero. A regression test covers cached reset.
A live 32-environment probe confirmed ordinary reset leaves all phases at 20,
whereas the forced reset returns all phases to zero. For the fixed-two-hand
generation-99 checkpoint, the probe's subsequent fresh evaluation averaged
81.44 steps for hand 0 and 64.88 for hand 1 (zero successes). The scene uses
32 independent environments rather than the original grouped 4096 scene,
so these numbers are not an exact replay of the training physics layout.

Do not compare old partial-episode metrics directly with corrected fresh
evaluation returns. Existing checkpoints can be reevaluated without retraining.

## Direct captured-MANO targets (2026-09-19)

Palm-geometry shared PPO/co-training now accepts `--mano-rollout`, defaulting
 to `artifacts/isaaclab_mano_residual/refined_mano/successful_rollout.npz`.
It computes five fingertip positions and terminal directions from the captured
MANO `hand_q` states using the existing MANO FK implementation. These targets
are supplied directly to each candidate's GPU IK solve, rather than computed
from the source WUJI seed's fingertips. Each candidate starts from neutral, then initializes each subsequent frame
from its own preceding solution. Temporal regularization (0.015) and +/-0.20
rad limits apply relative to that previous solution only; frame zero has neither.
Wrist orientation follows captured MANO orientation with a candidate-specific
neutral-geometry alignment. Default iterations are now 14, matching the original
MANO-to-robot solver. No saved WUJI joint or wrist trajectory is read.

GPU fingertip FK uses `l_wrist`. Solved wrist poses are converted to the
virtual-root/base pose expected by the reference using the URDF fixed transform.

A run writes `mano_task_targets.npz` with all five MANO targets, thumb/index
world-space goal poses (semantic +Z terminal direction), the exact captured
`object_pose_wxyz`, and the capture path/SHA256. All frames are taken in order from the captured MANO rollout. Both candidate IK and
reference packaging use this target bundle. Candidate topology/contact offsets
still come from the prototype. `prepare_wuji_parametric_training_assets.py`
receives `--task-targets` and replaces template object and thumb/index targets;
legacy external callers without this argument retain the template behavior.
Do not combine `--task-targets` and `--fixed-reference`.

This changes future palm-geometry co-training runs. Existing checkpoints,
references, and prototype banks are not rewritten, and the generated hand
mass/inertia differences described in the investigation remain unchanged.

## Seed-independent sequential candidate IK

For palm-geometry shared PPO, omit `--seed-trajectory`; old commands supplying
it remain accepted but the file is ignored (with a startup message). Legacy
morphology-only tooling still supports its historical seed-based solver.
`wuji_sequential_retarget.py` batches candidates but traverses frames in time
order. It mirrors `retarget_all_hands.build_hand/solve_frame`: neutral pose,
unit-vector wrist alignment, five tip position/direction errors, DLS damping
0.025, direction scale 0.15, max update magnitude 0.12, temporal weight 0.015
after frame zero, and per-frame joint-change bound 0.20 plus joint limits.
The candidate's previous solution is the only temporal anchor.

The implementation uses parametric Torch kinematics and the mathematically
equivalent primal DLS system, not the original MuJoCo model or a claim of
bitwise solver identity. On the source hand's first eight captured frames,
mean fingertip position errors differed from the original solver by less than
0.06 mm. Full task success is a separate physics validation.

## Original source USD with fresh Torch references (2026-09-19)

`train_wuji_hybrid_sac_morphology.py --original-source-hand` runs one fixed
original WUJI asset through the shared PPO training/evaluation loop. It requires
`--num-morphologies 1`, positive `--ppo-cycles-per-generation`, and
`--ppo-observation-mode palm_geometry`. It automatically forces the zero source
vector and skips SAC updates. No prototype bank or generated USD is needed.

The original USD is `artifacts/isaaclab_all_hands_residual/assets/wuji_hand_2/hand.usd`.
Its matching prepared URDF and reference topology are retained. Each generation
still runs neutral-initialized sequential Torch IK on `--mano-rollout`; the
reference packager matches finger columns by joint name, replaces wrist/joint
commands and captured object targets, and preserves the original contact-link
names and offsets. The simulation uses ordinary single-hand physics replication,
with the original 385D surface-landmark observation and no morphology context.

Example (from the repo root, inside the Isaac Lab Conda environment):

```bash
set -o pipefail
RUN_NAME="wuji_original_usd_torch_ppo_$(date +%Y%m%d_%H%M%S)"
DEXCODESIGN_OBJECT_USD_PATH="$PWD/artifacts/isaaclab_mano_residual/assets/g04_1.usd" \
./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py \
  --original-source-hand --num-morphologies 1 \
  --train-envs-per-morphology 4096 --eval-envs-per-morphology 32 \
  --ppo-cycles-per-generation 64 --ppo-rollout-multiplier 1 --generations 100 \
  --ppo-observation-mode palm_geometry --geometry-inward-direction-mode kinematic_normal \
  --mano-rollout "$PWD/artifacts/isaaclab_mano_residual/refined_mano/successful_rollout.npz" \
  --retarget-iterations 14 --optimizer-backend skrl --sac-updates 0 \
  --seed 42 --continue-after-success --headless \
  --wandb --wandb-project DexCoDesign --wandb-group wuji-original-usd-torch-ppo \
  --wandb-run-name "$RUN_NAME" --output-root "$PWD/artifacts/wuji_sac/$RUN_NAME" \
  2>&1 | tee "$RUN_NAME.log"
```

The existing logger still appends `_sac` to the outer W&B run name, even though
morphology optimizer updates are skipped. PPO and fresh deterministic evaluation
metrics continue to use that run. Generation boundaries still recreate the
simulation and load the preceding PPO checkpoint; this mode does not make the
outer loop equivalent to uninterrupted standalone PPO.

Validation: 13 focused tests passed. A two-generation, four-environment smoke
run completed at `artifacts/diagnostics/original_source_hand_torch_smoke`,
including PPO checkpoint reload, 385D observations, original 22 hand colliders,
phase-zero deterministic evaluation, and skipped SAC updates. No long training
was launched.

## Episode-weighted training reward plots (2026-09-20)

W&B-enabled shared PPO now additionally logs completed-episode return and length
means over a persistent training-step window. Set
`--ppo-episode-log-window-steps 1600` (the default). This is independent of the
16-step PPO rollout and the number of PPO cycles per generation. The existing
high-frequency metrics remain unchanged.

Use these new charts with `PPO/train_steps` on the x-axis:

- `PPO/Completed episodes / Return mean (window)`
- `PPO/Completed episodes / Length mean (window)`
- `PPO/Completed episodes / Count (window)`
- `PPO/Completed episodes / Window steps`
- `PPO/Completed episodes / Interrupted episodes (total)`

The logger accumulates raw rewards and lengths per environment, includes each
finished episode exactly once (including the terminal reward), and weights
means by episode counts. It does not use SKRL's 100-record deque or average
per-step means of completed-episode statistics. Distributed runs reduce sums
and counts across all ranks. Windows are nonoverlapping and persist across
runner/scene recreation; a final partial window is reported with its actual
step count. Episode accumulators are discarded before evaluation, while the
count of interrupted episodes is retained. Episodes still running at a logging
boundary can complete in a later window; no zero return is fabricated when a
window has no completions.

The added plots summarize stochastic training experience and can span changing
morphology populations and policies. They reduce short-window noise but do not
remove completion-time selection effects or guarantee smooth curves. Frozen
phase-zero evaluation remains the primary performance measure. No reward,
termination, rollout, optimizer, or checkpoint behavior was changed. Historical
W&B records cannot be reconstructed into these exact episode-weighted metrics
from the old aggregate values alone; new runs get these additional charts.

## 2026-09-21: Separate WUJI palm/base contact hulls

The source rigid-part graph concatenates `r_base_link` and `r_wrist` into part
zero. Cooking that combined surface as one convex hull filled the indentation
between the mounting base and palm. WUJI exporters now emit two collision
elements on the same rigid link (22 hand colliders total, no extra joint).
`morphology/wuji_palm_collision.py` uses verified source-link face provenance,
restores base vertices from the source, and retains the designed palm shell.
Topology-changing remeshing fails explicitly rather than silently using the
wrong partition. Materialized and runtime USD overlays cancel palm affine
transforms on the named/tagged fixed base. The 385D observation size and action
layout remain unchanged.

The corrected cached bank for local experiments is:

```
artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70_split_base_v1
```

Use that directory with `--prototype-bank-root`. The old bank and previous
training runs are unchanged. `prepare_wuji_split_palm_bank.py OLD_BANK
--output-root NEW_BANK` creates the correction without rerunning retargeting.
The copied bank shares immutable compiled graphs, reference motions and finger
mesh dependencies with the old bank; keep the old bank available. Source bank
USDs are flattened into new local assets before changing the palm colliders.
The trainer warns when a legacy bank lacks the partition signature.

Validation: all 32 corrected prototypes have 22 collision meshes and identical
base vertices; original and corrected base/palm hulls returned PhysX
`RESULT_VALID`; exporter smoke produced 22 collision elements with 20 finger
joints; regression tests cover fixed base, editable palm, provenance rejection,
and cancellation of repeated runtime affine transforms. This is geometry and
collision-cooking validation, not a new training-quality measurement. Generated
mass/inertia and finger-surface differences from the original USD still remain.


## 2026-09-21: Grouped hand physics overrides and contact-buffer fix

Runtime debugging found a major difference between original-source PPO and
morphology-batch PPO: `_spawn_grouped_morphology_sources()` used a bare
`UsdFileCfg` to load the hand super-environment, then constructed `Articulation`
with `spawn=None`. The configured hand spawn properties never ran. Live grouped
hands consequently retained gravity and USD solver iterations 32/1 instead of
configured gravity-off and 8/2; the configured 1 m/s depenetration limit was also
missing. Self-collision was already disabled. Actuator gains, armatures and
limits matched at runtime.

The super-environment spawner now forwards hand physics properties before
cloning and before separately spawning objects. This does not change the bank,
hand mass/inertia, reference construction, observation, reward, or PPO settings.
A 4096-env validation also exposed contact-patch buffer overflow during
synchronized evaluation (175395 required versus default 163840). The environment
now reserves 524288 patches. The overflowed exploratory run was stopped; reported
training results come from a new clean run.

Controlled frozen-policy evaluation, same checkpoint and 16 phase-zero
deterministic episodes: old grouped generated hand 97.00 mean steps / 81.17
return; corrected generated hand 323.62 / 614.67; original USD 334.12 / 626.96.
Single-setting and paired ablations show interacting solver/gravity effects;
gravity-only does not resolve the failure. Generated runtime landmark FK matches
the observation FK within 0.6 micrometres, ruling out a gross USD joint-axis
interpretation mismatch in these tested rollouts.

Fresh five-generation tests used seed 42, 4096 environments, 64 PPO cycles of
16 steps per generation, 32 deterministic evaluation episodes, 14 retarget
iterations, 385D observations and no SAC updates. Original baseline reproduced
all five historical points exactly. At generation 5:

| Asset/settings | Mean episode return | Mean episode steps |
|---|---:|---:|
| Original USD | 203.16 | 144.88 |
| Generated, old physics (previous run) | 101.93 | 98.03 |
| Generated, corrected physics | 183.95 | 136.38 |

The corrected generated curve increases from 96.44 to 106.28, 120.00, 125.00,
and 136.38 steps; maximum in the final evaluation is 172 steps. Both new runs
completed with no contact-buffer overflow or traceback. This restores early
learning beyond the old ~98-step plateau, but is not evidence of full task
completion or identical multiseed convergence. Geometry and inertial differences
listed in the earlier audit remain.

Clean run: `artifacts/wuji_sac/debug_corrected_4096_5gen_20260921_patch524288`.
Baseline: `artifacts/wuji_sac/debug_original_4096_5gen_20260921`.
Evidence, scripts, raw live-physics snapshots, ablations, and plots:
`artifacts/analysis/wuji_runtime_debug/README.md`.
Generated generation-0 USD hashes and reference arrays match the failed run
exactly; the fix operates at scene spawning. Existing training commands use it
automatically. Regression plus related evaluation tests: 8 passed.
