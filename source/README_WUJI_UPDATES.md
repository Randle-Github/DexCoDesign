# WUJI residual PPO and morphology co-training updates

Updated 2026-09-22. Changes since `fd73d0c` ("Scale WUJI co-training reference
storage and add PPO reward logging"). See the
[co-training handoff](../docs/wuji_385_sac_ppo_cotraining_handoff.md) for commands
and the full workflow.

## Generated-hand runtime physics

**Bug:** grouped spawning loaded the hand super-environment with a bare
`UsdFileCfg`, then constructed the articulation with `spawn=None`. The configured
hand physics overrides never ran, despite matching configuration values.

| Hand property | Old grouped path | Corrected / original path |
|---|---|---|
| Gravity | Not disabled | Disabled |
| Solver position / velocity iterations | 32 / 1 from USD | 8 / 2 |
| Maximum depenetration velocity | Configured override omitted | 1 m/s |
| Self-collision | Disabled | Disabled |

`ManoResidualEnv._spawn_grouped_morphology_sources()` now forwards hand spawn
properties before cloning and before separately spawning objects. Object gravity
remains enabled. This fix does not alter hand masses, inertia, references,
reward terms, PPO hyperparameters, or the 385D observation layout.

4096-environment validation also exceeded the default 163840 GPU rigid contact
patches (PhysX requested at least 175395). The environment now reserves
`gpu_max_rigid_patch_count=2**19` (524288). This is contact-buffer capacity, not
solver iterations. The overflowing exploratory run was excluded and restarted.

## Fresh evaluation after PPO training

**Bug:** SKRL's Isaac Lab wrapper physically reset only on its first `reset()`;
later calls returned cached observations. Evaluation after training continued
episodes already in progress. A high final reference phase therefore did not
establish survival from phase zero under the frozen evaluation policy.

The shared evaluator now re-arms the wrapper reset, refreshes observations, and
checks starting phases are zero. The frozen policy uses mean actions. Each
selected environment contributes only its first episode, with actual
`episode_steps_min/max/mean`, `evaluation_start_phase`, and
`evaluation_action_mode`, including per-hand W&B metrics.

Older evaluation curves should not be treated as fresh full-episode evaluations.
This does not invalidate all historical PPO training rewards or make their
checkpoints unusable.

With 64 PPO cycles and 16 rollout steps, each generation trains for 1024 vector
environment steps. `generation_094.pt` is saved after the final cycle of
generation 94, immediately before evaluation; it is not the best intermediate
cycle. Loading the preceding generation preserves shared-policy learning.
Generation 94 thus represents 6080 total PPO cycles with these settings.

## Direct captured-MANO references and sequential Torch IK

The palm-geometry shared-PPO path takes `--mano-rollout` as its task source.
`wuji_mano_task_targets.py` extracts fingertip targets using captured MANO FK and
uses the **actual captured object poses** as object references. Packaging no
longer mixes these hand targets with an older prototype bank's object motion.
Capture paths and SHA256 hashes record provenance.

`wuji_sequential_retarget.py` batches candidates and traverses frames in order.
Each candidate starts at neutral joints and regularizes to its own previous
solution. Wrist orientation follows MANO with a candidate-specific fixed
neutral-frame alignment; wrist translation and finger joints are solved. The
default is 14 iterations. No saved WUJI trajectory supplies targets or an anchor
in this path. `--seed-trajectory` remains accepted but is ignored with a message;
legacy morphology-only tooling retains seed-based behavior.

The objective and limits follow the original sequential solver, but Torch FK
and the primal damped-least-squares solve are not bitwise MuJoCo equivalence.
`--original-source-hand` provides an original-USD comparison using fresh Torch
references in the same PPO loop, with asset generation and SAC updates disabled.
Fixed generated populations use `--fixed-ppo-vectors`; the generated zero source
design can be held with `--force-source-morphology --fixed-palm-prototype 0`.

Asset preparation is renamed to `prepare_wuji_parametric_training_assets.py`.
Call sites use this name; `prepare_wuji_parametric_usd_smoke.py` remains a CLI shim.

## Separate base and palm contact hulls

The canonical palm combines the source base and wrist/palm links. One combined
convex hull filled the indentation between them. Exporters now emit separate
base and palm collision elements on the same rigid link: 22 hand collision
meshes instead of 21, with no additional joint. Partitioning verifies source-link
face provenance and rejects incompatible remeshing.

The base remains fixed; the palm shell stays editable. Materialized USD and
runtime overlays cancel palm scale/yaw transforms on the tagged base.
`prepare_wuji_split_palm_bank.py OLD_BANK --output-root NEW_BANK` creates a
corrected bank copy. The local corrected bank is:

```text
artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70_split_base_v1
```

The trainer warns about banks lacking `source_base_and_palm_v1`. Code updates
alone do not change existing cached collision hulls. Corrected banks may share
immutable finger meshes, compiled graphs and references with the old bank;
retain those dependencies. The old bank is not rewritten.

## Episode-weighted PPO logging

`--ppo-episode-log-window-steps 1600` adds W&B metrics:

- `PPO/Completed episodes / Return mean (window)`
- `PPO/Completed episodes / Length mean (window)`
- Completed count, actual window steps, and cumulative interrupted-episode count.

Each completed episode has equal weight and includes its terminal reward.
Nonoverlapping windows persist across generations. Unfinished episodes lost at
scene recreation are counted separately, never joined across resets. Empty
completion windows do not fabricate zero returns. Distributed runs reduce sums
and counts. The older high-frequency metrics remain available.

This changes logging only, not PPO rollout size or optimization. Window means
still have completion-time selection effects; fresh frozen evaluation remains
the more direct checkpoint-performance measure.

## Visibility, continuous playback, and manifest validation

Some generated assets have visual meshes only for palm/base. The old display
helper returned when it found any visual mesh, leaving the 20 finger-link
collision meshes hidden as `guide`. It now checks each link and exposes its
collision surfaces when visuals are absent, without adding collision shapes.
Deinstancing happens during scene setup **before** PhysX tensor initialization;
doing it afterward could invalidate views and cause `getRootTransforms` errors.

- Evaluation supports `--show-hand-geometry` and optional `--stochastic` actions;
  deterministic mean actions remain the default, without policy updates.
- `play.py --show-hand-geometry --video --video_length 900 --keep-playing` records
  one local clip and continues playback afterward. Videos go to the checkpoint
  run's `videos/play/` directory. Playback disables exit-on-success capture.
- Running `play.py` with `DexCoDesign-WUJI-PalmGeometry-Residual-Direct-Eval-v0`
  preserves failure thresholds, resetting completed episodes while the app
  stays open. The separate Play task relaxes failure thresholds and should not
  be used for quantitative evaluation.
- For ordinary non-grouped manifests, row count must match `--num_envs`.
  Changing `--num_envs` alone does not resize assets, references or geometry.
  The previous 64-row/8-env mismatch produced 3840 versus 480 landmark entries;
  it now raises an actionable error before scene creation. Grouped training
  retains its distinct super-environment replication layout.
- Eight environments for two hands require four complete rows per hand. Simply
  taking the first eight of 32 consecutive hand-0 replicas omits hand 1.
  W&B `hand_0` is the source design and `hand_1` the fixed variant.
- Checkpoints do not embed the manifest, USD assets, or reference trajectories;
  these dependencies must exist locally.

## Validation and remaining differences

Repository-level `tests/`: **74 passed** in the Isaac Lab Python environment
with USD libraries available and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Coverage
includes fresh resets, episode statistics, references, retargeting, parametric
geometry, grouped spawn overrides and manifest-size validation. This is not the
entire upstream Isaac Lab test suite.

Local headless playback checks with two and eight environments loaded generation
094, reset successfully, rendered fingers and wrote 30-step MP4 clips. Visibility
inspection exposed 20 finger meshes while keeping the collision count at 22.

Controlled frozen-policy evaluation used the same checkpoint and 16 fresh,
phase-zero deterministic episodes:

| Asset/runtime settings | Mean return | Mean episode steps |
|---|---:|---:|
| Generated, old grouped physics | 81.17 | 97.00 |
| Generated, corrected physics | 614.67 | 323.62 |
| Original USD | 626.96 | 334.12 |

Fresh five-generation runs used 4096 environments and 64 PPO cycles per
generation. Final mean steps were 136.38 (corrected generated), 144.88 (original),
and 98.03 (prior generated run). These single-seed tests establish the bug's
effect, not identical convergence or universal task completion. Local detailed
evidence is under `artifacts/analysis/wuji_runtime_debug/` and is not shipped.

Generated mass/inertia still differ from the original: approximately 1.05 kg
total versus 0.6897 kg; palm/body mass 0.35 kg with diagonal inertia 8e-4 kg m^2;
each finger link 0.035 kg with diagonal inertia 1.5e-5 kg m^2. Generated centers
of mass are at link origins, and these constants are not recomputed when the
geometry changes. Joint friction also differs. Gravity-off does not eliminate
inertial effects; the fixes do not establish full dynamic equivalence.

Phase-zero evaluation replicas currently have no deliberate initial-pose
perturbations, so they are not independent randomized initial conditions.
Deterministic policy actions and a fixed seed do not guarantee bitwise GPU
physics reproducibility. Local checkpoints, banks, videos and W&B data are not
included in this source update.
