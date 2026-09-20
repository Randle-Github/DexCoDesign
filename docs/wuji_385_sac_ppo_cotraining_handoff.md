# WUJI 385-D SAC + PPO co-training handoff

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
- `temp/hocap_mano_replay/scripts/prepare_wuji_parametric_usd_smoke.py`
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
