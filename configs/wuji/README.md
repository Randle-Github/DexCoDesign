# WUJI Skynet experiment configurations

| Configuration | Designs per generation | Training environments per design | Total training environments | SAC updates per generation |
| --- | ---: | ---: | ---: | ---: |
| `fixed8_palm0_ppo_skynet.yaml` | Same eight saved designs | 512 | 4096 | Disabled |
| `fixed32_palm0_ppo_skynet.yaml` | Same 32 saved designs | 128 | 4096 | Disabled |
| `codesign8_palm0_sac_ppo_skynet.yaml` | Eight proposals | 512 | 4096 | 64 |
| `codesign32_palm0_sac_ppo_skynet.yaml` | 32 proposals | 128 | 4096 | 64 |

The eight-hand configuration uses
`artifacts/wuji_sac/fixed_populations/wuji_random8_palm0_seed42.npy` with shape
`(8, 23)`. The separate 32-hand configuration uses `wuji_random32_palm0_seed42.npy`
in the same directory with shape `(32, 23)`. Its first eight rows exactly match
the eight-hand population; 24 additional random designs extend the same
seed-42 sequence. All use palm prototype zero. These vectors remain fixed
across generations, while assets and references are prepared each generation.
The artifact files are not tracked by Git and must also be present on Skynet.

## Joint-space PPO counterparts

The existing WUJI joint-space observation is **86D**, not 85D:
26 current joint positions + 6 current thumb/index positions + 7 object pose
values + 14 reference thumb/index pose values + 26 reference joint positions
and 7 reference object pose values. The controller still outputs the same
26 residual actions.

| Palm-geometry configuration | Joint-space configuration |
| --- | --- |
| `fixed32_palm0_ppo_skynet.yaml` | `fixed32_palm0_joint_space_ppo_skynet.yaml` |
| `codesign8_palm0_sac_ppo_skynet.yaml` | `codesign8_palm0_joint_space_sac_ppo_skynet.yaml` |
| `codesign32_palm0_sac_ppo_skynet.yaml` | `codesign32_palm0_joint_space_sac_ppo_skynet.yaml` |

Each counterpart uses `PPO_OBSERVATION_MODE: legacy`,
`MORPHOLOGY_CONTEXT: false`, and a fresh PPO policy (`PPO_CHECKPOINT: null`).
It appends `--retarget-per-generation` to `args`. That flag is necessary to
keep the same direct MANO retargeting and actual rollout object reference in
legacy mode; `RETARGET_PER_PROPOSAL: true` alone does not select this branch in
the shared-PPO trainer. Asset preparation and scene recreation still run each
generation, and PPO checkpoints carry forward within each run.

The new configurations have distinct `run_name_prefix` and `WANDB_GROUP`
values, with the same W&B entity/project. Their training parameters match the
corresponding palm-geometry configuration. The fixed-32 comparison uses exactly
the same saved design vectors. Co-design runs can select different hands as
their separately trained controllers produce different SAC rewards.

At preparation time, the three Skynet palm-geometry YAMLs had been switched
to `partition: overcap`, `account: overcap`, `qos: short`; the new joint-space
YAMLs preserve that routing and the A40/128 GB/24-hour request. The local
palm-geometry YAMLs still specify the lab allocation. To submit a joint-space
variant to the lab instead, change its partition and account to
`ravichandar-lab` (keep `qos: short` and `gpus: a40:1`).

Example:

```bash
cd /coc/flash5/yhan389/DexCoDesign
python3 scripts/tools/submit_wuji_training.py configs/wuji/fixed32_palm0_joint_space_ppo_skynet.yaml --dry-run
python3 scripts/tools/submit_wuji_training.py configs/wuji/fixed32_palm0_joint_space_ppo_skynet.yaml
```

The original palm-geometry co-design configurations use the existing trainer
and batch launcher. They keep the corrected prototype bank, 385D observation, seed 42,
64 PPO cycles per generation, 16-step PPO rollout (multiplier 1), 32 deterministic
evaluation episodes per design, 14 retargeting iterations, 1600-step episode
logging window, and 100 generations. Each requests one A40 on the lab account,
128 GB RAM, and a 24-hour time limit. Requesting 24 hours does not guarantee that
all 100 generations finish within that allocation.

## Enable design-policy learning

The essential settings are:

```yaml
FIXED_PPO_VECTORS: null
FORCE_SOURCE_MORPHOLOGY: false
OPTIMIZER_BACKEND: skrl
SAC_UPDATES: 64
SAC_BATCH_SIZE: 64
```

Keeping a fixed vector file or forcing the source morphology bypasses
`optimizer.observe(...)`, even if `SAC_UPDATES` is positive. Clearing the file
also means these runs do not start with the eight saved random designs. They
use the existing proposal mixture: SAC samples, uniform samples, elite
mutations, and elite replay. Counts are rounded to the population size (the
default 0.05 elite replay fraction rounds to zero with eight designs). The
source-hand baseline occupies slot zero each generation.

`FIXED_PALM_PROTOTYPE: 0` fixes the discrete prototype choice. It does not freeze
the remaining design coordinates, including the continuous palm affine edits.

Each generation prepares and retargets that generation's designs, creates a
simulation scene, trains the shared PPO controller for 64 cycles, then freezes
it and evaluates fresh deterministic episodes from phase zero. The mean return
of each design's 32 episodes supplies one morphology reward to SAC. SAC then
runs 64 gradient steps; PPO weights carry forward to the next generation.

`SAC_BATCH_SIZE` counts design/reward replay entries, not the 4096 PPO
environments. The current SKRL sampler uses fewer entries until the replay
buffer contains 64. The 0.01 SAC reward scale and proposal mixture values are
the existing batch-launcher defaults, made explicit in the new configurations.

Both co-design sizes initialize PPO and SAC from scratch. To initialize only
PPO from a checkpoint, set `PPO_CHECKPOINT` to a path available on Skynet.
Its observation mode and input dimension must match the selected variant
(385D palm geometry or 86D legacy). This does not resume an old SAC optimizer.

## Submit

On Skynet:

```bash
cd /coc/flash5/yhan389/DexCoDesign
python3 scripts/tools/submit_wuji_training.py configs/wuji/codesign8_palm0_sac_ppo_skynet.yaml --dry-run
python3 scripts/tools/submit_wuji_training.py configs/wuji/codesign8_palm0_sac_ppo_skynet.yaml
```

For the 32-design experiment, substitute
`configs/wuji/codesign32_palm0_sac_ppo_skynet.yaml`.

W&B uses `njyunhai-georgia-tech/DexCoDesign` with separate co-design groups and
timestamped run names. The historical `_sac` suffix is appended by the trainer.
Outputs are under `artifacts/wuji_sac/${RUN_NAME}`. The submission tool clears
stale experiment exports before applying the YAML.

Validation: all seven configurations passed the submission helper's dry run.
The three joint-space variants also passed Skynet `sbatch --test-only` checks;
five submission tests and nine evaluation/retargeting tests passed. No training
job was launched as part of creating these files.
