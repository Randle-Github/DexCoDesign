# WUJI training on Skynet

Runtime investigation started September 22, 2026. This note concerns Slurm,
paths, runtime libraries, collision cooking caches, and logging. No training
Python, reward, observation, optimizer, or physics parameters were edited for
this investigation. Short diagnostic jobs override only their requested test
size/duration and explicitly selected experiment arguments.

## Checkout and environment

- Repository: `/coc/flash5/yhan389/DexCoDesign` (use this physical path on compute nodes).
- Branch: `Yunhai/integrate-all`, commit `36ec10d`, also the GitHub branch head
  at the time of inspection.
- Conda: `/coc/flash5/yhan389/miniconda3`, environment
  `DexCoDesign_sim4_5_lab_2_3_2`.
- Isaac Sim: `/coc/flash5/yhan389/apps/isaacsim-4.5.0` via `_isaac_sim`.
- Core packages match the desktop: Torch `2.7.0+cu128`, SKRL `2.1.0`,
  Gymnasium `1.2.1`, NumPy `1.26.4`. Cluster W&B is `0.29.0`.
- The desktop's `--retarget-per-generation` joint-space change was deliberately
  not copied during the original runtime-only task. On September 23, the user
  requested joint-space variants of the three Skynet experiments, so this
  existing trainer change was synced separately with the new YAMLs. Those
  variants explicitly pass the flag to retain per-generation MANO retargeting
  with the existing 86D legacy observation. The palm-geometry branch continues
  to retarget automatically. The joint-space configurations and trainer support
  are included together in this update.

## Failures reproduced

The baseline launcher completed two tiny PPO generations on `omgwth`, `deebot`,
and `clippy` (jobs `3931299`–`3931301`), but PhysX reported
`Unable to create convex mesh` and the UJITSO datastore reported missing blocks.
All three evaluated to phase 4. Exit status zero alone therefore does not
validate this training.

A controlled repeat on `deebot` (job `3931308`) used the same source code,
asset, and numerical training arguments, with the two cache switches below.
The convex-mesh failures disappeared and final evaluation reached phase 56.
This is a runtime regression check, not evidence of a learned successful policy.

Earlier logs also contain missing `libGLU.so.1`, startup native crashes, missing
data files, incorrect Slurm working directories, timeouts, and killed processes.
Those older failures do not alone establish that a node is unusable today.

## Launcher changes

`train_hand_hybrid_sac_morphology.sbatch` now:

- Stops on command/pipeline failures and resolves the repository's physical path.
- Checks GLU loading and records Python/package versions, Git commit, and GPU list.
- Runs a CUDA allocation/matrix-operation check before starting Isaac Sim.
- Temporarily excludes `bishop`: job `3931352` received an L40S with UUID
  `GPU-acdc37a9-ebba-2b42-4002-42119fe50097` and immediately reported
  `cudaErrorECCUncorrectable`, then Vulkan initialization failed and Isaac Sim
  segfaulted. This is evidence about that allocation, not all L40S hardware.
  Revisit the exclusion after administrators verify the node's GPU health.
- Uses a separate Kit user configuration file for each job.
- Defaults to direct collision cooking, bypassing the broken shared cache:
  `--/physics/cooking/ujitsoCollisionCooking=false` and
  `--/persistent/physics/useLocalMeshCache=false`.
  Set `PHYSX_DISABLE_MESH_CACHE=0` only for a deliberate cache comparison.
- Raises the file-descriptor soft limit up to 65,536 within the hard limit and
  disables core dumps to avoid filling shared storage after native failures.
- Reports `/usr/bin/time -v` resource usage when available.
- Forwards trailing command-line arguments without changing existing numerical
  defaults, so an explicit experiment can pass `--seed 42`, for example.
- When W&B is enabled, launches the unchanged program through
  `scripts/tools/run_isaac_with_wandb.py`. The wrapper calls `wandb.finish()`
  immediately before the program's existing `SimulationApp.close()` call.
  It preserves script arguments and forwards the original close arguments.

The cache switches affect how collision data is built/retrieved. Collision
approximation, geometry, gravity, solver iteration counts, and all PPO/SAC
parameters are unchanged by this patch.

## Slurm and paths

Use a login shell for remote submissions: the login node's non-login PATH does
not include Slurm. Its `scontrol --json` plugin is broken; use the text output.

Create `slurm_logs` **before** submitting; Slurm opens its output before the
batch script runs. Export absolute `/coc/flash5/...` asset paths. An exported
`/nethome/yhan389/flash/...` bank path failed on a compute node even though it
was present through the login node's home-directory alias.

The lab queue initially blocked tests with `QOSGrpGRES`. Overcap submission is:

```bash
cd /coc/flash5/yhan389/DexCoDesign
mkdir -p slurm_logs
sbatch --partition=overcap --account=overcap --qos=scavenger_qos \
  train_hand_hybrid_sac_morphology.sbatch
```

This bare command keeps the launcher's existing experiment defaults; export
your intended experiment first. Overcap is preemptible. It is useful for these
bounded tests but does not guarantee uninterrupted long training. Do not assume
the outer training driver automatically resumes an interrupted Slurm job.

### Lab QOS, wall time, and GPU quota (verified September 22)

The partition reports `MaxTime=04:00:00`, but this is **not** the effective
maximum when using the lab's `short` or `long` QOS. Both have the
`PartitionTimeLimit` flag, which allows overriding the partition time limit
([Slurm documentation](https://slurm.schedmd.com/sacctmgr.html)). `short` permits
up to two days, and `long` up to seven days. The account association allows both.
Earlier advice that lab jobs must finish in four hours missed this QOS override.

The default YAML now requests `ravichandar-lab`, `short`, one A40, and
`time: "1-00:00:00"` (24 hours). A matching `sbatch --test-only` request was
accepted with an estimated placement on `clippy`; no actual job was submitted.
The batch script itself still defaults to four hours when used without an
explicit time override. Requesting 24 hours is a runtime allowance, not a
minimum runtime: completed training exits earlier.

The lab partition's QOS currently grants eight A40 GPUs in aggregate and zero
L40S GPUs. Node membership alone does not establish GPU entitlement. Use A40
for the lab allocation; the L40S runtime test used overcap. No `nodelist` is
required: let Slurm select a compatible node, retaining the temporary `bishop`
exclusion. Pinning `dave` can reproduce the tested A40 node, but may wait longer.

## W&B

Use the same destination as desktop experiments:

```bash
export WANDB_ENABLED=1 WANDB_MODE=online
export WANDB_ENTITY=njyunhai-georgia-tech WANDB_PROJECT=DexCoDesign
```

Existing cluster credentials authenticated successfully. The first diagnostic
run was created at
<https://wandb.ai/njyunhai-georgia-tech/DexCoDesign/runs/ahzw1t5b>.
Its collision errors make it an invalid training baseline; use it only as
proof of connectivity. Diagnostic jobs use group `skynet-runtime-diagnostics`
and unique run names, without overwriting the desktop runs.

The first completed eight-hand jobs exposed a shutdown issue: cloud runs still
showed `running` with an older generation, despite complete local result JSONs.
Isaac Sim's default native fast shutdown bypassed Python's W&B cleanup. Trying
normal Kit teardown (`--/app/fastShutdown=false`, job `3931364`) completed PPO
but then segfaulted during native shutdown. Syncing the abandoned binary log
afterward also reported `unexpected EOF`.

The launcher therefore keeps fast shutdown and flushes W&B **before** closing
Isaac Sim via the small runtime wrapper. This avoids modifying the trainer,
reward/observation code, or optimizer behavior. The wrapper's close order,
failure status, and no-active-run cases were checked with a simulated abrupt
close; an actual full-size job validates the final cloud upload.

The wrapper imports the same `SimulationApp` class used by `AppLauncher`, wraps
only its final `close()` method, then executes the original script with its
original arguments. It does not intercept environment steps, PPO updates,
retargeting, checkpoints, or evaluation. W&B is not imported before Isaac
initialization, because some of its dependencies come from Isaac's bundles.

## Assets staged for the real eight-hand tests

Copied the corrected split-base bank, `wuji_random8_palm0_seed42.npy` (and its
metadata), and `refined_mano/successful_rollout.npz`. Rebased paths only in
the copied bank's URDFs, JSON, and compiled-directory symlink. The corrected
bank still depends on the existing original bank for reference files and
compiled finger geometry. No source training files were changed by this copy.

## Completed eight-hand smoke test

Job `3931345` on **dave (A40)** completed two generations in 6m30s:
eight fixed designs, four training environments/hand, two evaluation episodes/hand,
two PPO cycles/generation, 14 retargeting iterations, seed 42, and 385D observations.
The saved results confirm retargeting in both generations and deterministic
evaluation starting at phase zero. The checkpoint was carried to generation 1.
Best evaluated phase was 96 in both generations, with no successful full task
episodes. This validates execution, not trained-policy quality.

W&B: <https://wandb.ai/njyunhai-georgia-tech/DexCoDesign/runs/ugz2zjs2>.
The run name contains `deebot` because the initially queued job was moved to
`dave` before it started; the actual Slurm node was `dave`.

## Full-size results and node recommendation

Both jobs below used eight fixed designs, 512 training environments/hand
(4,096 total), 32 evaluation episodes/hand, 64 PPO cycles/generation, rollout
multiplier 1, retargeting iterations 14, seed 42, and 385D observations. SAC
updates were disabled. Each generation retargeted/rebuilt the scene; evaluations
started at phase zero. The Slurm allocation was one GPU, 16 CPU cores, and 128 GiB.

| Node | GPU | Job | Completed generations | Elapsed | Process peak RSS | Exit |
| --- | --- | --- | --- | --- | --- | --- |
| dave | A40 | 3931356 | 3 | 18m06s | 8.30 GiB | 0 |
| dynamics | L40S | 3931375 | 1 | 4m03s | 7.40 GiB | 0 |

These are tested choices for this environment. No convex-creation failures,
CUDA errors, or Python tracebacks occurred in the two successful job logs.
Keep `bishop` excluded pending GPU health repair; its failing allocation is
documented above. Other nodes have not all been validated with the full workload.
The RSS figures come from `/usr/bin/time`, not aggregate host RAM or GPU VRAM;
the launcher's existing 128 GiB request is unchanged.

On `dave`, mean fresh deterministic evaluation returns across all eight hands
were 148.82, 248.19, and 397.20; mean episode lengths were 124.48, 185.54, and
266.63 steps. Generation 2 had 2 successful episodes out of 256. This establishes
short-run execution and learning progress, not convergence or a guarantee of an
uninterrupted 100-generation run.

The final launcher, including the W&B close wrapper, was tested by job `3931375`:
<https://wandb.ai/njyunhai-georgia-tech/DexCoDesign/runs/c1lqd6bi>.
The API confirmed **finished**, generation 0, mean evaluation return
131.2879457473755, and mean episode length 118.21484375. These values match the
saved `shared_ppo_results.json`. The earlier A40 run predates the W&B shutdown
fix; use its local JSONs for complete results rather than assuming its cloud
summary contains the last generation.

Evidence is saved on both machines under `artifacts/analysis/skynet_runtime_debug/`:
`validation_summary.json`, `wandb_verified.json`, and the successful job logs.
The runtime updates are versioned with this note. The diagnostic results above
used the training code from `36ec10d`; these updates do not change that trainer.

## Reproduce the full-size bounded validation

### Submit from YAML (no terminal exports required)

The configuration `configs/wuji/fixed8_palm0_ppo_skynet.yaml` contains the same
fixed-eight-hand experiment, with 100 generations and a 24-hour
`ravichandar-lab` / `short` request. Changing the allocation does not change
the training budget.
Edit `env` for experiment settings, `slurm` for resource requests, and `args`
for the seed and completed-episode logging window. For example, change
`GENERATIONS: 100` to `GENERATIONS: 3` for the bounded test below.

```bash
cd /coc/flash5/yhan389/DexCoDesign

# Preview only; does not create run directories or submit a job.
python3 scripts/tools/submit_wuji_training.py \
  configs/wuji/fixed8_palm0_ppo_skynet.yaml --dry-run

# Submit once ready. The helper prints the job ID and monitoring commands.
python3 scripts/tools/submit_wuji_training.py \
  configs/wuji/fixed8_palm0_ppo_skynet.yaml
```

The login node's `/usr/bin/python3` already has PyYAML; Isaac/Conda activation
still happens inside the existing batch script. `${REPO_ROOT}` resolves to the
physical checkout containing the helper, and `${RUN_NAME}` is the configured
prefix plus a unique UTC timestamp. No shell expansion or command substitution
is performed inside YAML. Boolean values become `1`/`0`; `null` clears an override.
Quote Slurm times and numeric entries in the `args` list.

The helper ignores inherited batch experiment overrides, `SBATCH_*` options,
and `DEXCODESIGN_*` overrides before applying the YAML. Other environment values,
including W&B credentials, are preserved. Unspecified experiment options use
the existing batch defaults. Each submission saves `submission.yaml` and
`submission.json` (resolved overrides, command, and job ID) in the output
directory; secrets from the inherited environment are not included in those
records. Reusing an output directory with an existing submission record fails
instead of overwriting it. This helper does not modify or import training code.

For a short lab test, keep the supplied allocation and reduce `GENERATIONS`
to fit the limit. For overcap, set `partition` and `account` to `overcap`,
`qos` to `scavenger_qos`, and `time` to `"24:00:00"` (preemptible).
With overcap, set `gpus: "l40s:1"` for L40S; optionally add `nodelist: dynamics` to request
the tested node explicitly (which may increase queue time). The batch script's
temporary `bishop` exclusion applies unless explicitly overridden.

### Equivalent environment-variable submission

This uses the desktop fixed-eight-hand settings, with only the number of
generations reduced for the runtime test. It starts a fresh run.

```bash
cd /coc/flash5/yhan389/DexCoDesign
mkdir -p slurm_logs
export BANK_ROOT="$PWD/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70_split_base_v1"
export FIXED_PPO_VECTORS="$PWD/artifacts/wuji_sac/fixed_populations/wuji_random8_palm0_seed42.npy"
export NUM_MORPHOLOGIES=8 TRAIN_ENVS_PER_MORPHOLOGY=512
export EVAL_ENVS_PER_MORPHOLOGY=32 PPO_CYCLES_PER_GENERATION=64
export PPO_OBSERVATION_MODE=palm_geometry MORPHOLOGY_CONTEXT=0
export RETARGET_PER_PROPOSAL=1 RETARGET_ITERATIONS=14
export PPO_ROLLOUT_MULTIPLIER=1 GENERATIONS=3 SAC_UPDATES=0
export WANDB_ENABLED=1 WANDB_MODE=online
export WANDB_ENTITY=njyunhai-georgia-tech WANDB_PROJECT=DexCoDesign
export WANDB_GROUP=skynet-runtime-diagnostics
export WANDB_RUN_NAME="skynet_fixed8_validation_$(date +%Y%m%d_%H%M%S)"

sbatch --partition=overcap --account=overcap --qos=scavenger_qos \
  --gpus=a40:1 --cpus-per-task=16 --mem=128G --time=00:45:00 \
  train_hand_hybrid_sac_morphology.sbatch \
  --seed 42 --ppo-episode-log-window-steps 1600
```

Remove the partition/account/QOS overrides to use the launcher's lab allocation
instead. It may wait with `QOSGrpGRES`. Avoid fixing a node name unless needed:
free slots changed during this investigation and pinned tests received overnight
start estimates. GPU/node suitability and current scheduler availability are
separate checks.

For a later 100-generation experiment, retain the same experiment exports and
set `GENERATIONS=100` plus a unique run name/group and pass `--time=1-00:00:00`
for a 24-hour allocation. The lab's `short` QOS permits this through its
`PartitionTimeLimit` flag, despite the partition's displayed four-hour limit.
Overcap also accepts a 24-hour request but remains preemptible. No 100-generation
job was launched during this investigation.
