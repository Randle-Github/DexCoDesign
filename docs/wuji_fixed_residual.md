# Fixed original WUJI residual PPO

`train_wuji_fixed_residual.sbatch` trains joint residual control on the original
left WUJI model at
`artifacts/isaaclab_all_hands_residual/assets/wuji_hand_2/hand.usd`.
It uses the existing retargeted reference and object asset. There is no
morphology generation, palm-prototype bank, or SAC optimizer in this job.

## Observation

The new task is `DexCoDesign-WUJI-PalmGeometry-Residual-Direct-v0`.
Its separate SKRL PPO actor and value network both consume the same 189 values
through `OBSERVATIONS`, with the existing running observation standardizer.

| Indices (Python slice) | Feature | Values |
| --- | --- | --- |
| `0:75` | Actual finger landmarks in the current palm frame | 75 |
| `75:150` | Reference finger landmarks in the current palm frame | 75 |
| `150:157` | Reference palm pose relative to the current palm | 7 |
| `157:164` | Current object pose relative to the current palm | 7 |
| `164:171` | Reference object pose relative to the current palm | 7 |
| `171:185` | Demonstration thumb/index goal poses relative to the current palm | 14 |
| `185:188` | Unit gravity direction in the current palm frame | 3 |
| `188:189` | Reference phase divided by the last frame index | 1 |

The 25 landmarks are the child-frame origins of the 20 finger joints, followed
by five physical tips. Both lists follow the original wrapped URDF order:
thumb, index, middle, ring, pinky, with each finger's joints ordered along its
chain. Tip positions include the fixed tip-joint translation on the distal
body, so they do not require fixed fingertip links to survive USD import.
Positions are in meters; pose quaternions use wxyz with a nonnegative scalar
component. The palm frame is the original model's registered palm body frame.

Reference landmarks and palm poses are computed once from `hand_q` using FK of
the original wrapped URDF. Reference points are expressed in the **current**
palm frame, retaining errors in palm motion. The independently stored
demonstration fingertip targets remain separate from reference FK geometry.
Gravity and the relative palm goal retain the information needed for wrist
control. This first geometry variant does not include joint angles, velocities,
or explicit morphology context.

The first observation checks URDF FK against the imported articulation's
landmarks and palm position (50 micrometer tolerance). This catches a
mismatched reference skeleton or USD asset before policy training.

The actions, residual scales, object-tracking/contact reward, network sizes,
and PPO settings are shared with the existing fixed-hand task. Successful and
farthest training rollouts are saved; success does not stop training, so both
observation modes can receive the same budget. These captures come from
training exploration and are not deterministic policy evaluation results.

## Launch and compare

First inspect the command without starting Isaac Sim or submitting a job:

```bash
DRY_RUN=1 bash train_wuji_fixed_residual.sbatch
```

On the cluster, use the existing simulation environment. Override `CONDA_SH`
and `CONDA_ENV` if its location differs from the hybrid morphology job:

```bash
# Small runtime test: builds both networks and performs two PPO updates.
NUM_ENVS=8 MAX_ITERATIONS=2 WANDB_ENABLED=0 sbatch train_wuji_fixed_residual.sbatch

# Matched training runs; repeat with RUN_SEED=43 and 44.
RUN_SEED=42 sbatch train_wuji_fixed_residual.sbatch
OBSERVATION_MODE=legacy RUN_SEED=42 sbatch train_wuji_fixed_residual.sbatch
```

Defaults are 1024 environments, 1000 PPO iterations, and seed 42. With the
existing rollout length of 16, this is 16,000 vector environment steps.
Outputs and PPO logs live under
`artifacts/wuji_fixed_residual/<observation_mode>/seed<seed>_<job_id>/`.
`REFERENCE_PATH`, `DEXCODESIGN_OBJECT_USD_PATH`, `OUTPUT_ROOT`, and
`PPO_CHECKPOINT` can be overridden. Resume only a checkpoint trained with the
same observation mode: legacy WUJI has 86 inputs and geometry has 189.

## Inspect a trained policy

Use the new play task for a complete reference video:

```bash
DEXCODESIGN_HAND_ID=wuji_hand_2 ./isaaclab.sh -p \
  scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-WUJI-PalmGeometry-Residual-Direct-Play-v0 \
  --algorithm PPO --num_envs 1 --checkpoint /path/to/checkpoints/best_agent.pt \
  --video --video_length 445 --headless
```

The corresponding `DexCoDesign-WUJI-PalmGeometry-Residual-Direct-Eval-v0`
task retains training failure thresholds. Use it with `play.py` to inspect
whether the deterministic policy completes the motion under those thresholds.
For the legacy checkpoint, use the existing
`DexCoDesign-Hand-Residual-Direct-Play-v0` or `-Eval-v0` task instead.
Export the same reference and object paths used for training if you overrode
them. Compare deterministic completion, object tracking errors, and contact
behavior alongside the training return and value-loss curves. A visually
complete play video alone is insufficient because the Play task disables
object-loss termination.

## Local verification

```bash
python -m pytest tests/test_palm_geometry_observation.py -q
```

These tests check frame transforms against independent rotation math,
translation/yaw invariance, retained target errors and phase, quaternion signs,
batched FK and joint indexing, and the recorded reference fingertip positions
when the prepared original WUJI assets are present. Simulator startup, actor
and critic updates, and learning performance require the Isaac Lab GPU runtime.
