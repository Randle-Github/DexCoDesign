# HO-Cap to MANO and WUJI: End-to-End Workflow

This directory implements the complete DexCoDesign hand-learning pipeline for
one reviewed HO-Cap sequence:

```text
extracted HO-Cap data
  -> compact left-hand/object subset
  -> MANO joint-space reference
  -> MANO residual PPO
  -> captured successful MANO rollout
  -> WUJI inverse-kinematics retargeting
  -> WUJI Isaac Lab reference and USD
  -> WUJI residual PPO
  -> optional WUJI morphology SAC / SAC+PPO
```

All commands below are run from the repository root:

```bash
cd /home/yhan389/Desktop/DexCoDesign
conda activate DexCoDesign_sim4_5_lab_2_3_2
```

After activation, invoke preprocessing utilities with plain `python`; no
separate virtual-environment interpreter is required.

The maintained local simulator combination is Isaac Sim 4.5, Isaac Lab 2.3.2,
and Python 3.10. Paths in this document are repository-relative unless an
absolute path is shown.

## 1. Pipeline inputs and selected HO-Cap sequence

The current experiment uses:

| Field | Value |
|---|---|
| Sequence | `subject_7/20231022_192832` |
| Frames | 446 at 30 Hz |
| Hand | left |
| HO-Cap hand slot | 1 (`[right, left]`) |
| Object slot | 0 |
| Object ID | `G04_1` |
| Label camera | `043422252387` |

The already-extracted HO-Cap dataset is expected under:

```text
/home/yhan389/Desktop/HO-Cap/datasets/
  subject_7/20231022_192832/
    poses_m.npy
    poses_o.npy
    043422252387/label_*.npz
  models/G04_1/
  calibration/mano/subject_7.yaml
  calibration/extrinsics/extrinsics_20231014.yaml
```

`poses_m.npy` stores 51 MANO values per frame:

```text
global rotation vector (3) + MANO hand pose (45) + translation (3)
```

The 21 official `hand_joints_3d` labels provide the geometric wrist and finger
targets. The 45 MANO pose channels are not directly copied into robot joints.

## 2. Create the compact local subset

Run:

```bash
python temp/hocap_mano_replay/scripts/prepare_extracted_subset.py \
  --hocap-root /home/yhan389/Desktop/HO-Cap/datasets \
  --output-root temp/hocap_mano_replay/data/subset
```

This function:

1. selects left-hand slot 1 and object slot 0;
2. transforms the official 21 hand joints from camera coordinates to the
   HO-Cap world frame;
3. copies the G04_1 meshes and required calibration files; and
4. writes metadata documenting the selected slots and coordinate frames.

Expected output:

```text
temp/hocap_mano_replay/data/subset/
  subject_7/20231022_192832/
    mano_pose_left.npy
    object_pose_G04_1.npy
    hand_joints_3d_left.npy
    subset.json
  models/G04_1/
    cleaned_mesh_2000.obj
    cleaned_mesh_10000.obj
    textured_mesh.obj
    textured_mesh.mtl
    textured_mesh_0.png
  calibration/
    mano/subject_7.yaml
    extrinsics/extrinsics_20231014.yaml
```

## 3. Build the MANO reference trajectory

Run the temporally warm-started IK solver:

```bash
python temp/hocap_mano_replay/scripts/prepare_isaaclab_reference.py \
  --sequence-root temp/hocap_mano_replay/data/subset/subject_7/20231022_192832 \
  --iterations 32
```

`--iterations 32` is the maximum number of IK updates for each frame. Frame
`t` starts from the solution at frame `t-1`, so the trajectory is temporally
continuous rather than solving every frame from a neutral pose.

The solver tracks the reviewed wrist orientation and the position and distal
direction of all five fingertips.

This stage uses the shared `solve_frame()` retargeting objective. For each
frame, the wrist orientation is copied from the reviewed HO-Cap/MANO target
and treated as a hard constraint. The optimization variables are the MANO
wrist translation and active finger joints. The least-squares objective
contains:

1. five physical fingertip-position errors;
2. five distal-finger direction errors, weighted with an equivalent length of
   `0.15 m`; and
3. a finger-joint temporal regularizer toward the preceding frame, weighted
   by `0.015` after the initial frame.

The damped least-squares coefficient is `0.025`. Each solver update is capped
at `0.12`, and after the initial frame each active finger joint is constrained
to remain within `0.20 rad` of its warm-start value, as well as within the
joint's physical limits. The targets in this step come from the raw reviewed
HO-Cap 21-point hand trajectory and MANO wrist pose. Object contact, collision,
and object-pose errors are not part of this kinematic IK objective.

Conceptually, the per-frame objective is:

```text
sum_f ||p_f(q, t) - p_f_target||^2
  + 0.15^2 sum_f ||d_f(q) - d_f_target||^2
  + 0.015^2 ||q - q_previous||^2
```

where `t` is wrist translation, `q` contains the active finger joints, `p_f`
is a physical fingertip position, and `d_f` is a distal-finger direction. The
temporal term is omitted on the initial frame.

It writes:

```text
temp/hocap_mano_replay/data/subset/subject_7/20231022_192832/
  isaaclab_reference.npz
  isaaclab_reference.retargeting.json
```

The reference NPZ contains:

| Key | Meaning |
|---|---|
| `joint_names` | Ordered 28-DoF MANO joint names |
| `hand_q` | 6 virtual wrist DoFs plus 22 finger DoFs |
| `hand_ctrl` | Initial position-control reference |
| `object_pose_wxyz` | Object position and WXYZ quaternion |
| `fingertip_pose_wxyz` | Thumb/index reference poses |
| `fingertip_link_names` | Bodies used for policy tracking |
| `fingertip_offsets` | Surface offsets from the terminal bodies |
| `fps` | 30 Hz |

The 6 wrist DoFs are virtual prismatic/rotational joints used by the simulator;
they are not additional joints from the human MANO model.

### MANO coordinate mapping

The reviewed local alignment uses a fixed `-90 degrees` rotation about the
hand-local X axis:

```text
R_world(t) = R_mano_trajectory(t) @ R_local_to_mano
```

HO-Cap's last three MANO values are translation parameters rather than the
wrist joint itself. For subject 7, the constant MANO shape offset is:

```text
J0 = [-0.096753545, 0.006290510, 0.006127380] meters
```

## 4. Prepare the MANO hand and object USDs

The MANO task loads:

```text
artifacts/isaaclab_mano_residual/assets/mano_left.usd
artifacts/isaaclab_mano_residual/assets/g04_1.usd
```

The MANO USD must be regenerated whenever its URDF or collision meshes change.
On this local Isaac Sim 4.5 installation, run the URDF converter without
`--headless` so that the URDF importer extension is available:

```bash
mkdir -p artifacts/isaaclab_mano_residual/assets

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  assets/robot_hands/direct_motor/mano/left/hand.urdf \
  artifacts/isaaclab_mano_residual/assets/mano_left.usd \
  --fix-base \
  --joint-stiffness 300 \
  --joint-damping 34.6410162 \
  --joint-target-type position
```

Create the dynamic object asset if it is absent:

```bash
./isaaclab.sh -p scripts/tools/convert_mesh.py \
  temp/hocap_mano_replay/data/subset/models/G04_1/cleaned_mesh_2000.obj \
  artifacts/isaaclab_mano_residual/assets/g04_1.usd \
  --collision-approximation convexDecomposition \
  --mass 0.015 \
  --headless
```

The current MANO source has finger collision capsules aligned with each
finger's local X axis. If those source collision shapes are changed, regenerate
the direct-motor MANO URDF before converting the USD.

## 5. Validate the open-loop MANO reference

Before training, run the reference with a zero residual action:

```bash
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/diagnose_zero_residual.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --output artifacts/isaaclab_mano_residual/diagnostics/zero_residual_local.json
```

This diagnostic uses one environment and obtains the reference path from the
MANO environment definition. The default path is the
`isaaclab_reference.npz` generated in section 3. It records per-frame object
pose errors, joint tracking errors, fingertip contacts, and the first failure
threshold crossing.

Use `--ignore-early-termination` when the complete 446-frame diagnostic is
needed even after the first failure.

## 6. Train the MANO residual policy

The registered task IDs are:

| Task | Purpose |
|---|---|
| `DexCoDesign-MANO-Residual-Direct-v0` | training with early termination |
| `DexCoDesign-MANO-Residual-Direct-Play-v0` | full visual replay; failure thresholds disabled |
| `DexCoDesign-MANO-Residual-Direct-Eval-v0` | strict full-reference evaluation |

The policy executes a normalized residual in `[-1, 1]`:

```text
joint_target[t] = reference_target[t] + residual_scale * action[t]
```

The action scales and the hand/support collision option are configured in:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/config/mano_residual_env.yaml
```

The PPO architecture and hyperparameters are in:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/agents/skrl_ppo_cfg.yaml
```

Start a fresh training run after changing a hand USD, collision mesh, object
USD, residual-action range, or collision setting:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-MANO-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless \
  --video \
  --video_length 1500 \
  --video_interval 5000
```

Training output is placed under:

```text
logs/skrl/mano_residual/<timestamp>_ppo_torch/
  checkpoints/
  params/env.yaml
  params/agent.yaml
  videos/
```

The current SKRL configuration uses separate policy/value networks, 16-step
rollouts, 5 PPO learning epochs, 4 mini-batches, W&B logging, and a checkpoint
interval of 2000 trainer timesteps (125 PPO updates with 16-step rollouts). The
environment YAML values are copied to the run and logged to W&B by the training
entry point. `--video_interval` is also measured in environment steps.

### MANO observations and actions

For `N=28` controlled joints, the observation has `2N + 34 = 90` values:

```text
current joint positions
+ current thumb/index fingertip positions
+ current object pose
+ goal thumb/index fingertip poses
+ goal joint positions
+ goal object pose
```

The 28 actions control 3 wrist translations, 3 wrist rotations, and 22 finger
joint residuals. Actions are clipped to `[-1, 1]` before applying their scales.

## 7. Evaluate and capture a refined MANO rollout

Set the checkpoint and desired capture path:

```bash
MANO_RUN="replace_with_the_mano_run_directory"
CHECKPOINT="$PWD/logs/skrl/mano_residual/$MANO_RUN/checkpoints/best_agent.pt"
CAPTURE="$PWD/artifacts/isaaclab_mano_residual/refined_mano/successful_rollout.npz"
mkdir -p "$(dirname "$CAPTURE")"
```

Evaluate 16 environments and save a successful rollout:

```bash
HAND_SUCCESS_TRAJECTORY_PATH="$CAPTURE" \
HAND_CAPTURE_BEST_SUCCESS=1 \
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_mano_checkpoint.py \
  --task DexCoDesign-MANO-Residual-Direct-Eval-v0 \
  --algorithm PPO \
  --checkpoint "$CHECKPOINT" \
  --output artifacts/isaaclab_mano_residual/refined_mano/evaluation.json \
  --num_envs 16 \
  --seed 42 \
  --headless \
  agent.agent.experiment.wandb=false
```

Evaluation succeeds when at least one environment reaches the last reference
phase without exceeding the training failure thresholds. The JSON reports the
success count, success rate, successful environment IDs, and per-environment
errors. Before creating the environment, the evaluator safely restores scalar
settings from the checkpoint run's `params/env.yaml`; Eval-only behavior,
explicit `env.*` overrides, `--num_envs`, `--seed`, and `--device` retain final
precedence. With `HAND_CAPTURE_BEST_SUCCESS=1`, the successful environment with
the largest pose-plus-contact return at that completion step is saved.

The captured NPZ contains the actual simulated rollout, not only policy
weights:

| Key | Meaning |
|---|---|
| `hand_q` | simulated MANO joint trajectory |
| `object_pose_wxyz` | simulated object trajectory |
| `actions` | normalized policy residuals |
| `joint_targets` | reference plus residual targets |
| `pose_reward` | per-frame pose reward |
| `contact_reward` | per-frame contact reward |
| `metadata_json` | success, environment ID, scales, and final errors |

If `HAND_SUCCESS_TRAJECTORY_PATH` is not set, evaluation writes only its JSON
summary; this is why earlier evaluation runs may not contain a rollout NPZ.
When capture is enabled, the evaluator continues through the terminal
transition and writes both the successful rollout NPZ and its matching JSON.
Training launchers retain their default stop-on-first-captured-success behavior.

To inspect a checkpoint interactively without capturing a trajectory:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task DexCoDesign-MANO-Residual-Direct-Play-v0 \
  --algorithm PPO \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  agent.agent.experiment.wandb=false
```

## 8. Retarget the refined MANO rollout to WUJI hand 2

Use the captured MANO `hand_q` trajectory as the kinematic target source:

```bash
RETARGET="$PWD/artifacts/all_hands_success_action_retarget"

python temp/hocap_mano_replay/scripts/retarget_captured_success_all_hands.py \
  --capture "$CAPTURE" \
  --output-dir "$RETARGET" \
  --ik-iterations 14 \
  --max-mean-tip-error-m 0.03 \
  --solve-only \
  --hands wuji_hand_2
```

For every frame, the function:

1. runs forward kinematics on the refined MANO hand;
2. extracts the MANO wrist and fingertip targets;
3. constrains WUJI wrist orientation to follow the mapped MANO orientation;
4. optimizes WUJI wrist translation and active finger joints;
5. applies mimic-joint relations before forward kinematics; and
6. warm-starts the next frame from the previous solution.

This stage calls the same shared `solve_frame()` routine used in Step 3, so
its core objective and solver constraints are the same:

```text
sum_f ||p_f(q, t) - p_f_target||^2
  + 0.15^2 sum_f ||d_f(q) - d_f_target||^2
  + 0.015^2 ||q - q_previous||^2
```

WUJI wrist orientation is copied through the MANO-to-WUJI orientation mapping
and is not optimized. WUJI wrist translation and active finger joints are
optimized using damped least squares (`0.025` damping), a maximum update of
`0.12`, joint limits, and a per-frame finger-joint change limit of `0.20 rad`
after the initial frame. Object contact, collision, object tracking, and
intermediate-link matching are not included in this IK objective.

The difference from Step 3 is the source and destination of the targets. Step
3 converts raw reviewed HO-Cap geometry into the initial simulator MANO
trajectory. Step 8 obtains its targets by running forward kinematics on the
residual-policy-refined MANO rollout, then solves for a mechanically different
WUJI hand trajectory.

`--solve-only` skips the MuJoCo visualization but still writes the trajectory.
With no `--max-frames` argument, every frame stored in `$CAPTURE` is used. The
option remains available as a positive frame cap for smoke tests. The solve is
accepted only when the mean fingertip error is at most 3 cm.

Expected output:

```text
artifacts/all_hands_success_action_retarget/
  retarget_report.json
  wuji_hand_2/
    retargeted_trajectory.npz
```

The WUJI trajectory includes finger `qpos`, solved wrist position and XYZW
orientation, object WXYZ poses, and per-frame IK errors.

## 9. Build the WUJI Isaac Lab reference and USD

Convert the retargeted trajectory into the generic-hand residual-RL schema:

```bash
WUJI_PREPARED="$PWD/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2"

python temp/hocap_mano_replay/scripts/prepare_all_hand_rl_references.py \
  --capture "$CAPTURE" \
  --retarget-dir "$RETARGET" \
  --output-dir artifacts/isaaclab_all_hands_residual/prepared \
  --hands wuji_hand_2
```

This generates:

```text
artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/
  hand_rl.urdf
  reference.npz
  manifest.json
```

`hand_rl.urdf` wraps the WUJI articulation with six virtual wrist joints.
`reference.npz` contains the WUJI joint/control reference, action-to-control
mapping, contact-link names, fingertip targets, and object trajectory.

Convert the wrapped URDF to USD locally:

```bash
mkdir -p artifacts/isaaclab_all_hands_residual/assets/wuji_hand_2

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "$WUJI_PREPARED/hand_rl.urdf" \
  artifacts/isaaclab_all_hands_residual/assets/wuji_hand_2/hand.usd \
  --fix-base \
  --joint-stiffness 300 \
  --joint-damping 34.6410162 \
  --joint-target-type position
```

Again, omit `--headless` for local Isaac Sim 4.5 URDF conversion.

## 10. Validate the WUJI reference in PhysX

The generic hand task is selected by `DEXCODESIGN_HAND_ID`. Explicit paths make
the test independent of defaults:

```bash
OBJECT_USD="$PWD/artifacts/isaaclab_mano_residual/assets/g04_1.usd"

DEXCODESIGN_HAND_ID=wuji_hand_2 \
DEXCODESIGN_REFERENCE_PATH="$WUJI_PREPARED/reference.npz" \
DEXCODESIGN_OBJECT_USD_PATH="$OBJECT_USD" \
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/diagnose_zero_residual.py \
  --task DexCoDesign-Hand-Residual-Direct-Play-v0 \
  --output artifacts/isaaclab_all_hands_residual/wuji_refined_zero_residual.json
```

This is open-loop validation: it tests the retargeted WUJI reference with zero
policy correction. It does not train a controller.

## 11. Train the WUJI residual policy

The generic-hand task registrations are:

| Task | Purpose |
|---|---|
| `DexCoDesign-Hand-Residual-Direct-v0` | generic hand training |
| `DexCoDesign-Hand-Residual-Direct-Play-v0` | generic hand visualization |
| `DexCoDesign-Hand-Residual-Direct-Eval-v0` | generic hand strict evaluation |

Train WUJI with the same residual-PPO implementation:

```bash
DEXCODESIGN_HAND_ID=wuji_hand_2 \
DEXCODESIGN_REFERENCE_PATH="$WUJI_PREPARED/reference.npz" \
DEXCODESIGN_OBJECT_USD_PATH="$OBJECT_USD" \
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task DexCoDesign-Hand-Residual-Direct-v0 \
  --algorithm PPO \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless \
  --video \
  --video_length 1500 \
  --video_interval 5000 \
  agent.agent.experiment.directory=wuji_residual \
  agent.agent.experiment.experiment_name=wuji_hand_2_refined
```

The directory override keeps WUJI runs separate from MANO runs:

```text
logs/skrl/wuji_residual/<timestamp>_ppo_torch_wuji_hand_2_refined/
```

Keep the three `DEXCODESIGN_*` variables set for training, play, and evaluation;
otherwise the generic environment may load a different hand or reference.

Evaluate and capture a successful WUJI rollout with:

```bash
WUJI_RUN="replace_with_the_wuji_run_directory"
WUJI_CHECKPOINT="$PWD/logs/skrl/wuji_residual/$WUJI_RUN/checkpoints/best_agent.pt"
WUJI_CAPTURE="$PWD/artifacts/isaaclab_all_hands_residual/wuji_hand_2/successful_rollout.npz"

DEXCODESIGN_HAND_ID=wuji_hand_2 \
DEXCODESIGN_REFERENCE_PATH="$WUJI_PREPARED/reference.npz" \
DEXCODESIGN_OBJECT_USD_PATH="$OBJECT_USD" \
HAND_SUCCESS_TRAJECTORY_PATH="$WUJI_CAPTURE" \
HAND_CAPTURE_BEST_SUCCESS=1 \
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_mano_checkpoint.py \
  --task DexCoDesign-Hand-Residual-Direct-Eval-v0 \
  --algorithm PPO \
  --checkpoint "$WUJI_CHECKPOINT" \
  --output artifacts/isaaclab_all_hands_residual/wuji_hand_2/evaluation.json \
  --num_envs 16 \
  --seed 42 \
  --headless \
  agent.agent.experiment.wandb=false
```

Despite its historical filename and `MANO_EVAL_*` console messages,
`evaluate_mano_checkpoint.py` also evaluates the registered generic WUJI task.

## 12. Optional: optimize WUJI morphology with SAC

Residual PPO optimizes motion for a fixed WUJI morphology. Morphology SAC is a
different outer optimization stage: it proposes hand-design vectors, retargets
the trajectory, evaluates each design in PhysX, and learns which morphology
produces the best rollout return.

Before running morphology SAC, build the current general-v3 certified palm
prototype bank:

```text
artifacts/wuji_physx_search/
  palm_prototype_bank_general_v3_source_star_0p70/
    vectors.npy
    vectors.schema.json
    gpu_retarget_all.npz
    prepared/physx_batch_manifest.json
    direct_physx_results.json
```

The legacy `palm_prototype_bank_32` directories do not contain the required
grammar signature and are rejected by the current trainer.

### Build the 32-palm bank locally

Use the refined WUJI trajectory produced in Step 8 as the source-hand warm
start:

```bash
BANK="$PWD/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70"
SEED="$PWD/artifacts/all_hands_success_action_retarget/wuji_hand_2/retargeted_trajectory.npz"

mkdir -p "$BANK"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
```

First, write the canonical general-v3 design matrix and its grammar signature:

```bash
python temp/hocap_mano_replay/scripts/write_wuji_palm_prototype_vectors.py \
  "$BANK/vectors.npy"
```

The 32 rows select ordered palm prototypes `0` through `31`. They span the
source-to-star fusion expansion range `0.0` through `0.70`; prototype zero is
the unmodified source WUJI palm. The command writes both `vectors.npy` and the
required `vectors.schema.json`.

Retarget the source WUJI trajectory to all 32 palm geometries on the GPU:

```bash
python temp/hocap_mano_replay/scripts/gpu_wuji_retarget.py \
  --seed-trajectory "$SEED" \
  --vectors "$BANK/vectors.npy" \
  --output "$BANK/gpu_retarget_all.npz" \
  --iterations 4 \
  --candidate-chunk 32 \
  --device cuda \
  --save-trajectories \
  --skip-proxy
```

Compile the hand graphs, collision geometry, URDFs, and candidate-specific
references. Four local workers reduce peak RAM compared with the 32-worker
cluster recipe; use `--workers 2` or `--workers 1` if necessary:

```bash
python temp/hocap_mano_replay/scripts/prepare_wuji_morphology_physx_batch.py \
  "$BANK/gpu_retarget_all.npz" \
  --output-root "$BANK/prepared" \
  --workers 4 \
  --limit 32
```

Convert all candidate URDFs to USD in one Isaac Sim process. On the local
Isaac Sim 4.5 installation, do not add `--headless` to this conversion because
the URDF importer extension is unavailable in that headless experience:

```bash
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/convert_wuji_morphology_physx_batch.py \
  "$BANK/prepared/physx_batch_manifest.json"
```

Finally, replay all 32 candidates in PhysX and write their certification
results. This stage can run headless because it loads the already converted
USDs:

```bash
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py \
  "$BANK/prepared/physx_batch_manifest.json" \
  --output "$BANK/direct_physx_results.json" \
  --headless
```

The result records every candidate's reward, completed trajectory phase,
object-pose errors, fingertip contact forces, and success flag. Verify the
required bank products before starting SAC:

```bash
test -s "$BANK/vectors.schema.json" && \
test -s "$BANK/prepared/physx_batch_manifest.json" && \
test -s "$BANK/direct_physx_results.json" && \
echo "WUJI palm prototype bank complete"
```

The equivalent 32-worker, 192-GB Slurm recipe is recorded in:

```text
temp/hocap_mano_replay/isaaclab/build_wuji_palm_prototype_bank.sbatch
```

Once the bank exists, submit pure morphology SAC:

```bash
scripts/dexcodesign/launch_hand_learning.sh sac \
  --hand wuji_hand_2 \
  --task-id g04_1_refined \
  --reference "$WUJI_PREPARED/reference.npz" \
  --object-usd "$OBJECT_USD" \
  --output "$PWD/artifacts/wuji_sac/g04_1_refined" \
  --population 128 \
  --generations 12 \
  --physics-batch-size 128
```

Use `sac-ppo` instead of `sac` to interleave morphology SAC with shared
residual-policy PPO updates:

```bash
scripts/dexcodesign/launch_hand_learning.sh sac-ppo \
  --hand wuji_hand_2 \
  --task-id g04_1_refined \
  --reference "$WUJI_PREPARED/reference.npz" \
  --object-usd "$OBJECT_USD" \
  --output "$PWD/artifacts/wuji_sac_ppo/g04_1_refined" \
  --population 128 \
  --generations 12 \
  --ppo-iterations 20 \
  --gpus 1
```

The launcher submits the 192-GB Slurm job. The underlying Python entry point is
`temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py`.

## 13. Function and file map

| File | Responsibility |
|---|---|
| `scripts/prepare_extracted_subset.py` | Extract the reviewed left-hand/object subset from downloaded HO-Cap data |
| `scripts/prepare_isaaclab_reference.py` | Warm-started MANO IK and Isaac Lab reference export |
| `isaaclab/prepare_assets.sh` | Batch MANO/object asset preparation; the local URDF conversion may need the non-headless command above |
| `isaaclab/diagnose_zero_residual.py` | Run an open-loop reference and write per-frame diagnostics |
| `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/mano_residual_env.py` | Shared MANO/generic-hand residual environment, observations, actions, reward, termination, and capture |
| `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/config/mano_residual_env.yaml` | Collision choice and residual-action scales |
| `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/agents/skrl_ppo_cfg.yaml` | PPO networks, optimizer, checkpointing, and W&B settings |
| `scripts/reinforcement_learning/skrl/train.py` | Train MANO or generic-hand PPO |
| `scripts/reinforcement_learning/skrl/play.py` | Load a checkpoint plus its saved configuration and visualize it |
| `isaaclab/evaluate_mano_checkpoint.py` | Strict multi-environment evaluation and optional successful-rollout capture |
| `scripts/retarget_captured_success_all_hands.py` | Retarget captured MANO FK targets to WUJI or other registered hands |
| `scripts/prepare_all_hand_rl_references.py` | Wrap robot URDFs with a virtual wrist and export generic-hand references |
| `isaaclab/replay_hand_rollout.py` | Replay an exactly captured generic-hand rollout |
| `scripts/dexcodesign/launch_hand_learning.sh` | Submit WUJI morphology SAC or SAC+PPO |
| `isaaclab/train_wuji_hybrid_sac_morphology.py` | Persistent Isaac Lab morphology optimization driver |

## 14. Reproducibility checklist

Before comparing experiments, record or verify:

- the exact HO-Cap sequence, hand slot, object slot, and frame count;
- `isaaclab_reference.retargeting.json` IK errors;
- MANO/WUJI URDF and USD versions;
- object USD collision approximation and mass;
- `disable_hand_support_collisions` and all residual-action scales;
- the saved `params/env.yaml` and `params/agent.yaml`;
- checkpoint path and Git commit;
- evaluation seed, number of environments, and success rate;
- the captured rollout NPZ used for WUJI retargeting; and
- the prototype-bank `vectors.schema.json` used by morphology SAC.

Do not compare a checkpoint trained with an old collider against results from a
new collider as if they were the same experiment. Rebuild the USD and retrain
after changing physical geometry.
