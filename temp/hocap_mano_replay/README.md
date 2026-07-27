# Temporary HO-Cap MANO/Object Replay

This isolated experiment downloads and extracts only the HO-Cap assets needed
for a small hand-object replay:

- `models.zip`: object meshes
- `poses.zip`: MANO and object trajectories
- `calibration.zip`: per-subject MANO shape parameters (10 KB archive)
- selected byte ranges from `labels.zip`: official 21-point hand trajectory

The full 1.5 GB label archive is not downloaded.  The subset downloader uses
HTTP byte ranges to extract one camera's 446 small label records, then applies
the official camera-to-world extrinsic.

## Selected subset

- Sequence: `subject_7/20231022_192832`
- Frames: 446
- Hand: left
- Manipulated object slot: 0
- Officially verified pose-slot-0 object ID: `G04_1`

The official `poses.zip` omits each sequence's `meta.yaml`, and the current
multi-gigabyte subject links return 404. Therefore the object slot-to-ID mapping
is isolated in `scripts/prepare_subset.py` and can be replaced without touching
the replay code. Pose arrays are copied without modification.

## Reproduce

```bash
.venv-morphology/bin/python temp/hocap_mano_replay/scripts/download_minimal.py
.venv-morphology/bin/python temp/hocap_mano_replay/scripts/prepare_subset.py
.venv-morphology/bin/python temp/hocap_mano_replay/scripts/download_label_subset.py
.venv-morphology/bin/python temp/hocap_mano_replay/scripts/prepare_isaaclab_reference.py
.venv-morphology/bin/mjpython temp/hocap_mano_replay/scripts/replay_mujoco.py
```

## Retarget all direct-motor hands

`scripts/retarget_all_hands.py` follows EgoEngine's semantic path: official
21-point hand labels are converted into a wrist pose and five fingertip poses,
then temporally warm-started IK retargets those poses to each normalized hand.
No object-surface snapping is used. The objective has only these terms:

- bottom-wrist orientation;
- four or five fingertip positions;
- the corresponding terminal-link/mesh orientations.

Wrist translation deliberately has no target. There are no intermediate-link,
contact, collision, or object-surface terms in the per-hand solve. Allegro V5
and MIDAS use four tips; the other hands use five.

The Isaac Lab reference and retargeting diagnostics are regenerated with:

```bash
.venv-morphology/bin/python \
  temp/hocap_mano_replay/scripts/prepare_isaaclab_reference.py
```

This writes `isaaclab_reference.npz` and
`isaaclab_reference.retargeting.json` beside the selected trajectory.

Before training, replay that reference with exactly zero residual on Skynet:

```bash
sbatch temp/hocap_mano_replay/isaaclab/diagnose_zero_residual.sbatch
```

The Isaac Lab task follows EgoEngine's residual formulation:

```text
ctrl[t] = ctrl_ref[t] + action[t] * action_scale
observation[t] = [hand_q[t], object_pose[t], ctrl_ref[t]]
```

Training samples 64-control-step windows. The action scales, object-mass
scaling, object-pose C-error reward, binary thumb-plus-other-fingertip contact
reward, and PPO hyperparameters match the Aria residual-RL configuration.

```bash
# Solve and save per-hand trajectories.
.venv-morphology/bin/python \
  temp/hocap_mano_replay/scripts/retarget_all_hands.py \
  --stride 4 --iterations 18 --solve-only

# Render the cached trajectories as one 4x4 comparison video.
.venv-morphology/bin/mjpython \
  temp/hocap_mano_replay/scripts/retarget_all_hands.py \
  --stride 4 --reuse-cache
```

Outputs:

```text
artifacts/hocap_all_hands_ik.mp4
artifacts/hocap_all_hands_ik_preview.png
artifacts/all_hands_ik_diagnostics.json
artifacts/all_hands_ik_cache/*_ik.npz
```

## MANO mapping semantics

HO-Cap stores each hand frame as 51 values:

```text
global rotation vector (3) + MANO PCA hand pose (45) + translation (3)
```

The 45 PCA channels are not assigned directly to robot joints.  Exact hand
semantics come from HO-Cap's official `hand_joints_3d` labels in MediaPipe/MANO
21-joint order.  Wrist position and all finger positions therefore remain
independent of the separately licensed MANO PCA basis.  The known global
rotation vector supplies wrist orientation, and the distal bone directions
supply terminal-link orientation targets.

Those targets are retargeted to the current generated direct-motor MANO URDF:

```text
assets/robot_hands/direct_motor/mano/left/hand.urdf
```

The older `assets/robot_hands/mano/source/*.xml` entry points are not used.
The direct-motor URDF has 28 scalar DoFs: 6 world-root DoFs and 22 articulated
finger DoFs.

### Root-frame alignment

The signed-axis review selected candidate B: `-90°` around the hand-local
`X` axis from the first replay mapping.

```text
R_local_to_mano = R_first @ RotX(-90 degrees)
```

The selected fixed basis is composed on the right of every original trajectory
root rotation:
`R_world(t) = R_mano_trajectory(t) @ R_local_to_mano`. Object poses are
unchanged; wrist positions use the MANO translation correction below.

HO-Cap's last three pose values are MANO translation parameters, not the wrist
joint itself. Manopth produces `wrist = translation + J0(shape)`. For subject 7,
official 3D wrist labels at frames 0 and 222 give the same constant:

```text
J0 = [-0.096753545, 0.006290510, 0.006127380] meters
```
