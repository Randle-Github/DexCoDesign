# Temporary HO-Cap MANO/Object Replay

This isolated experiment downloads and extracts only the HO-Cap assets needed
for a small hand-object replay:

- `models.zip`: object meshes
- `poses.zip`: MANO and object trajectories
- `calibration.zip`: per-subject MANO shape parameters (10 KB archive)

No RGB, depth, segmentation, or label archives are required.

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
.venv-morphology/bin/mjpython temp/hocap_mano_replay/scripts/replay_mujoco.py
```

## Retarget all direct-motor hands

`scripts/retarget_all_hands.py` keeps the verified MANO replay unchanged and
retargets it to the other 13 normalized left hands. Its damped least-squares IK
optimizes the free 6-DoF wrist and finger joints together. The objective has
only these terms:

- bottom-wrist orientation;
- four or five fingertip positions;
- the corresponding terminal-link/mesh orientations.

Wrist translation deliberately has no target. There are no intermediate-link,
contact, collision, or object-surface objectives. Allegro V5 and MIDAS use four
tips; the other hands use five.

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

## MANO mapping status

HO-Cap stores each hand frame as 51 values:

```text
global rotation vector (3) + MANO PCA hand pose (45) + translation (3)
```

Exact 778-vertex reconstruction requires the separately licensed
`MANO_LEFT.pkl` and `MANO_RIGHT.pkl`, as documented by the official HO-Cap
toolkit. Those files are not present in this repository.

The temporary video therefore replays object pose and wrist pose exactly, then
maps the hand PCA motion to the repository's current generated direct-motor
MANO URDF:

```text
assets/robot_hands/direct_motor/mano/left/hand.urdf
```

The older `assets/robot_hands/mano/source/*.xml` entry points are not used by
this experiment. The direct-motor URDF has 28 scalar DoFs: 6 world-root DoFs
and 22 articulated finger DoFs. Once licensed MANO files are provided, the
approximate finger mapping can be replaced by direct MANO vertex generation;
no point-to-point retargeting is needed for MANO-to-MANO replay.

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
