# ARCTIC minimal dataset

ARCTIC is kept separate from TACO under `datasets/arctic_v1`. DexCoDesign uses
ARCTIC for bimanual manipulation of articulated objects and deliberately does
not download RGB images, videos, image features, preprocessed visual splits,
SMPL-X bodies, or baseline checkpoints.

Current prepared result:

```text
source sequences:             301
raw MANO/object files:        602
selected demonstrations:      100
selected frames at 10 Hz:     24,380
articulated object classes:   11
total local size:             approximately 177 MB
```

## Local contents

The checked local asset subset contains all 11 ARCTIC objects:

```text
box, capsulemachine, espressomachine, ketchup, laptop, microwave,
mixer, notebook, phone, scissors, waffleiron
```

For each object it stores only:

```text
bottom.obj       fixed/root rigid part
top.obj          articulated rigid part
<object>.urdf    valid two-link URDF
```

This is 22 meshes and 11 URDFs, approximately 70 MB. The assets come from the
ARCTIC URDF resource linked by the official ARCTIC repository. ARCTIC is
non-commercial; retain its upstream terms.

## Articulated-object representation

An ARCTIC object is not stored as one deforming mesh. It is represented as:

```text
floating root pose in the world
          |
       bottom
          |
          +-- revolute joint q, axis (0, 0, -1), origin (0, 0, 0)
                    |
                   top
```

Per frame, the state is:

```text
root position:       3 values, metres
root orientation:    quaternion wxyz
internal joint q:    1 value, radians
```

The source `*.object.npy` format is seven values per frame:

```text
[q_rad, root_axis_angle_x, root_axis_angle_y, root_axis_angle_z,
 root_translation_x_mm, root_translation_y_mm, root_translation_z_mm]
```

The top link is first rotated by `q` about the canonical negative Z axis, and
the root SE(3) transform is then applied to both links. This separation is
important: object motion and object articulation must not be collapsed into a
single six-dimensional pose.

The machine-readable definition is generated at:

```text
datasets/arctic_v1/manifests/articulated_objects.json
```

Joint ranges are refreshed from all downloaded trajectories. For example, the
observed scissors range is currently `[0.0, 1.437]` radians. The initial
`[0, pi]` convention is only a fallback when no trajectories are present.

## Minimal official motion download

Register and accept the license at <https://arctic.is.tue.mpg.de/>, then set:

```bash
export ARCTIC_USERNAME='your-email'
export ARCTIC_PASSWORD='your-password'
```

Download only `raw_seqs.zip` and selectively extract `*.mano.npy` and
`*.object.npy`:

```bash
python scripts/datasets/arctic/download_arctic_minimal.py
```

The official archive is approximately 215 MB. Camera trajectories and SMPL-X
files inside it are skipped during extraction. The archive is SHA-256 verified
and removed after extraction unless `--keep-archives` is supplied. Official
`meta.zip` is unnecessary because the compact object assets are already
present; it can be requested explicitly with `--include-official-meta`.

No MANO download is performed: DexCoDesign already contains the left and right
MANO assets used by the TACO/Isaac pipeline.

## Build the benchmark

Regenerate URDFs and the object graph manifest:

```bash
python scripts/datasets/arctic/prepare_arctic_objects.py
```

Build approximately 100 balanced, image-free demonstrations at 10 Hz:

```bash
python scripts/datasets/arctic/prepare_arctic_subset.py --count 100 --stride 3
```

Each compact trajectory contains:

- left and right MANO global orientation, pose, translation, shape and fitting error;
- object floating-root position and quaternion;
- the object's internal revolute-joint position;
- explicit object graph semantics and units.

Selection is balanced across object identities and prefers manipulation/use
sequences over static grab sequences. TACO and ARCTIC manifests remain separate
so rigid-object and articulated-object metrics cannot be accidentally mixed.

## Bimanual residual mode

ARCTIC records both hands.  `env.bimanual_mode=true` therefore spawns both
physical MANO articulations and controls both with one residual PPO policy.
The action is the concatenation of the selected primary side (normally
`right`) and the opposite side, so the canonical MANO pair has 56 learned
residual actions. The observation contains current and goal state for both
hands plus one shared object state. Both hands participate in PhysX contacts.
Single-hand mode remains unchanged; single-hand checkpoints are not loaded as
bimanual checkpoints.

Prepare a synchronized pair:

```bash
python scripts/datasets/arctic/prepare_arctic_bimanual_item.py \
  --canonical-trajectory datasets/arctic_v1/canonical_100_tabletop/000_box_s08/trajectory.npz \
  --output artifacts/arctic_bimanual/000_box_s08
```

Training requires both paths:

```bash
SAMPLE_ID=000_box_s08 OBJECT_ID=box \
  sbatch scripts/datasets/arctic/train_arctic_bimanual.sbatch
```

Both sides receive independently learned residual corrections around their own
synchronized references. Neither hand is a fixed or reference-only support
hand.

## Files

```text
scripts/datasets/arctic/download_arctic_minimal.py
scripts/datasets/arctic/prepare_arctic_objects.py
scripts/datasets/arctic/prepare_arctic_subset.py
scripts/datasets/arctic/prepare_arctic_bimanual_item.py
scripts/datasets/arctic/train_arctic_bimanual.sbatch
docs/ARCTIC_DATASET.md
```
