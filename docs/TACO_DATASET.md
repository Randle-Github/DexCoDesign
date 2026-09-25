# TACO pose subset

TACO is kept separate from the existing HO-Cap experiment data:

```text
datasets/
└── taco_v1/
    ├── taco_info.csv
    ├── raw/
    │   ├── hand_poses_3d/<triplet>/<sequence>/hand_joints.npy
    │   ├── object_poses/<triplet>/<sequence>/{tool,target}_*.npy
    │   └── object_models/*_cm.obj
    └── manifests/
        ├── summary.json
        ├── selected_100.jsonl
        └── rejected.jsonl

temp/hocap_mano_replay/        # existing HO-Cap work; never mixed with TACO
```

The local `datasets/` directory is ignored by Git. Code and this management
contract are tracked; licensed raw data and generated subsets are not.

## Scope

Only three modalities are used:

- `Hand_Poses_3D.zip`: left/right 21-joint trajectories;
- `Object_Poses.zip`: tool and target 6-DoF trajectories;
- `Object_Models.zip`: meshes for object IDs actually used by selected demos.

RGB, depth, segmentation, and multi-camera videos are intentionally excluded.
The compact pose downloads are about 200 MB before object meshes, rather than
hundreds of GB for the video dataset.

## Existing HO-Cap counts

These counts are different stages, not interchangeable claims:

- 64 sequences indexed in `hocap_sequence_catalog.json`;
- 9 sequences listed in `benchmark_tasks.json`;
- 5 prepared task directories currently present under `data/tasks/`.

## Selection policy

The default TACO subset contains 100 demos from 100 distinct
`<action, tool, object>` triplets. Selection is deterministic and applies:

1. modality and shape validation;
2. finite-value and SE(3) transform validation;
3. aligned frame counts and atomic-duration bounds;
4. hand/object temporal-jump rejection;
5. minimum meaningful object motion;
6. ranking by compact duration, smoothness, motion, and metadata quality;
7. diversity cap per triplet.

This follows the practical lesson from EgoEngine: dataset membership does not
make a sequence robot-executable. Retargeting and simulation remain validation
gates. In particular, a pose-only audit cannot prove that an object is stable on
a table. Before PPO/SAC use, each selected sequence must pass a short simulator
settling test with gravity, collision geometry, mass, friction, and the initial
object orientation. Failed demos remain recorded with a rejection reason.

The local subset runs this gate in MuJoCo for both the tool and target. It drops
each mesh onto a plane using the demo's first-frame orientation, then rejects
objects that rotate more than 15 degrees or drift more than 3 cm.

## Rebuild

Download the four small/non-video inputs, build the subset, and run the physical
gate:

```bash
.venv-morphology/bin/python scripts/datasets/taco/download_taco_minimal.py \
  --root datasets/taco_v1

.venv-morphology/bin/python scripts/datasets/taco/prepare_taco_subset.py \
  --root datasets/taco_v1 --count 100

.venv-morphology/bin/python scripts/datasets/taco/audit_taco_settling.py \
  --root datasets/taco_v1
```

By default the archives are deleted after the selected files are extracted.
Use `--keep-archives` only while debugging the ingestion pipeline.

## Provenance

- Official instructions: <https://github.com/leolyliu/TACO-Instructions>
- TACO paper/project: <https://taco2024.github.io/>
- License: CC BY 4.0

The Hugging Face mirror supplies the compact `Hand_Poses_3D.zip`; provenance,
archive sizes, and SHA-256 hashes are written to `manifests/summary.json`.
