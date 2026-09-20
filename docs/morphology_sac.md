# WUJI morphology SAC

The morphology optimizer is intentionally split across two Python packages in
this monorepo. `dexcodesign` owns the grammar and compilation pipeline, while
the Isaac Lab code owns physical evaluation, SAC updates, and optional shared
residual PPO updates. Install `source/dexcodesign` editable before launching so
all code imports the same authoritative grammar implementation.

The focused runtime is:

- `source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/`: batched physical evaluation task;
- `temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py`: persistent Isaac trainer;
- `temp/hocap_mano_replay/scripts/skrl_sac_wuji_morphology.py`: default SKRL SAC optimizer;
- `temp/hocap_mano_replay/scripts/hybrid_sac_wuji_morphology.py`: preserved custom optimizer backend;
- `temp/hocap_mano_replay/isaaclab/wuji_parametric_usd.py`: runtime USD overlays;
- `scripts/dexcodesign/launch_hand_learning.sh`: Slurm launcher for `sac` and `sac-ppo`.

The implementation currently supports `wuji_hand_2`. It requires two generated
inputs that are deliberately not committed: a certified palm-prototype bank and
a seed/fixed reference trajectory. Supply them using `BANK_ROOT` and
`SEED_TRAJECTORY` (or `--reference` through the launcher). Training outputs must
remain under `artifacts/`.

Inspect the launcher without submitting a job:

```bash
scripts/dexcodesign/launch_hand_learning.sh sac \
  --reference /path/to/reference.npz \
  --object-usd /path/to/object.usd \
  --output artifacts/wuji_sac/run \
  --population 128 \
  --generations 12 \
  --dry-run
```

The launcher uses the active Conda environment by default. Set `CONDA_ENV` and
`CONDA_SH` explicitly when submitting from a shell where Conda is not active.
