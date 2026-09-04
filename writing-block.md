# DexCoDesign Quick Start on Skynet

## 1. Clone and activate the environment

```bash
mkdir -p /coc/flash7/$USER/workspace
cd /coc/flash7/$USER/workspace

git clone -b codex/all-hands-rl-code \
  https://github.com/Randle-Github/DexCoDesign.git

cd DexCoDesign
git lfs pull

source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
```

## 2. Link the shared runtime assets

The assets are read-only. All training outputs are saved in the new user’s workspace.

```bash
SRC=/coc/flash7/yliu3735/workspace/DexCoDesign

mkdir -p \
  artifacts/isaaclab_all_hands_residual \
  artifacts/isaaclab_mano_residual \
  artifacts/wuji_physx_search

ln -sfn "$SRC/artifacts/hand_morphology" \
  artifacts/hand_morphology

ln -sfn "$SRC/artifacts/isaaclab_all_hands_residual/assets" \
  artifacts/isaaclab_all_hands_residual/assets

ln -sfn "$SRC/artifacts/isaaclab_all_hands_residual/prepared" \
  artifacts/isaaclab_all_hands_residual/prepared

ln -sfn "$SRC/artifacts/isaaclab_mano_residual/assets" \
  artifacts/isaaclab_mano_residual/assets

ln -sfn "$SRC/artifacts/isaaclab_mano_residual/runtime_libs" \
  artifacts/isaaclab_mano_residual/runtime_libs

ln -sfn \
  "$SRC/artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70" \
  artifacts/wuji_physx_search/palm_prototype_bank_general_v3_source_star_0p70
```

Set the G04 task inputs:

```bash
REF="$PWD/artifacts/isaaclab_all_hands_residual/prepared/wuji_hand_2/reference.npz"
OBJ="$PWD/artifacts/isaaclab_mano_residual/assets/g04_1.usd"
```

## 3. Run experiments

### PPO only

```bash
scripts/dexcodesign/launch_hand_learning.sh ppo \
  --reference "$REF" \
  --object-usd "$OBJ" \
  --output "$PWD/runs/ppo" \
  --hand wuji_hand_2 \
  --num-envs 4096 \
  --max-iterations 1200
```

### Morphology SAC only

```bash
scripts/dexcodesign/launch_hand_learning.sh sac \
  --reference "$REF" \
  --object-usd "$OBJ" \
  --output "$PWD/runs/sac" \
  --population 128 \
  --generations 20
```

### Four-GPU SAC–PPO co-design

```bash
scripts/dexcodesign/launch_hand_learning.sh sac-ppo \
  --reference "$REF" \
  --object-usd "$OBJ" \
  --output "$PWD/runs/sac_ppo" \
  --population 128 \
  --generations 20 \
  --ppo-iterations 20 \
  --gpus 4 \
  --morphology-replicas 32
```

## 4. Monitor jobs

```bash
squeue -u "$USER"
```

The shared `palm_prototype_bank_general_v3_source_star_0p70` only needs to be generated once and can be reused by all Skynet users.