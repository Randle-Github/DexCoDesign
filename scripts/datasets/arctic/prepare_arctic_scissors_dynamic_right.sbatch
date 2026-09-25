#!/usr/bin/env bash
#SBATCH --job-name=arctic-dynamic-ref
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/arctic_scissors_training/prepare-dynamic-%j.out

set -euo pipefail

ROOT=/coc/flash7/yliu3735/workspace/DexCoDesign
OUT="$ROOT/artifacts/arctic_mano_dynamic/00_scissors"
source /coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh
conda activate codesign
cd "$ROOT"
mkdir -p "$OUT" artifacts/arctic_scissors_training

python scripts/datasets/arctic/prepare_arctic_ik_trajectory.py \
  --canonical-trajectory datasets/arctic_v1/canonical_100/009_scissors_s08/trajectory.npz \
  --side right \
  --iterations 48 \
  --priority-target-ids 4 8 \
  --priority-weight 8.0 \
  --output "$OUT/trajectory_ik_right_fixedtip_dynamic.npz"

python scripts/datasets/arctic/prepare_arctic_rl_reference.py \
  --trajectory "$OUT/trajectory_ik_right_fixedtip_dynamic.npz" \
  --hand-urdf assets/robot_hands/direct_motor/mano/right/hand.urdf \
  --side right \
  --target-fps 30 \
  --world-translation 0 0 -0.9516945 \
  --output "$OUT/isaaclab_reference_right_fixedtip_dynamic_tabletop_30hz.npz"

printf 'ARCTIC_DYNAMIC_RIGHT_REFERENCE_COMPLETE output=%s\n' \
  "$OUT/isaaclab_reference_right_fixedtip_dynamic_tabletop_30hz.npz"
