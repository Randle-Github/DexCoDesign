#!/usr/bin/env bash
#SBATCH --job-name=wuji-persist
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/persistent-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
MANIFEST="${MANIFEST:?set MANIFEST}"
OUTPUT="${OUTPUT:?set OUTPUT}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh -p \
  temp/hocap_mano_replay/isaaclab/smoke_persistent_wuji_batches.py \
  "${MANIFEST}" --output "${OUTPUT}" --repeats 2 --headless
