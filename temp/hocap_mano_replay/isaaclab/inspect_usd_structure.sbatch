#!/usr/bin/env bash
#SBATCH --job-name=inspect-hand-usd
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/wuji_physx_search/inspect-usd-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
USD="${USD:?set USD}"
FLATTEN_OUTPUT="${FLATTEN_OUTPUT:-}"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
command=(./isaaclab.sh -p temp/hocap_mano_replay/isaaclab/inspect_usd_structure.py
  "${USD}" --headless)
if [[ -n "${FLATTEN_OUTPUT}" ]]; then
  command+=(--flatten-output "${FLATTEN_OUTPUT}")
fi
"${command[@]}"
