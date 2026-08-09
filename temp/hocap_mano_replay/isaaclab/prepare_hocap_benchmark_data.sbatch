#!/usr/bin/env bash
#SBATCH --job-name=hocap-data
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/data-%j.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
mkdir -p artifacts/hocap_multitask/slurm
python temp/hocap_mano_replay/scripts/prepare_hocap_benchmark.py --ik-iterations 32
touch artifacts/hocap_multitask/.data_ready
printf 'HOCAP_BENCHMARK_DATA_READY\n'
