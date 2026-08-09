#!/usr/bin/env bash
#SBATCH --job-name=hocap-retarget
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=hk47,spd-13,voltron,ig-88,sonny
#SBATCH --output=/coc/flash7/yliu3735/workspace/DexCoDesign/artifacts/hocap_multitask/slurm/retarget-%A_%a.out

set -euo pipefail
REPO_ROOT="/coc/flash7/yliu3735/workspace/DexCoDesign"
TASK_IDS=(g04_pick_place g15_vertical_local g06_rotation_heavy g19_mixed_reorientation g02_horizontal_reorientation)
TASK_ID="${TASK_IDS[${SLURM_ARRAY_TASK_ID:?missing array index}]}"
CAPTURE="${REPO_ROOT}/temp/hocap_mano_replay/data/tasks/${TASK_ID}/isaaclab_reference.npz"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/prepared/${TASK_ID}"
RETARGET_ROOT="${REPO_ROOT}/artifacts/hocap_multitask/retarget/${TASK_ID}"
HANDS=(wuji_hand_2)
if [[ "${TASK_ID}" == "g04_pick_place" ]]; then
  HANDS+=(midas_hand sharpa_wave_01 orca_hand_v2)
fi
source "/coc/flash7/yliu3735/anaconda3/etc/profile.d/conda.sh"
conda activate codesign
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/artifacts/isaaclab_all_hands_residual/reference_python_deps:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}" "${RETARGET_ROOT}"
python temp/hocap_mano_replay/scripts/retarget_captured_success_all_hands.py \
  --capture "${CAPTURE}" --output-dir "${RETARGET_ROOT}" --max-frames 446 \
  --max-mean-tip-error-m 0.03 --ik-iterations 14 --solve-only --hands "${HANDS[@]}"
python temp/hocap_mano_replay/scripts/prepare_all_hand_rl_references.py \
  --capture "${CAPTURE}" --retarget-dir "${RETARGET_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" --hands "${HANDS[@]}"
touch "${OUTPUT_ROOT}/.ready"
printf 'HOCAP_RETARGET_READY task=%s hands=%s\n' "${TASK_ID}" "${HANDS[*]}"
