#!/usr/bin/env bash
set -euo pipefail

# Step 1: create /vePFS workspace, copy repo, pretrained weights, Table3 data,
# labels, and splits to the A800 local disk. This script does not delete source
# or destination data.

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
REPO_DST="${REPO_DST:-${ROOT}/repo/brain_network_decoder}"
DATA_DST="${DATA_DST:-${ROOT}/data}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
ONE_ROOT="${ONE_ROOT:-outputs/one_${RUN_TS}}"

SRC_REPO="${SRC_REPO:-/mnt/dataset3/nzh/lcm_ds/brain_network_decoder}"
SRC_PRETRAIN="${SRC_PRETRAIN:-${SRC_REPO}/pretrain_weights_fold0}"
SRC_DATASET1="${SRC_DATASET1:-/mnt/dataset1}"
SRC_DATASET4="${SRC_DATASET4:-/mnt/dataset4}"

SRC_ROI="${SRC_DATASET4}/DATASETS/fmri_pretraining/fmri_dataset/roi"
DST_ROI="${DATA_DST}/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi"
SRC_LABEL="${SRC_DATASET1}/ningzh/labels"
DST_LABEL="${DATA_DST}/dataset1/ningzh/labels"

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing file: ${path}" >&2
    exit 1
  fi
}

copy_dir() {
  local src="$1"
  local dst="$2"
  require_dir "${src}"
  mkdir -p "${dst}"
  rsync -a --info=progress2 "${src%/}/" "${dst%/}/"
}

copy_file() {
  local src="$1"
  local dst_dir="$2"
  require_file "${src}"
  mkdir -p "${dst_dir}"
  rsync -a --info=progress2 "${src}" "${dst_dir}/"
}

echo "==> Creating workspace under ${ROOT}"
mkdir -p \
  "${ROOT}/repo" \
  "${ROOT}/envs" \
  "${ROOT}/conda_pkgs" \
  "${DATA_DST}/dataset1" \
  "${DATA_DST}/dataset3" \
  "${DATA_DST}/dataset4"

cat > "${ROOT}/run_table3_one.env" <<EOF
export A800_ROOT=${ROOT}
export REPO_DST=${REPO_DST}
export DATA_DST=${DATA_DST}
export RUN_TS=${RUN_TS}
export ONE_ROOT=${ONE_ROOT}
export TABLE3_DATASET1_ROOT=${DATA_DST}/dataset1
export TABLE3_DATASET3_ROOT=${DATA_DST}/dataset3
export TABLE3_DATASET4_ROOT=${DATA_DST}/dataset4
EOF

echo "==> Saved run variables to ${ROOT}/run_table3_one.env"

echo "==> Copying repo from ${SRC_REPO} to ${REPO_DST}"
if [[ -f "${REPO_DST}/finetune_table3.py" && ! -d "${SRC_REPO}" ]]; then
  echo "Source repo ${SRC_REPO} not found, but ${REPO_DST} already contains the repo. Skipping repo copy."
elif [[ "$(cd "${SRC_REPO}" 2>/dev/null && pwd -P)" == "$(cd "${REPO_DST}" 2>/dev/null && pwd -P)" ]]; then
  echo "Source repo and destination repo are the same path. Skipping repo copy."
else
  require_dir "${SRC_REPO}"
  mkdir -p "${REPO_DST}"
  rsync -a --info=progress2 \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'data_old_outputs' \
    --exclude 'LP_MLP/outputs' \
    --exclude 'pretrain_weights_fold0' \
    "${SRC_REPO%/}/" "${REPO_DST%/}/"
fi

echo "==> Copying pretrained weights"
copy_dir "${SRC_PRETRAIN}" "${REPO_DST}/pretrain_weights_fold0"

echo "==> Copying dataset4 Table3 ROI data and splits"
copy_dir "${SRC_ROI}/ABIDE/AAL" "${DST_ROI}/ABIDE/AAL"
copy_dir "${SRC_ROI}/ABIDE/Schaefer2018_100_crop_split" "${DST_ROI}/ABIDE/Schaefer2018_100_crop_split"

copy_dir "${SRC_ROI}/NKI/AAL" "${DST_ROI}/NKI/AAL"
copy_dir "${SRC_ROI}/NKI/100ROI_split" "${DST_ROI}/NKI/100ROI_split"

copy_dir "${SRC_ROI}/SALD/AAL" "${DST_ROI}/SALD/AAL"
copy_dir "${SRC_ROI}/SALD/100ROI_split" "${DST_ROI}/SALD/100ROI_split"

copy_dir "${SRC_ROI}/ABCD/AAL" "${DST_ROI}/ABCD/AAL"
copy_dir "${SRC_ROI}/HCP/AAL" "${DST_ROI}/HCP/AAL"

copy_dir "${SRC_ROI}/BHRC/AAL" "${DST_ROI}/BHRC/AAL"
copy_dir "${SRC_ROI}/BHRC/100ROI_split" "${DST_ROI}/BHRC/100ROI_split"

copy_dir "${SRC_ROI}/PPMI/AAL" "${DST_ROI}/PPMI/AAL"
copy_dir "${SRC_ROI}/PPMI/100ROI" "${DST_ROI}/PPMI/100ROI"

copy_dir "${SRC_ROI}/ADNI/AAL/CN" "${DST_ROI}/ADNI/AAL/CN"
copy_dir "${SRC_ROI}/ADNI/AAL/MCI" "${DST_ROI}/ADNI/AAL/MCI"
copy_dir "${SRC_ROI}/ADNI/AAL/AD" "${DST_ROI}/ADNI/AAL/AD"
copy_file "${SRC_ROI}/ADNI/AAL/MCI_train_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"
copy_file "${SRC_ROI}/ADNI/AAL/MCI_val_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"
copy_file "${SRC_ROI}/ADNI/AAL/MCI_test_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"
copy_file "${SRC_ROI}/ADNI/AAL/AD_train_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"
copy_file "${SRC_ROI}/ADNI/AAL/AD_val_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"
copy_file "${SRC_ROI}/ADNI/AAL/AD_test_abs_AAL.txt" "${DST_ROI}/ADNI/AAL"

echo "==> Copying dataset1 labels"
copy_file "${SRC_LABEL}/age/ABIDE.csv" "${DST_LABEL}/age"
copy_file "${SRC_LABEL}/age/NKI.csv" "${DST_LABEL}/age"
copy_file "${SRC_LABEL}/age/SALD.csv" "${DST_LABEL}/age"

copy_file "${SRC_LABEL}/sex/ABCD.csv" "${DST_LABEL}/sex"
copy_file "${SRC_LABEL}/sex/HCP.csv" "${DST_LABEL}/sex"
copy_file "${SRC_LABEL}/sex/BHRC.csv" "${DST_LABEL}/sex"

copy_file "${SRC_LABEL}/disease/PPMI.csv" "${DST_LABEL}/disease"
copy_file "${SRC_LABEL}/disease/adni_list.xlsx" "${DST_LABEL}/disease"

copy_file "${SRC_LABEL}/education/NKI.csv" "${DST_LABEL}/education"

echo "==> Size summary"
du -sh "${DATA_DST}/dataset1" "${DATA_DST}/dataset4" "${REPO_DST}/pretrain_weights_fold0"

echo "==> .npy count check"
for d in \
  ABIDE/AAL NKI/AAL SALD/AAL ABCD/AAL HCP/AAL BHRC/AAL PPMI/AAL \
  ADNI/AAL/CN ADNI/AAL/MCI ADNI/AAL/AD
do
  echo "${d}"
  echo -n "  source: "; find "${SRC_ROI}/${d}" -type f -name '*.npy' | wc -l
  echo -n "  local : "; find "${DST_ROI}/${d}" -type f -name '*.npy' | wc -l
done

echo "==> Step 1 done. Next:"
echo "source ${ROOT}/run_table3_one.env"
echo "bash ${REPO_DST}/scripts/a800_02_create_envs.sh"
