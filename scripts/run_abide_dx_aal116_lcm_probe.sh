#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dataset3/nzh/fmri_baseline/brain_network_decoder
source ~/.bashrc || true
conda activate lcm

python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --feature_source lcm_frozen \
  --mode both \
  --seeds 23 24 25 26 27 \
  --out_dir outputs/abide_dx_aal116_lcm_probe
