#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dataset3/nzh/fmri_baseline/brain_network_decoder
source ~/.bashrc || true
conda activate lcm

python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --seeds 4 44 444 \
  --epochs 50 \
  --batch_size 16 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune
