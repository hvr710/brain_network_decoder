#!/usr/bin/env bash
set -euo pipefail

cd /vePFS-0x0d/nzh/baseline/brain_network_decoder
source /vePFS-0x0d/nzh/envs/lcm/bin/activate

python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /vePFS-0x0d/nzh/data/dataset1/ningzh/labels/age/ABIDE.csv \
  --split_dir /vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/split \
  --feature_source lcm_frozen \
  --mode both \
  --seeds 23 24 25 26 27 \
  --feature_batch_size 8 \
  --num_workers 8 \
  --out_dir outputs/abide_dx_aal116_lcm_probe
