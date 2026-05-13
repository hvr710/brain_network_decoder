#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dataset3/nzh/fmri_baseline/brain_network_decoder

bash scripts/run_abide_dx_aal116_lcm_probe.sh
bash scripts/run_abide_dx_aal116_lcm_finetune.sh

source ~/.bashrc || true
conda activate lcm
python scripts/render_abide_dx_aal116_tables.py \
  --probe_dir outputs/abide_dx_aal116_lcm_probe \
  --finetune_dir outputs/abide_dx_aal116_lcm_finetune \
  --feature_source lcm_frozen \
  --out_dir outputs/abide_dx_aal116_tables
