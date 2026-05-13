#!/usr/bin/env bash
set -euo pipefail

cd /vePFS-0x0d/nzh/baseline/brain_network_decoder

bash scripts/run_abide_dx_aal116_lcm_probe_volcano.sh
bash scripts/run_abide_dx_aal116_lcm_finetune_volcano.sh

source /vePFS-0x0d/nzh/envs/lcm/bin/activate
python scripts/render_abide_dx_aal116_tables.py \
  --probe_dir outputs/abide_dx_aal116_lcm_probe \
  --finetune_dir outputs/abide_dx_aal116_lcm_finetune \
  --feature_source lcm_frozen \
  --out_dir outputs/abide_dx_aal116_tables
