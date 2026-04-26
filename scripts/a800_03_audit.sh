#!/usr/bin/env bash
set -euo pipefail

# Step 3: run local-copy audit before training.

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
ENV_FILE="${ROOT}/run_table3_one.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "Missing ${ENV_FILE}; run a800_01_prepare_repo_data.sh first." >&2
  exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source "${ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ROOT}/envs/lcm"
cd "${REPO_DST}"

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids all \
  --folds 1 \
  --seeds 4 \
  --device cuda:0 \
  --audit_only \
  --output_root "${ONE_ROOT}/audit_${RUN_TS}" \
  --task_dir_suffix "${RUN_TS}"

echo "==> Audit files"
find "${ONE_ROOT}/audit_${RUN_TS}" -name audit.json | sort

echo "==> Audit summary/warnings"
find "${ONE_ROOT}/audit_${RUN_TS}" -name train.log -exec grep -H "Split summary\\|Audit warning" {} \; | sort || true

echo "==> Step 3 done. If ADNI missing_files/missing_labels are 0, start training:"
echo "bash ${REPO_DST}/scripts/a800_run_step_in_tmux.sh launch_1 ${REPO_DST}/scripts/a800_04_launch_tmux.sh"
