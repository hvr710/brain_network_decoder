#!/usr/bin/env bash
set -euo pipefail

# Step 2: create the lcm and bsem conda environments under /vePFS.

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
ENV_FILE="${ROOT}/run_table3_one.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

REPO_DST="${REPO_DST:-${ROOT}/repo/brain_network_decoder}"

cd "${REPO_DST}"
bash scripts/bootstrap_a800_envs.sh

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source "${ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ROOT}/envs/lcm"

python -c "import torch, torch_geometric, torch_scatter; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m py_compile finetune_table3.py table3_dataset.py table3_utils.py launch_table3_queue.py summarize_table3.py

echo "==> Step 2 done. Next:"
echo "source ${ENV_FILE}"
echo "bash ${REPO_DST}/scripts/a800_03_audit.sh"
