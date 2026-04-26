#!/usr/bin/env bash
set -euo pipefail

# Launch one A800 preparation/training step in a detached tmux session.
# Usage:
#   bash scripts/a800_run_step_in_tmux.sh prep_1 scripts/a800_01_prepare_repo_data.sh

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <tmux_session> <script_path> [script args...]" >&2
  exit 1
fi

SESSION="$1"
shift
SCRIPT_PATH="$1"
shift || true

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
ENV_FILE="${ROOT}/run_table3_one.env"
START_DIR="$(pwd -P)"

if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="${START_DIR}/${SCRIPT_PATH}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -c "${START_DIR}" "
set -euo pipefail
export A800_ROOT='${A800_ROOT:-}'
export REPO_DST='${REPO_DST:-}'
export DATA_DST='${DATA_DST:-}'
export RUN_TS='${RUN_TS:-}'
export ONE_ROOT='${ONE_ROOT:-}'
export SRC_REPO='${SRC_REPO:-}'
export SRC_PRETRAIN='${SRC_PRETRAIN:-}'
export SRC_DATASET1='${SRC_DATASET1:-}'
export SRC_DATASET4='${SRC_DATASET4:-}'
export AGE_GPU='${AGE_GPU:-}'
export SEX_GPU='${SEX_GPU:-}'
export DISEASE_GPU='${DISEASE_GPU:-}'
export BATCH_SIZE='${BATCH_SIZE:-}'
export GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS:-}'
if [[ -f ${ENV_FILE} ]]; then
  source ${ENV_FILE}
fi
if [[ -n \${REPO_DST:-} && -d \${REPO_DST:-} ]]; then
  cd \${REPO_DST}
fi
bash '${SCRIPT_PATH}' $*
"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
