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

if [[ $# -gt 0 ]]; then
  printf -v SCRIPT_ARGS_Q '%q ' "$@"
else
  SCRIPT_ARGS_Q=""
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

LOG_DIR="${ROOT}/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

tmux new-session -d -s "${SESSION}" -c "${START_DIR}" "
set -euo pipefail
mkdir -p '${LOG_DIR}'
exec > >(tee -a '${LOG_FILE}') 2>&1
export A800_ROOT='${A800_ROOT:-}'
export REPO_DST='${REPO_DST:-}'
export DATA_DST='${DATA_DST:-}'
export RUN_TS='${RUN_TS:-}'
export ONE_ROOT='${ONE_ROOT:-}'
export SRC_REPO='${SRC_REPO:-}'
export SRC_PRETRAIN='${SRC_PRETRAIN:-}'
export SRC_DATASET1='${SRC_DATASET1:-}'
export SRC_DATASET4='${SRC_DATASET4:-}'
export HUOSHAN_HOST='${HUOSHAN_HOST:-}'
export HUOSHAN_ROOT='${HUOSHAN_ROOT:-}'
export COPY_REPO='${COPY_REPO:-}'
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
echo '=== Session ${SESSION} started at '\"\$(date '+%F %T')\"' ==='
rc=0
bash '${SCRIPT_PATH}' ${SCRIPT_ARGS_Q} || rc=\$?
echo
if [[ \$rc -eq 0 ]]; then
  echo '=== Session ${SESSION} completed successfully ==='
else
  echo '=== Session ${SESSION} failed with exit code' \$rc '==='
fi
echo 'Log file: ${LOG_FILE}'
echo 'Session is staying open. Type exit when you are done inspecting it.'
exec bash
"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Log file: ${LOG_FILE}"
