#!/usr/bin/env bash
set -euo pipefail

# Step 4: launch the three official tmux jobs.
# Override GPUs if needed:
#   AGE_GPU=cuda:0 SEX_GPU=cuda:1 DISEASE_GPU=cuda:2 bash scripts/a800_run_step_in_tmux.sh launch_1 scripts/a800_04_launch_tmux.sh

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
ENV_FILE="${ROOT}/run_table3_one.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "Missing ${ENV_FILE}; run a800_01_prepare_repo_data.sh first." >&2
  exit 1
fi

AGE_GPU="${AGE_GPU:-cuda:0}"
SEX_GPU="${SEX_GPU:-cuda:1}"
DISEASE_GPU="${DISEASE_GPU:-cuda:2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

start_session() {
  local session="$1"
  local gpu="$2"
  local task_ids="$3"
  local group="$4"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    echo "Attach with: tmux attach -t ${session}" >&2
    return
  fi

  tmux new-session -d -s "${session}" "
source ${ENV_FILE}
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ${ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate ${ROOT}/envs/lcm
cd ${REPO_DST}
python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids ${task_ids} \
  --folds 1 \
  --seeds 4,44,444 \
  --device ${gpu} \
  --batch_size ${BATCH_SIZE} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --output_root ${ONE_ROOT}/${group}_${RUN_TS} \
  --task_dir_suffix ${RUN_TS}
"
  echo "Started ${session} on ${gpu}: ${task_ids}"
}

echo "==> Launching tmux sessions with RUN_TS=${RUN_TS}"
start_session age_1 "${AGE_GPU}" "abide_age,nki_age,sald_age" "age"
start_session sex_1 "${SEX_GPU}" "abcd_sex,hcp_sex,bhrc_sex" "sex"
start_session disease_1 "${DISEASE_GPU}" "ppmi_pd_diagnosis,adni_mci,adni_ad,nki_education" "disease"

echo
echo "==> Attach commands"
echo "tmux attach -t age_1"
echo "tmux attach -t sex_1"
echo "tmux attach -t disease_1"
echo
echo "==> Output root"
echo "${ONE_ROOT}"
