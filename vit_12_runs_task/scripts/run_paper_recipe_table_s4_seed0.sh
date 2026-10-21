#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paper_recipe_common.sh
source "${SCRIPT_DIR}/paper_recipe_common.sh"
paper_init_environment

: "${GPUS:=0 1 2 3}"
: "${APPROVAL_FILE:=${TASK_DIR}/paper_recipe_approvals/baseline_approval.json}"
: "${RUNS_DIR:=${TASK_DIR}/paper_recipe_runs}"
: "${SUMMARY_DIR:=${TASK_DIR}/paper_recipe_summaries}"
paper_require_task_path "${APPROVAL_FILE}"
paper_require_task_path "${RUNS_DIR}"
paper_require_task_path "${SUMMARY_DIR}"
mkdir -p "${RUNS_DIR}" "${SUMMARY_DIR}"

"${PYTHON}" -u "${TASK_DIR}/scripts/approve_paper_recipe_baseline.py" \
  --verify-only --approval "${APPROVAL_FILE}"
APPROVED_EVAL_ENV_HASH="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["evaluation_environment_hash"])' "${APPROVAL_FILE}")"
"${PYTHON}" -u "${TASK_DIR}/scripts/verify_paper_initialization.py" \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --seeds 0 \
  --output "${TASK_DIR}/configs/paper_initialization_hashes.json"
"${PYTHON}" -c '
import json,sys
approval=json.load(open(sys.argv[1],encoding="utf-8"))
hashes=json.load(open(sys.argv[2],encoding="utf-8"))
expected="1ea7e99480dc31523e62aa8af5c496204931c15052e4fc660277e9acba45d022"
actual=hashes["seeds"]["0"]["hash"]
assert approval["implementation_hash"] == hashes["implementation_hash"]
assert approval["baseline_initialization_hash"] == actual
assert actual == expected, (actual, expected)
' "${APPROVAL_FILE}" "${TASK_DIR}/configs/paper_initialization_hashes.json"

# Reused rows are a hard precondition.  This verifies their original locks,
# checkpoints, eval.json artifacts, completion markers, and approved evaluation
# environment without modifying either old run.
for reused in pgd_at_linf10 pda_no_proj_w0p1; do
  reused_dir="${RUNS_DIR}/${reused}_seed0"
  if [ ! -f "${reused_dir}/.complete.json" ]; then
    echo "Required reused Table S4 run is incomplete: ${reused_dir}" >&2
    exit 3
  fi
  "${PYTHON}" -u "${TASK_DIR}/scripts/summarize_deit_table_s4_seed0.py" \
    --validate-run "${reused_dir}" \
    --expected-evaluation-environment-hash "${APPROVED_EVAL_ENV_HASH}" >/dev/null
done
echo "Validated both reused Table S4 seed-0 rows before starting supplemental training."

EXPERIMENTS=(
  s4_pgd_at_linf5
  s4_avg_four_source
  s4_avg_five_source
  s4_pcgrad_four_source
  s4_pcgrad_five_source
  s4_pcgrad_matched_two_source
  s4_mgda_four_source
  s4_mgda_five_source
  s4_mgda_matched_two_source
)
# Training code remains seed-generic.  This supplemental launcher intentionally
# schedules only seed 0; future expansion can change this array to (0 1 2).
SEEDS=(0)
read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "GPUS must provide at least one GPU assignment." >&2
  exit 2
fi
if [ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -ne "${#GPU_LIST[@]}" ]; then
  echo "GPU assignments must be distinct: ${GPUS}" >&2
  exit 2
fi

S4_EVAL_ATTACKS=(
  clean fgsm_linf pgd_linf_20 pgd_linf_50 autoattack_linf
  pgd_l2_20 pgd_l2_50 autoattack_l2
)

prepare_locks() {
  local run_dir="$1"
  local base_lock="${run_dir}/implementation_lock.json"
  local table_lock="${run_dir}/table_s4_implementation_lock.json"
  if [ -f "${base_lock}" ]; then
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
      --verify-lock "${base_lock}" >/dev/null
  else
    if [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "Table S4 run has artifacts but no base implementation lock: ${run_dir}" >&2
      return 3
    fi
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
      --write-lock "${base_lock}" >/dev/null
  fi
  if [ -f "${table_lock}" ]; then
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.supplemental_table_s4 \
      --verify-lock "${table_lock}" >/dev/null
  else
    if [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 ! -name implementation_lock.json -print -quit)" ]; then
      echo "Table S4 run has artifacts but no supplemental implementation lock: ${run_dir}" >&2
      return 3
    fi
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.supplemental_table_s4 \
      --write-lock "${table_lock}" >/dev/null
  fi
}

reject_unknown_partial_artifacts() {
  local run_dir="$1"
  local unknown
  unknown="$(find "${run_dir}" -mindepth 1 -maxdepth 1 \
    ! -name console.log \
    ! -name planned_config.json \
    ! -name config.json \
    ! -name implementation_lock.json \
    ! -name table_s4_implementation_lock.json \
    ! -name training_complete.json \
    ! -name train_log.csv \
    ! -name batch_diagnostics.csv \
    ! -name best.pt \
    ! -name last.pt \
    ! -name eval.json \
    ! -name .complete.json \
    -print -quit)"
  if [ -n "${unknown}" ]; then
    echo "Unknown partial artifact would be mixed into Table S4 run: ${unknown}" >&2
    return 3
  fi
}

train_run() {
  local experiment="$1" seed="$2" run_dir="$3"
  reject_unknown_partial_artifacts "${run_dir}"
  if [ -f "${run_dir}/training_complete.json" ]; then
    echo "Training already complete: ${run_dir}"
    return 0
  fi
  local resume_args=()
  if [ -f "${run_dir}/last.pt" ]; then
    resume_args=(--resume "${run_dir}/last.pt")
  elif [ -f "${run_dir}/config.json" ]; then
    echo "Config exists without a resumable last.pt: ${run_dir}" >&2
    return 3
  fi
  if [ ! -f "${run_dir}/planned_config.json" ]; then
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.supplemental_table_s4 \
      --write-plan --experiment "${experiment}" --seed "${seed}" \
      --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --num-workers "${NUM_WORKERS}" \
      --output "${run_dir}/planned_config.json"
  fi
  local command=(
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.supplemental_table_s4
    --experiment "${experiment}" --run-kind formal_12 --save-dir "${run_dir}"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --data-dir "${DATA_DIR}"
    --epochs 40 --batch-size 64 --seed "${seed}" --num-workers "${NUM_WORKERS}"
    --device cuda --no-download "${resume_args[@]}"
  )
  paper_print_command "${command[@]}"
  "${command[@]}"
}

table_s4_eval() {
  local run_dir="$1" seed="$2"
  local current_eval_env_hash
  current_eval_env_hash="$("${PYTHON}" -c 'from pareto_attack_dg.eval import evaluation_environment; print(evaluation_environment(include_autoattack=True)["environment_hash"])')"
  if [ "${current_eval_env_hash}" != "${APPROVED_EVAL_ENV_HASH}" ]; then
    echo "Evaluation environment changed after baseline approval: approved=${APPROVED_EVAL_ENV_HASH} current=${current_eval_env_hash}" >&2
    return 4
  fi
  if [ ! -f "${run_dir}/eval.json" ]; then
    local command=(
      "${PYTHON}" -u eval.py --checkpoint "${run_dir}/best.pt" --dataset cifar10
      --eval_attacks "${S4_EVAL_ATTACKS[@]}" --eps_linf 8/255 --alpha_linf 2/255
      --eps_l2 0.5 --alpha_l2 0.1 --batch_size 64 --data_dir "${DATA_DIR}" --no_download
      --num_workers "${NUM_WORKERS}" --eval_seed "${seed}" --device cuda
      --save_json "${run_dir}/eval.json"
    )
    paper_print_command "${command[@]}"
    "${command[@]}"
  else
    echo "Table S4 evaluation already exists; validating without overwrite: ${run_dir}/eval.json"
  fi
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
    --verify-evaluation "${run_dir}/eval.json" --checkpoint "${run_dir}/best.pt" \
    --expected-seed "${seed}" --expected-eps-linf 8/255 --expected-eps-l2 0.5 \
    --expected-environment-hash "${APPROVED_EVAL_ENV_HASH}" \
    --required-attacks "${S4_EVAL_ATTACKS[@]}" >/dev/null
}

run_one() {
  local gpu="$1" experiment="$2" seed="$3"
  local run_dir="${RUNS_DIR}/${experiment}_seed${seed}"
  paper_require_task_path "${run_dir}"
  mkdir -p "${run_dir}"
  prepare_locks "${run_dir}"
  reject_unknown_partial_artifacts "${run_dir}"
  if [ -f "${run_dir}/.complete.json" ]; then
    "${PYTHON}" -u "${TASK_DIR}/scripts/summarize_deit_table_s4_seed0.py" \
      --validate-run "${run_dir}" \
      --expected-evaluation-environment-hash "${APPROVED_EVAL_ENV_HASH}" >/dev/null
    echo "Skipping strictly validated complete Table S4 run: ${run_dir}"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  {
    echo "experiment=${experiment} seed=${seed} gpu=${gpu} recipe=paper supplement=deit-table-s4-seed0-symmetric-aggregation-v1"
    train_run "${experiment}" "${seed}" "${run_dir}"
    table_s4_eval "${run_dir}" "${seed}"
    "${PYTHON}" -u "${TASK_DIR}/scripts/summarize_deit_table_s4_seed0.py" \
      --mark-complete "${run_dir}" \
      --expected-evaluation-environment-hash "${APPROVED_EVAL_ENV_HASH}"
    echo "table_s4_formal_run_complete=${run_dir}"
  } 2>&1 | tee -a "${run_dir}/console.log"
}

JOB_EXPERIMENTS=()
JOB_SEEDS=()
for experiment in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOB_EXPERIMENTS+=("${experiment}")
    JOB_SEEDS+=("${seed}")
  done
done

run_worker() {
  local worker_index="$1" gpu="$2" job_index
  for ((job_index=worker_index; job_index<${#JOB_EXPERIMENTS[@]}; job_index+=${#GPU_LIST[@]})); do
    run_one "${gpu}" "${JOB_EXPERIMENTS[$job_index]}" "${JOB_SEEDS[$job_index]}"
  done
}

pids=()
for worker_index in "${!GPU_LIST[@]}"; do
  run_worker "${worker_index}" "${GPU_LIST[$worker_index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  echo "At least one Table S4 training/evaluation failed; no final 11-row table was generated." >&2
  exit "${status}"
fi

"${PYTHON}" -u "${TASK_DIR}/scripts/summarize_deit_table_s4_seed0.py" \
  --runs-dir "${RUNS_DIR}" --output-dir "${SUMMARY_DIR}" \
  --expected-evaluation-environment-hash "${APPROVED_EVAL_ENV_HASH}"
echo "All 9 new runs and 2 reused rows passed; DeiT-Tiny Table S4 seed 0 is complete."
