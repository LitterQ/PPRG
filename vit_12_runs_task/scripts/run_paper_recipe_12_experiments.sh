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
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --seeds 0 1 2 \
  --output "${TASK_DIR}/configs/paper_initialization_hashes.json"
"${PYTHON}" -c 'import json,sys; approval=json.load(open(sys.argv[1],encoding="utf-8")); hashes=json.load(open(sys.argv[2],encoding="utf-8")); assert approval["implementation_hash"]==hashes["implementation_hash"]; assert approval["baseline_initialization_hash"]==hashes["seeds"]["0"]["hash"],(approval["baseline_initialization_hash"],hashes["seeds"]["0"]["hash"])' "${APPROVAL_FILE}" "${TASK_DIR}/configs/paper_initialization_hashes.json"

EXPERIMENTS=(pgd_at_linf10 pda_no_proj_w0p1 pda_no_proj_w0p2 pda_proj_w0p2)
SEEDS=(0 1 2)
read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
  echo "GPUS must provide at least four assignments; got: ${GPUS}" >&2; exit 2
fi
if [ "$(printf '%s\n' "${GPU_LIST[@]:0:4}" | sort -u | wc -l)" -ne 4 ]; then
  echo "The first four GPU assignments must be distinct: ${GPUS}" >&2; exit 2
fi

BASE_EVAL_ATTACKS=(
  clean fgsm_linf pgd_linf_20 pgd_linf_50 cw_linf_20 cw_linf_50 autoattack_linf
  pgd_l2_20 pgd_l2_50 cw_l2_20 cw_l2_50 autoattack_l2
)
LINF_SWEEP_ATTACKS=(fgsm_linf pgd_linf_20 pgd_linf_50 cw_linf_20 cw_linf_50)
L2_SWEEP_ATTACKS=(pgd_l2_20 pgd_l2_50 cw_l2_20 cw_l2_50)
LINF_EPS=(4/255 8/255 12/255 16/255)
LINF_ALPHA=(1/255 2/255 3/255 4/255)
LINF_LABEL=(eps4 eps8 eps12 eps16)
L2_EPS=(0.25 0.5 0.75 1.0)
L2_ALPHA=(0.05 0.1 0.15 0.2)
L2_LABEL=(eps0p25 eps0p5 eps0p75 eps1p0)

train_run() {
  local experiment="$1" seed="$2" run_dir="$3"
  if [ -f "${run_dir}/training_complete.json" ]; then
    echo "Training already complete: ${run_dir}"
    return 0
  fi
  local resume_args=()
  if [ -f "${run_dir}/last.pt" ]; then
    resume_args=(--resume "${run_dir}/last.pt")
  elif [ -f "${run_dir}/config.json" ]; then
    echo "Config exists without resumable last.pt: ${run_dir}" >&2
    return 3
  elif [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 ! -name console.log ! -name planned_config.json ! -name implementation_lock.json -print -quit)" ]; then
    echo "Unrecognized partial artifacts would be mixed into a new run: ${run_dir}" >&2
    return 3
  fi
  if [ ! -f "${run_dir}/planned_config.json" ]; then
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.config \
      --experiment "${experiment}" --seed "${seed}" --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
      --num-workers "${NUM_WORKERS}" --run-kind formal_12 --epochs 40 --batch-size 64 \
      --output "${run_dir}/planned_config.json"
  fi
  local command=(
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.train
    --experiment "${experiment}" --run-kind formal_12 --save-dir "${run_dir}"
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --data-dir "${DATA_DIR}"
    --epochs 40 --batch-size 64 --seed "${seed}" --num-workers "${NUM_WORKERS}" --device cuda --no-download
    "${resume_args[@]}"
  )
  paper_print_command "${command[@]}"
  "${command[@]}"
}

base_eval() {
  local run_dir="$1" seed="$2"
  local current_eval_env_hash
  current_eval_env_hash="$("${PYTHON}" -c 'from pareto_attack_dg.eval import evaluation_environment; print(evaluation_environment(include_autoattack=True)["environment_hash"])')"
  if [ "${current_eval_env_hash}" != "${APPROVED_EVAL_ENV_HASH}" ]; then
    echo "Evaluation environment changed after baseline approval: approved=${APPROVED_EVAL_ENV_HASH} current=${current_eval_env_hash}" >&2
    return 4
  fi
  if [ -f "${run_dir}/eval.json" ]; then
    echo "Base evaluation already exists: ${run_dir}/eval.json"
  else
    local command=(
      "${PYTHON}" -u eval.py --checkpoint "${run_dir}/best.pt" --dataset cifar10
      --eval_attacks "${BASE_EVAL_ATTACKS[@]}" --eps_linf 8/255 --alpha_linf 2/255
      --eps_l2 0.5 --alpha_l2 0.1 --batch_size 64 --data_dir "${DATA_DIR}" --no_download
      --num_workers "${NUM_WORKERS}" --eval_seed "${seed}" --device cuda --save_json "${run_dir}/eval.json"
    )
    paper_print_command "${command[@]}"
    "${command[@]}"
  fi
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
    --verify-evaluation "${run_dir}/eval.json" --checkpoint "${run_dir}/best.pt" \
    --expected-seed "${seed}" --expected-eps-linf 8/255 --expected-eps-l2 0.5 \
    --expected-environment-hash "${APPROVED_EVAL_ENV_HASH}" \
    --required-attacks "${BASE_EVAL_ATTACKS[@]}" >/dev/null
}

epsilon_sweep() {
  local run_dir="$1" seed="$2" sweep_dir="${run_dir}/epsilon_sweep_eval"
  mkdir -p "${sweep_dir}"
  local idx output command
  for idx in "${!LINF_EPS[@]}"; do
    output="${sweep_dir}/linf_${LINF_LABEL[$idx]}.json"
    if [ ! -f "${output}" ]; then
      command=(
        "${PYTHON}" -u eval.py --checkpoint "${run_dir}/best.pt" --dataset cifar10
        --eval_attacks "${LINF_SWEEP_ATTACKS[@]}" --eps_linf "${LINF_EPS[$idx]}"
        --alpha_linf "${LINF_ALPHA[$idx]}" --eps_l2 0.5 --alpha_l2 0.1 --batch_size 64
        --data_dir "${DATA_DIR}" --no_download --num_workers "${NUM_WORKERS}" --eval_seed "${seed}" \
        --device cuda --save_json "${output}"
      )
      paper_print_command "${command[@]}"
      "${command[@]}"
    fi
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
      --verify-evaluation "${output}" --checkpoint "${run_dir}/best.pt" \
      --expected-seed "${seed}" --expected-eps-linf "${LINF_EPS[$idx]}" --expected-eps-l2 0.5 \
      --required-attacks "${LINF_SWEEP_ATTACKS[@]}" >/dev/null
  done
  for idx in "${!L2_EPS[@]}"; do
    output="${sweep_dir}/l2_${L2_LABEL[$idx]}.json"
    if [ ! -f "${output}" ]; then
      command=(
        "${PYTHON}" -u eval.py --checkpoint "${run_dir}/best.pt" --dataset cifar10
        --eval_attacks "${L2_SWEEP_ATTACKS[@]}" --eps_linf 8/255 --alpha_linf 2/255
        --eps_l2 "${L2_EPS[$idx]}" --alpha_l2 "${L2_ALPHA[$idx]}" --batch_size 64
        --data_dir "${DATA_DIR}" --no_download --num_workers "${NUM_WORKERS}" --eval_seed "${seed}" \
        --device cuda --save_json "${output}"
      )
      paper_print_command "${command[@]}"
      "${command[@]}"
    fi
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
      --verify-evaluation "${output}" --checkpoint "${run_dir}/best.pt" \
      --expected-seed "${seed}" --expected-eps-linf 8/255 --expected-eps-l2 "${L2_EPS[$idx]}" \
      --required-attacks "${L2_SWEEP_ATTACKS[@]}" >/dev/null
  done
}

run_one() {
  local gpu="$1" experiment="$2" seed="$3"
  local run_dir="${RUNS_DIR}/${experiment}_seed${seed}"
  paper_require_task_path "${run_dir}"
  local lock="${run_dir}/implementation_lock.json"
  mkdir -p "${run_dir}"
  if [ -f "${lock}" ]; then
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity --verify-lock "${lock}" >/dev/null
  else
    if [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "Formal run has artifacts but no implementation lock: ${run_dir}" >&2
      return 3
    fi
    "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity --write-lock "${lock}" >/dev/null
  fi
  if [ -f "${run_dir}/.complete.json" ]; then
    "${PYTHON}" -u "${TASK_DIR}/scripts/summarize_vit_runs.py" --validate_run "${run_dir}" >/dev/null
    echo "Skipping validated complete run: ${run_dir}"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  {
    echo "experiment=${experiment} seed=${seed} gpu=${gpu} recipe=paper"
    train_run "${experiment}" "${seed}" "${run_dir}"
    base_eval "${run_dir}" "${seed}"
    epsilon_sweep "${run_dir}" "${seed}"
    "${PYTHON}" -u "${TASK_DIR}/scripts/summarize_vit_runs.py" --validate_run "${run_dir}"
    "${PYTHON}" -c 'import hashlib,json,sys; from datetime import datetime,timezone; from pathlib import Path; p=Path(sys.argv[1]); cfg=json.load((p/"config.json").open(encoding="utf-8")); lock=json.load((p/"implementation_lock.json").open(encoding="utf-8")); names=["planned_config.json","config.json","implementation_lock.json","training_complete.json","train_log.csv","batch_diagnostics.csv","best.pt","last.pt","eval.json",*[str(x.relative_to(p)) for x in sorted((p/"epsilon_sweep_eval").glob("*.json"))]]; hashes={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names}; marker={"validated":True,"completed_at":datetime.now(timezone.utc).isoformat(),"run_kind":"formal_12","paper_recipe_hash":cfg["paper_recipe_hash"],"implementation_hash":lock["implementation_hash"],"artifact_sha256":hashes}; (p/".complete.json").write_text(json.dumps(marker,indent=2,sort_keys=True)+"\n",encoding="utf-8")' "${run_dir}"
    echo "formal_run_complete=${run_dir}"
  } 2>&1 | tee -a "${run_dir}/console.log"
}

run_group() {
  local gpu="$1" experiment="$2" seed
  for seed in "${SEEDS[@]}"; do
    run_one "${gpu}" "${experiment}" "${seed}"
  done
}

pids=()
for idx in "${!EXPERIMENTS[@]}"; do
  run_group "${GPU_LIST[$idx]}" "${EXPERIMENTS[$idx]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [ "${status}" -ne 0 ]; then
  echo "At least one formal run failed; no aggregate summary was regenerated." >&2
  exit "${status}"
fi

"${PYTHON}" -u "${TASK_DIR}/scripts/summarize_vit_runs.py" \
  --runs_dir "${RUNS_DIR}" --output_dir "${SUMMARY_DIR}"
"${PYTHON}" -u "${TASK_DIR}/scripts/record_paper_recipe_status.py"
echo "All 12 paper-recipe runs and full evaluation matrices completed: ${RUNS_DIR}"
