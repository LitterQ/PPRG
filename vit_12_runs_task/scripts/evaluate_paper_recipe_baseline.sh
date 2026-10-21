#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paper_recipe_common.sh
source "${SCRIPT_DIR}/paper_recipe_common.sh"
paper_init_environment

: "${GPU:=0}"
: "${BASELINE_DIR:?Set BASELINE_DIR to the completed paper_recipe_baseline run directory}"
BASELINE_DIR="$(realpath -m -- "${BASELINE_DIR}")"
paper_require_task_path "${BASELINE_DIR}"
case "${BASELINE_DIR}" in
  "${TASK_DIR}/paper_recipe_baseline"/*) ;;
  *) echo "BASELINE_DIR is not under paper_recipe_baseline: ${BASELINE_DIR}" >&2; exit 2 ;;
esac
test -f "${BASELINE_DIR}/training_complete.json"
test -f "${BASELINE_DIR}/last.pt"
: "${EVAL_OUTPUT:=${BASELINE_DIR}/baseline_eval_official.json}"
: "${EVAL_LOG:=${BASELINE_DIR}/baseline_eval_official_console.log}"
paper_require_task_path "${EVAL_OUTPUT}"
paper_require_task_path "${EVAL_LOG}"
if [ -f "${EVAL_OUTPUT}" ]; then
  echo "Refusing to overwrite existing official evaluation: ${EVAL_OUTPUT}" >&2; exit 3
fi
export CUDA_VISIBLE_DEVICES="${GPU}"

command=(
  "${PYTHON}" -u eval.py --checkpoint "${BASELINE_DIR}/last.pt" --dataset cifar10
  --eval_attacks clean cw_linf_20 pgd_linf_20 pgd_linf_100 autoattack_linf
  --eps_linf 8/255 --alpha_linf 2/255 --eps_l2 0.5 --alpha_l2 0.1
  --batch_size 256 --data_dir "${DATA_DIR}" --no_download --num_workers "${NUM_WORKERS}"
  --eval_seed 0 --device cuda --save_json "${EVAL_OUTPUT}"
)
paper_print_command "${command[@]}" | tee "${EVAL_LOG}"
"${command[@]}" 2>&1 | tee -a "${EVAL_LOG}"
"${PYTHON}" -u -m vit_12_runs_task.paper_recipe.integrity \
  --verify-evaluation "${EVAL_OUTPUT}" --checkpoint "${BASELINE_DIR}/last.pt" \
  --expected-seed 0 --expected-eps-linf 8/255 --expected-eps-l2 0.5 \
  --required-attacks clean cw_linf_20 pgd_linf_20 pgd_linf_100 autoattack_linf >/dev/null
"${PYTHON}" -c 'import json,math,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); keys=("clean_acc","cw_linf_20_acc","pgd_linf_20_acc","pgd_linf_100_acc","autoattack_linf_acc"); assert all(isinstance(p["metrics"].get(k),(int,float)) and math.isfinite(p["metrics"][k]) for k in keys),p["metrics"]; assert p["evaluation_semantics"]["paper_official_linf"] is True; assert all(v==10000 for v in p["dataloader"]["actual_samples_per_attack"].values()); print({k:p["metrics"][k] for k in keys})' "${EVAL_OUTPUT}"
"${PYTHON}" -u "${TASK_DIR}/scripts/record_paper_recipe_status.py"
echo "Evaluation complete; review metrics before explicit approval. No approval marker was created."
