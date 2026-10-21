#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paper_recipe_common.sh
source "${SCRIPT_DIR}/paper_recipe_common.sh"
paper_init_environment

: "${GPU:=0}"
: "${RUN_TAG:=manual1}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_TAG contains invalid characters: ${RUN_TAG}" >&2; exit 2
fi
RUN_DIR="${TASK_DIR}/paper_recipe_validation/pgd_at_linf10_seed0_1ep_${RUN_TAG}"
paper_require_task_path "${RUN_DIR}"
if [ -d "${RUN_DIR}" ] && [ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Refusing to overwrite validation directory: ${RUN_DIR}" >&2; exit 3
fi
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" -u -m vit_12_runs_task.paper_recipe.config \
  --experiment pgd_at_linf10 --seed 0 --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --num-workers "${NUM_WORKERS}" --run-kind validation_only --epochs 1 --batch-size 64 \
  --output "${RUN_DIR}/planned_config.json" | tee "${RUN_DIR}/console.log"

train_command=(
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.train
  --experiment pgd_at_linf10 --run-kind validation_only --save-dir "${RUN_DIR}"
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --data-dir "${DATA_DIR}"
  --epochs 1 --batch-size 64 --seed 0 --num-workers "${NUM_WORKERS}" --device cuda --no-download
)
paper_print_command "${train_command[@]}" | tee -a "${RUN_DIR}/console.log"
"${train_command[@]}" 2>&1 | tee -a "${RUN_DIR}/console.log"

eval_command=(
  "${PYTHON}" -u eval.py --checkpoint "${RUN_DIR}/last.pt" --dataset cifar10
  --eval_attacks clean pgd_linf_2 --eps_linf 8/255 --alpha_linf 2/255
  --batch_size 64 --data_dir "${DATA_DIR}" --no_download --num_workers "${NUM_WORKERS}"
  --max_eval_batches 4 --device cuda --save_json "${RUN_DIR}/validation_eval.json"
)
paper_print_command "${eval_command[@]}" | tee "${RUN_DIR}/eval_console.log"
"${eval_command[@]}" 2>&1 | tee -a "${RUN_DIR}/eval_console.log"

"${PYTHON}" -c 'import csv,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=list(csv.DictReader((p/"train_log.csv").open(encoding="utf-8"))); assert len(rows)==1, rows; assert int(rows[0]["train_samples_seen"])==50000, rows[0]; assert (p/"VALIDATION_ONLY_NOT_A_BASELINE").is_file(); print("validation_artifacts_ok")' "${RUN_DIR}"
"${PYTHON}" -u "${TASK_DIR}/scripts/record_paper_recipe_status.py"
echo "Validation only (not a baseline): ${RUN_DIR}"
