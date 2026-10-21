#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paper_recipe_common.sh
source "${SCRIPT_DIR}/paper_recipe_common.sh"
paper_init_environment

: "${GPU:=0}"
: "${RUN_TAG:=officialinit_v2}"
: "${RESUME:=auto}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_TAG contains invalid characters: ${RUN_TAG}" >&2; exit 2
fi
RUN_DIR="${TASK_DIR}/paper_recipe_baseline/pgd_at_linf10_seed0_40ep_${RUN_TAG}"
paper_require_task_path "${RUN_DIR}"
if [ -f "${RUN_DIR}/training_complete.json" ]; then
  echo "Baseline training is already complete; refusing to rerun: ${RUN_DIR}" >&2; exit 3
fi
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

resume_args=()
if [ -f "${RUN_DIR}/last.pt" ]; then
  if [ "${RESUME}" != "auto" ] && [ "${RESUME}" != "yes" ]; then
    echo "Partial baseline exists. Set RESUME=auto (recommended) or choose a new RUN_TAG." >&2; exit 3
  fi
  resume_args=(--resume "${RUN_DIR}/last.pt")
elif [ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 ! -name console.log ! -name planned_config.json -print -quit)" ]; then
  echo "Non-resumable partial output exists in ${RUN_DIR}; choose a new RUN_TAG." >&2; exit 3
fi

if [ ! -f "${RUN_DIR}/planned_config.json" ]; then
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.config \
    --experiment pgd_at_linf10 --seed 0 --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --num-workers "${NUM_WORKERS}" --run-kind manual_baseline --epochs 40 --batch-size 64 \
    --output "${RUN_DIR}/planned_config.json" | tee "${RUN_DIR}/console.log"
else
  echo "Using existing planned_config.json for strict resume." | tee -a "${RUN_DIR}/console.log"
fi

command=(
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.train
  --experiment pgd_at_linf10 --run-kind manual_baseline --save-dir "${RUN_DIR}"
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --data-dir "${DATA_DIR}"
  --epochs 40 --batch-size 64 --seed 0 --num-workers "${NUM_WORKERS}" --device cuda --no-download
  "${resume_args[@]}"
)
paper_print_command "${command[@]}" | tee -a "${RUN_DIR}/console.log"
"${command[@]}" 2>&1 | tee -a "${RUN_DIR}/console.log"
test -f "${RUN_DIR}/training_complete.json"
"${PYTHON}" -u "${TASK_DIR}/scripts/record_paper_recipe_status.py"
echo "Training complete but NOT approved: ${RUN_DIR}"
echo "Next, manually run evaluate_paper_recipe_baseline.sh with BASELINE_DIR=${RUN_DIR}"
