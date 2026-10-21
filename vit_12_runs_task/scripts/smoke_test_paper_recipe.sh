#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=paper_recipe_common.sh
source "${SCRIPT_DIR}/paper_recipe_common.sh"
paper_init_environment

: "${GPU:=0}"
: "${SMOKE_OUTPUT_DIR:=${TASK_DIR}/paper_recipe_smoke}"
: "${DEVICE:=cuda}"
: "${RESUME_INCOMPLETE_SMOKE:=0}"
paper_require_task_path "${SMOKE_OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

command=(
  "${PYTHON}" -u "${TASK_DIR}/scripts/smoke_test_paper_recipe.py"
  --python "${PYTHON}"
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
  --output-dir "${SMOKE_OUTPUT_DIR}"
  --device "${DEVICE}"
)
if [ "${RESUME_INCOMPLETE_SMOKE}" = "1" ]; then
  command+=(--resume-incomplete-smoke)
fi
paper_print_command "${command[@]}"
"${command[@]}"
"${PYTHON}" -u "${TASK_DIR}/scripts/record_paper_recipe_status.py"
