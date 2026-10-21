#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${DATA_DIR:=data}"
: "${OUTPUT_ROOT:=vit_12_runs_task/paper_recipe_runs}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"
: "${EVALUATE:=1}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the paper DeiT-Tiny source checkpoint}"

experiments=(
  s4_pgd_at_linf5
  s4_avg_four_source s4_avg_five_source
  s4_pcgrad_four_source s4_pcgrad_five_source s4_pcgrad_matched_two_source
  s4_mgda_four_source s4_mgda_five_source s4_mgda_matched_two_source
)
download_args=()
if [ "${NO_DOWNLOAD}" = "1" ]; then download_args+=(--no-download); fi

for experiment in "${experiments[@]}"; do
  run_dir="${OUTPUT_ROOT}/${experiment}_seed0"
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.supplemental_table_s4 \
    --experiment "${experiment}" --run-kind manual_baseline --save-dir "${run_dir}" \
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" --data-dir "${DATA_DIR}" \
    --epochs 40 --batch-size 64 --seed 0 --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" "${download_args[@]}"
  if [ "${EVALUATE}" = "1" ]; then
    CHECKPOINT="${run_dir}/best.pt" DATASET=cifar10 DATA_DIR="${DATA_DIR}" BATCH_SIZE=64 \
      NUM_WORKERS="${NUM_WORKERS}" DEVICE="${DEVICE}" NO_DOWNLOAD="${NO_DOWNLOAD}" \
      bash scripts/eval/evaluate_checkpoint.sh
  fi
done
