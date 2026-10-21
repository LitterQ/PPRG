#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${EXPERIMENT:=pda_no_proj_w0p1}"
: "${SEEDS:=0 1 2}"
: "${DATA_DIR:=data}"
: "${OUTPUT_ROOT:=vit_12_runs_task/paper_recipe_runs}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the adapted source DeiT-Tiny checkpoint}"

case "${EXPERIMENT}" in
  pgd_at_linf10|pda_no_proj_w0p1|pda_no_proj_w0p2|pda_proj_w0p2) ;;
  *) echo "Unsupported DeiT experiment: ${EXPERIMENT}" >&2; exit 2 ;;
esac

download_args=()
if [ "${NO_DOWNLOAD}" = "1" ]; then download_args+=(--no-download); fi
read -r -a seed_list <<< "${SEEDS}"

for seed in "${seed_list[@]}"; do
  run_dir="${OUTPUT_ROOT}/${EXPERIMENT}_seed${seed}"
  "${PYTHON}" -u -m vit_12_runs_task.paper_recipe.train \
    --experiment "${EXPERIMENT}" --run-kind manual_baseline \
    --save-dir "${run_dir}" --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" --epochs 40 --batch-size 64 --seed "${seed}" \
    --num-workers "${NUM_WORKERS}" --device "${DEVICE}" "${download_args[@]}"
done
