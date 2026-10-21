#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${RUNS_DIR:=vit_12_runs_task/paper_recipe_runs}"
: "${OUTPUT_DIR:=vit_12_runs_task/paper_recipe_landscape_absolute}"
: "${DATA_DIR:=data}"
: "${DEVICE:=auto}"
: "${GPUS:=}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"

extra_args=()
if [ -n "${GPUS}" ]; then extra_args+=(--gpus "${GPUS}"); fi
if [ "${NO_DOWNLOAD}" = "1" ]; then extra_args+=(--no_download); fi

"${PYTHON}" -u -m vit_12_runs_task.analysis.loss_landscape_v3_multi \
  --runs_dir "${RUNS_DIR}" --output_dir "${OUTPUT_DIR}" --data_dir "${DATA_DIR}" \
  --seed 0 --grid_size 15 --radius 0.2 --direction_seed 20260818 \
  --num_samples 256 --batch_size 64 --subset_seed 20250816 \
  --primary_attack pgd_linf_10 --auxiliary_attack pgd_l2_5 \
  --unseen_attacks pgd_linf_20,pgd_l2_20,cw_linf_20 \
  --plot_attacks pgd_linf_10,cw_linf_20,pgd_l2_20 \
  --eps_linf 0.03137254901960784 --alpha_linf 0.00784313725490196 \
  --eps_l2 0.5 --alpha_l2 0.1 --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" "${extra_args[@]}"
