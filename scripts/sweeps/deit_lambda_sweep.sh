#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${MODE:=both}"
: "${SEEDS:=0 1 2}"
: "${DATA_DIR:=data}"
: "${OUTPUT_ROOT:=vit_12_runs_task/paper_recipe_runs}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"
: "${EVALUATE:=1}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the paper DeiT-Tiny source checkpoint}"

case "${MODE}" in pda) families=(pda);; pda_proj) families=(pda_proj);; both) families=(pda pda_proj);; *) exit 2;; esac
weights=(0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90)
read -r -a seed_list <<< "${SEEDS}"
download_args=()
if [ "${NO_DOWNLOAD}" = "1" ]; then download_args+=(--no-download); fi

canonical_label() {
  case "$1" in
    0.05) echo 0p05;; 0.10) echo 0p1;; 0.15) echo 0p15;; 0.20) echo 0p2;; 0.30) echo 0p3;;
    0.40) echo 0p4;; 0.50) echo 0p5;; 0.60) echo 0p6;; 0.70) echo 0p7;; 0.80) echo 0p8;; 0.90) echo 0p9;;
  esac
}

for family in "${families[@]}"; do
  for weight in "${weights[@]}"; do
    label="$(canonical_label "${weight}")"
    experiment="pda_no_proj_w${label}"
    if [ "${family}" = pda_proj ]; then experiment="pda_proj_w${label}"; fi
    case "${experiment}" in
      pgd_at_linf10|pda_no_proj_w0p1|pda_no_proj_w0p2|pda_proj_w0p2)
        module=vit_12_runs_task.paper_recipe.train;;
      pda_proj_w0p1)
        module=vit_12_runs_task.paper_recipe.supplemental_proj_w0p1;;
      pda_no_proj_w0p05|pda_no_proj_w0p15|pda_no_proj_w0p3|pda_proj_w0p05|pda_proj_w0p15|pda_proj_w0p3)
        module=vit_12_runs_task.paper_recipe.supplemental_lambda_sweep;;
      *) module=vit_12_runs_task.paper_recipe.supplemental_high_lambda_sweep;;
    esac
    for seed in "${seed_list[@]}"; do
      run_dir="${OUTPUT_ROOT}/${experiment}_seed${seed}"
      "${PYTHON}" -u -m "${module}" --experiment "${experiment}" --run-kind manual_baseline \
        --save-dir "${run_dir}" --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
        --data-dir "${DATA_DIR}" --epochs 40 --batch-size 64 --seed "${seed}" \
        --num-workers "${NUM_WORKERS}" --device "${DEVICE}" "${download_args[@]}"
      if [ "${EVALUATE}" = "1" ]; then
        CHECKPOINT="${run_dir}/best.pt" DATASET=cifar10 DATA_DIR="${DATA_DIR}" BATCH_SIZE=64 \
          NUM_WORKERS="${NUM_WORKERS}" DEVICE="${DEVICE}" NO_DOWNLOAD="${NO_DOWNLOAD}" \
          bash scripts/eval/evaluate_checkpoint.sh
      fi
    done
  done
done
