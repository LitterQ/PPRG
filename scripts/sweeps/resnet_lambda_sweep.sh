#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${MODE:=both}"
: "${SEEDS:=0 1 2}"
: "${DATA_DIR:=data}"
: "${OUTPUT_ROOT:=runs/resnet18_lambda_sweep}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"
: "${EVALUATE:=1}"

case "${MODE}" in pda) families=(pda);; pda_proj) families=(pda_proj);; both) families=(pda pda_proj);; *) exit 2;; esac
weights=(0.05 0.10 0.15 0.20 0.30)
read -r -a seed_list <<< "${SEEDS}"
data_args=(--data_dir "${DATA_DIR}" --num_workers "${NUM_WORKERS}")
if [ "${NO_DOWNLOAD}" = "1" ]; then data_args+=(--no_download); fi

canonical_label() {
  case "$1" in
    0.05) echo 0p05;; 0.10) echo 0p1;; 0.15) echo 0p15;; 0.20) echo 0p2;; 0.30) echo 0p3;;
  esac
}

for family in "${families[@]}"; do
  for weight in "${weights[@]}"; do
    label="$(canonical_label "${weight}")"
    for seed in "${seed_list[@]}"; do
      run_dir="${OUTPUT_ROOT}/${family}_w${label}_seed${seed}"
      if [ "${family}" = pda ]; then
        method_args=(--method avg_attack_dg --attack_domains pgd_linf_10 pgd_l2_5 \
          --domain_weights pgd_linf_10:1.0,pgd_l2_5:"${weight}" --no-normalize-domain-weights)
      else
        method_args=(--method primary_projected_aux_l2 --attack_domains pgd_linf_10 pgd_l2_5 \
          --primary_domain pgd_linf_10 --aux_domain pgd_l2_5 --aux_weight "${weight}")
      fi
      "${PYTHON}" -u train.py --dataset cifar10 --model resnet18 "${method_args[@]}" \
        --epochs 100 --batch_size 128 --lr 0.1 --momentum 0.9 --weight_decay 5e-4 \
        --optimizer sgd --scheduler cosine --eps_linf 8/255 --alpha_linf 2/255 \
        --eps_l2 0.5 --alpha_l2 0.1 --seed "${seed}" --device "${DEVICE}" \
        "${data_args[@]}" --save_dir "${run_dir}"
      if [ "${EVALUATE}" = "1" ]; then
        CHECKPOINT="${run_dir}/best.pt" DATASET=cifar10 MODEL=resnet18 DATA_DIR="${DATA_DIR}" \
          BATCH_SIZE=128 NUM_WORKERS="${NUM_WORKERS}" DEVICE="${DEVICE}" NO_DOWNLOAD="${NO_DOWNLOAD}" \
          bash scripts/eval/evaluate_checkpoint.sh
      fi
    done
  done
done
