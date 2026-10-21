#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${DATASET:=cifar10}"
: "${MODEL:=resnet18}"
: "${EXPERIMENT:=pda_0p1}"
: "${SEEDS:=0 1 2}"
: "${DATA_DIR:=data}"
: "${OUTPUT_ROOT:=runs/${DATASET}_${MODEL}}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=2}"
: "${NO_DOWNLOAD:=0}"

case "${DATASET}/${MODEL}" in
  cifar10/resnet18|cifar100/resnet18|cifar10/wrn_28_10) ;;
  *) echo "Unsupported main setting: DATASET=${DATASET} MODEL=${MODEL}" >&2; exit 2 ;;
esac

case "${EXPERIMENT}" in
  pgd_at)
    run_name="pgd_at_linf10"
    method_args=(--method pgd_at --train_attack pgd_linf_10)
    ;;
  pda_0p1)
    run_name="pda_no_proj_w0p1"
    method_args=(--method avg_attack_dg --attack_domains pgd_linf_10 pgd_l2_5 \
      --domain_weights pgd_linf_10:1.0,pgd_l2_5:0.1 --no-normalize-domain-weights)
    ;;
  pda_0p2)
    run_name="pda_no_proj_w0p2"
    method_args=(--method avg_attack_dg --attack_domains pgd_linf_10 pgd_l2_5 \
      --domain_weights pgd_linf_10:1.0,pgd_l2_5:0.2 --no-normalize-domain-weights)
    ;;
  *) echo "EXPERIMENT must be pgd_at, pda_0p1, or pda_0p2" >&2; exit 2 ;;
esac

data_args=(--data_dir "${DATA_DIR}" --num_workers "${NUM_WORKERS}")
if [ "${NO_DOWNLOAD}" = "1" ]; then data_args+=(--no_download); fi
read -r -a seed_list <<< "${SEEDS}"

for seed in "${seed_list[@]}"; do
  run_dir="${OUTPUT_ROOT}/${run_name}_seed${seed}"
  "${PYTHON}" -u train.py \
    --dataset "${DATASET}" --model "${MODEL}" "${method_args[@]}" \
    --epochs 100 --batch_size 128 --lr 0.1 --momentum 0.9 --weight_decay 5e-4 \
    --optimizer sgd --scheduler cosine \
    --eps_linf 8/255 --alpha_linf 2/255 --eps_l2 0.5 --alpha_l2 0.1 \
    --random_start --seed "${seed}" --device "${DEVICE}" \
    "${data_args[@]}" --save_dir "${run_dir}"
done
