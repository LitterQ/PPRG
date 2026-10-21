#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${CHECKPOINT:?Set CHECKPOINT to the checkpoint to evaluate}"
: "${DATASET:=cifar10}"
: "${MODEL:=}"
: "${DATA_DIR:=data}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=2}"
: "${DEVICE:=auto}"
: "${NO_DOWNLOAD:=0}"
: "${OUTPUT_DIR:=$(dirname -- "${CHECKPOINT}")/epsilon_sweep_eval}"

mkdir -p "${OUTPUT_DIR}"
model_args=()
if [ -n "${MODEL}" ]; then model_args=(--model "${MODEL}"); fi
data_args=(--data_dir "${DATA_DIR}" --num_workers "${NUM_WORKERS}")
if [ "${NO_DOWNLOAD}" = "1" ]; then data_args+=(--no_download); fi

linf_eps=(4/255 8/255 12/255 16/255)
linf_alpha=(1/255 2/255 3/255 4/255)
linf_label=(eps4 eps8 eps12 eps16)
for idx in "${!linf_eps[@]}"; do
  "${PYTHON}" -u eval.py --checkpoint "${CHECKPOINT}" --dataset "${DATASET}" "${model_args[@]}" \
    --eval_attacks fgsm_linf pgd_linf_20 pgd_linf_50 cw_linf_20 cw_linf_50 \
    --eps_linf "${linf_eps[$idx]}" --alpha_linf "${linf_alpha[$idx]}" \
    --eps_l2 0.5 --alpha_l2 0.1 --batch_size "${BATCH_SIZE}" --device "${DEVICE}" \
    "${data_args[@]}" --save_json "${OUTPUT_DIR}/linf_${linf_label[$idx]}.json"
done

l2_eps=(0.25 0.5 0.75 1.0)
l2_alpha=(0.05 0.1 0.15 0.2)
l2_label=(eps0p25 eps0p5 eps0p75 eps1p0)
for idx in "${!l2_eps[@]}"; do
  "${PYTHON}" -u eval.py --checkpoint "${CHECKPOINT}" --dataset "${DATASET}" "${model_args[@]}" \
    --eval_attacks pgd_l2_20 pgd_l2_50 cw_l2_20 cw_l2_50 \
    --eps_linf 8/255 --alpha_linf 2/255 --eps_l2 "${l2_eps[$idx]}" \
    --alpha_l2 "${l2_alpha[$idx]}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" \
    "${data_args[@]}" --save_json "${OUTPUT_DIR}/l2_${l2_label[$idx]}.json"
done
