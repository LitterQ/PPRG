#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT}"

: "${PYTHON:=python}"
: "${CHECKPOINT:?Set CHECKPOINT to a release-compatible best.pt or last.pt}"
: "${DATASET:=cifar10}"
: "${MODEL:=}"
: "${DATA_DIR:=data}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=2}"
: "${DEVICE:=auto}"
: "${NO_DOWNLOAD:=0}"
: "${OUTPUT:=$(dirname -- "${CHECKPOINT}")/eval.json}"

model_args=()
if [ -n "${MODEL}" ]; then model_args=(--model "${MODEL}"); fi
data_args=(--data_dir "${DATA_DIR}" --num_workers "${NUM_WORKERS}")
if [ "${NO_DOWNLOAD}" = "1" ]; then data_args+=(--no_download); fi

"${PYTHON}" -u eval.py \
  --checkpoint "${CHECKPOINT}" --dataset "${DATASET}" "${model_args[@]}" \
  --eval_attacks clean fgsm_linf pgd_linf_20 pgd_linf_50 cw_linf_20 cw_linf_50 \
    autoattack_linf pgd_l2_20 pgd_l2_50 cw_l2_20 cw_l2_50 autoattack_l2 \
  --eps_linf 8/255 --alpha_linf 2/255 --eps_l2 0.5 --alpha_l2 0.1 \
  --batch_size "${BATCH_SIZE}" --device "${DEVICE}" "${data_args[@]}" \
  --save_json "${OUTPUT}"
