#!/usr/bin/env bash

# Sourced by paper-recipe launchers. Every caller must set -euo pipefail first.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${TASK_DIR}/.." && pwd -P)"

paper_require_task_path() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  case "${resolved}" in
    "${TASK_DIR}"|"${TASK_DIR}"/*) ;;
    *) echo "Path escapes vit_12_runs_task: ${resolved}" >&2; return 2 ;;
  esac
}

paper_require_project_path() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  case "${resolved}" in
    "${PROJECT_ROOT}"|"${PROJECT_ROOT}"/*) ;;
    *) echo "Path escapes project root: ${resolved}" >&2; return 2 ;;
  esac
}

paper_print_command() {
  printf 'resolved_command:'
  printf ' %q' "$@"
  printf '\n'
}

paper_init_environment() {
  cd "${PROJECT_ROOT}"
  : "${PYTHON:=python}"
  : "${DATA_DIR:=${PROJECT_ROOT}/data}"
  : "${NUM_WORKERS:=2}"
  : "${PRETRAINED_CHECKPOINT:=${TASK_DIR}/pretrained/deit_tiny_patch16_224_fb_in1k.pt}"
  : "${EXPECTED_PRETRAINED_SHA256:=a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97}"
  : "${TIMM_WHEEL:=}"

  if ! command -v "${PYTHON}" >/dev/null 2>&1 && [ ! -x "${PYTHON}" ]; then
    echo "Python is not executable or on PATH: ${PYTHON}" >&2; return 2
  fi
  if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
    echo "Missing pretrained checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    return 2
  fi
  if [ -n "${TIMM_WHEEL}" ] && [ ! -f "${TIMM_WHEEL}" ]; then
    echo "TIMM_WHEEL does not exist: ${TIMM_WHEEL}" >&2
    return 2
  fi
  local actual_sha
  actual_sha="$(sha256sum -- "${PRETRAINED_CHECKPOINT}" | awk '{print $1}')"
  if [ "${actual_sha,,}" != "${EXPECTED_PRETRAINED_SHA256,,}" ]; then
    echo "Pretrained SHA256 mismatch: expected ${EXPECTED_PRETRAINED_SHA256}, got ${actual_sha}" >&2
    return 2
  fi

  export TORCH_HOME="${TASK_DIR}/.cache/torch"
  export HF_HOME="${TASK_DIR}/.cache/huggingface"
  export XDG_CACHE_HOME="${TASK_DIR}/.cache/xdg"
  export CUDA_CACHE_PATH="${TASK_DIR}/.cache/cuda"
  export MPLCONFIGDIR="${TASK_DIR}/.cache/matplotlib"
  export TMPDIR="${TASK_DIR}/tmp"
  export PYTHONPYCACHEPREFIX="${TASK_DIR}/.cache/pycache"
  if [ -n "${TIMM_WHEEL}" ]; then
    export PYTHONPATH="${TIMM_WHEEL}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  else
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  fi
  export PYTHONUNBUFFERED=1
  mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${CUDA_CACHE_PATH}" \
    "${MPLCONFIGDIR}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

  "${PYTHON}" -c 'import sys,torch,torchvision,timm; assert timm.__version__ == "1.0.15"; print({"python":sys.executable,"torch":torch.__version__,"torchvision":torchvision.__version__,"timm":timm.__version__,"cuda":torch.cuda.is_available()})'
}
