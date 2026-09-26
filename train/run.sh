#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash train/run.sh {sft|opd|rl} [config.yaml] [extra arguments...]" >&2
  exit 2
fi

STAGE="$1"
shift

case "${STAGE}" in
  sft)
    if [[ $# -gt 0 ]]; then
      CONFIG="$1"
      shift
    else
      CONFIG="train/sft/config.yaml"
    fi
    exec python train/sft/train.py --config "${CONFIG}" "$@"
    ;;
  opd)
    if [[ $# -gt 0 ]]; then
      CONFIG="$1"
      shift
    else
      CONFIG="train/opd/config.yaml"
    fi
    exec torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
      train/opd/train.py --config "${CONFIG}" "$@"
    ;;
  rl)
    if [[ $# -gt 0 ]]; then
      CONFIG="$1"
      shift
    else
      CONFIG="train/rl/config.yaml"
    fi
    exec python train/rl/train.py --config "${CONFIG}" "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}. Expected sft, opd, or rl." >&2
    exit 2
    ;;
esac
