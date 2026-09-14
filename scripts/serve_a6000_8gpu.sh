#!/usr/bin/env bash
# Example: DeepSeek-V4.1-Flash on 8× Ampere sm_86 GPUs (e.g. A6000 48GB), PCIe, no NVLink.
set -euo pipefail
CKPT="${CKPT:-/mnt/ssd/models/DeepSeek-V4.1-Flash}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8000}"
ARCH="${ARCH:-sm_86}"
export DSV41_EP_RELAY="${DSV41_EP_RELAY:-0}"

MODE="${1:-pipeline}"  # pipeline | ep
if [[ "$MODE" == "ep" ]]; then
  exec python -m dsv41.serve --ckpt "$CKPT" --devices "$DEVICES" --ep --port "$PORT" --cuda-arch "$ARCH"
else
  exec python -m dsv41.serve --ckpt "$CKPT" --devices "$DEVICES" --port "$PORT" --cuda-arch "$ARCH"
fi
