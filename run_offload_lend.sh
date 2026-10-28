#!/usr/bin/env bash
# run_offload_lend.sh: "lend mode" LLM — DeepSeek-V4.1-Flash on ONE GPU with the FP4 experts in pinned host RAM
# (--offload-experts cpu), used by scripts/gpu_lender.py while the other GPUs are lent to the H3 fleet.
# Slow (host->GPU expert traffic per token) but keeps the API up. GPU from LEND_LLM_GPU (default 2).
set -eo pipefail
cd /mnt/ssdraid/git/deepseekv4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1
export DSV41_MAX_SEQS=${LEND_MAX_SEQS:-2}
export DSV41_CED=0
export DSV41_REPETITION_PENALTY=1.05
export DSV41_FREQUENCY_PENALTY=0.02
export DSV41_PRESENCE_PENALTY=0.0
export DSV41_PENALTY_WINDOW=2048
export DSV41_PROGRESSIVE_PENALTY=0.0
export DSV41_BAN_CYCLES=1
export DSV41_LOOP_DETECT=1
export DSV41_MIN_LOOP_MATCH=48
export DSV41_MIN_LOOP_CYCLE=1
export DSV41_INTERACTIVE_MAX_NEW=${LEND_MAX_NEW:-8192}
export DSV41_LONG_PROMPT_MAX_NEW=${LEND_MAX_NEW:-8192}
export DSV41_PREFIX_CACHE_DIR=/dev/shm/dsv41-prefix-cache-lend
export DSV41_PREFIX_CACHE_ENTRIES=4
export DSV41_PREFIX_CACHE_GB=8
export DSV41_PREFIX_TMPFS_ENTRIES=4
export DSV41_PREFIX_TMPFS_GB=8
DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated"
[ -d "$DEFAULT_CKPT" ] || DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-1017"
CKPT="${DSV41_CKPT:-$DEFAULT_CKPT}"
exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt "$CKPT" \
    --devices "${LEND_LLM_GPU:-2}" \
    --offload-experts cpu \
    --hot-experts "${LEND_HOT_EXPERTS:-32}" \
    --max-seq-len "${LEND_MAX_SEQ_LEN:-65536}" \
    --max-seqs "${LEND_MAX_SEQS:-2}" \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
