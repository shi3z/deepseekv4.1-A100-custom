#!/usr/bin/env bash
# Profile 2: 1,000,000 Context Robust Milestone (Max context, CED, strict allocation, OOM-proof)
set -e

cd /mnt/ssdraid/git/deepseekv4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Chunk pipeline parallelism
export DSV41_HC_PREFILL_CHUNK=2048
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_SPARSE_ATTN_CHUNK=128

# Causal Encoder-Decoder (CED): saves 47% prefill compute
export DSV41_CED=1
export DSV41_CED_TAIL_WINDOW=128

# Memory safety guards for 1M tokens
export DSV41_EXACT_CACHE_GROW=1
export DSV41_PREFIX_DEDUP_MIRRORS=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_EP_COMPACT_XQ=1
export DSV41_LOOP_DETECT=1

# Strict slot allocation: 1 decode slot to minimize KV batch dimension to 2
export DSV41_MAX_SEQS=2

DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated"
if [ ! -d "$DEFAULT_CKPT" ]; then
    DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash"
fi
CKPT="${DSV41_CKPT:-$DEFAULT_CKPT}"

exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt "$CKPT" \
    --devices 2,3,0,1 \
    --ep \
    --ep-shards 92,95,99,98 \
    --max-seq-len 1048576 \
    --max-seqs 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
