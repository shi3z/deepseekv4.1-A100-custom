#!/usr/bin/env bash
# run_64slots.sh: 64-way Concurrent Decode Streams on 4x A100 GPUs
# Context: 8,192 (8K tokens)
# Concurrency: 64 parallel decode streams (max_seqs 65 = slot 0 scratchpad + slots 1..64 decode)
set -eo pipefail

cd /mnt/ssdraid/git/deepseekv4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefill & pipeline chunking
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128
export DSV41_LAYER_COUNTS=10,10,10,10

# 8K Context Horizon
export DSV41_EP_GRAPH_TOKENS=8192
export DSV41_EP_CAND_TOKENS=8192
export DSV41_CACHE_INIT_TOKENS=8192
export DSV41_EP_PREALLOC_TOKENS=8192
export DSV41_EXACT_CACHE_GROW=1

# 64 Concurrent Decode Streams (max_seqs 65: slot 0 scratchpad + slots 1..64 concurrent decode)
export DSV41_MAX_SEQS=65

# Communication & Cache Acceleration
export DSV41_EP_COMPACT_XQ=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_PREFIX_DEDUP_MIRRORS=1
export DSV41_CED=1

# Repetition & Stability Safeguards
export DSV41_REPETITION_PENALTY=1.05
export DSV41_FREQUENCY_PENALTY=0.02
export DSV41_PRESENCE_PENALTY=0.0
export DSV41_PENALTY_WINDOW=2048
export DSV41_PROGRESSIVE_PENALTY=0.0
export DSV41_BAN_CYCLES=1
export DSV41_LOOP_DETECT=1
export DSV41_MIN_LOOP_MATCH=48
export DSV41_MIN_LOOP_CYCLE=1
export DSV41_INTERACTIVE_MAX_NEW=4096
export DSV41_LONG_PROMPT_MAX_NEW=4096

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
    --max-seq-len 8192 \
    --max-seqs 65 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
