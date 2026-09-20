#!/usr/bin/env bash
# run_256k.sh: 256K Context Horizon & 128K Output Generation on 4x A100 GPUs
# Context: 262,144 (256K tokens)
# Max Output: 131,072 (128K generation tokens)
# Concurrency: 4 parallel decode streams (max_seqs 5 = slot 0 scratchpad + slots 1..4 decode)
set -eo pipefail

cd /mnt/ssdraid/git/deepseekv4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefill & pipeline chunking
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128
export DSV41_LAYER_COUNTS=10,10,10,10

# 256K Context Horizon
export DSV41_EP_GRAPH_TOKENS=32768
export DSV41_EP_CAND_TOKENS=262144
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1
export DSV41_CACHE_GROW_CHUNK=4096

# 4 Concurrent Decode Streams (max_seqs 5: slot 0 scratchpad + slots 1..4 decode)
export DSV41_MAX_SEQS=5

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
export DSV41_LOOP_DETECT=0

# 128K Output Generation Limit
export DSV41_INTERACTIVE_MAX_NEW=131072
export DSV41_LONG_PROMPT_MAX_NEW=131072

export DSV41_MODEL_NAME="deepseek-v4.1-flash-abliterated"
export DSV41_VISION_DEVICE="cuda:4"

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
    --max-seq-len 262144 \
    --max-seqs 5 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
