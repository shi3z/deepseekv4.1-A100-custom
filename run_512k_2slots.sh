#!/usr/bin/env bash
# run_512k_2slots.sh: 512K Context with 2 Concurrent Decode Streams on 4x A100 GPUs
# Context: 524,288 (512K tokens)
# Concurrency: 2 parallel decode streams (max_seqs 3 = slot 0 prefill scratchpad + slots 1..2 decode); 512K x 5 rows OOMed at a 499K prompt (2026-10-28)
# Aggregate Throughput: Scales to 180-240+ tok/s under parallel multi-agent load
set -eo pipefail

cd /mnt/ssdraid/git/deepseekv4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefill & pipeline chunking
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128
export DSV41_LAYER_COUNTS=10,10,10,10

# 512K Context Horizon
export DSV41_EP_GRAPH_TOKENS=524288
export DSV41_EP_CAND_TOKENS=524288
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1

# 2 Concurrent Decode Streams (max_seqs 3): compressed KV 4.7 GiB per device at 512K (7.8 GiB with 5 rows did not fit next to the prefill temporaries)
export DSV41_MAX_SEQS=3

# Communication & Cache Acceleration
export DSV41_EP_COMPACT_XQ=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_PREFIX_DEDUP_MIRRORS=1
export DSV41_CED=0  # exact prefill for everything (user decision 2026-10-05); =1 is the approximate CED tail for text prompts, images are always exact (results/image_crosstalk_report.md)

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
export DSV41_INTERACTIVE_MAX_NEW=65536
export DSV41_LONG_PROMPT_MAX_NEW=65536

# Prefix Cache (Host RAM & tmpfs)
export DSV41_PREFIX_CACHE_DIR=/dev/shm/dsv41-prefix-cache
export DSV41_PREFIX_CACHE_ENTRIES=16
export DSV41_PREFIX_CACHE_GB=32
export DSV41_PREFIX_TMPFS_ENTRIES=16
export DSV41_PREFIX_TMPFS_GB=32
export DSV41_PREFIX_BLOCK_REPLAY=1
export DSV41_PREFIX_BLOCK_SIZE=512
export DSV41_PREFIX_BLOCK_MIN=16
export DSV41_PREFIX_ANCHOR_STRIDE=1024
export DSV41_PREFIX_ANCHOR_MAX=2

DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated"
if [ ! -d "$DEFAULT_CKPT" ]; then
    DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-1017"
fi
CKPT="${DSV41_CKPT:-$DEFAULT_CKPT}"

exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt "$CKPT" \
    --devices 2,3,0,1 \
    --ep \
    --ep-shards 92,95,99,98 \
    --max-seq-len 524288 \
    --max-seqs 3 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
