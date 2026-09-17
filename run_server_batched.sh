#!/usr/bin/env bash
set -e

cd /mnt/ssdraid/git/deepseekv4.1

export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=2048
export DSV41_HC_PREFILL_CHUNK=2048
export DSV41_EP_COMPACT_XQ=1
export DSV41_EP_GRAPH_TOKENS=1048576
export DSV41_EP_CAND_TOKENS=1048576
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1
export DSV41_MAX_SEQS=5

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

exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt /mnt/ssd/models/DeepSeek-V4.1-Flash \
    --devices 2,3,0,1 \
    --ep \
    --ep-shards 96,96,96,96 \
    --max-seq-len 1048576 \
    --max-seqs 5 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0

