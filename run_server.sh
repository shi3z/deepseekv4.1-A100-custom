#!/usr/bin/env bash
set -e

cd /mnt/ssdraid/git/deepseekv4.1

export DSV41_MOE_PREFILL_CHUNK=256
export DSV41_ENGRAM_PREFILL_CHUNK=256
export DSV41_HC_PREFILL_CHUNK=2048
export DSV41_EP_COMPACT_XQ=1
export DSV41_EP_GRAPH_TOKENS=131072
export DSV41_EP_CAND_TOKENS=1048576
export DSV41_CACHE_INIT_TOKENS=1048576
export DSV41_EP_PREALLOC_TOKENS=1048576
export DSV41_EXACT_CACHE_GROW=1
export DSV41_MAX_SEQS=9

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
    --devices 2,0,1,4,5,6,7,3 \
    --ep \
    --ep-shards 48,48,48,48,48,48,48,48 \
    --max-seq-len 1048576 \
    --max-seqs 9 \
    --host 127.0.0.1 \
    --port 8000 \
    --mtp 0
