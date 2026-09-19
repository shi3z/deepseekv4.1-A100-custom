#!/usr/bin/env bash
# run_mtp_4gpu.sh: DeepSeek-V4.1 with MTP (DSpark) Speculative Decoding on 4x A100 GPUs
# Achieves ~85-93 tok/s single-stream decode throughput (1.35x - 1.45x speedup over standard decode)
set -eo pipefail

cd /mnt/ssdraid/git/deepseekv4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Bound context length to 64K to ensure sufficient VRAM headroom for DSpark on the last GPU
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1

# Concurrency: 1 speculative decode stream (slot 0: prefill scratchpad + slot 1: speculative decode)
export DSV41_MAX_SEQS=2

# Communication & cache speedups
export DSV41_EP_COMPACT_XQ=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_PREFIX_DEDUP_MIRRORS=1
export DSV41_CED=1
export DSV41_LOOP_DETECT=1
export DSV41_MIN_LOOP_MATCH=48
export DSV41_MIN_LOOP_CYCLE=1
export DSV41_REPETITION_PENALTY=1.05
export DSV41_FREQUENCY_PENALTY=0.02
export DSV41_PRESENCE_PENALTY=0.0
export DSV41_PENALTY_WINDOW=2048

# Unload threshold for long prompts (fallback to plain decode if prompt exceeds 64K)
export DSV41_MTP_LONG_PROMPT_LIMIT=65536
export DSV41_MTP_UNLOAD_ON_LONG=1

DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated"
if [ ! -d "$DEFAULT_CKPT" ]; then
    DEFAULT_CKPT="/mnt/ssd/models/DeepSeek-V4.1-Flash"
fi
CKPT="${DSV41_CKPT:-$DEFAULT_CKPT}"

# ARCHITECTURAL KEY FOR 4-GPU MTP:
# In a 4-GPU setup (--devices 2,3,0,1), DSpark lives entirely on the last GPU (cuda:1).
# We rebalance expert shards to 100,100,100,84 (giving cuda:1 only 84 experts instead of 96+),
# freeing ~15.4 GiB of VRAM on cuda:1. This allows the 3 DSpark blocks (~13.3 GiB) and LM head
# to fit comfortably alongside the backbone layers without OOM.
exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt "$CKPT" \
    --devices 2,3,0,1 \
    --ep \
    --ep-shards 100,100,100,84 \
    --max-seq-len 65536 \
    --max-seqs 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 5
