#!/usr/bin/env bash
# run_1m_6slots.sh: 1M context on 4x A100, with per-slot capacities so the big slot costs nothing for short sessions
# Context: 1,048,576 (1M tokens), slot capacities "1m,64k,64k,64k,512k" (DSV41_SLOT_CAPS):
#   slot 0 = the prefill scratchpad, which must be the largest, so it bounds the longest prompt (1M)
#   slot 4 = 512K, slots 1-3 = 64K each; the largest decode slot is listed LAST on purpose: the flat cache buffer
#   holds (first unit of the last slot + the largest slot's units), so putting 512K first would waste 35 % of it
#   instead of 22 % (measured with dsv41/slots.py: 13.44 GiB against 11.25).
# Compressed cache: 11.25 GiB over the four GPUs, 2.81 per GPU -- LESS than the 512K x 5 profile's 12.50 / 3.13,
# because the capacity sum (1,792K token-rows) is smaller than 5 x 512K (2,560K).  Tables under a slot layout are
# allocated once at full size and never grow, so the old+new growth peak that wedged the server on 2026-11-04
# cannot happen here at all.
# Timing to expect: a 1M cold prefill is ~35-40 min with the CED tail (256K measured at 451 s), and decode at 1M
# runs at 5-6 tok/s because every token reads the whole compressed cache (421K measured at 12.5 tok/s).  Long
# sessions are only practical through the prefix cache, which restored 419K in 12.8 s.
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
# Both must match max_seq_len at 1M.  EP_CAND_TOKENS sizes the candidate mask the decode indexer writes, so a
# smaller value is not a memory saving but a shape error ("size of tensor a (524304) must match b (1048592)",
# 2026-11-04); the buffer is one bool per token, 5 MiB at 1M.  EP_GRAPH_TOKENS below max_seq_len makes ep.py clone
# the RoPE tables per block instead of sharing the model's (ep.py:599), which costs more than it saves.  What made
# the earlier 1M startups die was neither: it was the decode indexer building [B, 1, H, T] in fp32 plus an fp32
# copy of the key cache, 3,200 MiB per indexer layer at T=1M, now 320 MiB in bf16 (DSV41_INDEX_BF16, decode.py).
export DSV41_EP_GRAPH_TOKENS=1048576
export DSV41_EP_CAND_TOKENS=1048576
# 2026-11-04: allocate the full 512K compressed cache at startup (12.50 GiB over all four GPUs, 3.12 of them on
# GPU 2) instead of growing into it.  Growing needs the old and the new table side by side, so a 499K prompt asked
# for 2.38 GiB on a GPU where another tenant's ASR server had taken 2.27 GiB, failed, and left the partly grown
# tables allocated: every request after that died with "1.31 MiB is free" until a restart.  Reserved up front, the
# transient peak never happens and no co-tenant can take the space first.
export DSV41_CACHE_INIT_TOKENS=1048576   # ignored under DSV41_SLOT_CAPS (those tables are allocated at full size)
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1

# 4 decode slots of mixed capacity (max_seqs 5: slot 0 scratchpad + slots 1..4 decode); the 512K slot is last so
# the flat cache buffer wastes the least (see the header).  max_seqs stays at 5, the value this machine is known to
# fit: at 6 the graph-capture buffers grow with the batch dimension and capture died on GPU 0 wanting 2.50 GiB with
# 1.40 GiB free (2026-11-04), even with the graph shapes clamped to 512K.
export DSV41_MAX_SEQS=5
export DSV41_SLOT_CAPS="1m,64k,64k,64k,512k"
# consumer-local compressed-cache placement (2026-10-28): each owner's KV/index cache only on the GPUs that read it
# (3.1 GiB/GPU at 512K x 5 rows instead of 7.8 with full mirrors) -- see load.consumer_cache_devices
export DSV41_CACHE_PLACEMENT=local
# long prefills run in segments; between segments the active slots decode (time budget) and one small waiting
# request may be admitted (slot 0 saved/restored around it) -- engine._interleave_step
export DSV41_PREFILL_SEGMENT_TOKENS=8192
export DSV41_INTERLEAVE_DECODE_MS=1500
export DSV41_INTERLEAVE_ADMIT=1
export DSV41_INTERLEAVE_MAX_NEW_TOKENS=16384

# Communication & Cache Acceleration
export DSV41_EP_COMPACT_XQ=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_PREFIX_DEDUP_MIRRORS=1
# CED tail on for text, exact for images.  The model code decides per request: `use_ced` requires `images is None`
# and >= DSV41_CED_MIN_TOKENS (2048), so the image cross-talk of 2026-10-04 cannot recur, and that gate was verified
# end to end on 2026-11-04 (a 2,296-token image request read the word in its image and ran at 436 tok/s, the
# exact-path rate from the audit's table; CED would have been 710-795).  Text recall was the open question the audit
# never measured; scripts/longctx_quality.py answers it: 15/15 needles at 32K, 128K and 256K across depths 0.1 to
# 0.9, identical to exact prefill, while the cold prefill falls 42-47 % (128K 357 -> 207 s, 256K 851 -> 451 s).
# Set DSV41_CED=0 to go back to exact prefill for everything.
export DSV41_CED=${DSV41_CED:-1}

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
# 2026-10-30: 64 entries (16 were all 7 MB entries of 700-token background requests, 0.14 GiB of the 32 GB budget,
# and every 150K+ snapshot had been evicted); prompts under MIN_STORE tokens are not stored; entries of
# PROTECT tokens or more are evicted last (engine._store_prefix_cache_entry, _prefix_tmpfs eviction)
export DSV41_PREFIX_CACHE_ENTRIES=64
export DSV41_PREFIX_CACHE_GB=32
export DSV41_PREFIX_TMPFS_ENTRIES=64
export DSV41_PREFIX_TMPFS_GB=32
export DSV41_PREFIX_MIN_STORE_TOKENS=4096
export DSV41_PREFIX_PROTECT_TOKENS=65536
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

# Expert placement, rebalanced 2026-11-04 from the /api/memory breakdown.  --ep-shards maps positionally to
# --devices, so this is cuda:2=92, cuda:3=100, cuda:0=97, cuda:1=95.  Before (92,95,99,98) the free VRAM was
# 3.10 / 5.64 / 1.53 / 1.40 GiB: the two GPUs with the most experts had the least room, and CUDA-graph capture for a
# 1M profile died there wanting 2.0-2.5 GiB.  One expert shard costs about 0.6 GiB (measured: cuda:0 had 4 more
# shards than cuda:3 and 2.65 GiB more weights), so moving 2 off cuda:0 and 3 off cuda:1 onto cuda:3 should leave
# roughly 2.6-3.2 GiB free everywhere.
exec /home/shi3z/.local/bin/python -u -m dsv41.serve \
    --ckpt "$CKPT" \
    --devices 2,3,0,1 \
    --ep \
    --ep-shards 92,100,97,95 \
    --max-seq-len 1048576 \
    --max-seqs 5 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0
