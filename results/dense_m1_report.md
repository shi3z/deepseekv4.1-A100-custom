# Single-stream decode: dense and expert M=1 kernels — investigation report

Date: 2026-09-28. Baseline (this branch, A100 80GB **PCIe** ×4, HBM2e 1512 MHz → 1.94 TB/s peak, 8K context,
greedy): S=1 59.9 tok/s (16.68 ms/token), S=2 87.1, S=8 220.8. Measured S=1 critical path
(`results/expert_replica_report.md` §5): dense attention/hc phase 53 %, owner expert phase 18 %, P2P 10 %.

## 1. Repository history: the tiled-FP4 / fp8_tcw kernel work

`git log --all` (plus `fsck --lost-found`: one dangling commit b654ac8, unrelated) shows every kernel commit is
reachable; nothing was removed. `dev` (and `main`) branched from `origin/compression-study` at ca93eb3
(2026-09-11, "chat / serve: --ep and --ep-shards") for the engine/server work; the kernel work continued on
`compression-study` and was never merged back. Dev **does** contain 5d1b3d5 + 83c0e46 (fp8_tcw.cu, fp8_tcg.cu,
k-permuted FP8 layout, fp4_tcw.cu) — `fp8_tcw` is in production and used for 17–64 rows.

| commit (compression-study only) | files | what | recorded result | relevance to S=1 / compatibility |
|---|---|---|---|---|
| 9e901a9 FP4 experts: tiled weight layout `[N/16][K/128][16 rows][64 B]`, scales alongside, all groups on the shared-memory kernel | cuda/fp4_tc.cu, fp4_tcw.cu, cukern.py, load.py, model.py, moe_kernels.py, quant.py, mtp_run.py, tests | load-time re-layout of every expert; one-tile variant of fp4_tcw for 1-token groups | 1 token/expert 1.02 → 1.46 TB/s (microbench); S=32 K=3 528 → 633 tok/s, S=256 1769 tok/s | touches load.py/model.py/moe_kernels.py that the vision/CED work rewrote (merge conflicts certain); the 1-token gain was measured with many groups active — in the EP owner only 1–2 of 6 groups are local, where block count, not layout, is the limit (§4) |
| 4e0fd85 FP8 dense: tiled layout option (`w8.tile`, DSV41_FP8_TILED) | cuda/fp8_tc.cu (+`tiled` arg), fp8_tcg/tcw.cu, w8.py, cukern.py, tests | a warp's 8 rows × 64 B per k-step become one 512-byte range | "~5 %" | small self-contained diff; re-measured today in isolation: 37.3 → 34.5 µs (wq_b), 30.8 → 27.9 (wo_a), 36.0 → 33.1 (wo_b) = −8 % on the big GEMVs, nothing on the small ones (table §2); superseded by the dedicated M=1 kernel below |
| 38f5b4b, b55753f, 51f9792, 00f9e87 (SwiGLU 256 cols, fused MoE dispatch, NT=2 expert tiles, bf16 dense copies ≥ 64 rows, indexer top-k shrink) | fused.py, p2p.cu, cukern.py, decode.py, load.py, w8.py, fp4_tcw.cu | batched-row (≥ 32 rows) work | 148 → 20 µs dispatch at 256 rows, +20 % for 9–32-token groups, S=256 1972 tok/s | no effect at S ≤ 8 |
| 05c77fa … 9a00459 | compression/, spark/ | expert compression (VQ) for DGX Spark | — | not for this box |

Why they never reached dev: the branch split predates them and the later dev commits (Jev, CED, vision, prefix
cache) rewrote the same files; there is no commit that removes them. Strategy C ("recover fp8_tcw") has nothing to
recover: for S ≤ 16 rows the dispatch takes the `fp8_gemm_tc8/16` path (x as the mma A operand, weights read once
for all rows), and that is the right path there; `fp8_tcw` only starts at 17 rows.

## 2. The dense S=1 path, reconstructed (plain layer 5 on cuda:2, Kineto trace of one token)

Every layer runs, on its owner GPU, `hc_sub → attention2 → hc_post2 → hc_sub → gate → experts → hc_post2`
(`decode.py:block2`/`ep.py:_owner_layer`). Weight formats: FP8 e4m3 bytes k-permuted within 16-k groups
(`w8.PERM_K`), E8M0 scales per 32×32 block; activations bf16 already rounded to the fp8 grid (`_norm_quant`).
Kernel `fp8_gemm_tc8` (`cuda/fp8_tc.cu`): 4 warps/block, one warp per 8 output rows, mma.m16n8k16 with the
single x row in a 16-row tile, split-K so that ≥ 4096 warps stream (`_splits_for`), fp32 partials in global
memory, fused epilogue (the last warp per 8-column tile sums the splits, bf16 out). 77 registers, 6 blocks/SM.

| # | op | activation (bf16) | weight | output | bytes read (MB) | bytes written | launches | in-model µs | isolated µs (graph) | GB/s | ncu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | wq_a | xq [1, 5120] | [1280, 5120] e4m3 + [40, 160] E8M0 | qr [1, 1280] | 6.75 | 14 split partials 72 KB + 2.5 KB | 1 (grid 40×14) | 10.4 | 9.4 | 700 | — |
| 2 | wq_b | qr [1, 1280] (after q_norm) | [32768, 1280] + [1024, 40] | q [1, 32768] = 64 heads × 512 | 41.9 | 64 KB | 1 (1024×1) | 36.9 | 37.3 | 1126 | — |
| 3 | wkv | xq [1, 5120] | [512, 5120] + [16, 160] | [1, 512] | 2.6 | 29 KB + 1 KB | 1 (16×14) | 7.6 | 7.5 | 348 | — |
| 4 | wo_a | o [8 groups, 4096] | [8192, 4096] block-diagonal (group_cols 1024) + [256, 128] | [1, 8192] | 33.6 | 131 KB + 16 KB | 1 (256×4) | 31.3 | 30.8 | 1091 | — |
| 5 | wo_b | [1, 8192] | [5120, 8192] + [160, 256] | [1, 5120] | 41.9 | 164 KB + 10 KB | 1 (160×8) | 33.9 | 36.0 | 1166 | 38.4 µs under ncu clocks, DRAM 56.7 % of peak, 1.98 waves, 31.7 % warps active |
| | **total** | | **121 MiB** | | 126.8 | | 5 | **120.1** | 121.0 | 1050 | |

(In the expert phase the shared expert uses the same kernel: w13 [4608, 5120] 23.5 µs, w2 [5120, 2304] 15.0 µs on a
side stream; the gate is a cuBLAS fp32 gemv over a bf16 [384, 5120] weight, 9.5 µs; the head is **bf16**
[129280, 5120] = 1.26 GB, 0.77 ms per token.)

### Timeline of one plain layer (layer 5, 165 µs of attention phase + 162 µs of MoE phase; µs, main stream unless noted)

```
  0.0  hc_mix            6.7  (side stream)     hyper-connection mix, fp32 [24, 20480] weights
  1.5  fill              1.8  (pre-mix buffer)
  3.4  hc_pre2           2.4                    residual mix -> x
  5.8  norm_quant2       3.0                    RMSNorm + fp8 rounding -> xq (+ xf)
  7.9  sinkhorn          8.9  (side stream)     hc pre/post/comb coefficients (needed only at hc_post)
  8.8  fp8_gemm_tc8     10.4   wq_a
 19.2  norm_quant        6.8   q_norm + fp8 rounding
 26.0  fp8_gemm_tc8     36.9   wq_b
 62.8  rope_dev          2.6   rope on q
 65.2  fp8_gemm_tc8      7.6   wkv
 72.8  kv_write          2.8   kv_norm + rope + window ring write
 75.6  add<long>         1.5   pos + 1              } compress_len, recomputed in every layer
 77.2  floordiv<long>    2.0   // ratio             }
 79.2  sattn2_split     11.7   attention over window ring + compressed cache (top-k)
 91.5  sattn2_combine    4.0
 95.5  fp8_gemm_tc8     31.3   wo_a (block-diagonal)
126.6  fake_quant        2.2   fp8 rounding of the o-projection input
128.6  fp8_gemm_tc8     33.9   wo_b
162.4  hc_post2          2.5   residual update              <- stamp 1 (attention phase 165 µs)
166.1  hc_mix / fill / hc_pre2 / norm_quant2 / sinkhorn   (same 5 kernels again for the FFN sub-block, 12 µs)
177.6  cublas gemvx      9.5   gate [384, 5120] bf16 -> fp32
187.2  gate_topk         5.9
194.6  p2p_multicast    11.6   route packet to 3 peers
207.5  fp8_gemm_tc8     23.5   shared w13 (side stream 2)   || 218.8 fp4_gemm_tc8 56.5  own experts w13 (2 experts)
263.7  swiglu_quant      6.4   (shared)                      || 278.4 swiglu_quant  7.9
269.9  fp8_gemm_tc8     15.0   shared w2                     || 286.1 fp4_gemm_tc8 22.8  own experts w2
310.2  p2p_sum_rows      2.6 ; 314 p2p_signal 6.4 ; 320.7 p2p_wait 4.4 ; 326.5 hc_post2 3.1
```

19 kernels in the attention phase, gaps ≈ 1 µs. The five GEMVs are 120 of 165 µs; the 14 small kernels are
45 µs, of which ~16 µs sit on side streams (hc_mix, sinkhorn) and do not lengthen the critical path.

### The four heavy layers (2, 8, 14, 20: KV source + indexer; layer 2 = 446 µs)

The indexer decode path (`decode.py:indexer`) is ~60 generic torch ops per layer: two cuBLAS gemv for the
compressor (22 µs), index/cat/where/arange/copy kernels (≈ 60 µs), `gatherTopK` 43 µs, `radixSortKVInPlace`
36 µs, sgemm for the score 9 µs, six `p2p_copy_row` for the cache mirrors (10 µs) …: **≈ 280 µs per heavy layer
beyond a plain layer, 1.1 ms per token (7 %)**. The compressor adds two gemv (11 µs each) and ~10 small ops.

### Which kernels can be fused (plain layer)

* `add<long>` + `floordiv<long>` (3.5 µs): `compress_len = (pos+1)//ratio` is identical in all 40 layers → compute
  once per token (or inside `sattn2`). `fill` (1.8 µs): a zeroed pre-mix buffer that can be a static tensor.
* `hc_post2` (attention) + `hc_pre2` + `norm_quant2` (FFN): residual update immediately followed by the next
  mix/norm/quant on the same 4×5120 vector → one kernel (−2 launches, ~5 µs).
* `norm_quant` after wq_a (6.8 µs for a 1280-vector — launch-latency bound), `rope_dev` after wq_b, `kv_write`
  after wkv, `fake_quant` after wo_a: each is a tiny elementwise pass over the GEMV output; a "last block"
  epilogue in the M=1 GEMV can do them (the M=1 kernel already needs no split-K epilogue).
* wq_a and wkv consume the same xq: one launch over the concatenated 1792 rows. wq_a → wq_b and wo_a → wo_b are
  dependent chains: a persistent launch with a device-side barrier would save one launch ramp/tail each.
  Together: 5 GEMV launches → 2 (+ the attention core), 14 small kernels → ~4.

## 3. Strategy A — dedicated M=1 FP8 GEMV (`cuda/fp8_gemv_m1.cu`, written and measured)

Design: no tensor cores (at one row the mma path uses 1/16 of its tile; the problem is a pure weight stream).
Block = 8 warps; KW warps share an output row (each a K/KW slice, reduced in shared memory) so any N yields
≥ ~600 blocks; a lane streams 16-byte weight chunks with U loads in flight, decodes e4m3 → bf16 exactly (the
fp8_tc bit placement), FMA-accumulates in fp32 and applies the E8M0 block scale once per 16-k chunk; direct
bf16 output (fp32 optional), no split-K partials, no counters, no atomics. 39–57 registers, 32 B smem,
6–8 blocks/SM. Block-diagonal (wo_a) supported via `group_cols`.

Isolated, CUDA-graph timed, weights rotated past the 40 MB L2 (`dsv41/bench_gemv_m1.py`, cuda:6, error = max
abs diff vs fp32 reference / max |ref| — identical for all three kernels, 1.8–2.7e-3 = bf16 output rounding):

| shape | MiB | dev `fp8_gemm_tc8` µs (GB/s) | historical tiled µs (GB/s) | M=1 kernel best µs (GB/s), config |
|---|---|---|---|---|
| wq_a 1280×5120 | 6.3 | 9.4 (700) | 9.5 (688) | **8.3** (793) k2u2 |
| wq_b 32768×1280 | 40.0 | 37.3 (1126) | 34.5 (1217) | **33.3** (1261) k1u4 |
| wkv 512×5120 | 2.5 | 7.5 (348) | 7.3 (359) | **5.1** (512) k4u4 |
| wo_a 8192×4096 (block-diag) | 32.0 | 30.8 (1091) | 27.9 (1203) | **27.7** (1212) k1u2 |
| wo_b 5120×8192 | 40.0 | 36.0 (1166) | 33.1 (1269) | **32.3** (1301) k1u2 |
| shared w13 4608×5120 | 22.5 | 22.7 (1039) | 21.7 (1090) | **19.9** (1186) k1u2 |
| shared w2 5120×2304 | 11.3 | 14.5 (815) | 14.5 (816) | **12.7** (926) k1u2 |
| head-sized 129280×5120 (FP8) | 632 | 494 (1340) | 452 (1467) | **405** (1637) k1u2 |
| **attention GEMVs per layer** | 121 | **121.0** | 112.3 | **106.7** (−12 %) |

ncu on wo_b (profiling clocks): dev kernel 38.4 µs, DRAM 56.7 % of peak, 77 regs, 1.98 waves, warps active
31.7 %; M=1 kernel 35.0 µs, DRAM 62.3 %, 39 regs, 0.99 waves, warps active 61 %. On a 632 MiB stream the M=1
kernel reaches 1.64 TB/s (85 % of peak); on 30–40 MiB it reaches 1.2–1.3 TB/s: the remaining gap is the fixed
ramp + tail of a ~30 µs launch (≈ 6 µs), i.e. launch count, which is Strategy B's territory.

## 4. The owner expert path at S=1

`experts_tc` at B=1: one group per (token, expert) pair, `fp4_gemm_tc8` (`cuda/fp4_tc.cu`, 4 warps × 8 rows,
mma with one x row, no split-K) with grid (N/32, 6 groups); only the groups whose expert lies in this GPU's shard
do work (≈ 1.5 of 6 at S=1), so **144–288 blocks of 128 threads stream 12–24 MiB** — parallelism-starved.
Isolated (cuda:6, graph timed, 6 one-token groups, n experts local):

| local experts | w13 + SwiGLU + w2 µs | GB/s of expert bytes | w13 kernel alone µs (GB/s) |
|---|---|---|---|
| 1 | 46.8 | 402 | 29.1 (405) |
| 2 | 59.1 | 637 | 36.6 (644) |
| 3 | 71.0 | 794 | 45.8 (773) |
| 6 | 128.6 | 877 | 85.2 (830) |

Even fully populated the kernel stays under 0.9 TB/s at one token per group; in-model (layer 5, two local
experts) the two launches took 56.5 + 22.8 µs. A dedicated M=1 FP4 kernel with the FP8 kernel's structure
(8-warp blocks, KW warps per row, 16-byte chunks = 32 nibbles = one scale block, exact E2M1 → bf16 decode from
fp4_tc.cu, fp32 FMA) gives ≥ 576 blocks per expert instead of 144 and needs no tensor-core tile; expected
1.2–1.4 TB/s → 1.5 experts × 17.9 MiB ≈ 20–23 µs per layer instead of ≈ 55 (−32 µs/layer ≈ −1.3 ms/token).
Same FP4 bytes, no dequantised copies, no extra HBM traffic.

## 5. Theoretical speed-ups (token = 16.68 ms, dense phase 8.85 ms)

| dense phase × | dense ms | token ms | tok/s |
|---|---|---|---|
| 1.0 (now) | 8.85 | 16.68 | 59.9 |
| 1.5 | 5.90 | 13.73 | 72.8 |
| 2 | 4.43 | 12.26 | 81.6 |
| 3 | 2.95 | 10.78 | 92.8 |
| 4 | 2.21 | 10.04 | 99.6 |
| 5 | 1.77 | 9.60 | 104.2 |
| 5.8 (S=8 per-token efficiency) | 1.53 | 9.36 | 106.9 |

What the dense phase can physically reach on this GPU: the 121 MiB of projections at the best measured streaming
rate (1.64 TB/s) = 77 µs, plus the attention core (~16 µs) and ~15 µs of fused small kernels ≈ 110 µs per plain
layer (now 165, ×1.5); heavy layers 446 → ~200 with fused indexer kernels. Average 222 → ~120 µs, i.e. dense
phase ≈ 1.85× → token ≈ 12.6 ms ≈ 79 tok/s from the dense side alone. Adding the expert-kernel rewrite
(−1.3 ms) gives ≈ 11.3 ms ≈ 88 tok/s; 100 tok/s additionally needs the host side (1 ms: 4 graph launches + sync
+ sampling) and part of the P2P budget.

## 6. Ranking (expected payoff at S=1 vs implementation complexity)

| rank | experiment | expected saving / token | evidence | complexity |
|---|---|---|---|---|
| 1 | **M=1 FP4 expert kernel** (grouped, split-K/8-warp blocks, same bytes) | −1.2 … −1.4 ms (≈ +8 %) | isolated 0.4–0.64 TB/s today; FP8 M=1 kernel design reaches 1.2–1.6 TB/s on the same access pattern | medium: one new CUDA kernel + dispatch for 1-token groups, existing kernel kept for larger groups |
| 2 | **M=1 FP8 GEMV** (Strategy A, kernel exists) for the 5 projections + shared expert | −0.55 ms GEMVs (+3 %), shared expert −0.2 ms if it becomes critical | measured −13.5 µs/layer isolated | low: dispatch on M == 1 behind a flag |
| 3 | indexer / compressor decode chain in the 4 heavy layers (fused score + selection kernels) | −0.8 … −1.0 ms (≈ +6 %) | 280 µs of ~60 torch ops per heavy layer | medium-high: custom top-k (4K→512) + score kernel; context-length dependent |
| 4 | small-kernel fusion in plain layers (compress_len hoist, static fill, hc_post2+hc_pre2+norm_quant2, GEMV epilogues) | −0.6 … −0.9 ms (+4–5 %) | 45 µs of small kernels per layer, ~25 on the critical path | low-medium: Triton kernels already exist, mostly merging |
| 5 | Strategy B launch fusion (wq_a‖wkv concat; persistent wq_a→wq_b, wo_a→wo_b chains) | −0.4 … −0.6 ms (+3 %) | ≈ 6 µs ramp/tail per launch measured | medium: device-side grid barrier in a graph |
| 6 | FP8 head (bf16 today, 1.26 GB per token) | −0.35 ms (+2 %) | 632 MiB at 1.6 TB/s = 0.4 ms | low, but a numerics change |
| 7 | host side: argmax inside the last graph, one sync | −0.3 … −0.5 ms | 1 ms measured outside the layers | medium |
| — | Strategy C (fp8_tcw) | 0 | already on dev, not on the S ≤ 16 path | — |
| — | historical tiled FP4/FP8 layouts | ≤ −0.3 ms (FP8) / unclear (FP4, block count is the limit) | −8 % on big GEMVs; the M=1 kernel already exceeds it | high merge cost |

Next: implement rank 1 and rank 2 together behind `DSV41_FP4_M1=1` and `DSV41_FP8_M1=1` (baseline unchanged
when unset), then benchmark S = 1, 2, 8 with the identical harness (`bench_replica.py`, same prompts, context,
generation length, clocks, topology).

## 7. Implementation and measurements (rank 1 + rank 2, then rank 3 + rank 4)

All behind environment flags, baseline path untouched when unset:

| flag | what |
|---|---|
| `DSV41_FP8_M1=1` | `cukern.fp8_gemm_tc` routes single-row calls (M = 1, incl. the block-diagonal wo_a) to `cuda/fp8_gemv_m1.cu` (heuristic KW/U per shape) |
| `DSV41_FP4_M1=1` | `experts_tc` runs one-token expert groups on `cuda/fp4_gemv_m1.cu` (B = 1 steps; `DSV41_FP4_M1_MIXED=1` also splits bucketed steps between the two kernels — measured slower at S=8) |
| `DSV41_DECODE_LEAN=1` | `compress_len` computed once per token per (device, ratio) instead of per layer; static zeroed partial buffer in `hc_pre_norm_quant2` (no `fill` kernel) |
| `DSV41_INDEX_FAST=1` | `fused3.py`: indexer score in one Triton kernel (fp4-quantized q × cached keys, relu, head weights, length/candidate masks) + `cuda/topk_select.cu` (radix select + bitonic sort, one launch) for the non-candidate-source indexer layers |

Correctness: FP8 M=1 vs the tensor-core path 9.1e-4 relative (bf16 output rounding, same as between the
existing kernels); FP4 M=1 vs `fp4_gemm_tc` 1.6e-6 relative on the fp32 outputs, foreign-group zeroing exact;
fused indexer: identical index sets and selected scores vs the torch chain (B = 1 and 2, 4,097 and 65,537 keys,
with and without candidate masks), isolated 122–160 µs → 39–47 µs at 4,097 keys, 298 → 102 µs at 65,537.

First series (`results/m1_bench.jsonl`, 200 steps, same prompts/context/steps as every previous table; the
ASR server of another user holds 0.4/2.3 GiB on GPUs 0/2 and runs intermittently, which explains the ±3 %
run-to-run noise seen at S=2):

| config | S=1 tok/s (ms) | S=2 | S=8 | S=1 attn+hc µs/layer | S=1 local expert GEMMs µs/layer |
|---|---|---|---|---|---|
| baseline | 59.9 (16.69) | 87.8 (22.77) | 222.2 (36.01) | 221.8 | 73.7 |
| + FP8 M=1 | **62.0 (16.12)** | 82.3 (24.29, noise: attn+hc identical, EP phases +4 µs) | 222.8 (35.90) | **206.4** | 71.8 |
| + FP8 M=1 + FP4 M=1 | **62.6 (15.97)** | 88.7 (22.54) | 215.6 (37.10, mixed mode) | 206.1 | **68.4** |

The FP8 M=1 kernel delivers the predicted −15 µs per layer on the dense phase (−0.57 ms/token, +3.5 %). The
FP4 one-token kernel adds only −3.4 µs per layer: as the isolated numbers showed (46.8 → 38.3 µs at one local
expert), the expert phase at S=1 is bound by the fixed cost of three dependent launches per chain (~7 µs each of
ramp/tail inside a 15–25 µs kernel), not by the streaming rate (the marginal rate per extra expert is already
~1.05 TB/s). The next step for the expert phase is therefore one persistent launch per chain (w13 → SwiGLU → w2
with a device-side barrier), and the same for the shared expert, which becomes the critical path once the routed
chain shrinks (39 µs on its side stream).

## 8. Second series: all four flags (`results/m1_bench.jsonl`, same harness)

| config | S=1 tok/s (ms/step) | S=2 | S=8 | S=1 attn+hc µs/layer | S=1 layer µs |
|---|---|---|---|---|---|
| baseline (two runs) | 59.9 (16.69), 59.8 (16.73) | 87.8 (22.77) | 222.2 (36.01) | 221.8 | 363–367 |
| FP8 M=1 | 62.0 (16.12) | 82.3* / 88.7 | 222.8 (35.90) | 206.4 | 351 |
| FP8 M=1 + FP4 M=1 | 62.6 (15.97) | 88.7 (22.54) | 215.6 (mixed mode; off by default at B > 1) | 206.1 | 344 |
| FP8 M=1 + FP4 M=1 + lean | 63.5 (15.75) | – | – | 203.1 | 341 |
| **all four (+ fused indexer)** | **67.5 (14.81)**, repeat 66.1 (15.12) | **92.4 (21.65)** | **238.6 (33.53)** | **183.4** | **321** |

\* the 82.3 point is the interference run (attention phase identical to the baseline; EP phases +4 µs); the
repeat with the same kernels gave 88.7.

S=1: 16.7 → 14.8 ms per token, **+12.7 %** (67.5 tok/s; 66.1 on the repeat, i.e. +10.5 % — the ±1 tok/s
spread is the box's noise). Attribution per token: FP8 M=1 −0.61 ms, FP4 M=1 −0.15, lean −0.22, fused indexer
−0.94. S=2 +5 %, S=8 +7 % (the fused indexer and the lean changes apply at every S; the FP8 M=1 kernel only at
S=1). Deterministic-kernel comparison (`DSV41_DETERMINISTIC=1`, 160 greedy tokens, per-layer residual capture):
baseline vs its own repeat 15/160 tokens identical, first divergent layer 20; all flags vs baseline 42/160, first
divergent layer 1 (0.0044 max |Δh| after the first FP8 M=1 GEMV — one bf16 ulp of a different fp32 accumulation
order), reaching 3.0 at the last layer vs 4.25 for the baseline against itself; first-step logits max |Δ| 1.69 vs
1.18, argmax identical. The new paths are inside the runtime's run-to-run envelope.

Remaining gap to 100 tok/s (10 ms): 4.8 ms. Measured composition of the 14.8 ms token now: dense phase
183 µs × 40 = 7.3 ms (GEMVs 107 µs at 1.2–1.3 TB/s, small kernels ~40, heavy layers ~200 µs extra), expert phase
75 µs × 40 = 3.0 ms (two 3-launch chains), P2P 1.3 ms, head 0.77, host ~1.0, gate/misc 1.4. Next in payoff
order: persistent fused expert chains (routed and shared: −1.5 to −1.8 ms), hc_post2 + hc_pre2 + norm fusion and
GEMV epilogues (−0.6), FP8 head (−0.35), GEMV launch fusion (−0.5), host side (−0.4).
