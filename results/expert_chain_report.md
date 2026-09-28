# Persistent fused expert chains (S=1) — experiment report

Date: 2026-09-28. Starting point: the all-flags S=1 path of `results/dense_m1_report.md` (FP8 M=1 GEMV, FP4
one-token expert kernel, lean bookkeeping, fused indexer): 67.5 tok/s, 14.82 ms/token; expert phase
(stamp 2 → 3, own experts ∥ shared) 72.9 µs per layer, local expert GEMMs (2 → 10) 69.3 µs.

## 1. The expert chain as executed today (S=1, one layer, owner GPU; from `ep.py:_owner_layer`, `decode.py:experts_tc`, `_shared_expert`, and the Kineto trace)

All kernels below are captured in the per-GPU CUDA graphs; no host synchronization anywhere in the chain. Two
streams: the routed chain on the main stream, the shared expert on side stream 2 (forked after the multicast,
joined before the partial wait). Inputs: `xqp` = the FFN-input activation, RMS-normed, fp8-rounded, bf16, in the
8-k permuted order (for the FP4 kernels) and `xq` the same in natural order (for the FP8 shared expert); `eid`
int32 [6] and `wt` fp32 [6] from the gate.

| # | stream | kernel (S=1, all flags) | launches / layer | in | weights | out | µs (isolated, graph) | depends on | intermediate traffic | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | main | `fp4_gemv_m1` (w13, groups = the 6 (token, expert) pairs; the 1–2 local ones do work) | 1 | xqp bf16 [1, 5120] | FP4 [n_shard, 4608, 2560 B] + E8M0 [n_shard, 4608, 160] | gu fp32 [6, 4608] (foreign rows untouched) | ≈ 18 (1 local expert) | gate | writes 110 KB | 576 blocks per local expert |
| R2 | main | `swiglu_quant` (PERMUTE, weight per pair) | 1 | gu | – | hqp bf16 [6, 2304] (8-k permuted, fp8-rounded) | 3 | R1 | reads 110 KB, writes 28 KB | 1-warp programs, 6 × 72 |
| R3 | main | `fp4_gemv_m1` (w2, zero_out for foreign groups) | 1 | hqp | FP4 [n_shard, 5120, 1152 B] + [n_shard, 5120, 72] | y fp32 [6, 5120] | ≈ 13 | R2 | reads 28 KB, writes 123 KB | 640 blocks per local expert |
| R4 | main | `p2p_sum_rows` | 1 | y | – | part fp32 [5120] (into `part_in[own row]`) | 2.6 | R3 | reads 123 KB, writes 20 KB | |
| S1 | side 2 | `fp8_gemv_m1` (shared w13) | 1 | xq bf16 [1, 5120] | FP8 [4608, 5120] + [144, 160] | gu_s bf16 [1, 4608] | 19.9 | gate | 9 KB | |
| S2 | side 2 | `swiglu_quant` (no weight, natural order) | 1 | gu_s | – | hs bf16 [1, 2304] | 3 | S1 | | |
| S3 | side 2 | `fp8_gemv_m1` (shared w2) | 1 | hs | FP8 [5120, 2304] + [160, 72] | ys bf16 [1, 5120] | 12.7 | S2 | | |
| – | main | stream join, `p2p_signal`, `p2p_wait`, `hc_post2_` (sums the 4 partial rows + ys into the residual) | | | | | | R4, S3, peers | | |

Peers run R1–R3 and write their partial straight into the owner's `part_in` row (`p2p_sum_rows` with a remote
destination), i.e. 4 launches per peer per layer. Owner critical path = max(R1+R2+R3+R4 ≈ 37 µs of kernels +
fixed costs, S1+S2+S3 ≈ 36 µs) ≈ 69 µs measured (stamps 2 → 10 = 69.3 µs, 2 → 3 = 72.9 µs); the peers' chain
(≈ 40 µs) is hidden behind it, the owner then waits ≈ 20 µs for the slowest peer's flag.

Per-token totals at S=1: 7 launches per layer on the owner + 4 on each of 3 peers = 19 expert launches per
layer, 760 per token; the routed chain's four dependent launches are on the critical path.

## 2. Designs

**A — fused per-expert chain.** One CTA group computes w13 → SwiGLU → w2 for one expert with the intermediates on
chip. At one token, w2's every output needs the complete 2304-vector h, which needs all 4608 w13 outputs, i.e.
all 12 MiB of w13 read first: a single CTA cannot stream 12 MiB in useful time (one SM sustains ~20 GB/s here,
i.e. 600 µs), so the CTA group must be the whole GPU and the "fusion point" is a grid-wide dependency. A is
therefore only realizable as a persistent grid with a barrier between the two projections — which is design B.

**B — persistent multi-expert scheduler.** One launch, grid sized from the occupancy API (4 blocks × 108 SMs
= 432 resident blocks, 256 threads, 64 registers, 32.5 KB static shared memory for the 7 possible SwiGLU vectors),
work items = 8-row groups of (expert, projection); phase 1 (w13 of every local expert and the shared expert)
→ grid barrier (sense-reversing counter, self-resetting, graph-replayable) → every block computes the SwiGLU +
fp8 rounding of every slot into shared memory → phase 2 (w2 rows; a warp owns an output row and sums over all
local experts in registers, so the routed partial needs no atomics and is deterministic). Two schedulers were
measured: a dynamic device counter (Design B as specified: `(expert, rows)` claimed by persistent CTAs) and
static assignment. With ~600 items per expert and 432 blocks, the ~1000 serialized claim atomics per phase cost
more than the imbalance they remove (109 vs 85 µs at two experts), so the shipped variant assigns statically.

**B' — two hardware-scheduled launches (implemented as `DSV41_PERSISTENT_EXPERT=1`).** Same kernels and items,
but phase 1 and phase 2 are separate launches with one block per item: the grid barrier, the residency
constraint, the redundant per-block SwiGLU and the static imbalance disappear; the SwiGLU of a slot is done by
the last block that finishes that slot (atomic finished-item counter, self-resetting) into an L2-resident
buffer, and phase 2 reads it from L2. 7 launches → 2 on the owner, 4 → 2 on each peer.

Routed and shared expert in one implementation: the same 8-row tensor-core fragment routine is instantiated for
FP4 (E2M1 nibbles, per-32 E8M0 scale folded as a bf16 factor 2^(s−1)) and FP8 (e4m3 bytes in the PERM_K order,
scale 2^(s−7)); a block executes one item type at a time, so there is no intra-warp divergence; the FP8 items
carry 2× the bytes of FP4 items (the persistent variant balances them by bytes).

Arithmetic mattered: the first version used per-element fp32 FMAs (one 16-byte chunk → 32 products per lane);
Nsight showed the one-token FP4 kernels of the previous step running at 57–61 % SM throughput and only 22–32 %
of DRAM peak, i.e. issue-bound on the decode + FMA work, not memory-bound. Switching the fused chain to the
mma.m16n8k16 fragment layout of `fp4_tc.cu`/`fp8_tc.cu` (one x row in the 16-row tile, 8 weight rows per warp,
the 8 warps of a block splitting K, partials reduced in shared memory) cut the per-expert marginal cost from
≈ 30 µs to ≈ 17 µs.

## 3. Correctness (synthetic experts, cuda:6, `/tmp/claude-1000/test_chain.py`)

vs the seven-launch chain, for 0, 1, 2, 3 and 6 local experts, both variants:

| output | max abs error | RMS error | cosine |
|---|---|---|---|
| routed partial (fp32 [5120], 1–6 experts) | 5.8e-11 … 1.3e-10 (max |ref| 1.7e-4 … 4.7e-4) | 4.7e-12 … 2.3e-11 | 1.00000000 |
| routed partial, 0 local experts | 0 (all-zero rows, as the baseline) | 0 | – |
| shared expert ys (bf16 [5120]) | 64 at |ref| 2.15e4 = one bf16 ulp | 12.4 (5.8e-4 relative) | 0.99999654 |
| peer mode (routed only) vs owner mode | identical bits | | |

The SwiGLU + e4m3 rounding reproduces `fused.py::_swiglu_quant_kernel` (clamp, silu·up, weight, bf16 cast,
per-32 amax, 2^ceil(log2(amax/448)), e4m3 rounding) with IEEE division and rint; the only differences are the
fp32 accumulation order of the dot products and `expf` vs Triton's exp.

## 4. Kernel metrics (cuobjdump / Nsight Compute, cuda:6, two local experts + shared)

| kernel | registers | static smem | local mem | blocks/SM | grid | duration (ncu clocks) | DRAM % of peak | SM throughput | warps active | tensor pipe |
|---|---|---|---|---|---|---|---|---|---|---|
| `expert_chain_p1` (phase 1) | 60 | 320 B | 0 | 4 | 4032 (one per 8-row item, surplus exit) | 72.5 µs, 48.7 MB | 36 % | 41 % | 48 % | 17 % |
| `expert_chain_p2` (phase 2) | 64 | 288 B | 0 | 4 | 1280 | 32.7 µs, 24.4 MB | 40 % | 45 % | 46 % | 18 % |
| `expert_chain` (persistent) | 64 | 32.5 KB | 0 | 4 (432 resident) | 432 | – | – | – | – | – |
| previous `fp4_gemv_m1` (w13 / w2) | 48 | 0 | 0 | 5 | 3456 / 3840 | 42.1 / 29.1 µs | 32 / 22 % | 57 / 61 % | 58 / 56 % | 0 |

Neither DRAM nor the SM pipes are saturated: at one token the kernels are latency-bound (4 blocks × 8 warps per
SM, a few 16-byte loads in flight per lane).

## 5. Isolated timing (cuda:6, CUDA graphs, weights rotated past L2; owner = routed + shared, peer = routed only)

| local experts | 7 launches serialized on one stream | two-launch fused (owner) | persistent (owner) | two-launch (peer) | persistent (peer) |
|---|---|---|---|---|---|
| 1 | 91.5 µs | **62.6** (865 GB/s of expert bytes) | 72.9 | 35.6 | 43.9 |
| 2 | 102.1 | **80.5** (907 GB/s) | 96.7 | 54.3 | 72.6 |
| 3 | 114.3 | **99.0** (927 GB/s) | 129.1 | 73.3 | 96.8 |

Accounting of the isolated difference at one expert (91.5 → 62.6 µs, −29 µs):

* launch / ramp / tail time removed: 5 launches × ≈ 6 µs ≈ 29 µs — the whole difference;
* intermediate global traffic removed: ≈ 0.5 MB per layer per GPU (gu, hqp, y, gu_s, hs), all L2-resident:
  < 1 µs;
* FP4 read time: 17.9 MiB per expert at the marginal ≈ 1.05 TB/s ≈ 17 µs per expert (identical in both);
* arithmetic: the mma routines run at 17 % tensor-pipe activity; the arithmetic is not on the critical path.

But the serialized 7-launch number is not what the model pays: in the runtime the shared chain (S1–S3, 36 µs)
runs on a side stream concurrently with the routed chain, so the owner's expert phase is ≈ 69 µs, not 91. The
two-launch fused chain at the model's average of 1.5 local experts is ≈ 71 µs (interpolated) — parity.

## 6. In-model benchmark (`results/chain_bench.jsonl`, bench_replica, 200 steps, all flags of the previous step on)

| config | S=1 tok/s (ms/token) | expert launches / layer (owner + 3 peers) | expert phase µs/layer (2 → 3) | local GEMMs (2 → 10) | wait partials | peer chain (max) |
|---|---|---|---|---|---|---|
| reference (all flags), 3 runs | 67.5 (14.82), 67.3 (14.86), 67.3 (14.85) | 7 + 3 × 4 = 19 | 72.9 / 71.6 / 66.1 | 69.3 / 68.0 / 62.4 | 19.5 / 21.1 / 29.3 | 76.9 / 77.7 / 81.9 |
| two-launch fused chain (`DSV41_PERSISTENT_EXPERT=1`), 3 runs | 63.5 (15.75)*, **67.6 (14.78)**, **67.5 (14.81)** | 2 + 3 × 2 = 8 | 72.8 / 70.6 / 77.1 | 71.8 / 69.6 / 76.2 | 20.2 / 20.9 / 15.5 | 73.1 / 77.3 / 72.2 |
| persistent single launch (`=2`) | 64.3 (15.55) | 1 + 3 × 1 = 4 | 84.9 | 83.8 | 26.4 | 91.2 |
| two-launch at S=2 / S=8 (fused chain inactive at B > 1) | 92.6 / 240.0 (parity with 92.4 / 238.6) | | | | | |

\* first run: the per-layer stamps of that run are identical to the reference (layer 320.2 vs 318.9 µs) while the
host step was 0.9 ms longer — the intermittent ASR job on GPUs 0/2; the two alternating repeats reproduce
parity (p50 14.64 / 14.61 ms vs 14.67 / 14.68).

**Result: the fused chain removes 11 of 19 expert launches per layer and the shared-expert side stream, at
identical wall-clock latency (67.5 → 67.5 tok/s; expert phase 66–73 µs/layer before and after).** The persistent
single-launch variant is 5 % slower (grid barrier, redundant SwiGLU prologue, residency-limited grid).

Why the launch-cost model overestimated the gain: the "≈ 7 µs per launch" is ramp + tail of a 15–25 µs kernel,
not host or graph overhead; a fused kernel keeps a ramp and a tail per phase (the w13 → w2 dependency is a
grid-wide barrier in any design), the shared chain was already off the critical path on its side stream, and
the new kernels, while streaming at up to 0.93 TB/s of expert bytes (vs 0.4–0.6 before), still sit at 36–40 %
of DRAM peak (latency-bound at 4 blocks × 8 warps per SM). The proof that the isolated gain is launch removal
holds (−29 µs = 5 launches × 6 µs, intermediate traffic < 1 µs); it simply does not translate to the model
because the model's critical path never contained those 5 launches serially.

Deliverables: `dsv41/cuda/expert_chain.cu` (both variants), `cukern.expert_chain` (occupancy-sized grid,
struct arguments, per-device scratch), `ep.py` owner/peer integration (`_fused_chain_ok`: single-row steps
without replicas), off by default. Numerics: section 3.

## 7. Critical-path profile of the optimized S=1 path (all flags; `results/decode_profile_S1_all.json`, unprofiled stamps from `chain_bench.jsonl`)

Token = 14.82 ms. Per layer (mean over 40, unprofiled): attention + hc 183 µs, hc_sub2 9, gate + top-k 16.5,
multicast 13.3, expert phase 70 (local GEMMs 63–76), wait for peers 20, combine 3.9 → layer 319 µs, 12.76 ms in
the layers; outside the layers 2.06 ms = head 0.77 + hops 0.09 + host launches / final sync / sampling ≈ 1.2.

| component | ms / token | share | nature |
|---|---|---|---|
| FP8 dense GEMVs (wq_a, wq_b, wkv, wo_a, wo_b; 121 MiB/layer) | 4.3 (107 µs/layer) | 29 % | HBM streaming at 1.2–1.3 TB/s of a 1.94 peak; floor ≈ 3.2 |
| expert phase (routed FP4 + shared FP8, 2 chains or fused) | 2.8 | 19 % | latency-bound one-token kernels (36–40 % DRAM); floor ≈ 1.6 |
| small kernels around the projections (hc_pre/post, norms, quant, rope, kv write, sattn2, sinkhorn; ~14/layer) | 1.4 | 9 % | launch-latency-bound (2–12 µs each), fusable in pairs |
| P2P: multicast issue 0.53 + waiting for the slowest peer 0.8 | 1.3 | 9 % | protocol; the peers' chains are hidden except their tail |
| host: 4 graph launches, final device sync, argmax/tolist | ≈ 1.2 | 8 % | host |
| heavy layers' remaining indexer/compressor work (4 layers) | ≈ 0.8 | 5 % | torch ops (cuBLAS gemvs, cache mirror copies) |
| head (bf16 1.26 GB) | 0.77 | 5 % | HBM streaming |
| gate GEMM + top-k | 0.66 | 4 % | cuBLAS fp32 gemv 9.5 µs + fused top-k 6 |
| hyper-connection mixes / sinkhorn (side stream) | (0.7 kernel time, ≈ 0.2 on the path) | 1 % | overlapped |

Ranking for the next target (largest first): the dense FP8 GEMVs (1.1 ms of headroom to the streaming floor,
reachable only by fusing the dependent pairs wq_a→wq_b / wo_a→wo_b and wq_a‖wkv into fewer, longer streams —
the same launch-ramp argument as here, but there the launches ARE serial on the critical path), then the
expert kernels' latency (deeper prefetch / more warps per SM: 1.2 ms of headroom), then the host side (1.2 ms)
and the small-kernel pairs (≈ 0.6 ms). P2P stays at 9 %.
