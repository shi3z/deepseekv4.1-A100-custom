# Dependent dense GEMV chains (S=1): reconstruction, fusion candidates, and the first fusion — report

Date: 2026-09-28. Starting point: the all-flags S=1 path (`DSV41_FP8_M1=1 DSV41_FP4_M1=1 DSV41_DECODE_LEAN=1
DSV41_INDEX_FAST=1`; `results/dense_m1_report.md`, `results/expert_chain_report.md`): 67.2–67.5 tok/s, 14.8–14.9 ms/token,
FP8 dense GEMVs ≈ 4.3 ms/token of the critical path. Model: DeepSeek-V4.1-Flash, dim 5120, 40 layers, 64 heads × 512,
q_lora_rank 1280, o_lora_rank 1024 × 8 groups. Everything below is behind `DSV41_FUSED_DENSE_CHAIN` (default 0 = unchanged).

Two numbers are kept apart throughout: **isolated kernel time** (CUDA-graph-captured launches on an idle A100 at
boost clocks, weights rotated over 8 copies so they stream from DRAM, `scratchpad/test_dense_chain.py`), and
**in-model token time** (`bench_replica`, 200 steps, all 4 GPUs). ncu numbers are at locked base clocks and are only
used for counters (bytes, instructions, registers, occupancy), not for durations.

## 1. The chains as executed today (S=1, one layer, owner GPU; `decode.py:attention2`, all flags)

All kernels are inside the per-GPU CUDA graph (launch gaps ≈ 0.1 µs, the graph replays them back to back). Weights
are FP8 e4m3 with 32×32 E8M0 block scales, k-permuted inside 16-k groups; activations are bf16 that have been
fake-quantized to fp8 per 32 (`x = q·s`, s = 2^ceil(log2(amax/448))). "Kineto µs" = per-kernel durations from
`results/decode_profile_S1_all_trace.json` (inflated by the profiler by roughly 1.2–1.5×); "isolated µs" = graph
timing on GPU 6 at the real shapes.

| chain | # | kernel (grid × 256 thr) | in (bytes) | weight bytes | out (bytes) | Kineto µs | isolated µs | can B start before A completes? |
|---|---|---|---|---|---|---|---|---|
| Q | A1 | `fp8_gemv_m1_k2u2` = **wq_a** [1280, 5120] (320 blk) | xq bf16 [5120] (10,240) | 6.55 MB + 0.2 MB scales | qr_raw bf16 [1280] (2,560) | 9.2 | 8.1 | – |
| Q | A2 | `_norm_quant_kernel` = q_norm + fq8 (1 block, Triton) | qr_raw | 2,560 (norm w) | qr bf16 [1280] (2,560) | 6.8–10.1 | 6.1 | no: needs RMS over all 1280 of A1's outputs |
| Q | A3 | `fp8_gemv_m1_k1u4` = **wq_b** [32768, 1280] (4096 blk) | qr | 41.9 MB + 1.3 MB | q bf16 [32768] (65,536) | 34.7 | 33.4 | no: every output row needs the full K=1280 vector, which needs A2 |
| Q | A4 | `_rope_dev_kernel` | q | cos/sin | q (in place) | 2.6 | – | |
| KV | B1 | `fp8_gemv_m1_k4u2` = **wkv** [512, 5120] (256 blk) | xq (same as A1) | 2.62 MB + 0.08 MB | kv_raw bf16 [512] (1,024) | 6.1 | 4.9 | independent of A1 (shares its input) |
| KV | B2 | `_kv_write_kernel` = kv_norm + rope + ring write | kv_raw | | cache row | 2.8 | – | |
| O | C1 | `fp8_gemv_m1_k1u2` = **wo_a** [8192, 4096] block-diag (group_cols 1024, 1024 blk) | o bf16 [8, 512] (8,192) | 33.5 MB + 1.0 MB | o_a bf16 [8192] (16,384) | 26–27.4 | – | – |
| O | C2 | `_fake_quant_kernel` = fq8 per 32 (32 blocks, Triton) | o_a | – | o_q bf16 [8192] (16,384) | 2.1 | 1.8 | no: per-32 amax needs C1's outputs of that group (a per-block dependency, but C1's blocks finish in arbitrary order) |
| O | C3 | `fp8_gemv_m1_k1u2` = **wo_b** [5120, 8192] (640 blk) | o_q | 41.9 MB + 1.3 MB | a bf16 [5120] (10,240) | 29.2–30.6 | 32.1 | no: full K=8192 per row |

Launches per layer on these chains: 9 (A1–A4, B1–B2, C1–C3); the ones a fusion can remove are A2, B1 (into A1) and
C2. Everything is one row (M=1): no tensor cores, the GEMVs are pure weight streams at 1.2–1.3 TB/s for the two large
ones and latency-bound (0.5–0.8 waves) for wq_a / wkv.

## 2. Intermediate storage

| intermediate | elements | BF16 | FP16 | FP32 | where it could live |
|---|---|---|---|---|---|
| qr_raw / qr (Q chain) | 1,280 | 2,560 B | 2,560 B | 5,120 B | registers: 10 warps' worth (1 element per lane); one block's smem: 2.5 KB of 164 KB; **not** distributed smem (A100 has no thread-block clusters / DSMEM) |
| kv_raw | 512 | 1,024 B | 1,024 B | 2,048 B | idem |
| o_a / o_q (O chain) | 8,192 | 16,384 B | 16,384 B | 32,768 B | one block's smem: 16 KB (fp32 32 KB would cap occupancy at 5 blocks/SM); registers: 64 warps |

The consumer GEMV (wq_b: 4096 blocks; wo_b: 640 blocks) needs the *whole* vector in *every* block, so on A100 the only
ways to hand it over are (a) global memory (L2-resident: 2.5–16 KB ≪ 40 MB — this is what the separate kernels do
today, the "intermediate traffic" is 2.5–16 KB written once and read from L2 by every block) or (b) recomputing the
producer per block. Because the producer of the Q chain is a 1280-element RMSNorm and the producer of the O chain is
a per-32 fp8 rounding, (b) is cheap in bytes (every block re-reads the 2.5/16 KB raw vector from L2, exactly what it
reads today) but not in instructions — see §5.

## 3. Candidate designs

**Design A — fused two-stage GEMV (A1→A2→A3 or C1→C2→C3 in one kernel).** GEMV A's outputs are spread over all its
blocks; GEMV B's every block needs all of them → a grid-wide dependency. On A100 that is either a grid barrier
(cooperative launch: the persistent expert chain of `results/expert_chain_report.md` showed the barrier + phase
boundary costs what the launch cost, and single-wave residency caps the grid at ~4 blocks/SM) or the second stage
re-does GEMV A per block (impossible: 6.5–33 MB of weights). Rejected for both chains.

**Design B — producer/consumer persistent CTAs.** Same grid-wide dependency: consumers cannot start a row before all
producer blocks finish, so they idle for the whole of GEMV A. Not provably cheaper than the ≈ 0.1 µs graph launch gap;
rejected (the spec only allows it if provably cheaper).

**Design C — concatenate projections sharing an input.** wq_a and wkv both consume `xq` (natural order, fp8-rounded
bf16 [5120]): concatenate rows → one [1792, 5120] GEMV (`W8.cat`, row counts are multiples of 32 so the scale rows
concatenate too), outputs split by row range. wq_b and wo_b share nothing with any other GEMV (different inputs);
wo_a is block-diagonal on a different input. Only the Q/KV pair qualifies.

**Design D (added) — prologue fusion.** Fuse the small elementwise producer (A2: RMSNorm+fq8; C2: fq8) into the
*consumer* GEMV's prologue: every block reads the raw vector, normalizes/rounds it into shared memory (K·2 bytes,
2.5 / 16 KB), then streams its rows from smem. This removes the tiny kernel's launch + ramp/tail and its 2.5–16 KB
write, and keeps GEMV B bit-identical. Cost: the producer is recomputed by every block (4096× for wq_b, 640× for wo_b).

Per-candidate estimate (before measuring; launch saving = one graph node ≈ 2–6 µs of ramp/tail for these tiny
kernels, traffic saving = the intermediate's write + read ≈ 5–33 KB, i.e. < 0.03 µs at DRAM speed — the traffic term
is negligible for every candidate and is listed to make that explicit):

| candidate | launches removed / layer | bytes avoided / layer | registers | smem / block | occupancy | expected µs / layer |
|---|---|---|---|---|---|---|
| C: wq_a‖wkv | 1 | xq read once instead of twice (10 KB, L2) | 44 (unchanged kernel) | 32 B | 5 blk/SM (unchanged) | −2 to −3 (wkv's own ramp/tail; both kernels are < 1 wave) |
| D-Q: q_norm+fq8 into wq_b | 1 | 2.5 KB write + 2.5 KB read | 70 (was 55) | 2.6 KB dyn | 3 blk/SM (was 4) | −4 to −6 if the prologue hides under the weight stream |
| D-O: fq8 into wo_b | 1 | 16 KB write + 16 KB read | 64 (was 39) | 16.4 KB dyn | 4 blk/SM (was 6) | −1 to −2 |

## 4. Theoretical upper bounds (second launch free, intermediate traffic zero, GEMV time unchanged)

Per layer, from the isolated chain timings at the real shapes (Q chain separate = 53.3 µs, O chain separate = 33.7 µs):

| chain | today | bound: keep only the weight streams | saving / layer | × 40 layers | ms/token (from 14.88) | tok/s (from 67.2) |
|---|---|---|---|---|---|---|
| Q (A1+A2+A3, B1) | 53.3 µs | wq_a‖wkv bytes at 1.3 TB/s (7.1 µs) + wq_b unchanged (33.4 µs) = 40.5 µs | −12.8 µs | −0.51 ms | 14.37 | 69.6 |
| O (C2+C3) | 33.7 µs | wo_b unchanged (32.1 µs) | −1.6 µs | −0.06 ms | 14.82 | 67.5 |
| both | | | −14.4 µs | −0.57 ms | 14.31 | 69.9 |

(The pre-measurement estimate from Kineto durations — Q −19 µs, O −8 µs, 72.8 tok/s — was inflated by the profiler's
per-kernel overhead; the graph-timed bound above is the one to hold the results against.) Even at the bound, dense
chain fusion is worth ≤ +4% at S=1: the chains are dominated by the two 42 MB weight streams, which fusion does not touch.

**Ranking** (payoff / complexity / numerical risk): 1. Design C on the Q chain — biggest measured saving, trivial
(a cached `W8.cat` + row slices), zero numerical risk (identical kernel, identical per-row arithmetic). 2. D-O — one
kernel with a fused prologue, bit-identical rounding, small payoff. 3. D-Q — same kernel, but 4096 replicas of an
RMSNorm; payoff depends entirely on whether the prologue hides under the stream (it does not, §5).

## 5. Implementation and isolated measurements

* `dsv41/cukern.py`: `FUSED_DENSE_CHAIN` (env `DSV41_FUSED_DENSE_CHAIN`), `fp8_gemv_m1_pre(x_raw, w8, s8, norm_w, eps, mode)`.
* `dsv41/cuda/fp8_gemv_m1.cu`: `fp8_gemv_m1_pre{1,2}_k{KW}u{U}` — the M=1 GEMV with the activation prologue fused
  (mode 1 = RMSNorm·w + per-32 fp8 rounding, mode 2 = rounding only), x staged in dynamic shared memory (K·2 bytes),
  the first U weight loads issued before the prologue, GEMV loop software-pipelined by one step. Rounding is exactly
  the Triton kernels' (amax ≥ 1e-4, s = 2^ceil(log2(amax/448)), e4m3 RNE of clamp(x/s), q·s → bf16; powers of two so
  x·(1/s) ≡ x/s). Two prologue layouts, each the faster one for its shape: mode 1 element-parallel over all 8 warps
  (amax via shuffles), mode 2 one lane per 32-group (amax in registers).
* `dsv41/decode.py:attention2` (only when B == 1, i.e. S=1; larger batches take the unchanged path):
  * value 1: **Q chain, Design C** — `[wq_a; wkv]` cached per layer as one `W8` (9.2 MB concatenated copy per layer, see the note below), one `fp8_gemv_m1`, outputs split by row range, then the
    unchanged `norm_quant` → `wq_b` → rope and `kv_write`.
  * value 2: + **O chain, D-O** — `fake_quant_fp8` + `wo_b` replaced by `fp8_gemv_m1_pre(mode 2)`.
  * value 3: + **D-Q** — `norm_quant` + `wq_b` replaced by `fp8_gemv_m1_pre(mode 1)` (record only, slower).

  Note on memory: the concatenated weight is a copy (`torch.cat`); the original `wq_a`/`wkv` tensors are still
  referenced by the module, so value ≥ 1 costs 9.2 MB per layer per GPU (10 layers per GPU → 92 MB). Re-pointing the
  originals at row slices of the concatenation would remove that; not done because the prefill path uses them.

Correctness (GPU 6, random weights/activations at the real shapes, fused vs the separate kernels):

| comparison | max abs diff | RMS diff | cosine | exact elements | argmax |
|---|---|---|---|---|---|
| concat GEMV rows 0–1279 vs wq_a alone | 0 | 0 | 1.0 | 100 % | same |
| concat GEMV rows 1280–1791 vs wkv alone | 0 | 0 | 1.0 | 100 % | same |
| wq_b with fused q_norm+fq8 (mode 1) vs norm_quant → wq_b | 0 | 0 | 1.0 | 100 % | same |
| wo_b with fused fq8 (mode 2) vs fake_quant → wo_b | 0 | 0 | 1.0 | 100 % | same |
| idem with a zero group and a 1e-5 group (amax floor) | 0 | 0 | 1.0 | 100 % | same |

Isolated kernel times (µs, CUDA graph, GPU 6, weights rotated past L2, real shapes; ±0.1):

| | separate | fused | Δ / layer |
|---|---|---|---|
| wq_a (8.09) + wkv (4.86) | 12.95 | wq_a‖wkv 10.10 | **−2.85** |
| norm_quant (6.13) + wq_b (33.39) | 39.52 | wq_b mode 1 **75.34** | **+41.8** |
| Q chain (4 launches) | 53.25 | value 1 (concat + separate norm_quant/wq_b): 50.4 (computed from the rows above); value 3 (2 launches): 85.2 | −2.85 / +32 |
| fake_quant (1.81) + wo_b (32.10) | 33.66 (chain measured) | wo_b mode 2 32.07 | **−1.6** |

ncu counters at the real shapes (GPU 6, locked base clocks — durations are not comparable to the graph timings above):

| kernel | grid | regs | blocks/SM (regs) | achieved occupancy | DRAM read | DRAM % of peak | instructions |
|---|---|---|---|---|---|---|---|
| wq_a `k2u2` | 320 | 44 | 5 | 34 % | 6.58 MB | 30 % | 1.64 M |
| wkv `k4u2` | 256 | 44 | 5 | 28 % | 2.64 MB | 15 % | 0.86 M |
| wq_a‖wkv `k2u2` | 448 | 44 | 5 | 46 % | 9.20 MB | 32 % | 2.29 M |
| norm_quant (Triton) | 1 | 62 | – | 5.6 % | 24 KB | 0.1 % | 5 K |
| wq_b `k1u4` | 4096 | 55 | 4 | 45 % | 42.0 MB | 52 % | 13.2 M |
| wq_b `pre1_k1u4` | 4096 | 70 | 3 | 35 % | 42.0 MB | 23 % | **34.1 M** |
| fake_quant (Triton) | 32 | 19 | – | 6.4 % | 22 KB | 0.3 % | 15 K |
| wo_b `k1u2` | 640 | 39 | 6 | 59 % | 42.0 MB | 65 % | 8.8 M |
| wo_b `pre2_k1u2` | 640 | 64 | 4 | 40 % | 42.0 MB | 52 % | 12.9 M |

No spills in any variant (`-Xptxas -v`: 0 bytes stack / spill for all `pre` kernels; 32 B static smem + K·2 dynamic).

**Why D-Q loses.** wq_b streams only 8 rows × 1,280 B = 10 KB per block (2.5 16-byte chunks per lane), so the
kernel is 1.65 M instructions for 42 MB. The fused prologue re-does the 1280-element RMSNorm + rounding in each of
the 4096 blocks; even after three rewrites (warp-per-group with shuffles; lane-per-group in registers; element-
parallel over all 8 warps with the weight prefetch issued first) it adds ≈ 5 K instructions per block = +21 M
instructions per layer, and the kernel goes from DRAM-bound to issue-bound (DRAM 52 % → 23 %, duration 33 → 75 µs).
The per-launch saving it was after (norm_quant ≈ 6 µs) cannot pay for that. The only way to make D-Q pay is fewer,
fatter blocks (e.g. 64 rows per block with the loop pipelined across rows) — a different GEMV tiling, out of scope for
the first fusion. D-O survives because wo_b streams 64 KB per block (32 chunks per lane): +4 M instructions on 8.8 M
is hidden under the stream at boost clocks (32.07 vs 32.10 µs alone) and the fake_quant launch (1.8 µs) is gone.

## 6. In-model measurements (S=1 / 2 / 8, all flags, 200 steps, `results/dense_chain_bench.jsonl`)

Runs alternate base / fused to expose interference (the ASR job on GPUs 0/2 runs intermittently and costs up to
±3–6 %); `attention_hc` is the stamp-timed phase inside the CUDA graph (hc_sub → attention2 → hc_post2, per layer,
averaged over the 200 steps) and is the honest per-layer measurement of the chain — it is not inflated by any profiler.

| run | S | tok/s | ms/token | p50 / p90 ms | attention_hc µs/layer | expert phase µs/layer | wait partials µs/layer |
|---|---|---|---|---|---|---|---|
| base r1 | 1 | 67.0 | 14.92 | 14.73 / 15.13 | 183.6 | 72.4 | 19.2 |
| **fused=1** (Q concat) r1 | 1 | **68.1** | **14.68** | 14.51 / 14.75 | **179.6** | 76.8 | 19.2 |
| base r2 | 1 | 63.1 (interference) | 15.85 | 15.64 / 15.91 | 183.2 | 72.8 | 21.9 |
| fused=1 r2 | 1 | 67.7 | 14.76 | 14.57 / 14.93 | 179.4 | 77.1 | 19.4 |
| fused=2 (+ wo_b prologue) | 1 | 67.4 | 14.84 | 14.78 / 14.95 | 181.5 | 72.1 | 21.0 |
| fused=3 (+ wq_b prologue, record) | 1 | 61.6 | 16.23 | 16.07 / 16.42 | 218.3 | 77.9 | 16.5 |
| fused=1 r3 | 1 | 68.0 | 14.71 | 14.50 / 14.85 | 179.7 | 74.1 | 20.0 |
| base r3 | 1 | 67.5 | 14.81 | 14.70 / 14.99 | 183.1 | 73.4 | 21.6 |
| base | 2 | 92.7 | 21.57 | 21.30 / 22.72 | 209.5 | 160.4 | 48.7 |
| fused=1 (B≠1: unchanged path) | 2 | 92.6 | 21.60 | 21.36 / 22.51 | 209.5 | 162.7 | 45.2 |
| base | 8 | 239.5 | 33.40 | 33.26 / 33.86 | 254.0 | 322.0 | 115.1 |
| fused=1 (unchanged path) | 8 | 223.7 (interference: same attention_hc, wait 135.7) | 35.76 | 35.67 / 36.08 | 253.9 | 313.7 | 135.7 |
| fused=1 r2 | 8 | 240.7 | 33.24 | 33.08 / 33.52 | 254.4 | 313.3 | 104.0 |
| base r2 | 8 | 239.0 | 33.47 | 33.26 / 33.71 | 254.1 | 332.2 | 89.7 |

**S=1, old vs new chain latency (in-model, stamps):** attention_hc 183.1–183.6 → 179.4–179.7 µs/layer = **−3.8 µs/layer**
(isolated prediction −2.85 µs for the concat + ≈ 1 µs of graph-node gap/ramp). **Full-token delta:** −0.15 ms/token
(3.8 × 40 layers), i.e. base 67.0–67.5 tok/s (14.81–14.92 ms) → fused 67.7–68.1 tok/s (14.68–14.76 ms): **+0.7–1.1 %**,
just above the run-to-run noise of the non-outlier runs (±0.3 tok/s) and confirmed by three alternating pairs.
Launches removed: 1 per layer (wkv); 40 per token on the owner path. Bytes avoided: the second read of xq
(10 KB per layer, from L2) — no DRAM bytes. The `fused=2` run adds the wo_b prologue and measures 181.5 µs/layer
(single run): the isolated −1.6 µs/layer does not survive in-model (the prologue's +4 M instructions per layer are
not fully hidden when the GPU is also serving the peers' expert phase); value 2 is not recommended. The `fused=3` run
confirms the isolated loss of the wq_b prologue in-model: +35 µs/layer, 61.6 tok/s.

**Correctness in-model (first decode step, the S=1 prompt; `--check-tokens` against base r1):**

| run vs base r1 | max abs logit diff | RMS diff (RMS ref 4.75) | cosine | argmax | greedy tokens identical (of 200, after 16 warm-up steps) |
|---|---|---|---|---|---|
| base r2 (the envelope) | 1.77 | 0.365 | 0.99761 | same | 18 |
| base r3 (the envelope) | 2.25 | 0.420 | 0.99665 | same | 9 |
| fused=1 r1 | 0.875 | 0.175 | 0.99939 | same | 4 |
| fused=1 r3 | 2.38 | 0.421 | 0.99758 | same | 69 |
| fused=2 | 1.28 | 0.271 | 0.99841 | same | 15 |
| fused=3 | 2.75 | 0.443 | 0.99691 | same | 0 |

The fused runs sit inside the baseline-vs-baseline envelope (the runtime is not bitwise reproducible run to run:
atomics in the expert combine and the P2P partial sums; the sample texts of the base runs differ from each other
the same way). The fusion itself is bitwise exact on its own kernels (§5), so no numerical change is expected or seen.

## 7. Re-profile of the full S=1 critical path with the fusion

Kineto profile of one S=1 step (`decode_profile.py`, `results/decode_profile_S1_dense.json` vs `results/decode_profile_S1_all.json`
from the expert-chain study; per-kernel durations are profiler-inflated, and a single captured step carries the P2P wait of
that particular step, so the two totals are not a before/after of the fusion — only the attention-projection category is):

| category (owner critical path, µs per token) | all flags (before) | all flags + `DSV41_FUSED_DENSE_CHAIN=1` | Δ |
|---|---|---|---|
| FP8 dense GEMV in attention+hc (wq_a‖wkv, wq_b, wo_a, wo_b, hc/gate projections) | 4,480 | 4,312 | **−168 (−4.2 µs / layer)** |
| FP8 dense GEMV in the local expert phase (shared expert) | 1,333 | 1,332 | 0 |
| RMSNorm (incl. norm_quant) | 584 | 602 | +18 (noise) |
| _fake_quant_kernel | 133 | 134 | 0 (value 1 keeps it) |
| _kv_write_kernel | 113 | 123 | +10 (noise) |
| attention core | 634 | 637 | 0 |
| P2P wait (spin) | 3,797 | 2,493 | −1,304 (step-to-step variance of the peer wait, not the fusion) |
| layers wall (owner sections) | 16,446 | 14,799 | (dominated by the wait line above) |
| host step incl. sync | 20,762 | 18,202 | |

Critical-path ranking after the fusion (Kineto-inflated µs per token, owner path, S=1): FP8 dense GEMVs 5,644 (attention
4,312 + shared expert 1,332) › cuBLAS GEMMs of the expert phase 2,910 › P2P wait 2,493 › hyper-connection mixes 777 ›
P2P send/signal 771 › sinkhorn 722 › attention core 637 › RMSNorm 602 › router top-k 571 › elementwise/copies 374 ›
indexer/compressor/rope 274 › combine 225 › fake_quant 134 › kv_write 123; inter-layer gaps 88; bf16 head 772. The
dense chains that fusion can still touch (norm_quant ≈ 15 µs/layer inflated ≈ 6 isolated, fake_quant 3.3 / 1.8,
kv_write 3 / 2.8) are together ≈ 10 µs/layer isolated = 0.4 ms/token = 2.7 % — the same order as the noise between
two benchmark runs. The 42 MB weight streams of wq_b and wo_b (≈ 65 µs/layer, 2.6 ms/token) and the expert phase remain
the only items worth more than a few percent at S=1.

## 8. Bottom line

* **Implemented and kept:** `DSV41_FUSED_DENSE_CHAIN=1` = the Q chain's Design C (wq_a‖wkv concatenated into one M=1
  GEMV), S=1 only, default off, bitwise-identical outputs. Measured: −3.8 µs/layer of attention phase, −0.15 ms/token,
  67.0–67.5 → 67.7–68.1 tok/s (+0.7–1.1 %). Cost: 9.2 MB of extra weight copies per layer per GPU (92 MB).
* **Implemented, measured, not recommended:** value 2 (fake_quant fused into wo_b's prologue, isolated −1.6 µs/layer,
  in-model no gain) and value 3 (q_norm fused into wq_b's prologue: 4096 replicas of a 1280-element RMSNorm make the
  kernel issue-bound, +35 µs/layer, 61.6 tok/s). The pre-measurement ranking put the wq_b prologue first because the
  design study assumed a 4096-row wq_b; the real wq_b is 32768 × 1280 (42 MB, 4096 blocks of only 10 KB each), which
  is exactly the shape where a replicated prologue cannot hide.
* **Design A / B** (fused two-stage GEMV, producer/consumer CTAs) are rejected on A100 for the reasons in §3: every
  consumer block needs every producer output, and the grid barrier or the idle consumers cost what the ≈ 0.1 µs graph
  launch gap costs, as the expert-chain experiment already showed.
* **Upper bound reached vs available:** the fusion took 3.8 of the 12.8 µs/layer bound of the Q chain; the rest is the
  norm_quant kernel (6 µs isolated, a single Triton block) and wq_b's ramp/tail, which only a re-tiled wq_b (fatter
  blocks with a pipelined multi-row loop, so the prologue is amortized) could recover — out of scope for the first
  fusion, and worth at most another ≈ 0.3 ms/token (2 %).
* **Per the spec, stop here:** the S=1 critical path was re-profiled with the fusion (§7). Its ranking is unchanged:
  the 42 MB wq_b / wo_b weight streams and the expert phase dominate; the remaining fusable dense chains are worth
  ≤ 2–3 % combined and are below the run-to-run noise of a single benchmark.
