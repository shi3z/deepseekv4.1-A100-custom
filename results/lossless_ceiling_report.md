# Lossless S=1 ceiling: exact speculative decoding (DSpark/MTP) and the remaining kernel/runtime levers — report

Date: 2026-09-28. Baseline for this study: the all-flags S=1 path plus `DSV41_FUSED_DENSE_CHAIN=1`
(`results/dense_chain_report.md`): 67.7–68.1 tok/s, 14.7 ms/token, 4 × A100 80GB PCIe, expert-parallel decode
(`--devices 2,3,0,1 --ep`). Constraint for everything here: **semantics-preserving only** — no weight format change,
no approximation, no altered routing. The MTP path uses the checkpoint's own DSpark draft head, and its acceptance
rule is exact (§1.3), so it is inside the constraint.

Vocabulary: a *pass* = one full-model step (one CUDA-graph replay per GPU); the verify pass carries K = 1 + drafts
rows. "Isolated" = kernel timed alone inside a CUDA graph on an idle GPU; "in-model" = `mtp_run` / `bench_replica`
wall-clock per step. ncu/Kineto durations are never reported as measured token time.

## 1. The existing MTP implementation (`dsv41/dspark.py`, `dsv41/mtp_run.py`, `dsv41/engine.py:_generate_mtp_locked`)

### 1.1 How drafts are generated

DSpark is DeepSeek-V4.1-Flash's trained draft head (tech report §2.4.3; checkpoint tensors `mtp.{0,1,2}.*`):
three transformer blocks (window attention over a 128-token ring of the *projected main hidden state* +
the draft block itself, 128-expert top-3 MoE, hyper-connections), fed with the embedding of the just-accepted token
followed by `dspark_block_size − 1 = 4` noise tokens, producing base logits for **5 draft positions in one parallel
(non-autoregressive) pass**. A first-order Markov head (`markov_head`: prev token → vocab bias) is applied left to
right with greedy argmax, and a confidence head gives a per-position acceptance estimate (computed but unused by our
runtime — the tech report's scheduler picks the verify length from it; we fix K by `--mtp`/`--drafts`).
Input state: the mean-over-hc hidden state at the *inputs* of the three target layers (`dspark_target_layer_ids`,
15,360 = 3 × 5,120 values per position), projected by `main_proj`, normalized, and written as kv into the draft
blocks' rings for every main position (`write_main_rows`, also for every verify row of the previous pass).
The batched version (`DSparkRows`) runs on the runtime's fused kernels inside one CUDA graph per S; it is ~40
launches per block, **3.9 ms per draft at S=1** in the earlier measurement (`results/mtp-S1-d5.log`).

### 1.2 How many draft tokens are proposed

Always 5 (the trained block size). The runtime verifies the first `drafts` of them: `--mtp 1..5` (server),
`--drafts 1..5` (`mtp_run`). **Draft lengths 6 and 8 are not available**: DSpark is trained for one 5-position block
per accepted token; a second block would have to be drafted from main-model hidden states of positions that have
not been verified yet (they do not exist until the verify pass), so it cannot be produced without changing the
model's semantics. The benchmark below therefore covers drafts 1, 2, 3, 4, 5.

### 1.3 How verification works, and whether it is exact

One verify pass = one ordinary batched decode step with K rows of the *same* sequence at positions p+1 … p+K:
row 0 = the accepted "bonus" token, rows 1..K−1 = the drafts. All K rows are written into the window ring, the
compressed-kv cache and the indexer state at their own positions *before* the attention kernels run; the attention
mask (`fused2._sattn2_split_kernel`: `valid = held ≤ plim` with `plim = pos` of the row) lets row i see exactly the
positions ≤ p+1+i, i.e. **causal within the pass**; the compressed/indexed part uses `compress_len = (pos+1)/ratio`
per row, also causal. Each row's logits are therefore the full model's logits for the prefix [… , t_p, bonus, d_1 … d_i].

Acceptance (`engine.py` lines 4863–4890, `mtp_run.py`): for every row i a token v_i is drawn from the row's logits
with the request's own sampler (`sample_token`: argmax at temperature 0, otherwise temperature/top-p sampling with the
request's generator); drafts are accepted while v_i == d_i; the emitted tokens are d_1 … d_acc followed by v_acc.

* **Greedy (temperature 0):** v_i = argmax of the full model's logits for the correct prefix, so every emitted
  token is exactly the token the full model would have selected — the accepted sequence is the greedy sequence of
  the full model by construction (the only caveat is §1.4).
* **Sampling:** this is the Leviathan/Chen speculative-sampling rule specialised to a deterministic draft. With a
  point-mass draft q = δ(d_i), the acceptance probability min(1, p(d_i)/q(d_i)) = p(d_i) is exactly the probability
  that an independent sample v_i ~ p equals d_i, and the residual distribution norm(max(0, p − q)) = p restricted to
  x ≠ d_i is exactly the law of v_i conditioned on v_i ≠ d_i — which is what emitting v_i on rejection produces.
  So the emitted sequence is an exact sample from the target model's distribution; no heuristic threshold is
  involved anywhere (the confidence head is not used for acceptance).
* **Rollback:** free by construction. Rejected rows leave their kv in the ring/caches, but the bookkeeping
  (`p_last = p + 1 + acc`, `written_max = p + K`, rows' `plim = pos`) makes every position > p_last invisible and the
  next pass overwrites those positions. No kernel runs to roll back.

### 1.4 Does greedy MTP produce *exactly* the baseline token sequence?

Only up to the runtime's own numerical reproducibility. The K-row pass runs the same weights through different
kernels than the 1-row pass (tensor-core FP8 GEMM for M = K rows instead of the M=1 GEMV, bucketed FP4 expert GEMMs
instead of the one-token kernel, K-row attention), so the fp32 accumulation order differs; and the baseline itself
is not bitwise reproducible from run to run (`results/dense_chain_report.md` §6: two baseline runs agree on 9–18 of
200 greedy tokens, first-step logits cosine 0.9966–0.9976, because of atomics in the expert combine / P2P partial
sums). Measured in §2.3: the first-step logits of the K-row pass vs the 1-row pass are within that envelope and the
argmax agrees. There are no deterministic kernels in this runtime today; making them deterministic (ordered
reductions in the expert combine and the P2P partial sums) is a separate, lossless, but throughput-costing change.

## 2. Measurements (S=1, `mtp_run`, `--ep-shards 100,100,100,84`, DSpark on cuda:1, 3 prompts × 200 tokens, all flags)

### 2.1 Draft length sweep (greedy; 3 prompts of `dsv41/batch_prompts16.txt`: offset 0 = MoE explanation, 3 = √2 proof, 6 = Rust code; 200 verify passes each, prompt 0 stops at EOS after ~105 tokens)

`results/mtp_s1_bench.jsonl`. Accepted = drafts accepted per pass; tokens/pass = accepted + 1 (the bonus token).
Verify = wall time of the K-row full-model pass (the CUDA-graph replay chain over the 4 GPUs incl. the logits copy);
draft = DSpark (one CUDA graph on cuda:1); other = host bookkeeping. Rollback cost = 0 (no kernel, §1.3).

| drafts (K rows) | tokens / pass | accepted / pass mean | p50 (per prompt) | p90 | full-reject rate | reject rate at draft position 1 / 2 / … | verify ms mean / p50 / p90 | draft ms | other ms | tok/s per prompt | tok/s token-weighted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (1) — baseline | 1.00 | – | – | – | – | – | 14.5 / 14.5 / 14.8 | 0 | 0 | 68.4 / 68.8 / 68.7 | **68.7** |
| 1 (2) | 1.90 | 0.90 | 1/1/1 | 1/1/1 | 10 % | 10 % | 21.0 / 20.9 / 22.3 | 4.1 | 0.7 | 68.3 / 72.8 / 76.8 | **73.8** |
| 2 (3) | 2.63 | 1.63 | 1/2/2 | 2/2/2 | 11 % | 11 / 26 % | 22.4 / 22.4 / 22.8 | 4.0 | 0.6 | 78.7 / 93.5 / 105.2 | **97.1** |
| 3 (4) | 3.28 | 2.28 | 1/3/3 | 3/3/3 | 13 % | 13 / 24 / 35 % | 25.1 / 25.1 / 25.7 | 4.1 | 0.7 | 79.2 / 104.8 / 121.2 | **109.6** |
| 4 (5) | 3.66 | 2.66 | 2/3/4 | 4/4/4 | 14 % | 14 / 29 / 41 / 50 % | 26.4 / 26.3 / 27.0 | 4.1 | 0.7 | 93.6 / 110.5 / 128.8 | **117.2** |
| 5 (6) | 3.80 | 2.80 | 2/3/3 | 5/5/5 | 19 % | 19 / 34 / 47 / 57 / 64 % | 28.8 / 28.3 / 30.5 | 4.1 | 1.1 | 81.8 / 112.1 / 116.9 | **111.4** |
| 6, 8 | not available (§1.2) | | | | | | | | | | |

Acceptance histograms (passes with 0, 1, …, K−1 accepted drafts), drafts = 4: prompt 0 [6, 13, 9, 6, 8], prompt 3
[34, 32, 29, 23, 82], prompt 6 [21, 23, 15, 11, 130]; drafts = 5: [13, 8, 13, 2, 2, 5], [40, 27, 23, 24, 20, 66],
[31, 32, 19, 20, 9, 89]. The distribution is bimodal: a pass either fails at the first draft (10–19 %) or runs to the
end of the block (40–65 %), so the mean is a poor summary — p50 is 3 and p90 is the block length.

**≥ 1.5 accepted tokens per full pass is achieved already**: 1.63 at 2 drafts, 2.28 at 3, 2.66 at 4, 2.80 at 5
(token-weighted over the three prompts; the hardest prompt alone gives 1.14 / 1.38 / 1.93 / 1.70). The throughput
optimum on these prompts is 4 drafts (117 tok/s token-weighted, 94–129 by prompt), because the 5th draft adds only
0.14 tokens per pass but 2.4 ms of verify time.

### 2.2 Where the verify time goes (why 1.5× tokens/pass gives 1.7× instead of 2.5×)

verify(K) − verify(1): **+6.5 ms for the second row**, then +1.4, +2.7, +1.3, +2.4 ms per further row. The jump at
K = 2 is the switch from the S=1 path to the batched path: none of the M=1 work applies (`DSV41_FP8_M1` → tensor-core
`fp8_gemm_tc` for M=K, `DSV41_FP4_M1` → the bucketed FP4 expert GEMM, the lean bookkeeping and fused indexer are
S=1-only), and the batched path has its own fixed costs (bucketing sort/scan/scatter, per-group launches). The
in-graph stamps of the S=2 decode (`results/dense_chain_bench.jsonl`, 21.6 ms/step, identical to verify(2) = 21.0 ms)
show it per layer: attention+hc 180 → 210 µs, **expert phase 74 → 160 µs**, peer wait 20 → 47 µs, layer 318 → 482 µs.
The expert phase doubles although its bytes only grow from ≤ 6 to ≤ 12 unique experts per layer (26 → 53 MB per GPU,
i.e. 20 → 40 µs of streaming at 1.3 TB/s): like the S=1 expert phase it is fixed-launch-cost bound, and the batched
path has more launches. Kineto profile of the 6-row pass (device 0, per step): `p2p_wait` 15.1 ms (idle, waiting on
the pipeline / peers), `fp4_gemm_tc8` 5.7 ms in 80 launches, `fp8_gemm_tc8` 1.8 ms in 73 launches, bucketing sort
0.5 ms, 246 elementwise copies 0.5 ms, index/scatter 0.6 ms; on the last device additionally the bf16 head at M=6
takes 1.55 ms in cuBLAS's `s16816gemm_64x64` (0.77 ms at M=1 — cuBLAS picks a tile that streams the 1.3 GB head twice).

A verify pass is therefore ≈ 1-row pass + 6.5 ms of batched-path overhead + ≈ 1.9 ms per extra row (expert bytes
and attention rows). The exact-MTP lever is not acceptance — it is making the K-row pass as lean as the 1-row pass:
one-token-group expert kernels for the ≤ K×6 routed rows (the FP4 M=1 kernel already handles one-token groups;
`DSV41_FP4_M1_MIXED` exists but was only measured at S=8), an M≤8 variant of the dense GEMV (the weights are the
bottleneck, K rows cost nothing extra in bytes), the lean bookkeeping and the fused indexer for K rows, and a proper
M≤8 head GEMM. None of that changes numerics beyond the existing kernel-selection differences.

### 2.3 Exactness: K-row pass vs 1-row pass (first verify step, identical prefix, row 0 logits; `results/mtp_s1_logits.*`)

| prompt | K rows | max abs diff vs K=1 | RMS diff | RMS of logits | cosine | argmax same | top-1 margin at K=1 |
|---|---|---|---|---|---|---|---|
| 0 | 2 / 3 / 4 / 5 / 6 | 2.13 / 2.22 / 1.88 / 2.11 / 1.94 | 0.42 / 0.37 / 0.36 / 0.43 / 0.39 | 5.76 | 0.99729 / 0.99793 / 0.99813 / 0.99727 / 0.99776 | yes ×5 | 1.25 |
| 3 | 2 / 3 / 4 / 5 / 6 | 1.88 / 1.41 / 1.38 / 1.88 / 1.63 | 0.40 / 0.29 / 0.29 / 0.37 / 0.34 | 4.68 | 0.99703 / 0.99804 / 0.99814 / 0.99750 / 0.99756 | no / no / yes / no / yes | **0.25** |
| 6 | 2 / 3 / 4 / 5 / 6 | 1.09 / 0.94 / 0.99 / 0.94 / 0.92 | 0.19 / 0.17 / 0.19 / 0.20 / 0.23 | 3.92 | 0.99882 / 0.99906 / 0.99889 / 0.99876 / 0.99828 | yes ×5 | 9.50 |

This is the same envelope as two 1-row baseline runs against each other (`results/dense_chain_report.md` §6: max abs
1.8–2.3, cosine 0.9966–0.9976). Prompt 3's first token sits 0.25 logits from the runner-up, and it flips between
baseline runs too (the baseline run and the drafts-3 run produced "The Irrationality of √2", the drafts-1/2/4 runs
"Proof that √2 is Irrational"). There is no K-specific bias; the greedy MTP sequence equals the greedy baseline
sequence to the extent the baseline equals itself, and would be bit-identical with deterministic kernels (§1.4).

## 3. Semantics-preserving kernel work on the two 42 MB streams (wq_b 32768 × 1280, wo_b 5120 × 8192)

Measured on GPU 6 (idle A100 80GB PCIe), CUDA-graph timing, weights rotated over 8 copies (DRAM-resident):

| probe | result |
|---|---|
| DRAM copy ceiling (`copy_`, 42 MB and 1 GB) | 1.57 / 1.55 TB/s (read + write) |
| the same M=1 GEMV kernel on the 632 MB head shape | 406.6 µs = **1.63 TB/s** (read-only stream, the kernel's own ceiling) |
| wq_b today (`k1u4`, 4096 blocks) | 33.4 µs = 1.26 TB/s |
| wo_b today (`k1u2`, 640 blocks) | 32.0 µs = 1.31 TB/s |
| tensor-core FP8 GEMM (`fp8_gemm_tc`) / tiled ldmatrix kernel on the same shapes | 35.2 / 34.5 µs (wq_b), 33.0 / 33.9 µs (wo_b) |

Variants tried (all bit-exact per chunk, only the accumulation grouping changes; `fp8_gemv_m1.cu`, `DEFR`):

| variant | wq_b | wo_b | sh_w13 | sh_w2 |
|---|---|---|---|---|
| current | 33.4 | 32.0 | 20.0 | 12.4 |
| KW/U sweep (`bench_gemv_m1`): best other | 36.5 (k1u2) | 32.4 (k4u4) | 20.8 (k2u2) | 13.7 (k2u4) |
| 4-warp blocks (finer tail), 1 row / warp | 33.5 (u4) | **31.6** (u2) | 19.5 | 12.2 |
| 2 rows / warp as one flat chunk range (u4) | 34.8 | 35.9 | 21.1 | 12.9 |
| 4 rows / warp (u4 / u8) | 36.9 / 35.9 | 43.2 / 50.5 | 26.4 / 32.4 | 14.4 / 17.0 |
| 8 rows / warp (u8) | 49.4 | 77.6 | 48.2 | 24.8 |

Reading: the current tiling is at its optimum for this kernel family (the 4-warp block is within 1 % on the
three large shapes, −1.3 % on wo_b). The gap to the kernel's own 1.63 TB/s streaming rate is **not** bandwidth
inefficiency inside the stream: 42 MB / 1.63 TB/s = 25.8 µs, i.e. **≈ 7 µs per kernel of fixed ramp + tail**
(block scheduling, the first DRAM round trip, the last wave draining). Two such kernels per layer → ≈ 14 µs/layer
= 0.56 ms/token is the whole remaining lossless headroom on the big streams, and recovering it needs the ramp of
one kernel to overlap the tail of the previous one. On A100 that requires either independent kernels on separate
streams (not the case: wq_b depends on q_norm(wq_a), wo_b on fq8(wo_a)) or programmatic dependent launch (Hopper
only). Weight prepacking / transposed layouts / vectorized loads are already in place (16-byte loads, k-permuted
FP8 with E8M0 scales folded once per chunk); scale fusion and intermediate reuse were the subject of the previous
report and are worth ≤ 0.15 ms/token. **Conclusion: wq_b/wo_b are within ≈ 20 % of the physical stream limit and
the remainder is not reachable losslessly on this GPU.** The multi-row and 4-warp variants stay in the source as an
experiment and are not wired in.

## 4. Profile of the rest of the S=1 critical path (owner GPU, per token)

From `results/decode_profile_S1_dense.json` (Kineto, inflated ≈ 1.2–1.5×) cross-checked with the in-graph stamps of
`bench_replica --trace` (not inflated; layer = 318 µs → 12.7 ms of the 14.7 ms token):

| item | measured | lossless lever | realistic saving |
|---|---|---|---|
| dense FP8 streams (attention 4.3 ms + shared expert 1.3 ms, Kineto) ≈ 107 µs/layer in stamps | 4.3 ms | ramp/tail overlap (not available on A100), see §3 | 0 – 0.2 ms |
| expert phase (own routed experts ∥ shared expert, stamps 2→3) | 72–77 µs/layer = 2.9–3.1 ms | fixed-launch-cost bound (`results/expert_chain_report.md`): fused chains measured at parity | 0 |
| P2P wait for peers' partials (stamps 3→4) | 19–22 µs/layer = 0.8 ms (S=1, quiet machine) | peers' chains are already overlapped; bounded by the slowest peer's expert phase | 0 – 0.2 ms |
| P2P send / signal / multicast issue | 13 + 30/40 µs/layer ≈ 0.6 ms | one fewer signal per layer (owner-side) | ≤ 0.1 ms |
| hyper-connection mixes + sinkhorn + RMSNorms + router top-k + elementwise | ≈ 3.0 ms Kineto (≈ 2.2 ms stamps) | launch fusion of the ~14 small kernels per layer into 3–4 (hc_pre+norm+quant already fused; sinkhorn+mix, gate+topk+multicast remain) | 0.3 – 0.5 ms |
| gate (`hc_sub2` 1→8 + gate/top-k 8→9) | 16.5 µs/layer = 0.66 ms | idem | included above |
| bf16 head (1.32 GB, 0.77 ms = 1.7 TB/s) | 0.77 ms | already at the stream limit; FP8 head would be lossy (excluded) | 0 |
| host launch + sync (host step − first-to-last stamp) | ≈ 1.1 ms (14.7 − 13.6) | async logits copy / pinned host buffers, overlapping the next step's `set_rows` | 0.3 – 0.5 ms |
| inter-layer / pipeline hops | 0.09 ms | – | 0 |

Sum of realistic lossless kernel/runtime savings: **0.6 – 1.3 ms/token → 14.7 → 13.4–14.1 ms → 71–75 tok/s.**

## 5. The two numbers

Both numbers are for S=1 on this 4 × A100 PCIe box with the current expert-parallel layout, greedy decoding, and
no change to weights, precision or semantics.

**1. Kernel/runtime optimizations only (no speculative decoding): 72–75 tok/s realistic** (today 68.7; from the
0.6–1.3 ms/token of lossless savings itemized in §4: small-kernel launch fusion around the hyper-connection /
router / norm kernels, host launch+sync overlap, P2P signalling). The physical bound — every weight byte streamed
once per token at 1.63 TB/s on each GPU in pipeline order, zero fixed costs — is ≈ 7.6 ms = 130 tok/s, but the
measured structure of this runtime (fixed ramp/tail per kernel, the expert phase's launch-bound floor, the pipeline
hand-offs) puts everything above ≈ 80 tok/s out of reach without changing what is computed. The two 42 MB dense
streams are within ≈ 20 % of their limit and their remaining 0.5 ms/token is not recoverable on A100 (§3).

**2. With exact MTP (DSpark drafts, exact greedy / exact speculative-sampling acceptance): measured today
110–117 tok/s token-weighted at 4–5 drafts (range 82–129 by prompt), realistic 130–150 tok/s** once the K-row verify
pass is given the S=1 path's lean kernels (§2.2: verify(5 rows) from 26.4 ms toward ≈ 19–21 ms, draft from 4.1 toward
≈ 2–3 ms; at the measured 3.66 tokens per pass that is 3.66 / (20 + 3 + 0.7) ms ≈ 150 tok/s, 130 with only the
verify path fixed). The acceptance side is already above the 1.5 tokens-per-pass question by a wide margin
(2.3–2.8 accepted drafts per pass); the throughput is bounded by the per-pass latency, not by acceptance, and is
task-dependent (code and formal text accept 3–4 drafts per pass, free-form explanation 1.4–1.9).

Recommended next step under the lossless constraint: the K-row verify path (`--mtp 4` in production instead of
`--mtp 0`, which alone is +60–70 % at S=1 on this evidence, then the lean multi-row kernels).
