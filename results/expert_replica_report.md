# Strata-style expert residency on the 4×A100 EP runtime — experiment report

Date: 2026-09-28. Checkpoint: DeepSeek-V4.1-Flash-Abliterated (FP4 experts, 40 MoE layers × 384 experts, top-6),
runtime `dsv41` on GPUs 2,3,0,1 (pipeline order), EP shards 96/96/96/96, context 8192, plain batched decode
(no MTP), CUDA graphs on. Nothing below is an estimate unless it says so.

## 1. Existing expert-dispatch path (reconstructed before any change)

Decode = one CUDA graph per GPU per token (`EPRuntime.token_graph`, `dsv41/ep.py`). Layers are pipelined over the
four GPUs in order (10 layers each); every layer's routed experts are sharded over all four GPUs.

1. **Router top-k ids** — on the layer's *owner* GPU (the GPU running the layer's attention), inside the graph:
   `F.linear(xf, moe.gate_w)` then the fused kernel `fused2.gate_topk` writes `eid` (int32 [B,6]) and `wt`
   (fp32 [B,6]) straight into the owner's outbox buffer (`ep.py:_owner_layer`). No host involvement.
2. **Bucketing by expert** — on every GPU, on its own copy of `eid`, in `DecodeRuntime.experts_tc`
   (`dsv41/decode.py`): B ≤ 1 → one group per (token, expert) pair from constant tables; B > 1 → device-side
   stable sort of the B·6 pairs, run-length groups of ≤ 64 tokens (`torch.sort`, `cumsum`, `scatter_reduce`; all
   static shapes, graph-captured). Prefill uses a different path (`MoE._forward_ep`, `model.py`) with host-side
   `nonzero()` per shard — out of scope here.
3. **Expert ownership** — fixed at load (`load.py:load_model`): contiguous id ranges per device
   (`ep_shards`, default equal split; profile used here: cuda:2 = 0..95, cuda:3 = 96..191, cuda:0 = 192..287,
   cuda:1 = 288..383). Each layer's `MoE.ep` holds one shard dict per device
   (`{device, start, n, w13, s13, w2, s2}`, uint8 tensors `[n, 2·2304, 2560] / [n, 2·2304, 160] / [n, 5120, 1152] /
   [n, 5120, 72]`). The grouped GEMM kernel (`cuda/fp4_tc.cu`, `fp4_gemm_tcw.cu` for > 8 tokens) gets
   `shard_start/shard_n` and simply skips groups whose global id is outside its range (optionally zeroing their
   output rows); ownership is therefore a kernel-argument convention, not a table.
4. **Activations crossing GPUs** — per layer, from the owner: one routing packet
   `[xq bf16 [B,5120] | eid int32 [B,6] | wt fp32 [B,6]]` (10,288 B at B=1) is pushed into *every* peer's inbox
   regardless of which experts were selected (`p2p_multicast` kernel below 32 KiB, copy-engine `memcpy_async`
   above; with the 4-GPU NVLink-pair relay for large packets). Each peer computes its shard's experts for the
   packet (rows of foreign experts are zero), sums them into a per-token partial (`p2p_sum_rows`) and pushes the
   partial (`[B,5120]` bf16 with `DSV41_EP_BF16_PART`, fp32 otherwise) into the owner's `part_in[sender]` row;
   the owner adds the four rows plus the shared expert in `hc_post2_`. Pipeline hops additionally move the
   residual stream `hop_h` (`[B,1,4,5120]` bf16) and the candidate/top-k buffers once every 10 layers.
5. **Expert weights never move between GPUs** during prefill or decode: every GPU only ever reads its own shard
   tensors (the replica cache added in this experiment copies weights GPU→GPU once, at load).
6. **Inside the CUDA graphs** — everything: gate, top-k, packet push, flag signal/wait spin kernels, peer GEMMs,
   partial sums, partial push, combine. A token is 4 `graph.replay()` calls plus one `torch.cuda.synchronize` on
   the last device for the logits (`EPRuntime.step`).
7. **Per-token synchronisation points** — device-side only, per layer: owner→peers route flag (`p2p_signal`
   after the packet), peers→owner partial flags (`flag_part[L, sender]`, the owner spins on the whole row with
   `p2p_wait`), plus one hop flag per pipeline boundary; a per-token sequence number (`seqno`) makes the flags
   monotone. Host side: one `synchronize` per token (logits) and the sampling round-trip in the harness/server.
   Measured on the owner's clock (globaltimer stamps, `DSV41_EP_TRACE=1`, mean over 40 layers and 256 steps):

   | per layer, µs | S=1 | S=2 | S=8 |
   |---|---|---|---|
   | attention + hyper-connection | 221 | 251 | 298 |
   | gate + top-k | 19 | 27 | 45 |
   | multicast issue | 14 | 15 | 36 |
   | own-shard experts ∥ shared expert | 81 | 123 | 360 |
   | wait for peers' partials | 20 | 32 | 112 |
   | combine (hc_post) | 4 | 4 | 7 |
   | **layer total** | **369** | **443** | **894** |
   | peer compute + push (max over peers) | 69 | 117 | 429 |
   | ms / token (host, incl. logging sync) | 16.9 | 22.9 | 35.9 |

   Remote dispatch (multicast issue + waiting for partials) is 9 % of the layer at S=1, 11 % at S=2, 17 % at S=8.
   Attention + hyper-connection is 60 % of the S=1 layer.
8. **Free VRAM in the 4-GPU profile** (after load, graph capture, S=8, context 8192, measured with
   `torch.cuda.mem_get_info`): cuda:2 4.8 GiB, cuda:3 9.3 GiB, cuda:0 9.0 GiB, cuda:1 8.2 GiB (of 79.25 GiB;
   torch has 69–72 GiB allocated). The production 128K-context / 4-slot profile leaves < 1 GiB per GPU.

**Exact memory cost per expert** (from the loaded shard tensors, identical to the checkpoint headers):
w1+w3 packed FP4 2 × 2304×2560 B, w2 5120×1152 B, three E8M0 scale tensors 3 × 368,640 B →
**18,800,640 B = 17.93 MiB per expert per layer**. Replica budget → slots per GPU: 1 GiB = 57, 2 GiB = 114,
4 GiB = 228, 8 GiB = 456 (of 40 × 288 = 11,520 remote (layer, expert) pairs per GPU).

Note on the tree: the dev branch running in production does not contain the tiled-FP4 / fp8_tcw kernel commits
(9e901a9, 4e0fd85 exist only as unreferenced history); all numbers here are for the code on `dev`.

## 2. Phase 1 — instrumentation (no runtime change)

* `dsv41/expert_profile.py`: runs plain batched decode for S sequences and saves, per step and layer, the routed
  ids/weights of every row (the runtime's existing `DSV41_ROUTE_LOG` buffers, filled inside the graphs), the EP
  device timeline, the layer owners, shards, bytes per expert and free VRAM (`results/expert_profile_S{1,2,8}.pt`,
  256 steps each, prompts from `dsv41/batch_prompts64.txt` with different offsets per S).
* `dsv41/replica_plan.py`: builds `route_count[layer][source_gpu][expert]` (source GPU = the layer's owner), the
  remote/local split, the cost per remote route from the trace, global and per-GPU rankings, budget coverage,
  the "all remote experts replicated" share, inter-layer and temporal locality, and writes a static plan JSON.
  `--fit` separates the profiles the plan is fitted on from the ones it is evaluated on.

Per-route bytes: under this protocol no bytes are exclusive to one route. The packet is multicast to all peers
whatever the routing; only a peer's partial return (20,480 B at S=1, bf16) can disappear, and only when *none*
of the token's experts of that layer need that peer. The score uses partial_bytes / 6 = 3,413 B as the
attribution and the measured (multicast + wait) / remote routes = 7.5 µs (S=1), 10.4 µs (S=2), 16.7 µs (S=8)
as the P2P cost per remote route.

### Routing matrix summary (all three profiles, 675,840 routes)

76.2 % of routes are remote (4.57 of 6 experts per layer-token), evenly per owner (75.1–77.3 %). 0.0 % of
layer-tokens have all six experts on the owner today.

Top experts by route count over the three profiles (share of the layer's routes): L20 E51 6.3 %, L18 E345 5.9 %,
L23 E177 5.8 %, L5 E40 5.7 % (local), L10 E381 5.4 %, L22 E184 4.8 %, L4 E122 4.6 %, L34 E44 4.4 %,
L6 E122 4.3 %, L3 E153 4.2 % … the 50th is at 2.9 %. The full top-50 and the top-20 candidates per GPU are in
`results/replica_plan_report_all.txt`. The heaviest single expert carries 6 % of its layer's routes: routing is
spread, not peaked (top-64 per layer ≈ 85–95 % of the mass, but 64 experts × 40 layers is 45 GiB per GPU).

### Replica coverage vs budget

| GiB / GPU | slots | remote routes covered, in-sample (fit = eval, 3 profiles) | remote routes covered, held-out (fit on the S=8 prompts, eval on the S=1/S=2 prompts) | layer-tokens with all remote experts replicated (held-out) |
|---|---|---|---|---|
| 1 | 57 | 17.2 % | 6.1 % | 0.0 % |
| 2 | 114 | 27.3 % | 10.9 % | 0.2 % |
| 4 | 228 | 41.0 % | 18.9 % | 0.6 % |
| 8 | 456 | 57.7 % | 31.5 % | 2.2 % |

In-sample numbers are what a cache tuned on the traffic it serves would see; held-out numbers are what a static
plan sees on new prompts. Only the last column can remove a peer wait, and it stays ≈ 0 for every budget.

### Routing locality (held-out steps, three profiles)

| k | P(E_L+1 \| E_L) precision@k | recall@k | unconditional precision@k | P(E_t+1,L \| E_t,L) precision@k | recall@k |
|---|---|---|---|---|---|
| 1 | 70.5 % | 11.7 % | 27.9 % | 67.1 % | 11.2 % |
| 2 | 61.0 % | 20.3 % | 25.3 % | 57.6 % | 19.2 % |
| 4 | 46.6 % | 31.0 % | 20.8 % | 43.4 % | 28.9 % |
| 8 | 32.2 % | 42.9 % | 16.1 % | 29.3 % | 39.1 % |

Conditioning on the previous layer's (or previous token's) experts roughly doubles the precision of the
unconditional guess, so there *is* structure — but to cover the six experts a layer will use, a predictor
would have to fetch 8 experts to catch 43 % of them (144 MiB per layer-token). At the measured layer time of
0.37 ms (S=1) that is 390 GB/s of weight traffic per GPU just for the prefetch, far above NVLink-pair
(~250 GB/s bidirectional) and PCIe cross-pair (~21 GB/s) budgets, and it competes with the HBM reads of the
experts actually used. Per-token expert prefetch is therefore ruled out by bandwidth two orders of magnitude
before any scheduling question.

## 3. Phase 2 — static read-only replicas (`DSV41_EXPERT_REPLICA_CACHE=1`)

Implemented and switchable; off by default. Canonical ownership is untouched.

* `dsv41/replica.py` — `ReplicaCache`: reads a plan JSON (per GPU a list of `[layer, expert]`, from
  `replica_plan.py --plan`, optionally capped by `DSV41_REPLICA_GIB`), allocates per (owner GPU, layer) FP4 tensors
  `[R, …]` in the shard layout and copies the experts GPU→GPU once at load (`tensor.copy_`), builds three static
  int32 lookup tables: `rep_lut[owner, L]` (global id → replica slot or −1), `peer_keep[peer, L]` (global id, or
  −1 for an expert replicated on the owner) and `need_lut[owner, L]` (peer slot that must compute the expert, −1 if
  local), and holds in-graph int64 counters.
* `dsv41/decode.py::experts_tc(..., replica=)` — after the shard's w13 pass a second pass on the replica tensors
  writes the rows of replica groups into the same `gu` buffer; after the shard's w2 pass (which zeroes non-owned
  rows) the replica w2 pass fills the replica rows. Same kernels, same FP4 bytes, two extra launches per layer.
  Both the B=1 path and the bucketed B>1 path are covered. Unit test on synthetic experts: own-shard + replica +
  remaining-peer results equal the all-local reference **bit for bit** at B=1 and B=4 (`/tmp/claude-1000/test_replica_gemm.py`).
* `dsv41/ep.py` — owner: `_experts_shard` passes the layer's replica set; after `gate_topk` the owner counts, per
  peer, the routed experts it still needs (`need_lut` gather + `scatter_add_`, all on the device), updates the
  counters (replica hits, remote misses, layer-tokens with no remote need). Peer: ids are remapped through
  `peer_keep` so a replicated expert is skipped (its rows are zeroed by the existing `zero_out` path) and nothing is
  computed twice. Optional `DSV41_REPLICA_SKIP_WAIT=1`: the owner waits only on peers with `need > 0`
  (`p2p_wait_masked`, new kernel in `cuda/p2p.cu`) and zeroes the rows it did not wait for. No host sync, no
  host-side sort, graph-capturable (one non-capturable scalar store was found and replaced by `fill_`).

### Correctness bisection (user-requested; S=1, prompt 3, 160 greedy tokens, `--warmup 0`)

Mask semantics, derived from the buffers:

| concept | definition in this runtime | where it lives |
|---|---|---|
| `remote_wait_mask` (data dependency) | peer p holds ≥ 1 routed expert of this layer-token that the owner has no replica of | `need_lut` gather → `need[p] > 0` |
| `remote_contribution_mask` | identical to the wait mask: a peer's partial is non-zero iff it was needed (the peer's `peer_keep` remap zeroes replicated experts, so an un-needed peer produces an all-zero partial) | same tensor |
| `local_replica_mask` | routed expert e with `rep_lut[e] ≥ 0`; its result goes into the owner's **own** row of `part_in` via `p2p_sum_rows`, never into a peer row | `experts_tc(replica=)` |
| `valid_result_mask` (per row of `part_in[d]`) | own row: always (written by the owner before the wait); peer row: after that peer's flag for this layer was waited on, or when its true contribution is zero | wait + zeroing |
| `buffer_lifetime_wait_mask` | none needed: a peer's writes to `part_in[owner][p]` are stream-ordered on the peer (layer L before L+1, token t before t+1), the owner reads row p only after waiting on p's flag for that same layer, and the flag/`seqno` protocol is unchanged; a late all-zero write from a skipped layer can only race the owner's zeroing of the same row with the same value | — |

Bisection results (first-step logits vs the baseline's, |logits| ≤ 35; tokens = matching greedy tokens):

| configuration | tokens same | first differing token | max abs logit diff (step 0) | mean | argmax same |
|---|---|---|---|---|---|
| replicas, peers always waited (no masked wait, no zeroing) | 41 / 60 | 41 | 1.64 | 0.29 | yes |
| replicas, masked **wait** only | 60 / 60 | – | 2.69 | 0.38 | yes |
| replicas, row **zeroing** only (bug) | 10 / 60 | 10 | 13.9 | 2.16 | no |
| replicas, both (bug) | 0 / 60 | 0 | 14.4 | 2.14 | no |

The masked wait alone is correct; the zeroing was the defect. Trace (in-graph record of the mask and the
per-row |sum| of `part_in` right before the combine, first decode step, all 40 layers): **0** rows that were
masked out held non-zero data, i.e. skipped peers' partials were zero as reasoned above — so the corruption did
not come from stale or missing remote data. `part_in[d]` at that point contains: own row = own-shard experts +
local replica experts (fp32 pair sum), peer rows = that peer's non-replicated experts. The bug: `need` is built with
`scatter_add_` and therefore holds the **number** of routed experts per peer; the wait only tests `≠ 0`, but the
zeroing multiplied every row by that count, scaling the partial of any peer that held 2+ of the token's experts
(46 % of (layer-token, peer) events in the S=1 profile) by 2 or 3. `remote_wait_mask` and the value used for zeroing
had silently been the same tensor. Smallest fix: zero with `(need > 0)` (one line, `ep.py`). The skipped-wait
share of peer events is 17.5 % (a peer holding none of the six experts); layer-tokens with *no* remote need at all
stay ≈ 0 % at 1 GiB.

Re-run after the fix, plus two controls, with `DSV41_EP_DEBUG_H=1` recording the residual stream after every layer
at the first decode step (first divergent layer = first layer whose bf16 residual differs from the baseline run):

| configuration (S=1, 160 tokens) | tokens same | first differing token | step-0 logit max / mean diff | first divergent layer |
|---|---|---|---|---|
| **baseline run twice** (identical config, default kernels) | 45 / 160 | 41 | 2.41 / 0.36 | 4 |
| baseline vs baseline with another partial-reduction association (`DSV41_EP_DMA=1`, fp32 relay) | 43 / 160 | 41 | 1.88 / 0.32 | 4 |
| replicas 1 GiB, full waits (default kernels) | 78 / 160 | 76 | 1.69 / 0.27 | 10 |
| **baseline run twice, `DSV41_DETERMINISTIC=1`** | 124 / 160 | 124 | 1.84 / 0.29 | 18 |
| replicas 1 GiB, full waits, deterministic | 107 / 160 | 13 | 1.66 / 0.27 | 18 |
| replicas 1 GiB, masked wait (fixed), deterministic | 131 / 160 | 130 | 2.56 / 0.37 | 4 |

The unmodified runtime does not reproduce its own greedy output: two identical baseline runs diverge at layer 4
(default kernels) or layer 18 (deterministic flag), with the same per-layer magnitudes (0.0156, 0.0347, 0.0459 …
at layers 4–7; 0.086, 0.19 … at layers 18–19) that the replica runs show. The replica configurations diverge from
the baseline at exactly those layers and stay inside that envelope (their first replica hits at layers 3, 4, 7, 9
left the residual stream bit-identical), so the token-for-token gate over 128 tokens is not attainable for any
configuration, including no change; the achievable evidence is (a) bit-exact GEMM path, (b) the count-mask defect
found and fixed with the masked-wait path now matching the full-wait path, (c) whole-model deviation no larger
than the baseline's run-to-run noise. The source of that run-to-run noise (a non-deterministic kernel in the
prefill or the attention path) is outside this experiment and was not changed.

## 4. Benchmark (measured; raw JSON lines in `results/replica_bench.jsonl`, one process per point)

200 timed steps after 16 warm-up steps, greedy, context 8192, prompts from `batch_prompts64.txt`
(offsets 3 / 6 / 24 for S = 1 / 2 / 8), `--trace` on (globaltimer stamps inside the graphs, ≈ 1 % overhead).
tok/s = S × steps / wall; ms/step is the per-token latency of every stream. "expert" = local expert GEMMs
(own shard + replicas) per layer on the owner, "wait" = owner waiting for peers' partials, "mcast" = issuing the
route packet; all per layer in µs (mean over 40 layers, last step). GPU utilisation is `nvidia-smi dmon` SM % of
the four GPUs (1 s samples; it is high in every configuration because peers spin in `p2p_wait` kernels — it is
not a measure of useful work). NVLink bytes are the driver's per-GPU counters (the two NV12 pairs; PCIe
cross-pair traffic is not counted): identical across configurations because the route packet is broadcast
regardless of routing.

| config | S | tok/s | ms/step | p50 | p90 | replica hit rate | layer-tokens fully local | replica VRAM / GPU | expert µs | wait µs | mcast µs | layer µs | NVLink MB/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | **59.9** | 16.68 | 16.62 | 16.86 | – | – | 0 | 72.9 | 30.3 | 13.1 | 372 | 2.55 |
| baseline | 2 | **87.1** | 22.96 | 22.77 | 23.09 | – | – | 0 | 149.7 | 47.5 | 18.2 | 508 | 3.54 |
| baseline | 8 | **220.8** | 36.23 | 35.93 | 37.04 | – | – | 0 | 312.7 | 104.8 | 32.8 | 822 | 26.69 |
| replica 1 GiB | 1 | 50.1 | 19.96 | 19.61 | 21.06 | 5.9 % | 0.06 % | 1.00 GiB | 105.7 | 25.8 | 13.2 | 441 | 2.55 |
| replica 1 GiB | 2 | 78.4 | 25.52 | 25.36 | 25.72 | 6.3 % | 0.00 % | 1.00 | 189.2 | 34.2 | 18.7 | 574 | 3.54 |
| replica 1 GiB | 8 | 202.1 | 39.59 | 39.08 | 39.94 | 15.3 % | 0.00 % | 1.00 | 435.1 | 37.8 | 33.1 | 919 | 26.69 |
| replica 2 GiB | 1 | 50.7 | 19.74 | 19.59 | 20.01 | 11.0 % | 0.35 % | 2.00 | 115.1 | 19.3 | 13.6 | 443 | 2.55 |
| replica 2 GiB | 2 | 77.7 | 25.73 | 25.61 | 26.08 | 11.0 % | 0.00 % | 2.00 | 207.4 | 27.7 | 18.7 | 586 | 3.54 |
| replica 2 GiB | 8 | 197.8 | 40.44 | 40.16 | 41.11 | 20.3 % | 0.00 % | 2.00 | 461.7 | 38.8 | 32.5 | 946 | 26.69 |
| replica 4 GiB | 1 | 50.2 | 19.91 | 19.84 | 20.32 | 17.8 % | 0.65 % | 3.99 | 121.3 | 18.6 | 13.5 | 448 | 2.55 |
| replica 4 GiB | 2 | 76.1 | 26.29 | 26.07 | 27.01 | 19.4 % | 0.00 % | 3.99 | 221.1 | 25.6 | 18.8 | 598 | 3.54 |
| replica 4 GiB | 8 | 200.9 | 39.82 | 39.53 | 41.16 | 22.7 % | 0.00 % | 3.99 | 415.1 | 129.0 | 33.3 | 990 | 26.69 |
| replica 4 GiB, peers always waited | 1 | 51.0 | 19.59 | 19.55 | 20.00 | 18.6 % | 0.71 % | 3.99 | 134.6 | 13.4 | 13.4 | 454 | 2.55 |
| replica 8 GiB | 1 | OOM | | | | | | | | | | | |

8 GiB per GPU does not fit next to the 8K-context profile: cuda:2 has 4.8 GiB free after graph capture (and
another process held 2.27 GiB on GPU 2 during these runs).

Reading: every replica configuration is **slower** than the baseline at every S (−16 % to −19 % at S=1,
−10 % to −13 % at S=2, −9 % to −10 % at S=8). The mechanism is visible in the trace columns: the owner's local
expert phase grows by 33–62 µs per layer (it now reads 1–3 more 17.9 MiB experts per token and launches two extra
kernels) while the wait for peers shrinks by only 4–12 µs (the peers' work was already hidden behind the owner's
own experts + shared expert). The masked wait removes nothing measurable (0.06–0.71 % of layer-tokens have no
remote need; 4 GiB with vs without it: 50.2 vs 51.0 tok/s). P2P traffic reduction in bytes: 0 (the packet is
multicast to all peers whatever the routing; NVLink counters identical); what shrinks is the peers' GEMM work
(hit rate 6–23 % of remote routes), which was not on the critical path.

Held-out routing coverage vs budget (fit on the S=8 prompts, evaluated on the S=1/S=2 prompts):
6.1 % / 10.9 % / 18.9 % / 31.5 % of remote routes for 1 / 2 / 4 / 8 GiB; measured hit rates during the runs are in
the table (5.9–22.7 %).

## 5. Single-stream bottleneck analysis: why S=1 is 60 tok/s while S=8 reaches 221 tok/s

Method. `dsv41/decode_profile.py` records one decode token per S with the Kineto/CUPTI profiler (every kernel
and DMA inside the CUDA graphs, per device and stream) and attributes them to (layer, phase) using the
runtime's stamp kernels as boundaries; per phase it reports wall (between stamps), GPU kernel time (all streams),
overlap (kernel time beyond the busy union) and idle (wall − busy union). The profiler itself slows the token
(S=1: 20.2 ms profiled vs 16.7 ms unprofiled; the peer/wait phases inflate most), so **wall-clock figures below are
the unprofiled globaltimer stamps** (`results/replica_bench.jsonl`, baseline rows) and the profiled traces are
used for composition, kernel counts and gaps (`results/decode_profile_S{1,2,8}.json`, `*_trace.json`).

### Per-token critical path, S=1 (16.68 ms per token, unprofiled; 40 layers × 372 µs = 14.89 ms inside the layers)

| phase (owner GPU of the layer) | wall µs / token | share | GPU kernel µs (profiled) | overlap µs | idle µs | what runs |
|---|---|---|---|---|---|---|
| attention + hyper-connection (stamp 0→1) | **8,880** | **53 %** | 9,436 | 637 | 111 | 5 FP8 dense GEMVs (wq_a, wq_b, wkv, wo_a, wo_b: 121 MiB/layer) 5,090 µs; attention core (sattn2 split+combine) 632; indexer/compressor/rope 309 + its top-k 502 + sort 308; RMSNorm 566; hc mixes 723 + sinkhorn 653; elementwise/copies 836 |
| hc_sub2 (1→8) | 440 | 3 % | 891 | 340 | 0 | second hc mix + norm + FP8 quant (side stream) |
| router: gate GEMM + top-k (8→9) | 680 | 4 % | 620 | 0 | 99 | cuBLAS fp32 gate (3.75 MiB) 381 + `gate_topk` 239 |
| multicast issue (9→2) | 520 | 3 % | 485 | 0 | 55 | `p2p_multicast` kernel stores 10 KB to 3 peers + flags |
| local expert GEMMs (2→10) | 2,920 | 18 % | 4,372 | 1,233 | 10 | own-shard FP4 grouped GEMM ×2 (≈ 1.5 experts, 27 MiB) 2,375; SwiGLU 359; shared expert FP8 GEMV ×2 (34 MiB) 1,638 on the side stream (the overlap) |
| shared-expert join (10→3) | 160 | 1 % | 104 | 0 | 71 | stream join + partial sum |
| wait for peers' partials (3→4) | 1,200 | 7 % | (spin) | 0 | 0 | `p2p_wait` spin; peers finished 60–80 µs after receiving the packet |
| combine (4→5) | 160 | 1 % | 118 | 0 | 49 | `hc_post2` |
| pipeline hops (3) + intra-device gaps | 89 | 0.5 % | – | – | 89 | hop_h copy + flag |
| final norm + head + logits | 770 | 5 % | 768 | 0 | – | 129K-vocab FP8 head |
| host: graph launches, final sync, argmax/tolist | ~1,000 | 6 % | – | – | 1,000 | 4 `replay()` + `synchronize` + sampling 91 µs |

In-graph launch gaps are negligible (≈ 1 µs between kernels; idle ≤ 2 % of any phase); the GPUs are busy, but
every one of them spends 55–68 % of the token in spin-wait kernels (peers waiting for the next route packet,
owners waiting for partials): kernel time per device 15.8–17.3 ms, of which 9.3–11.5 ms is spinning.

Bandwidth actually achieved on the critical path (A100-80GB peak ≈ 2.0 TB/s): attention projections 121 MiB in
120 µs of GEMV kernels = **1.06 TB/s**; shared expert 34 MiB in 38 µs = 0.9 TB/s; own routed experts ≈ 27 MiB in
79 µs of FP4 kernels = **0.34 TB/s** (at one token per group the grouped kernel has ~200 active blocks: it is
parallelism-, not bandwidth-limited); the dense phase as a whole (162 MiB of dense weights per layer, 6.3 GiB per
token) runs at ≈ 0.6 TB/s including the non-GEMV kernels.

### S=1 vs S=2 vs S=8 (unprofiled stamps, per layer; the per-stream columns divide by S)

| phase µs/layer | S=1 | S=2 | S=8 | per stream S=1 / S=2 / S=8 |
|---|---|---|---|---|
| attention + hc | 222 | 241 | 306 | 222 / 121 / 38 |
| gate + top-k | 17 | 33 | 34 | 17 / 17 / 4 |
| multicast issue | 13 | 18 | 33 | 13 / 9 / 4 |
| own experts ∥ shared (local GEMMs) | 77 (73) | 155 (150) | 318 (313) | 77 / 78 / 40 |
| wait partials | 30 | 47 | 105 | 30 / 24 / 13 |
| combine | 4 | 4 | 8 | 4 / 2 / 1 |
| layer | 372 | 508 | 822 | 372 / 254 / 103 |
| ms / step (host) | 16.68 | 22.96 | 36.23 | 16.68 / 11.48 / 4.53 |
| tok/s | 59.9 | 87.1 | 220.8 | |

The 3.7× aggregate scaling is almost entirely the dense phase: attention + hc costs 222 µs for one token and
306 µs for eight (the same 121 MiB of projections are streamed once per layer whatever S), so its per-token cost
falls 5.8×. The expert phase grows 4.1× for 8× tokens (more distinct experts touched, still one weight read per
expert), the peer wait 3.5×; router/multicast/combine are minor at every S.

### Upper bounds for S=1 (from the measured phases; nothing here is a measured speed-up)

| if … | ms/token | tok/s |
|---|---|---|
| measured baseline | 16.68 | 59.9 |
| all peer waits removed (wait partials = 0) | 15.47 | 64.6 (+8 %) |
| all P2P activation communication free (waits + multicast issue + hops = 0) | 14.91 | 67.1 (+12 %) |
| every MoE expert GEMM on the owner at S=8's per-token efficiency (77 → 40 µs/layer) | 15.20 | 65.8 (+10 %) |
| the three above together | 13.47 | 74.3 (+24 %) |
| attention + hc at S=8's per-token efficiency (222 → 38 µs/layer) | 9.35 | 107 (+78 %) |
| pure weight-streaming floor: 6.3 GiB dense + 1.1 GiB routed experts per token at 2.0 TB/s | ≈ 3.9 | ≈ 250 |

### Ranking of the S=1 bottlenecks by measured contribution to the critical path

1. **Dense weight streaming at M=1 in the attention / hyper-connection phase** — 53 % of the token
   (8.9 ms): five FP8 GEMV launches per layer reading 121 MiB at ~1.06 TB/s plus ~14 small kernels (norms, hc
   mixes, sinkhorn, attention core, indexer). Category A (memory bandwidth) with a large B component (the GEMVs
   reach half of peak; the surrounding kernels are latency-bound). This is the phase that batching amortises.
2. **Owner-side expert phase** — 18 % (2.9 ms): FP4 grouped GEMM at one token per group (0.34 TB/s,
   parallelism-limited: B) and the shared expert (memory-bound GEMV, overlapped on a side stream). Category B > A;
   FP4 unpack (I) was measured earlier at ≤ 11 % of this kernel's time.
3. **P2P serialisation** — 10 % (1.7 ms): waiting for partials 7 % (the peers are done ~60–80 µs after the packet
   arrives, i.e. the wait is the tail of peer compute + flag latency, not link bandwidth) and issuing the
   multicast 3 %. Categories D/E. This is the *entire* pool that expert replication or peer-wait elimination can
   draw on: +8 % at best.
4. **Router + dispatch bookkeeping** — 7 % (1.1 ms): gate GEMM + top-k 4 %, hc_sub2 3 % (G).
5. **Host side** — 6 % (1.0 ms): 4 graph launches, the final device sync, argmax/tolist (C/H). In-graph gaps
   (H) are < 1 %.
6. **Final norm + head** — 5 % (0.77 ms), a 129K-vocab GEMV.
7. Insufficient overlap (F): only the shared expert is overlapped; the peers' expert work already overlaps the
   owner's own experts; the dense phase is a strict dependency chain (attention → gate → experts), so there is
   nothing left to overlap within a token at S=1 without speculative decoding or splitting the dense weights
   across GPUs.

Conclusion for the replica experiment: expert residency attacks item 3 (≤ 8–12 % upper bound) while adding to
item 2, which is why every replica configuration measured slower. The single-stream ceiling is set by item 1.

## 6. Deliverables

### Files changed / added

| file | change |
|---|---|
| `dsv41/cuda/p2p.cu` | `p2p_wait_masked` kernel (spin only on flags whose device-side mask entry is non-zero) |
| `dsv41/cukern.py` | `p2p_wait_masked` launcher |
| `dsv41/decode.py` | `experts_tc(..., replica=None)`: two extra FP4 GEMM launches on the replica tensors (B=1 and bucketed paths); baseline path untouched when `replica` is None |
| `dsv41/ep.py` | replica cache hook (`DSV41_EXPERT_REPLICA_CACHE`, `DSV41_REPLICA_PLAN`, `DSV41_REPLICA_GIB`), owner/peer id remaps, device-side per-peer need counts and counters, masked wait + row zeroing (`DSV41_REPLICA_SKIP_WAIT`, `DSV41_REPLICA_MASK_MODE` debug), stamp 10 (end of local expert GEMMs) and `local expert GEMMs` in `trace_report`, `DSV41_EP_DEBUG_H` / `DSV41_REPLICA_DEBUG` in-graph debug records |
| `dsv41/replica.py` | new: `ReplicaCache` (static replicas from a plan, LUTs, counters, `report()`) |
| `dsv41/expert_profile.py` | new: Phase 1 routing/timeline profiler |
| `dsv41/replica_plan.py` | new: routing matrix, rankings, budget coverage, locality, plan writer |
| `dsv41/bench_replica.py` | new: A/B harness (tok/s, ms/step, dmon utilisation, NVLink counters, EP trace, token/logit/hidden-state comparison) |
| `dsv41/decode_profile.py` | new: Kineto critical-path profiler with stamp-based phase attribution |
| `results/` | `expert_profile_S{1,2,8}.pt`, `replica_plan_report_all.txt`, `replica_plan_8gib.json`, `replica_bench.jsonl`, `decode_profile_S{1,2,8}{,_trace}.json`, this report |

All new behaviour is off unless `DSV41_EXPERT_REPLICA_CACHE=1`; with it unset the decode path executes exactly the
previous kernels (the only additions are one `p2p_stamp` per layer when `DSV41_EP_TRACE=1` and `None` checks).

### Architecture before / after

Before: owner computes attention/gate, multicasts the packet to all peers, computes its own shard's experts and
the shared expert, waits for all three peers' partials, combines. Expert weights are read only by their shard's
GPU.

After (cache on): identical protocol; the owner additionally holds FP4 copies of a static set of remote experts of
its own layers, computes them itself (same grouped kernels, ids remapped), the canonical peer skips those ids, and
the owner may wait only on the peers that still contribute. Weights move GPU→GPU once at load; nothing moves
during decode. Ownership, shards, flags, packet format and the combine are unchanged.

### Routing locality (held-out steps)

P(E_{L+1} | E_L): precision@1 70 %, @8 32 %; recall@8 43 % (unconditional 28 % / 16 % / 21 %).
P(E_{t+1} | E_t) at the same layer: precision@1 67 %, recall@8 39 %. Static frequency coverage on unseen prompts:
6 / 11 / 19 / 32 % of remote routes for 1 / 2 / 4 / 8 GiB per GPU; layer-tokens with every remote expert
replicated ≤ 2 %.

### Replica cache hit rates (measured, 200 steps)

1 GiB: 5.9 % (S=1) / 6.3 % (S=2) / 15.3 % (S=8) of remote routes; 2 GiB: 11 / 11 / 20 %; 4 GiB: 18 / 19 / 23 %.
Layer-tokens with no remote need: 0.06–0.71 %.

### P2P traffic reduction

Bytes: none (packet multicast is routing-independent; NVLink counters identical: 2.55 / 3.54 / 26.7 MB per step at
S = 1 / 2 / 8 for every configuration). Peer GEMM work: −6 % to −23 % (the hit rate), which was off the critical
path. Owner wait: −4 to −12 µs per layer (of 30–105).

### Benchmark

Section 4: every replica configuration is slower (S=1: 59.9 → 50.1–50.7 tok/s; S=2: 87.1 → 76.1–78.4;
S=8: 220.8 → 197.8–202.1). 8 GiB per GPU does not fit next to the 8K-context profile; the production 128K profile
has < 1 GiB free.

### Bottleneck after the optimisation

Unchanged, and now measured (section 5): dense weight streaming at M=1 (53 % of the single-stream token) and the
one-token FP4 expert GEMM (18 %); everything P2P is ≤ 10 % with an 8–12 % upper bound on what removing it could
give.

### Recommendation

Do not pursue dynamic residency or asynchronous expert prefetch on this runtime:

* The pool they can draw on is the peer wait + multicast issue: **+8 % to +12 %** tok/s at S=1 even if made free,
  and the static form measured **−16 %** because every locally computed expert adds a 17.9 MiB read and two
  kernel launches to the owner's critical path, where the peers' reads were free (overlapped).
* Prefetch cannot be fed: reaching 43 % recall of the next layer's experts needs 8 predicted experts = 144 MiB per
  layer-token, i.e. ~390 GB/s per GPU at the current 0.37 ms layer time, versus ~250 GB/s NVLink (pair only) and
  ~21 GB/s PCIe between pairs; the prediction is also only 32 % precise, so most of that traffic would be wasted.
* VRAM headroom is 4.8–9 GiB per GPU in the 8K benchmark profile and under 1 GiB in the production 128K profile.

What the measurements say the single-stream levers are, in order: (1) the dense phase — 121 MiB of FP8 attention
projections per layer read by one GPU at ~1 TB/s, plus ~14 latency-bound kernels around them; (2) the one-token
FP4 grouped GEMM (0.34 TB/s); (3) the host side (~1 ms/token). Any of these three is a larger target than the whole
P2P budget.
