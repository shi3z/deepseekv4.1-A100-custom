# Decode-path fusions on one DGX Spark: 18.49 -> 20.15 tok/s

Everything here is measured on the box at the shapes the decode step really runs, in one process
with one arena, arms interleaved. Numbers from a second process are not comparable on this box:
a change in the size of a resident allocation moves the 94 GB expert arena to a different offset
and that, not the change, is what a cross-process A/B measures (NOTES.md 2026-09-11).

Config held fixed throughout: `EXPERT_FORMAT=cb3 DSV41_DENSE_FP4=attn DSV41_HEAD_FMT=fp4
DSV41_FUSED_ATTN=0 DSV41_BLOCK=1`, arena 94 GB (6,503 of 15,360 routed experts resident).

## What was adopted

| change | ms/step | switch | evidence it is safe |
|---|---|---|---|
| `wq_a` + `wkv` as one projection | -2.23 % | `DSV41_FUSE_QKV` | kernel max\|d\| = 0; 300 generated tokens identical |
| per-shape decode tile for fp4_linear | -0.99 % | weight `.tile` attribute | 300 generated tokens identical |
| router gate: bf16 storage, fp32 accumulate | -0.97 % | `DSV41_GATE_KERNEL` | 0 top-6 changes in 71,044 routing decisions |
| SwiGLU chain in one kernel | -0.37 % | `DSV41_FUSE_SWIGLU` | bit-identical, T = 1..8 |
| routed + shared merge in one kernel | -0.26 % | `DSV41_FUSE_MERGE` | bit-identical, T = 1..8 |
| RMSNorm in one kernel | -2.33 % | `DSV41_FUSE_RMSNORM` | 0 top-6 changes in 21,360 decisions |
| RoPE in one kernel | -1.04 % | `DSV41_FUSE_ROPE` | 4 of 8,584 calls differ by one bf16 ulp |
| `hc_post` in one kernel | -0.90 % | `DSV41_FUSE_HC` | 1 bf16 ulp |

The last three together: 87.964 -> 84.422 ms (-4.03 %), and 19.44 -> 20.15 tok/s with each arm
warmed to its own hit = 1.0.

## What was rejected, and why

* **Router gate in bf16 end to end.** 2.3 % faster and it moves 6.67 % of the layer-0 expert picks
  (0.97 % over all layers, so about one token in three routes somewhere differently). The fp32
  accumulation is not decoration.
* **`wo_a` in FP4.** +2.6 % for +1.5 % PPL, and dropping it made the end-to-end faster, not slower.
* **Shared FFN tile tuning.** The three GEMMs already run at 191-217 GB/s; 0.17 ms/step was all
  there was.
* **A standalone merge kernel on its own.** The merge is 0.8 ms/step of launches, not bytes: the
  tensors are 40 kB.

## Two measurement traps this work kept falling into

* **Byte counts.** A verify block is two token positions routing independently, so a layer reads
  9.875 distinct experts on average, not 6. Dividing kernel time by the 6-expert figure said CB3
  ran at 148 GB/s when it runs at 190 -- the microbenchmark it was supposedly missing.
* **Microbenchmarks that stay in L2.** Timing one weight 200 times reports 418 GB/s on a box whose
  DRAM peak is 234. Every shape benchmark here rotates through enough copies to miss L2.

## Files

* `a100-vq/` -- the measurement scripts, as run on the Spark.
* `engine-fusions.patch` -- the engine diff (`engine/`, `tools/`, `start.sh`).
* `elem_fused.py`, `gate_gemm.py` -- the new kernels, dropped into `work/tools/`.
* `chatui.py` -- a browser front end; serves the page and proxies `/v1/*` to the engine, which
  binds 127.0.0.1 only.

`start.sh` also got a fix: it sourced `.env` *after* reading the environment, so `DSV41_*` set on
the command line was silently overridden. Every A/B through the server before that fix measured
whatever `.env` said (`attn,wo_a` + fp8 head), including the 18.14 tok/s figure this study had been
treating as a dense-fp8 baseline.

`engine/v41_engine.py` had a second bug, unrelated to any of this: the sampled decode loop walked
`range(5)` drafts whatever `DSV41_BLOCK` said, so every request with a temperature crashed at the
second drafted token under `DSV41_BLOCK=1`. Greedy decoding never reaches that path.
