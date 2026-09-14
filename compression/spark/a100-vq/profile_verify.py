"""Profile one verify step of the current best configuration. No code changes, measurement only.

The fast path is CUDA-graphed, so a kernel-level profiler must be told to look inside the graphs
(`--cuda-graph-trace=node`). NVTX ranges mark the verify step, the draft, and -- inside the step --
each host slot resolve, because with the device slot LUT off (it needs every routed expert
resident, which a 36%-of-experts arena is not) the step is 40 graph replays interleaved with 40
host round trips, and the host time between replays is exactly what a GPU-only summary misses.
"""
from __future__ import annotations

import os
import sys
import time

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
c = eng.config()
fd, m = eng.fast, eng.model
print(f"dense_fp4={c['dense_fp4']} head={c['head_fmt']} FUSED_ATTN={FD.FUSED_ATTN} "
      f"T_VERIFY={FD.T_VERIFY} lut={fd.lut is not None} graphs={fd.use_graphs} "
      f"GRAPH_SEGMENTS={FD.GRAPH_SEGMENTS} arena_slots={c['arena_slots']}", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
st = m.store.stats
print(f"hit {st['hits']/max(st['hits']+st['misses'],1):.4f}", flush=True)

pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}

# host-side cost of the 40 slot resolves, counted without a single sync
orig_resolve = fd._resolve
resolve_s = [0.0]
resolve_n = [0]


def timed_resolve(L):
    t = time.perf_counter()
    torch.cuda.nvtx.range_push(f"resolve{L}")
    orig_resolve(L)
    torch.cuda.nvtx.range_pop()
    resolve_s[0] += time.perf_counter() - t
    resolve_n[0] += 1


fd._resolve = timed_resolve


def one():
    m.c.len = pos
    torch.cuda.nvtx.range_push("verify")
    fd.step(block, pos, rows)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("draft")
    fd.draft(t0tok, pos - 1, 0.0)
    torch.cuda.nvtx.range_pop()


for _ in range(20):
    one()
torch.cuda.synchronize()

N = int(os.environ.get("PROF_STEPS", "50"))
resolve_s[0] = 0.0; resolve_n[0] = 0
torch.cuda.profiler.start()
t0 = time.perf_counter()
for _ in range(N):
    one()
torch.cuda.synchronize()
wall = time.perf_counter() - t0
torch.cuda.profiler.stop()
print(f"\n{N} (verify+draft) pairs in {wall*1000:.2f} ms -> {wall*1000/N:.3f} ms per pair",
      flush=True)
print(f"host slot resolve: {resolve_s[0]*1000:.2f} ms total, {resolve_n[0]} calls, "
      f"{resolve_s[0]*1000/N:.3f} ms per step ({100*resolve_s[0]/wall:.1f} % of wall)", flush=True)
