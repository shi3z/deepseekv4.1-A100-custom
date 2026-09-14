"""Device-side breakdown of one fully-resident decode step.

No pruning: the arena is warmed by generating the same prompt twice, which is the condition the
benchmark's warm rows measure (hit 1.0, no NVMe traffic). The step is the DSpark 6-token verify
block, replayed from its CUDA graph, so what the profiler sees is what the GPU actually runs.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))

from engine.v41_engine import V41Engine                      # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print("config:", {k: eng.config()[k] for k in ("expert_format", "arena_slots", "kernel")}, flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                          # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):                       # warm the arena to hit 1.0
    for _ in eng.generate(ids, max_tokens=120, temperature=0.0):
        pass
st = eng.model.store.stats
print(f"store: hits {st['hits']} misses {st['misses']} -> hit {st['hits']/max(st['hits']+st['misses'],1):.4f}",
      flush=True)

m, fd = eng.model, eng.fast
pos = m.c.len
tok = int(ids[-1])
d, q = fd.draft(tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def one():
    m.c.len = pos
    fd.step(block, pos, rows)


N = int(os.environ.get("STEPS", "120"))
for _ in range(10):
    one()
torch.cuda.synchronize()
if fd.rs_uniq is not None:
    fd.route_stats_reset()
    for _ in range(20):
        one()
    torch.cuda.synchronize()
    rs = fd.route_stats_report()
    if rs:
        per = rs["total"]
        MB = 14.454784
        print(f"\nrouted experts per step: {per:.1f} distinct over 40 layers "
              f"({rs['mean']:.2f} per layer) = {per*MB/1000:.2f} GB of expert payload "
              f"(w1+w3 {per*MB*2/3/1000:.2f} GB for the up kernel, w2 {per*MB/3/1000:.2f} GB for down)",
              flush=True)

# ---- whole-step wall time, CUDA events, median over N
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
for a, b in ev:
    a.record(); one(); b.record()
torch.cuda.synchronize()
ms = sorted(a.elapsed_time(b) for a, b in ev)
print(f"\nstep wall (CUDA events, {N} steps): median {ms[N//2]:.3f} ms   p95 {ms[int(N*0.95)]:.3f} ms   "
      f"min {ms[0]:.3f}   max {ms[-1]:.3f}", flush=True)

# ---- per-kernel, from the profiler, over a shorter run
from torch.profiler import ProfilerActivity, profile           # noqa: E402
NP = int(os.environ.get("PROF_STEPS", "20"))
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(NP):
        one()
    torch.cuda.synchronize()
rows_ = []
for e in prof.key_averages():
    if e.self_device_time_total > 0:
        rows_.append((e.self_device_time_total / NP, e.count / NP, e.key))
rows_.sort(reverse=True)
tot = sum(r[0] for r in rows_)
print(f"\nper-step device time: {tot:.1f} us over {len(rows_)} distinct kernels", flush=True)
print(f"{'us/step':>9} {'%':>6} {'calls':>7}  kernel")
for us, n, k in rows_[:24]:
    print(f"{us:9.1f} {us/tot*100:6.1f} {n:7.1f}  {k[:86]}")
print(f"\nGPU busy {tot/1000:.3f} ms of {ms[N//2]:.3f} ms wall  -> idle/launch gap "
      f"{ms[N//2]-tot/1000:.3f} ms ({(1-tot/1000/ms[N//2])*100:.1f} %)", flush=True)
