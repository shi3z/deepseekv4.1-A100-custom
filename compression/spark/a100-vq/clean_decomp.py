"""The same step with no profiler attached, to size the profiler's own dilation.

nsys with --cuda-graph-trace=node instruments every node of every graph; the pair it reports is
10 ms longer than the one the A/B harness measures, so the timeline's proportions need an
undilated anchor before any of its absolute numbers go in a report.
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
                keep_free_gb=12.0, prune_keep=None, expert_format="cb3")
fd, m = eng.fast, eng.model
c = eng.config()
print(f"dense_fp4={c['dense_fp4']} head={c['head_fmt']} lut={fd.lut is not None} "
      f"T_VERIFY={FD.T_VERIFY} arena_slots={c['arena_slots']}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}

orig = fd._resolve
acc = {"resolve": 0.0, "n": 0, "copy": 0.0}
store_resolve = m.store.resolve


def timed_resolve(L):
    t = time.perf_counter()
    idx = fd.route_idx
    t1 = time.perf_counter()
    slots = store_resolve(L, idx, False)
    t2 = time.perf_counter()
    fd.slots.copy_(slots)
    t3 = time.perf_counter()
    acc["resolve"] += t2 - t1
    acc["copy"] += t3 - t2
    acc["n"] += 1


fd._resolve = timed_resolve


def one():
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0tok, pos - 1, 0.0)


for _ in range(20):
    one()
torch.cuda.synchronize()
N = 100
acc.update(resolve=0.0, n=0, copy=0.0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    one()
torch.cuda.synchronize()
wall = time.perf_counter() - t0
print(f"\nno profiler, {N} pairs: {wall*1000/N:.3f} ms per (verify+draft)")
print(f"  store.resolve (python) {acc['resolve']*1000/N:7.3f} ms/pair "
      f"({100*acc['resolve']/wall:5.1f} %)  {acc['n']/N:.0f} calls/pair")
print(f"  slots H2D copy         {acc['copy']*1000/N:7.3f} ms/pair "
      f"({100*acc['copy']/wall:5.1f} %)")

# and the two halves separately, with events
def ev(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted(x.elapsed_time(y) for x, y in e)


def vstep():
    m.c.len = pos
    fd.step(block, pos, rows)


sv = ev(vstep)
dv = ev(lambda: fd.draft(t0tok, pos - 1, 0.0), 100)
print(f"  verify (cuda events)   {sv[100]:7.3f} ms   draft {dv[50]:6.3f} ms   "
      f"sum {sv[100]+dv[50]:7.3f} ms")
