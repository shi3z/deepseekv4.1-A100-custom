"""What the ~8,000 kernels of a verify step are, and what the 40 host round trips actually cost.

Two measurements the nsys timeline cannot give on its own:

* `resolve` already separates its own two halves -- `d2h_sync_s` is the blocking device->host copy
  of the routed expert ids, which is a wait for the GPU, and `host_set_s` is the numpy/dict work
  that follows it. Only the second is serial cost the GPU cannot hide.
* The graphs replay kernels with no Python attached, so op names and tensor shapes have to come
  from the eager path the graphs were captured from. Running the same `_layer_a/_resolve/_layer_b`
  sequence with `use_graphs=False` under torch.profiler gives every op its shapes and call site.
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
                arena_gb=94.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
print(f"dense_fp4={eng.config()['dense_fp4']} head={eng.config()['head_fmt']} "
      f"lut={fd.lut is not None} T_VERIFY={FD.T_VERIFY}", flush=True)
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


def one():
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0tok, pos - 1, 0.0)


for _ in range(20):
    one()
torch.cuda.synchronize()

st = m.store.stats
keys = ("d2h_sync_s", "host_set_s", "route_s")
before = {k: st.get(k, 0.0) for k in keys}
N = 100
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    one()
torch.cuda.synchronize()
wall = time.perf_counter() - t0
print(f"\n=== the 40 host round trips, {N} pairs, {wall*1000/N:.3f} ms per pair")
for k in keys:
    v = (st.get(k, 0.0) - before[k]) * 1000 / N
    print(f"  {k:14s} {v:8.3f} ms/pair  ({100*v/(wall*1000/N):5.1f} % of pair)")
d2h = (st['d2h_sync_s'] - before['d2h_sync_s']) * 1000 / N
hs = (st['host_set_s'] - before['host_set_s']) * 1000 / N
print(f"  -> d2h_sync is a WAIT for the GPU; host_set is serial work the GPU cannot hide")
print(f"  -> per layer: d2h {d2h/40*1000:7.1f} us   host_set {hs/40*1000:7.1f} us", flush=True)

# --------------------------------------------------------------- eager op anatomy
print("\n=== op anatomy of one verify step, eager (the code the graphs were captured from)",
      flush=True)
fd.use_graphs = False
m.c.len = pos
fd.step(block, pos, rows)          # warm the eager path
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity                    # noqa: E402
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0tok, pos - 1, 0.0)
    torch.cuda.synchronize()
fd.use_graphs = True
ka = prof.key_averages(group_by_input_shape=True)
rowsd = []
for e in ka:
    if e.device_time_total <= 0 and e.count == 0:
        continue
    rowsd.append((e.key, str(e.input_shapes)[:52], e.count,
                  e.device_time_total / 1e3, e.self_device_time_total / 1e3))
rowsd.sort(key=lambda r: -r[4])
print(f"{'op':34s} {'shapes':52s} {'n':>6s} {'self ms':>9s} {'us each':>9s}")
tot = sum(r[4] for r in rowsd)
for k, sh, n, _t, s in rowsd[:60]:
    print(f"{k[:34]:34s} {sh:52s} {n:6d} {s:9.3f} {s/max(n,1)*1000:9.1f}")
print(f"\ntotal self CUDA over {len(rowsd)} distinct (op, shape): {tot:.3f} ms; "
      f"launches {sum(r[2] for r in rowsd)}", flush=True)
prof.export_chrome_trace(os.path.expanduser("~/prof/eager_step.json"))
print("wrote ~/prof/eager_step.json", flush=True)
