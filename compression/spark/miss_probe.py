"""Where do depth=1's residual demand misses come from?"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))

from engine import model as M                                 # noqa: E402
from engine.v41_engine import V41Engine                       # noqa: E402


def disk():
    for ln in open("/proc/diskstats"):
        f = ln.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0


md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False, arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")),
                keep_free_gb=12.0, prune_keep=None,
                io_threads=int(os.environ.get("IO_THREADS", "12")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
m, store = eng.model, eng.model.store
ids = []
for line in open(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl")):
    ids += eng.tokenizer.encode(json.loads(line)["text"], add_special_tokens=False)
t = torch.tensor(ids[:int(os.environ.get("NTOK", "512"))], dtype=torch.long, device=eng.device)

route = {}
m.tap = lambda name, L, x: route.__setitem__(L, x.clone()) if name == "route_idx" else None
eng._reset(); m.begin_prompt(); m.forward(t, 0, prefill=True, need_logits=False)
m.tap = None
M.ORACLE_DEPTH = 1
print(f"# recorded routing for {len(route)} layers", flush=True)

for run in range(3):
    m.oracle = route
    m._pf_done = {}
    store.io.update(demand_hits=0, demand_loads=0, spec_hits=0, spec_loads=0, dup_loads=0)
    store.io["seen"] = set()
    store.miss_why = {}
    if store.trace is not None:
        store.trace.clear()
    eng._reset(); m.begin_prompt()
    b0 = disk(); torch.cuda.synchronize(); t0 = time.perf_counter()
    m.forward(t, 0, prefill=True, need_logits=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    m.oracle = None
    io = store.io
    dh, dl = io["demand_hits"], io["demand_loads"]
    print(f"run{run+1}: {dt*1000:8.1f} ms  NVMe {(disk()-b0)/1e9:5.1f} GB  "
          f"demand miss {dl} of {dh+dl} ({dl/max(dh+dl,1)*100:.2f} %)  spec {io['spec_loads']}  "
          f"dup {io['dup_loads']}", flush=True)
    tot = sum(store.miss_why.values()) or 1
    for k in sorted(store.miss_why, key=lambda x: -store.miss_why[x]):
        print(f"    {k:20s} {store.miss_why[k]:5d}  {store.miss_why[k]/tot*100:5.1f} %", flush=True)
    # which layers do the misses land in?
    if store.trace:
        byl = {}
        for (L, _e), r in store.trace.items():
            if r.get("reloaded"):
                byl[L] = byl.get(L, 0) + r["reloaded"]
        top = sorted(byl.items(), key=lambda kv: -kv[1])[:6]
        print("    misses by layer (top 6): " + ", ".join(f"L{L}:{n}" for L, n in top), flush=True)
