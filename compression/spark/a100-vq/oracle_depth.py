"""How far ahead does an expert prefetch have to see, and is deeper actually better?

One process, one model load, one recorded routing, so every depth is measured under identical
conditions. Depth 0 is resident-first with no prefetch (the control); depth d starts the experts of
layers L+1..L+d while layer L runs. The routing comes from a recorded pass, not a predictor: this
is the ceiling, and no predictor can beat it.

The counters separate the two things that look the same in a latency number -- a depth that
overlaps more I/O, and a depth that simply reads more bytes.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))

import cb3_store                                              # noqa: E402
from engine import model as M                                 # noqa: E402
from engine.v41_engine import V41Engine                       # noqa: E402

STRIDE = 14454784


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
NTOK = int(os.environ.get("NTOK", "512"))
t = torch.tensor(ids[:NTOK], dtype=torch.long, device=eng.device)
print(f"# {NTOK}-token prefill, arena {eng.config()['arena_slots']} slots, "
      f"record {STRIDE/1e6:.2f} MB", flush=True)

route = {}


def tap(name, L, x):
    if name == "route_idx":
        route[L] = x.clone()


def one(depth, oracle):
    M.ORACLE_DEPTH = depth
    m.oracle = oracle if depth > 0 else None
    m._pf_done = {}
    store.io.update(demand_hits=0, demand_loads=0, spec_hits=0, spec_loads=0, dup_loads=0)
    store.io["seen"] = set()
    m.stats["layer_ms"] = {}
    ld0 = store.stats.get("load_s", 0.0)
    p = cb3_store.PROBE
    b0, busy0, n0 = disk(), (p.busy if p else 0.0), (p.n if p else 0)
    eng._reset(); m.begin_prompt()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    m.forward(t, 0, prefill=True, need_logits=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    io = dict(store.io)
    lm = m.stats.get("layer_ms", {})
    m.oracle = None
    return {"s": dt, "gb": (disk() - b0) / 1e9,
            "busy": (p.busy - busy0) if p else 0.0, "reads": (p.n - n0) if p else 0,
            "load_s": store.stats.get("load_s", 0.0) - ld0,
            "rest_ms": sum(v[0] for v in lm.values()), "moe_ms": sum(v[1] for v in lm.values()),
            **{k: v for k, v in io.items() if k != "seen"}}


def show(tag, r):
    tot = r["demand_loads"] + r["spec_loads"]
    dh, dl = r["demand_hits"], r["demand_loads"]
    print(f"  {tag:16s} {r['s']*1000:8.1f} ms  NVMe {r['gb']:6.1f} GB {r['gb']/r['s']:5.2f} GB/s  "
          f"busy {r['busy']/r['s']*100:4.0f}%  loads {tot:5d} (spec {r['spec_loads']:5d}, "
          f"dup {r['dup_loads']:4d})  resident-at-MoE {dh/max(dh+dl,1)*100:5.1f}%  "
          f"wait {r['load_s']*1000:7.0f} ms  rest {r['rest_ms']:7.0f} ms  moe {r['moe_ms']:7.0f} ms",
          flush=True)


m.tap = tap
one(0, None)
m.tap = None
print(f"# recorded routing for {len(route)} layers", flush=True)
one(0, None)                                    # settle

# ------------------------------------------------------------------ the routing itself
sets = {L: set(route[L].reshape(-1).tolist()) for L in route}
n = len(sets)
print("\n# expert-set overlap between layer L and L+d (mean over layers), and |union L+1..L+d|")
for d in (1, 2, 4):
    ov = [len(sets[L] & sets[L + d]) / len(sets[L]) for L in range(n - d)]
    un = [len(set().union(*[sets[L + i] for i in range(1, d + 1)])) for L in range(n - d)]
    print(f"  d={d}: overlap {sum(ov)/len(ov)*100:5.1f} %   |future set| {sum(un)/len(un):6.1f} "
          f"experts (one layer alone: {sum(len(s) for s in sets.values())/n:.1f})", flush=True)

print("\n# three runs per depth")
res = {}
for depth in (0, 1, 2, 4):
    rs = []
    for i in range(3):
        r = one(depth, route)
        show(f"depth={depth} run{i+1}", r)
        rs.append(r)
    res[depth] = rs

base = sum(r["s"] for r in res[0]) / 3
print("\n| depth | latency | vs depth0 | NVMe GB/s | busy% | bytes read | spec% | dup | resident-at-MoE | MoE wait |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d, rs in res.items():
    s = sum(r["s"] for r in rs) / 3
    gb = sum(r["gb"] for r in rs) / 3
    busy = sum(r["busy"] for r in rs) / 3
    tot = sum(r["demand_loads"] + r["spec_loads"] for r in rs) / 3
    sp = sum(r["spec_loads"] for r in rs) / 3
    dup = sum(r["dup_loads"] for r in rs) / 3
    dh = sum(r["demand_hits"] for r in rs) / 3
    dl = sum(r["demand_loads"] for r in rs) / 3
    w = sum(r["load_s"] for r in rs) / 3
    print(f"| {d} | {s:.2f} s | {(s-base)/base*100:+.1f} % | {gb/s:.2f} | {busy/s*100:.0f} % | "
          f"{gb:.1f} GB | {sp/max(tot,1)*100:.0f} % | {dup:.0f} | {dh/max(dh+dl,1)*100:.1f} % | "
          f"{w*1000:.0f} ms |")
