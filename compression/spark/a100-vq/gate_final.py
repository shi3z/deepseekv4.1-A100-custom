"""MTP routing preservation, and an end-to-end where each arm gets its own warm arena.

The interleaved end-to-end was confounded: the two arms generate different text, so they evict
each other's experts and the hit rate (0.9905 vs 0.9940) moved more than the step time did. Here
each arm is warmed to its own steady state before it is measured, and the hit rate is printed so
a residual difference is visible rather than folded into tok/s.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
PROMPTS = [json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt")) if l.strip()][:4]
ENC = []
for p in PROMPTS:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ENC.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))

# ---- MTP + main routing, both recorded through GATE_CMP
for ids in ENC[:2]:
    list(eng.generate(ids, max_tokens=32, temperature=0.0, ignore_eos=True))
fd.use_graphs = False
FD.GATE_CMP = []
for ids in ENC:
    list(eng.generate(ids, max_tokens=120, temperature=0.0, ignore_eos=True))
rec = FD.GATE_CMP
FD.GATE_CMP = None
fd.use_graphs = True
main = collections.defaultdict(lambda: [0, 0, 0])
mtp = collections.defaultdict(lambda: [0, 0, 0])
emax = 0.0
for L, a, b, marg, serr in rec:
    d = mtp if L >= 100 else main
    emax = max(emax, float(serr))
    for t in range(a.shape[0]):
        x, y = a[t].tolist(), b[t].tolist()
        r = d[L]
        r[0] += 1; r[1] += set(x) == set(y); r[2] += x == y
for nm, d in (("main 40 layers", main), ("MTP 3 blocks", mtp)):
    tot = sum(r[0] for r in d.values())
    ss = sum(r[1] for r in d.values())
    so = sum(r[2] for r in d.values())
    if not tot:
        print(f"{nm}: no records"); continue
    print(f"{nm:16s} decisions {tot:7d}  set mismatch {tot-ss:5d} ({100*(tot-ss)/tot:.5f} %)  "
          f"order mismatch {tot-so:5d} ({100*(tot-so)/tot:.5f} %)", flush=True)
print(f"  max |score difference| {emax:.3e}", flush=True)

# ---- end to end, each arm warmed on its own
ids0 = ENC[0]
out = {}
for on in (False, True, False, True):
    FD.GATE_KERNEL = on
    fd.graphs.clear(); torch.cuda.synchronize()
    for _ in range(3):
        list(eng.generate(ids0, max_tokens=300, temperature=0.0, ignore_eos=True))
    v, h = [], []
    for _ in range(3):
        list(eng.generate(ids0, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        v.append(s["decode_tok_s"]); h.append(s["expert_hit_rate"])
    out.setdefault(on, []).append((st.mean(v), st.pstdev(v), min(h), s["accept_len_mean"]))
    print(f"  GATE_KERNEL={int(on)}  {st.mean(v):6.3f} tok/s sd {st.pstdev(v):5.3f}  "
          f"hit {min(h):.4f}  accept {s['accept_len_mean']:.3f}  {[round(x,2) for x in v]}",
          flush=True)
a = [x[0] for x in out[False]]
b = [x[0] for x in out[True]]
print(f"\n  A {st.mean(a):6.3f}   B {st.mean(b):6.3f}   "
      f"{100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %  (compare only if hit matches)")
