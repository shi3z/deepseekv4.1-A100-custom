"""How often does a bf16 gate GEMM pick a different top-6 from the same activation?"""
from __future__ import annotations

import collections
import os
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
                arena_gb=60.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)

fd.use_graphs = False
FD.GATE_CMP = []
list(eng.generate(ids, max_tokens=int(os.environ.get("NTOK", "200")),
                  temperature=0.0, ignore_eos=True))
rec = FD.GATE_CMP
FD.GATE_CMP = None
fd.use_graphs = True

per_layer = collections.defaultdict(lambda: [0, 0, 0])   # L -> [n, set_same, order_same]
for L, a, b in rec:
    for t in range(a.shape[0]):
        x, y = a[t].tolist(), b[t].tolist()
        r = per_layer[L]
        r[0] += 1
        r[1] += set(x) == set(y)
        r[2] += x == y
tot = sum(r[0] for r in per_layer.values())
ss = sum(r[1] for r in per_layer.values())
so = sum(r[2] for r in per_layer.values())
print(f"\nsame activation, two gate dtypes: {tot} top-6 picks over {len(per_layer)} layers")
print(f"  identical SET   {ss:7d} ({100*ss/tot:6.2f} %)   changed {100*(1-ss/tot):5.2f} %")
print(f"  identical ORDER {so:7d} ({100*so/tot:6.2f} %)")
L0 = per_layer[0]
print(f"  layer 0 (inputs bit-identical): set same {100*L0[1]/L0[0]:6.2f} %  "
      f"changed {100*(1-L0[1]/L0[0]):5.2f} %   n={L0[0]}")
worst = sorted(per_layer.items(), key=lambda kv: kv[1][1] / kv[1][0])[:8]
print("  worst layers (set-change rate): " +
      " ".join(f"L{L}:{100*(1-r[1]/r[0]):.0f}%" for L, r in worst))
