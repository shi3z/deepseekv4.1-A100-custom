"""Routing preservation for the bf16-storage / fp32-accumulate gate, over ja / en / code."""
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
                arena_gb=60.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
PROMPTS = [json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt")) if l.strip()][:6]
ENC = []
for p in PROMPTS:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ENC.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))
for ids in ENC[:2]:
    list(eng.generate(ids, max_tokens=32, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}  prompts {len(ENC)}", flush=True)

fd.use_graphs = False
FD.GATE_CMP = []
NT = int(os.environ.get("NTOK", "180"))
ntok = 0
for ids in ENC:
    ntok += len(list(t for b in eng.generate(ids, max_tokens=NT, temperature=0.0, ignore_eos=True)
                     for t in b))
rec = FD.GATE_CMP
FD.GATE_CMP = None
fd.use_graphs = True

per_layer = collections.defaultdict(lambda: [0, 0, 0])
mm_margin, ok_margin = [], []
score_err = 0.0
for L, a, b, marg, serr in rec:
    score_err = max(score_err, float(serr))
    for t in range(a.shape[0]):
        x, y = a[t].tolist(), b[t].tolist()
        r = per_layer[L]
        r[0] += 1
        same = set(x) == set(y)
        r[1] += same
        r[2] += x == y
        (ok_margin if same else mm_margin).append(float(marg[t]))
tot = sum(r[0] for r in per_layer.values())
ss = sum(r[1] for r in per_layer.values())
so = sum(r[2] for r in per_layer.values())
print(f"\ntokens generated {ntok}, routing decisions {tot} over {len(per_layer)} layers")
print(f"  max |score difference|        {score_err:.3e}")
print(f"  identical SET   {ss:8d}  mismatch {tot-ss:6d}  rate {100*(tot-ss)/tot:.5f} %")
print(f"  identical ORDER {so:8d}  mismatch {tot-so:6d}  rate {100*(tot-so)/tot:.5f} %")
bad = {L: r for L, r in per_layer.items() if r[1] < r[0]}
print(f"  layers with any set change: {len(bad)} of {len(per_layer)}" +
      ("  " + " ".join(f"L{L}:{r[0]-r[1]}" for L, r in sorted(bad.items())) if bad else ""))
if mm_margin:
    mm_margin.sort()
    print(f"\n  margin at mismatches: mean {st.mean(mm_margin):.3e} "
          f"p50 {mm_margin[len(mm_margin)//2]:.3e} max {mm_margin[-1]:.3e}")
ok_margin.sort()
print(f"  margin at matches:    mean {st.mean(ok_margin):.3e} "
      f"p1 {ok_margin[len(ok_margin)//100]:.3e} p50 {ok_margin[len(ok_margin)//2]:.3e}")
