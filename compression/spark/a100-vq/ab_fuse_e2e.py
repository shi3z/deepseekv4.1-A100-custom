"""Settle the end-to-end number for the fused projection: five interleaved generations per arm.

The step-time A/B is already tight (sd 0.06 ms), but decode_tok_s came out with a cold first
reading in one arm and a slow middle one in the other, so the two arms are alternated here and
every reading is kept.
"""
from __future__ import annotations

import collections
import os
import statistics as st
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import FP4Weight                               # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W, fd = eng.model.W, eng.fast
for b in list(W.layers) + list(W.mtp):
    qa, kv = b.wq_a, b.wkv
    b.wqkv = FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                       torch.cat([qa.s, kv.s], 0).contiguous(), qa.N + kv.N, qa.K)
    b.wqkv_split = qa.N
print(f"dense_fp4={eng.config()['dense_fp4']} head={eng.config()['head_fmt']} "
      f"arena_slots={eng.config()['arena_slots']}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)


def set_mode(on):
    FD.FUSE_QKV = on
    fd.graphs.clear()
    torch.cuda.synchronize()


for on in (False, True):                       # warm both arms' graphs and the arena
    set_mode(on)
    for _ in range(2):
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
print(f"warm: hit {eng.model.store.hit_rate():.4f}", flush=True)

res = collections.defaultdict(list)
for i in range(5):
    for on in (False, True):
        set_mode(on)
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        res[on].append(s["decode_tok_s"])
        print(f"  round{i+1} FUSE_QKV={int(on)}  {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
o, f = res[False], res[True]
print(f"\n  off  mean {st.mean(o):6.3f} median {st.median(o):6.3f} sd {st.pstdev(o):5.3f}  {o}")
print(f"  on   mean {st.mean(f):6.3f} median {st.median(f):6.3f} sd {st.pstdev(f):5.3f}  {f}")
print(f"  fused: {100*(st.median(f)-st.median(o))/st.median(o):+6.2f} % on median tok/s")
