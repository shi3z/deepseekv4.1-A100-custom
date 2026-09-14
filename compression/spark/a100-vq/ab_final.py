"""hc_post on its own, then all three new fusions together, with each arm warmed to its own hit=1.

Interleaved tok/s cannot be read when the arms generate different text: they evict each other's
experts and the hit rate moves more than the step time does. Each arm gets its own warm-up here.
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
import elem_fused as EF                                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
print(f"gate={FD.GATE_KERNEL} qkv={FD.FUSE_QKV} swiglu={EF.SWIGLU} merge={EF.MERGE} "
      f"rms={EF.RMSNORM} rope={EF.ROPE} hc={EF.HC_POST}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)


def arm(rms, rope, hc):
    EF.RMSNORM, EF.ROPE, EF.HC_POST = rms, rope, hc
    fd.graphs.clear(); torch.cuda.synchronize()


arm(True, True, False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0 = int(ids[-1])
d, _ = fd.draft(t0, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0], device="cuda"), d.clone()])
hsh = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hsh[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def ev(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for x, y in e:
        x.record(); fn(); y.record()
    torch.cuda.synchronize()
    return sorted(x.elapsed_time(y) for x, y in e)


def pair():
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0, pos - 1, 0.0)


print("\nhc_post alone (rms+rope already on), interleaved")
res = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    arm(True, True, on)
    v = ev(pair)
    res[on].append(v[100])
    print(f"  run{i+1} hc={int(on)}  {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
a, b = res[False], res[True]
print(f"  A mean {st.mean(a):8.3f} sd {st.pstdev(a):5.3f}   B mean {st.mean(b):8.3f} "
      f"sd {st.pstdev(b):5.3f}   {st.mean(b)-st.mean(a):+7.3f} ms "
      f"({100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %)", flush=True)

print("\nall three off vs all three on, interleaved ms/step")
res2 = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    arm(on, on, on)
    v = ev(pair)
    res2[on].append(v[100])
    print(f"  run{i+1} all={int(on)}  {v[100]:8.3f} ms", flush=True)
a, b = res2[False], res2[True]
print(f"  A mean {st.mean(a):8.3f} sd {st.pstdev(a):5.3f}   B mean {st.mean(b):8.3f} "
      f"sd {st.pstdev(b):5.3f}   {st.mean(b)-st.mean(a):+7.3f} ms "
      f"({100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %)", flush=True)

print("\nclean tok/s, each arm warmed to its own hit=1", flush=True)
out = {}
for on in (False, True, False, True):
    arm(on, on, on)
    for _ in range(3):
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
    v, h = [], []
    for _ in range(3):
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        v.append(s["decode_tok_s"]); h.append(s["expert_hit_rate"])
    out.setdefault(on, []).append(st.mean(v))
    print(f"  all={int(on)}  {st.mean(v):6.3f} tok/s sd {st.pstdev(v):5.3f}  hit {min(h):.4f}  "
          f"accept {s['accept_len_mean']:.3f}  {[round(x,2) for x in v]}", flush=True)
A, B = st.mean(out[False]), st.mean(out[True])
print(f"\n  A {A:6.3f}   B {B:6.3f}   {100*(B-A)/A:+6.2f} %")
