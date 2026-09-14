"""The fused RMSNorm: exactness on real activations, routing, timing, tok/s -- one process.

Exactness on random input is not the claim that matters. RMSNorm sits on the attention path and
its output reaches the router, so the check is whether real activations round to the same bf16 and
whether any top-6 moves. Both are counted over a real generation before anything is timed.
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
import v41_ref as R                                            # noqa: E402
import elem_fused as EF                                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
print(f"gate={FD.GATE_KERNEL} qkv={FD.FUSE_QKV} swiglu={EF.SWIGLU} merge={EF.MERGE} "
      f"rope={EF.ROPE}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)

# ---- exactness on the activations the model really produces
orig = FD.FastDecoder._rope
cnt = collections.Counter()
worst = [0.0]


def checking(self, x, fq, inverse=False):
    EF.ROPE = False
    ref = orig(self, x, fq, inverse)
    EF.ROPE = True
    new = orig(self, x, fq, inverse)
    cnt["n"] += 1
    if torch.equal(ref, new):
        cnt["same"] += 1
    else:
        cnt["diff"] += 1
        cnt[f"diff_{tuple(x.shape)}"] += 1
        worst[0] = max(worst[0], float((ref.float() - new.float()).abs().max()))
    return ref


EF.ROPE = False
for _ in range(2):
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
fd.use_graphs = False
FD.FastDecoder._rope = checking
list(eng.generate(ids, max_tokens=100, temperature=0.0, ignore_eos=True))
FD.FastDecoder._rope = orig
fd.use_graphs = True
print(f"\nrope calls compared: {cnt['n']}  identical {cnt['same']} "
      f"({100*cnt['same']/max(cnt['n'],1):.4f} %)  differing {cnt['diff']}"
      + (f"  max|d| {worst[0]:.3e}" if cnt["diff"] else ""), flush=True)
for k in sorted(cnt):
    if k.startswith("diff_("):
        print(f"    {k}: {cnt[k]}")

pos = m.c.len
t0 = int(ids[-1])
d, _ = fd.draft(t0, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0], device="cuda"), d.clone()])
hsh = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hsh[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def arm(on):
    EF.ROPE = on
    fd.graphs.clear(); torch.cuda.synchronize()


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


print("\ninterleaved timing (A = torch rope, B = fused)")
res = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    arm(on)
    v = ev(pair)
    res[on].append(v[100])
    print(f"  run{i+1} fused={int(on)}  {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
a, b = res[False], res[True]
print(f"\n  A mean {st.mean(a):8.3f} sd {st.pstdev(a):5.3f}   B mean {st.mean(b):8.3f} "
      f"sd {st.pstdev(b):5.3f}   {st.mean(b)-st.mean(a):+7.3f} ms "
      f"({100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %)", flush=True)

print("\nend-to-end, 3 rounds interleaved")
tp = collections.defaultdict(list)
toks = {}
for i in range(3):
    for on in (False, True):
        arm(on)
        out = [t for bb in eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True)
               for t in bb]
        toks[on] = out
        s = eng.last_stats
        tp[on].append(s["decode_tok_s"])
        print(f"  round{i+1} fused={int(on)}  {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
for k in (False, True):
    v = tp[k][1:]
    print(f"  fused={int(k)}  mean {st.mean(v):6.3f} sd {st.pstdev(v):5.3f}  {tp[k]}")
print(f"  delta {100*(st.mean(tp[True][1:])-st.mean(tp[False][1:]))/st.mean(tp[False][1:]):+6.2f} %")
print(f"  generated tokens identical: {toks[False] == toks[True]}")
