"""The three new fusions together: step time, routing, tokens, tok/s, hit and NVMe, 1000 tokens.

Routing is compared the only way that stays clean for changes this far upstream: one verify block
is run through each arm from the *same* cache state, so both see identical inputs and any top-6
that moves moved because of the kernels. The state is then advanced by one arm and the comparison
repeated, so the sample is many positions rather than one.
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
      f"rms={EF.RMSNORM} rope={EF.ROPE} hc={EF.HC_POST}  arena_slots={eng.config()['arena_slots']}",
      flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて600字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)


def arm(on):
    EF.RMSNORM = EF.ROPE = EF.HC_POST = on
    fd.graphs.clear(); torch.cuda.synchronize()


arm(False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True))
print(f"warm hit {m.store.hit_rate():.4f}", flush=True)

# ---------- routing: same cache state, one step per arm, many positions
orig_la = FD.FastDecoder._layer_a
REC = []


def traced(self, L, sh):
    orig_la(self, L, sh)
    REC.append((L, self.route_idx.detach().clone()))


pos = m.c.len
t0 = int(ids[-1])
d0, _ = fd.draft(t0, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0], device="cuda"), d0.clone()])
hsh = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hsh[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
fd.use_graphs = False
FD.FastDecoder._layer_a = traced
n = same_set = same_ord = 0
bad = collections.Counter()
for it in range(int(os.environ.get("ROUTE_STEPS", "24"))):
    both = []
    for on in (False, True):
        EF.RMSNORM = EF.ROPE = EF.HC_POST = on
        REC.clear()
        m.c.len = pos
        fd.step(block, pos, rows)
        both.append([(L, i.cpu()) for L, i in REC])
    for (L, a), (L2, b) in zip(*both):
        for t in range(a.shape[0]):
            x, y = a[t].tolist(), b[t].tolist()
            n += 1
            same_set += set(x) == set(y)
            same_ord += x == y
            if set(x) != set(y):
                bad[L] += 1
    EF.RMSNORM = EF.ROPE = EF.HC_POST = False       # advance the state with one arm
    m.c.len = pos
    lg, _ = fd.step(block, pos, rows)
    nxt = int(lg[-1].argmax())
    block = torch.cat([block[1:], torch.tensor([nxt], device="cuda")])
FD.FastDecoder._layer_a = orig_la
fd.use_graphs = True
print(f"\nrouting, same cache state, {n} decisions over {n//80} steps x 40 layers")
print(f"  top-6 SET   mismatch {n-same_set} ({100*(n-same_set)/n:.5f} %)")
print(f"  top-6 ORDER mismatch {n-same_ord} ({100*(n-same_ord)/n:.5f} %)")
if bad:
    print("  layers: " + " ".join(f"L{k}:{v}" for k, v in bad.most_common(8)))

# ---------- ms/step, interleaved
m.c.len = pos
d0, _ = fd.draft(t0, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0], device="cuda"), d0.clone()])


def ev(fn, k=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(k)]
    for x, y in e:
        x.record(); fn(); y.record()
    torch.cuda.synchronize()
    return sorted(x.elapsed_time(y) for x, y in e)


def pair():
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0, pos - 1, 0.0)


print("\nms/step, interleaved", flush=True)
r = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    arm(on)
    v = ev(pair)
    r[on].append(v[100])
    print(f"  run{i+1} all={int(on)}  {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
a, b = r[False], r[True]
print(f"  A {st.mean(a):8.3f} sd {st.pstdev(a):5.3f}   B {st.mean(b):8.3f} sd {st.pstdev(b):5.3f}"
      f"   {st.mean(b)-st.mean(a):+7.3f} ms ({100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %)")

# ---------- tok/s, each arm warmed to its own hit=1, 1000 tokens
N = int(os.environ.get("GEN", "1000"))
print(f"\nclean tok/s, {N} tokens, each arm warmed on its own", flush=True)
out, toks = {}, {}
for on in (False, True, False, True):
    arm(on)
    for _ in range(2):
        list(eng.generate(ids, max_tokens=N, temperature=0.0, ignore_eos=True))
    v, h, nv, ac = [], [], [], []
    for _ in range(2):
        o = [t for bb in eng.generate(ids, max_tokens=N, temperature=0.0, ignore_eos=True)
             for t in bb]
        toks[on] = o
        s = eng.last_stats
        v.append(s["decode_tok_s"]); h.append(s["expert_hit_rate"])
        nv.append(s["expert_misses"]); ac.append(s["accept_len_mean"])
    out.setdefault(on, []).append(st.mean(v))
    print(f"  all={int(on)}  {st.mean(v):6.3f} tok/s  accept {st.mean(ac):.3f}  hit {min(h):.4f}  "
          f"misses {nv[-1]}  nvme {s['nvme_gb']:.2f} GB  {[round(x,2) for x in v]}", flush=True)
A, B = st.mean(out[False]), st.mean(out[True])
print(f"\n  A {A:6.3f}   B {B:6.3f}   {100*(B-A)/A:+6.2f} %")
print(f"  generated tokens identical: {toks[False] == toks[True]}  "
      f"({len(toks[False])} vs {len(toks[True])})")
