"""A / A+swiglu / A+swiglu+merge, one change at a time, one process, one arena."""
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
fd, m, W = eng.fast, eng.model, eng.model.W
print(f"gate_kernel={FD.GATE_KERNEL} fused_qkv={hasattr(W.layers[0],'wqkv')} "
      f"head_tile={getattr(W.head,'tile',None)}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)

ARMS = {"A base": (False, False), "B +swiglu": (True, False), "C +swiglu+merge": (True, True)}


def arm(sw, mg):
    EF.SWIGLU, EF.MERGE = sw, mg
    fd.graphs.clear(); torch.cuda.synchronize()


arm(False, False)
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


print("\ninterleaved timing, 3 rounds of A B C")
res = collections.defaultdict(list)
for r in range(3):
    for nm, (sw, mg) in ARMS.items():
        arm(sw, mg)
        v = ev(pair)
        res[nm].append(v[100])
        print(f"  round{r+1} {nm:17s} {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
base = st.mean(res["A base"])
print()
for nm in ARMS:
    v = res[nm]
    print(f"  {nm:17s} mean {st.mean(v):8.3f} sd {st.pstdev(v):5.3f}  "
          f"{100*(st.mean(v)-base)/base:+6.2f} % vs A")

print("\nend-to-end, 3 rounds of A B C")
tp = collections.defaultdict(list)
toks = {}
for r in range(3):
    for nm, (sw, mg) in ARMS.items():
        arm(sw, mg)
        out = [t for bb in eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True)
               for t in bb]
        toks[nm] = out
        s = eng.last_stats
        tp[nm].append(s["decode_tok_s"])
        print(f"  round{r+1} {nm:17s} {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
b0 = st.mean(tp["A base"][1:])
print()
for nm in ARMS:
    v = tp[nm][1:]
    print(f"  {nm:17s} mean {st.mean(v):6.3f} sd {st.pstdev(v):5.3f}  "
          f"{100*(st.mean(v)-b0)/b0:+6.2f} % vs A   {tp[nm]}")
print(f"\n  tokens A==B {toks['A base']==toks['B +swiglu']}   "
      f"A==C {toks['A base']==toks['C +swiglu+merge']}")
