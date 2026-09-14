"""All three arms against one baseline, in one process, so the deltas can finally be added.

The fusion and the tile were each measured in their own process and their shared configuration --
fused, generic tile -- came out 88.92 ms in one and 83.26 ms in the other. Both A/Bs are internally
consistent, so each delta stands on its own, but 6.4 % of unexplained offset between them means
they cannot be summed. Here A, B and C are interleaved in one arena.

The control matters as much as the arms: `set_*` clears and recaptures the graphs every time, so
an off->off pair is run first to see whether recapture alone moves the logits. Without it the
tile arm's max|d| of 2.8 cannot be attributed to the tile.
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
W, fd, m = eng.model.W, eng.fast, eng.model
BLOCKS = list(W.layers) + list(W.mtp)
for b in BLOCKS:
    qa, kv = b.wq_a, b.wkv
    b.wqkv = FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                       torch.cat([qa.s, kv.s], 0).contiguous(), qa.N + kv.N, qa.K)
    b.wqkv_split = qa.N
TILED = [b.wqkv for b in BLOCKS] + ([W.head] if isinstance(W.head, FP4Weight) else [])
print(f"arena_slots={eng.config()['arena_slots']}  tiled weights {len(TILED)}", flush=True)


def arm(fuse: bool, tile: bool):
    FD.FUSE_QKV = fuse
    for w in TILED:
        if tile:
            w.tile = (16, 1, 3)
        elif hasattr(w, "tile"):
            del w.tile
    fd.graphs.clear()
    torch.cuda.synchronize()


sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
arm(False, False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def logits_of():
    m.c.len = pos
    lg, _ = fd.step(block, pos, rows)
    torch.cuda.synchronize()
    return lg.detach().float().clone()


print("\nequality (A = plain, B = fused, C = fused+tile)")
arm(False, False); a1 = logits_of()
arm(False, False); a2 = logits_of()           # control: recapture only
print(f"  A vs A (recapture control) max|d| {float((a2-a1).abs().max()):.3e}  "
      f"bit-identical {bool(torch.equal(a1, a2))}", flush=True)
arm(True, False); b = logits_of()
print(f"  A vs B (fusion)            max|d| {float((b-a1).abs().max()):.3e}  "
      f"bit-identical {bool(torch.equal(a1, b))}", flush=True)
arm(True, True); c = logits_of()
print(f"  B vs C (tile)              max|d| {float((c-b).abs().max()):.3e}  "
      f"bit-identical {bool(torch.equal(b, c))}  "
      f"argmax agree {float((c.argmax(-1) == b.argmax(-1)).float().mean()):.4f}", flush=True)


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
    fd.draft(t0tok, pos - 1, 0.0)


ARMS = {"A plain": (False, False), "B fused": (True, False), "C fused+tile": (True, True)}
print("\ninterleaved timing, 3 rounds of A B C")
res = collections.defaultdict(list)
for r in range(3):
    for name, (f_, t_) in ARMS.items():
        arm(f_, t_)
        v = ev(pair)
        res[name].append(v[100])
        print(f"  round{r+1} {name:13s} {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
base = st.mean(res["A plain"])
print()
for name in ARMS:
    v = res[name]
    print(f"  {name:13s} mean {st.mean(v):8.3f} sd {st.pstdev(v):5.3f}  "
          f"{100*(st.mean(v)-base)/base:+6.2f} % vs A")

print("\nend-to-end, 3 rounds of A B C")
tp = collections.defaultdict(list)
toks = {}
for r in range(3):
    for name, (f_, t_) in ARMS.items():
        arm(f_, t_)
        out = [t for bb in eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True)
               for t in bb]
        toks[name] = out
        s = eng.last_stats
        tp[name].append(s["decode_tok_s"])
        print(f"  round{r+1} {name:13s} {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
b0 = st.mean(tp["A plain"][1:])
print()
for name in ARMS:
    v = tp[name][1:]
    print(f"  {name:13s} mean {st.mean(v):6.3f} sd {st.pstdev(v):5.3f}  "
          f"{100*(st.mean(v)-b0)/b0:+6.2f} % vs A   {tp[name]}")
print(f"\n  generated tokens  A==B {toks['A plain']==toks['B fused']}   "
      f"A==C {toks['A plain']==toks['C fused+tile']}")
