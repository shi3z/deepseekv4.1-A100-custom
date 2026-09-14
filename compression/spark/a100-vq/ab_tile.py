"""Second change, measured on its own: the per-shape decode tile, on top of the fused projection.

Baseline for this A/B is the fused path with the generic tile, which is what the previous A/B
adopted. Only `tile` attributes are added or removed between arms -- same weights, same arena,
same graphs cleared each way. BLOCK_N splits the N dimension and each output still accumulates
over the whole K in the same order, so the result should be bit-identical; that is checked, not
assumed.
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
FD.FUSE_QKV = True
BLOCKS = list(W.layers) + list(W.mtp)
for b in BLOCKS:
    qa, kv = b.wq_a, b.wkv
    b.wqkv = FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                       torch.cat([qa.s, kv.s], 0).contiguous(), qa.N + kv.N, qa.K)
    b.wqkv_split = qa.N
head = W.head
print(f"head is {type(head).__name__}  arena_slots={eng.config()['arena_slots']}", flush=True)
TILE = (16, 1, 3)
TARGETS = [b.wqkv for b in BLOCKS] + ([head] if isinstance(head, FP4Weight) else [])
print(f"tile hint would apply to {len(TARGETS)} weights", flush=True)


def set_tile(on: bool):
    for w in TARGETS:
        if on:
            w.tile = TILE
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


set_tile(False); a = logits_of()
set_tile(True); b_ = logits_of()
print(f"\noutput equality: max|d| {float((b_-a).abs().max()):.3e}  "
      f"bit-identical {bool(torch.equal(a, b_))}", flush=True)


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


print("\ninterleaved A/B (off = generic tile, on = per-shape tile); both arms have FUSE_QKV on")
res = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    set_tile(on)
    v = ev(pair)
    res[on].append(v[100])
    print(f"  run{i+1} tile={int(on)}  pair median {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
o, f = res[False], res[True]
print(f"\n  off mean {st.mean(o):8.3f} sd {st.pstdev(o):5.3f}   "
      f"on mean {st.mean(f):8.3f} sd {st.pstdev(f):5.3f}   "
      f"{st.mean(f)-st.mean(o):+7.3f} ms ({100*(st.mean(f)-st.mean(o))/st.mean(o):+6.2f} %)",
      flush=True)

print("\nend-to-end, 4 rounds interleaved")
tp = collections.defaultdict(list)
for i in range(4):
    for on in (False, True):
        set_tile(on)
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        tp[on].append(s["decode_tok_s"])
        print(f"  round{i+1} tile={int(on)}  {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
for k in (False, True):
    v = tp[k][1:]
    print(f"  tile={int(k)}  mean {st.mean(v):6.3f} sd {st.pstdev(v):5.3f}  {tp[k]}")
print(f"  delta {100*(st.mean(tp[True][1:])-st.mean(tp[False][1:]))/st.mean(tp[False][1:]):+6.2f} %")
