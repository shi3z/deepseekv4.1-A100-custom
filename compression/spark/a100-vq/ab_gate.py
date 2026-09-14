"""End-to-end A/B of the bf16-storage fp32-accumulate router gate, on top of the fused projection.

Everything else is held: same engine, same arena, same warm cache, same prompt, same block size,
same head format, same expert store, same wq_a+wkv fusion. Only FD.GATE_KERNEL changes, and the
arms are interleaved A B A B so drift lands on both.
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

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m, W = eng.fast, eng.model, eng.model.W
print(f"dense_fp4={eng.config()['dense_fp4']} head={eng.config()['head_fmt']} "
      f"fused_qkv={hasattr(W.layers[0], 'wqkv')} arena_slots={eng.config()['arena_slots']}",
      flush=True)
# Phase 1: the fp32 copies the checkpoint's bf16 gate is expanded into
gb = sum(w.gate_w.numel() * 4 for w in W.layers) + sum(w.gate_w.numel() * 4 for w in W.mtp)
gb16 = gb // 2
print(f"gate fp32 resident {gb/2**20:7.1f} MB   bf16 would be {gb16/2**20:7.1f} MB   "
      f"main N={W.layers[0].gate_w.shape[0]} mtp N={W.mtp[0].gate_w.shape[0]} "
      f"K={W.layers[0].gate_w.shape[1]}", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)


def arm(on):
    FD.GATE_KERNEL = on
    fd.graphs.clear()
    torch.cuda.synchronize()


arm(False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


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


print("\ninterleaved timing, A = cuBLAS fp32 gate, B = bf16-storage fp32-accum")
res = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    arm(on)
    v = ev(pair)
    res[on].append(v[100])
    print(f"  run{i+1} GATE_KERNEL={int(on)}  {v[100]:8.3f} ms  p95 {v[190]:8.3f}", flush=True)
a, b = res[False], res[True]
print(f"\n  A mean {st.mean(a):8.3f} sd {st.pstdev(a):5.3f}   B mean {st.mean(b):8.3f} "
      f"sd {st.pstdev(b):5.3f}   {st.mean(b)-st.mean(a):+7.3f} ms "
      f"({100*(st.mean(b)-st.mean(a))/st.mean(a):+6.2f} %)", flush=True)

print("\nend-to-end, 4 rounds interleaved")
tp = collections.defaultdict(list)
toks = {}
for i in range(4):
    for on in (False, True):
        arm(on)
        out = [t for bb in eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True)
               for t in bb]
        toks[on] = out
        s = eng.last_stats
        tp[on].append(s["decode_tok_s"])
        print(f"  round{i+1} GATE_KERNEL={int(on)}  {s['decode_tok_s']:6.2f} tok/s  "
              f"accept {s['accept_len_mean']:.3f}  hit {s['expert_hit_rate']:.4f}", flush=True)
for k in (False, True):
    v = tp[k][1:]
    print(f"  GATE_KERNEL={int(k)}  mean {st.mean(v):6.3f} sd {st.pstdev(v):5.3f}  {tp[k]}")
print(f"  delta {100*(st.mean(tp[True][1:])-st.mean(tp[False][1:]))/st.mean(tp[False][1:]):+6.2f} %")
print(f"  generated tokens identical: {toks[False] == toks[True]}")
