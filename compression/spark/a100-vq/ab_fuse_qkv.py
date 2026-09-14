"""Same-process A/B of the fused wq_a+wkv projection, with the output checked for equality.

One engine, one arena, one warm cache, one prompt; only FD.FUSE_QKV changes between timings, and
the arms are interleaved off/on/on/off/off/on so an order effect would show. The fused weight is
attached once, before any timing, so neither arm pays for building it.
"""
from __future__ import annotations

import collections
import os
import statistics as st
import sys
import time

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
                arena_gb=float(os.environ.get("ARENA_GB", "40")), transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")), keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W, fd, m = eng.model.W, eng.fast, eng.model
c = eng.config()
print(f"dense_fp4={c['dense_fp4']} head={c['head_fmt']} T_VERIFY={FD.T_VERIFY} "
      f"FUSED_ATTN={FD.FUSED_ATTN} FUSE_QKV flag present={hasattr(FD, 'FUSE_QKV')}", flush=True)

n_fused = n_skip = 0
for b in list(W.layers) + list(W.mtp):
    qa, kv = getattr(b, "wq_a", None), getattr(b, "wkv", None)
    if isinstance(qa, FP4Weight) and isinstance(kv, FP4Weight) and qa.K == kv.K:
        b.wqkv = FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                           torch.cat([qa.s, kv.s], 0).contiguous(),
                           qa.N + kv.N, qa.K)
        b.wqkv_split = qa.N
        n_fused += 1
    else:
        n_skip += 1
print(f"fused weight attached to {n_fused} blocks, skipped {n_skip} "
      f"(+{n_fused * (W.layers[0].wqkv.w.numel() + W.layers[0].wqkv.s.numel())/2**20:.0f} MB)",
      flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def set_mode(on: bool):
    FD.FUSE_QKV = on
    fd.graphs.clear()
    torch.cuda.synchronize()


def logits_of():
    m.c.len = pos
    lg, _ = fd.step(block, pos, rows)
    torch.cuda.synchronize()
    return lg.detach().float().clone()


set_mode(False); off_lg = logits_of()
set_mode(True); on_lg = logits_of()
diff = (on_lg - off_lg).abs()
print(f"\noutput equality: max|d| {float(diff.max()):.3e}  mean|d| {float(diff.mean()):.3e}  "
      f"bit-identical {bool(torch.equal(on_lg, off_lg))}  "
      f"argmax agree {float((on_lg.argmax(-1) == off_lg.argmax(-1)).float().mean()):.4f}", flush=True)


def ev(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted(x.elapsed_time(y) for x, y in e)


def pair():
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0tok, pos - 1, 0.0)


print("\ninterleaved A/B (off = two kernels, on = fused)")
res = collections.defaultdict(list)
for i, on in enumerate((False, True, True, False, False, True)):
    set_mode(on)
    v = ev(pair)
    res[on].append(v[100])
    print(f"  run{i+1} FUSE_QKV={int(on)}  pair median {v[100]:8.3f} ms  p95 {v[190]:8.3f}",
          flush=True)
o, f = res[False], res[True]
print(f"\n  off  mean {st.mean(o):8.3f} median {st.median(o):8.3f} sd {st.pstdev(o):6.3f}")
print(f"  on   mean {st.mean(f):8.3f} median {st.median(f):8.3f} sd {st.pstdev(f):6.3f}")
print(f"  fused vs two-kernel: {st.mean(f)-st.mean(o):+7.3f} ms "
      f"({100*(st.mean(f)-st.mean(o))/st.mean(o):+6.2f} %)", flush=True)

print("\nend-to-end, resident, 300 tokens x 3 per arm")
for on in (False, True, False, True):
    set_mode(on)
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
    tps = []
    for _ in range(3):
        list(eng.generate(ids, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        tps.append((s["decode_tok_s"], s["accept_len_mean"], s["expert_hit_rate"]))
    print(f"  FUSE_QKV={int(on)}  tok/s " + " ".join(f"{t:.2f}" for t, _, _ in tps) +
          f"   accept {tps[-1][1]:.3f}  hit {tps[-1][2]:.4f}", flush=True)
