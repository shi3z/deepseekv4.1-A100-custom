"""Dense-FP4 A/B inside one process, because a cross-process one measures the arena's placement.

NOTES.md 2026-09-11: the head's bf16-vs-fp8 A/B reversed sign between processes -- a 0.66 GB
smaller head moves every later allocation, and an 89 GB streaming kernel is sensitive to where its
pages sit. Dense FP4 shrinks resident allocations the same way (the engine logs 8.34 vs 7.70 GiB
after weights), so every ms/step number taken by loading a second engine is suspect.

Here the arena is allocated once, with dense fp8, and the projections are re-quantized in place
between timings. The fp8 originals are kept so the run ends where it started: two `off` readings
bracket the FP4 arms and bound any drift.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
os.environ["DSV41_DENSE_FP4"] = "off"          # load fp8; the swaps below do the rest
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
# v41_ref does `from fp8_linear import ...` with tools/ on the path, so `tools.fp8_linear` would be
# a SECOND module object with a different FP8Weight class and every isinstance below would be False.
from fp4_linear import quantize_fp8_to_fp4, quantize_fp8_grouped_to_fp4  # noqa: E402
from fp8_linear import FP8Weight, FP8GroupedWeight             # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "80")), transient_slots=64,
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print(f"FUSED_ATTN={FD.FUSED_ATTN}  dense_fp4={eng.config()['dense_fp4']}  "
      f"head={eng.config()['head_fmt']}  arena_slots={eng.config()['arena_slots']}", flush=True)

ATTN = ("wq_a", "wq_b", "wkv", "wo_b")
W = eng.model.W


def blocks():
    out = list(W.layers)
    for m in W.mtp:
        out += [m] + [v for v in vars(m).values() if hasattr(v, "wq_a")]
    return [b for b in out if hasattr(b, "wq_a")]


BL = blocks()
orig = [{n: getattr(b, n) for n in ATTN + ("wo_a",)} for b in BL]
print(f"blocks with dense projections: {len(BL)}", flush=True)


def set_fmt(attn_fp4: bool, woa_fp4: bool):
    n_conv = 0
    for b, o in zip(BL, orig):
        for n in ATTN:
            w = o[n]
            if attn_fp4 and isinstance(w, FP8Weight):
                setattr(b, n, quantize_fp8_to_fp4(w)); n_conv += 1
            else:
                setattr(b, n, w)
        w = o["wo_a"]
        if woa_fp4 and isinstance(w, FP8GroupedWeight):
            b.wo_a = quantize_fp8_grouped_to_fp4(w); n_conv += 1
        else:
            b.wo_a = w
    eng.fast.graphs.clear()                    # the captured launches point at the old tensors
    torch.cuda.synchronize()
    want = (4 * len(BL) if attn_fp4 else 0) + (len(BL) if woa_fp4 else 0)
    assert n_conv == want, f"converted {n_conv} tensors, expected {want}"
    return n_conv


sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):                                             # warm the arena to hit 1.0
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
st = eng.model.store.stats
print(f"hit {st['hits']/max(st['hits']+st['misses'],1):.4f}", flush=True)

m, fd = eng.model, eng.fast
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def timed(label, n=200):
    def one():
        m.c.len = pos
        fd.step(block, pos, rows)
    for _ in range(20):
        one()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(n)]
    for a, b in ev:
        a.record(); one(); b.record()
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in ev)
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"{label:22s} median {ms[n//2]:8.3f} ms   p95 {ms[int(n*0.95)]:8.3f}   min {ms[0]:8.3f}"
          f"   free {free:6.2f} GB", flush=True)
    return ms[n // 2]


res = {}
for label, a, w in (("off (fp8) #1", False, False), ("attn fp4", True, False),
                    ("attn,wo_a fp4", True, True), ("off (fp8) #2", False, False)):
    n = set_fmt(a, w)
    print(f"  [{label}] re-quantized {n} tensors", flush=True)
    res[label] = timed(label)
b1, b2 = res["off (fp8) #1"], res["off (fp8) #2"]
base = (b1 + b2) / 2
print(f"\nbaseline drift between the two fp8 readings: {100*(b2-b1)/b1:+.2f} %", flush=True)
for k in ("attn fp4", "attn,wo_a fp4"):
    print(f"{k:22s} {100*(res[k]-base)/base:+6.2f} % vs mean fp8 baseline {base:.3f} ms", flush=True)
