"""LM-head format A/B inside one process, on top of the dense configuration under test.

Same reason as fp4_inproc.py: an fp8 head is 0.66 GB smaller than a bf16 one, and NOTES.md
2026-09-11 records that measuring this across two processes reversed its sign -- the arena landed
at a different offset and that, not the head, was what moved. Here one engine is loaded, the dense
projections are set once, and only the head object changes between timings.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
os.environ["DSV41_DENSE_FP4"] = "off"
os.environ["DSV41_HEAD_FMT"] = "bf16"
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, quantize_to_fp4    # noqa: E402
from fp8_linear import FP8Weight, quantize_to_fp8              # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "80")), transient_slots=64,
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
W, fd = eng.model.W, eng.fast
DENSE = os.environ.get("AB_DENSE", "attn")
if DENSE == "attn":                                            # the dense arm this A/B sits on
    n = 0
    for b in list(W.layers) + [m for m in W.mtp]:
        for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
            w = getattr(b, nm, None)
            if isinstance(w, FP8Weight):
                setattr(b, nm, quantize_fp8_to_fp4(w)); n += 1
    print(f"dense: re-quantized {n} attn tensors to fp4", flush=True)
head_bf16 = W.head.to(torch.bfloat16) if torch.is_tensor(W.head) else W.head
print(f"FUSED_ATTN={FD.FUSED_ATTN}  head loaded as {type(W.head).__name__}  "
      f"arena_slots={eng.config()['arena_slots']}", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
st = eng.model.store.stats
print(f"hit {st['hits']/max(st['hits']+st['misses'],1):.4f}", flush=True)

m = eng.model
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def set_head(fmt):
    h = (head_bf16 if fmt == "bf16" else
         quantize_to_fp8(head_bf16) if fmt == "fp8" else quantize_to_fp4(head_bf16))
    W.head = h
    fd.head_bf16 = h
    fd.graphs.clear()
    torch.cuda.synchronize()


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
    # the draft is the head's second read of the step; time it on its own too
    torch.cuda.synchronize()
    dv = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(50)]
    for a, b in dv:
        a.record(); fd.draft(t0tok, pos - 1, 0.0); b.record()
    torch.cuda.synchronize()
    dms = sorted(a.elapsed_time(b) for a, b in dv)
    print(f"{label:16s} step median {ms[n//2]:8.3f} ms  p95 {ms[int(n*0.95)]:8.3f}   "
          f"draft {dms[25]:6.3f} ms   free {torch.cuda.mem_get_info()[0]/2**30:6.2f} GB", flush=True)
    return ms[n // 2], dms[25]


res = {}
for fmt in ("bf16", "fp8", "fp4", "bf16"):
    set_head(fmt)
    res.setdefault(fmt, []).append(timed(f"head {fmt}"))
b = [x[0] for x in res["bf16"]]
print(f"\nbf16 drift between the two readings: {100*(b[1]-b[0])/b[0]:+.2f} %", flush=True)
base_s = sum(b) / 2
base_d = sum(x[1] for x in res["bf16"]) / 2
for fmt in ("fp8", "fp4"):
    s, dd = res[fmt][0]
    print(f"head {fmt:4s}: step {100*(s-base_s)/base_s:+6.2f} %  ({base_s:.3f} -> {s:.3f} ms)   "
          f"draft {100*(dd-base_d)/base_d:+6.2f} %  ({base_d:.3f} -> {dd:.3f} ms)", flush=True)
