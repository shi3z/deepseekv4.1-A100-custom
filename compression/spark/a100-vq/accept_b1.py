"""Acceptance at the operating block size, over every prompt, one process, one arena.

At DSV41_BLOCK=1 there is a single drafted position, so the fp4 head's argmax error can only cost
acceptance once per step -- unlike the 5-draft block, where it is paid at every position. Whether
that leaves the fp4 head ahead is the whole question, and one prompt cannot answer it: accept
length varies by +-0.1 between prompts.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys
import time

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
os.environ["DSV41_DENSE_FP4"] = "off"
os.environ["DSV41_HEAD_FMT"] = "bf16"
os.environ.setdefault("DSV41_BLOCK", "1")
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, quantize_to_fp4    # noqa: E402
from fp8_linear import FP8Weight, quantize_to_fp8              # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "80")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")),
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
W, fd, m = eng.model.W, eng.fast, eng.model
n = 0
for b in list(W.layers) + list(W.mtp):
    for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
        w = getattr(b, nm, None)
        if isinstance(w, FP8Weight):
            setattr(b, nm, quantize_fp8_to_fp4(w)); n += 1
assert n == 172
head_bf16 = W.head.to(torch.bfloat16) if torch.is_tensor(W.head) else W.head
print(f"T_DRAFT={FD.T_DRAFT} T_VERIFY={FD.T_VERIFY} FUSED_ATTN={FD.FUSED_ATTN} "
      f"attn fp4={n} tensors  arena_slots={eng.config()['arena_slots']}", flush=True)
assert FD.T_DRAFT == 1, f"this run is for the 1-draft block, got {FD.T_DRAFT}"


def set_head(fmt):
    h = quantize_to_fp8(head_bf16) if fmt == "fp8" else quantize_to_fp4(head_bf16)
    W.head, fd.head_bf16 = h, h
    fd.graphs.clear(); torch.cuda.synchronize()


sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
ENC = []
for p in [json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt")) if l.strip()]:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ENC.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))

set_head("fp8")
for ids in ENC:
    list(eng.generate(ids, max_tokens=24, temperature=0.0, ignore_eos=True))
print(f"arena warmed: hit {eng.model.store.hit_rate():.4f}", flush=True)

res = {}
for fmt in ("fp8", "fp4", "fp8"):                  # fp8 twice: the arms bracket each other
    set_head(fmt)
    hist, per_prompt, tok, wall = [], [], 0, 0.0
    for ids in ENC:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = [t for b in eng.generate(ids, max_tokens=120, temperature=0.0, ignore_eos=True)
               for t in b]
        torch.cuda.synchronize(); wall += time.perf_counter() - t0
        tok += len(out)
        h = eng.last_stats.get("accepted_hist", [])
        hist += h
        per_prompt.append(sum(h) / len(h) + 1)
    res.setdefault(fmt, []).append((hist, per_prompt, tok, wall))
    print(f"head {fmt}: accept_len_mean {sum(hist)/len(hist)+1:.4f}  steps {len(hist)}  "
          f"P(accept pos1) {sum(hist)/len(hist):.4f}  {tok} tok in {wall:.1f}s "
          f"({tok/wall:.3f} tok/s, hit {eng.model.store.hit_rate():.4f})", flush=True)

a1, a2 = res["fp8"][0][1], res["fp8"][1][1]
b = res["fp4"][0][1]
print(f"\nfp8 repeat drift, per prompt: mean {100*st.mean([(y-x)/x for x, y in zip(a1, a2)]):+.2f} %")
base = [(x + y) / 2 for x, y in zip(a1, a2)]
d = [(y - x) / x for x, y in zip(base, b)]
print(f"fp4 vs fp8 accept_len, paired over {len(d)} prompts: "
      f"mean {100*st.mean(d):+.2f} %  sd {100*st.pstdev(d):.2f} %  "
      f"se {100*st.pstdev(d)/len(d)**0.5:.2f} %  worse in {sum(1 for x in d if x < 0)}/{len(d)}")
print(f"fp8 accept_len {st.mean(base):.4f}   fp4 accept_len {st.mean(b):.4f}")
