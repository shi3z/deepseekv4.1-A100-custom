"""Does the bf16 router gate pick the same experts?

The weights and activations are bit-identical in either dtype (the checkpoint stores the gate in
BF16 and f32() only promotes it), so the single added error is the 5120-term dot product being
rounded to bf16 on write. A top-6 over 384 scores can move on that, and if it does the MoE reads
different experts -- which changes the output, not just its rounding. Recorded per layer, per
token, over a real greedy generation, with graphs off so the routing is visible.
"""
from __future__ import annotations

import collections
import os
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
                arena_gb=float(os.environ.get("ARENA_GB", "60")), transient_slots=400,
                keep_free_gb=12.0, prune_keep=None, expert_format="cb3")
fd, m, W = eng.fast, eng.model, eng.model.W
FD.FUSE_QKV = True
for b in list(W.layers) + list(W.mtp):
    qa, kv = b.wq_a, b.wkv
    b.wqkv = FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                       torch.cat([qa.s, kv.s], 0).contiguous(), qa.N + kv.N, qa.K)
    b.wqkv_split = qa.N
print(f"gate stored dtype in engine: {W.layers[0].gate_w.dtype}, "
      f"bf16 copy {W.layers[0].gate_w.to(torch.bfloat16).dtype}", flush=True)
exact = torch.equal(W.layers[0].gate_w, W.layers[0].gate_w.to(torch.bfloat16).float())
print(f"fp32 gate == bf16(gate).float() : {exact}   "
      f"(true means the fp32 copy carries no extra information)", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)

orig_layer_a = FD.FastDecoder._layer_a
REC = []
STEP = [0]


def traced_layer_a(self, L, sh):
    orig_layer_a(self, L, sh)
    if L == 0:                      # a step starts at layer 0; key on it so the two arms align
        STEP[0] += 1
    REC.append((STEP[0], L, self.route_idx.detach().clone(), self.route_w.detach().clone()))


def run(bf16: bool, n_tok: int):
    FD.GATE_BF16 = bf16
    fd.graphs.clear()
    fd.use_graphs = False
    FD.FastDecoder._layer_a = traced_layer_a
    REC.clear(); STEP[0] = 0
    toks = [t for b in eng.generate(ids, max_tokens=n_tok, temperature=0.0, ignore_eos=True)
            for t in b]
    FD.FastDecoder._layer_a = orig_layer_a
    fd.use_graphs = True
    return toks, {(st, L): (i.cpu(), w.cpu()) for st, L, i, w in REC}


N = int(os.environ.get("NTOK", "120"))
t32, r32 = run(False, N)
t16, r16 = run(True, N)
print(f"\nrecorded {len(r32)} layer-calls (fp32) / {len(r16)} (bf16) for {N} tokens", flush=True)
print(f"generated tokens identical: {t32 == t16}  ({len(t32)} vs {len(t16)})", flush=True)

keys = sorted(set(r32) & set(r16))
print(f"  aligned on {len(keys)} (step, layer) keys of {len(r32)} / {len(r16)} recorded")
same_set = same_order = tot = 0
per_layer_bad = collections.Counter()
wmax = 0.0
for k in keys:
    L = k[1]
    (i32, w32), (i16, w16) = r32[k], r16[k]
    for t in range(i32.shape[0]):
        tot += 1
        a, b = i32[t].tolist(), i16[t].tolist()
        if a == b:
            same_order += 1
        if set(a) == set(b):
            same_set += 1
        else:
            per_layer_bad[L] += 1
    wmax = max(wmax, float((w32 - w16).abs().max()))
print(f"\ntop-6 selections compared: {tot} (layer x token)")
print(f"  identical SET   {same_set:7d}  ({100*same_set/tot:7.3f} %)")
print(f"  identical ORDER {same_order:7d}  ({100*same_order/tot:7.3f} %)")
print(f"  max |routing weight delta| {wmax:.3e}")
if per_layer_bad:
    print("  layers with set changes: " +
          " ".join(f"L{L}:{c}" for L, c in per_layer_bad.most_common(12)))
