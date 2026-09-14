"""Routing preservation for the fused RMSNorm, and a tok/s measured with each arm warmed alone."""
from __future__ import annotations

import collections, json, os, statistics as st, sys
import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK); sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
import elem_fused as EF                                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
P = [json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt")) if l.strip()][:4]
ENC = []
for p in P:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ENC.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))
EF.RMSNORM = True
for ids in ENC[:2]:
    list(eng.generate(ids, max_tokens=32, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)

fd.use_graphs = False
FD.RMS_CMP = []
for ids in ENC:
    list(eng.generate(ids, max_tokens=120, temperature=0.0, ignore_eos=True))
rec = FD.RMS_CMP
FD.RMS_CMP = None
fd.use_graphs = True
n = same_set = same_ord = y_eq = 0
bad = collections.Counter()
for L, a, b, ye in rec:
    y_eq += int(ye)
    for t in range(a.shape[0]):
        x, z = a[t].tolist(), b[t].tolist()
        n += 1
        same_set += set(x) == set(z)
        same_ord += x == z
        if set(x) != set(z):
            bad[L] += 1
print(f"\nrouter decisions from the same hc_pre: {n} over {len(rec)} layer-calls")
print(f"  y bit-identical in {y_eq}/{len(rec)} layer-calls ({100*y_eq/len(rec):.3f} %)")
print(f"  top-6 SET   identical {same_set} mismatch {n-same_set} ({100*(n-same_set)/n:.5f} %)")
print(f"  top-6 ORDER identical {same_ord} mismatch {n-same_ord} ({100*(n-same_ord)/n:.5f} %)")
if bad:
    print("  layers with set changes: " + " ".join(f"L{k}:{v}" for k, v in bad.most_common(10)))

print("\nend-to-end, each arm warmed on its own", flush=True)
ids0 = ENC[0]
out = {}
for on in (False, True, False, True):
    EF.RMSNORM = on
    fd.graphs.clear(); torch.cuda.synchronize()
    for _ in range(3):
        list(eng.generate(ids0, max_tokens=300, temperature=0.0, ignore_eos=True))
    v, h = [], []
    for _ in range(3):
        list(eng.generate(ids0, max_tokens=300, temperature=0.0, ignore_eos=True))
        s = eng.last_stats
        v.append(s["decode_tok_s"]); h.append(s["expert_hit_rate"])
    out.setdefault(on, []).append(st.mean(v))
    print(f"  fused={int(on)}  {st.mean(v):6.3f} tok/s sd {st.pstdev(v):5.3f}  hit {min(h):.4f}  "
          f"accept {s['accept_len_mean']:.3f}  {[round(x,2) for x in v]}", flush=True)
a, b = st.mean(out[False]), st.mean(out[True])
print(f"\n  A {a:6.3f}   B {b:6.3f}   {100*(b-a)/a:+6.2f} %")
