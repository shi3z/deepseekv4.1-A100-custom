"""Would one [1792, 5120] projection beat wq_a [1280] and wkv [512] run separately?

fastdecode's `_layer_a` applies both to the same x, so they can share a K loop. That matters here
because neither is bandwidth-bound: wkv moves 1.56 MB and still takes 40-46 us, the same as wq_a's
3.91 MB, which is what a serial K=5120 loop costs whatever the byte count. Fusing pays that loop
once. Measured, not assumed -- and against the tuned separate configs, not the default ones.
"""
from __future__ import annotations

import os
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, fp4_linear         # noqa: E402
from fp8_linear import FP8Weight                               # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
os.environ["DSV41_DENSE_FP4"] = "off"
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=30.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
b0 = eng.model.W.layers[0]
M = FD.T_VERIFY
COPIES = 12
qa, kv = b0.wq_a, b0.wkv
print(f"wq_a w{tuple(qa.w.shape)} s{tuple(qa.s.shape)}   "
      f"wkv w{tuple(kv.w.shape)} s{tuple(kv.s.shape)}", flush=True)
K = qa.K if hasattr(qa, "K") else qa.w.shape[1]
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

fused8 = FP8Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                   torch.cat([qa.s, kv.s], 0).contiguous())
print(f"fused  w{tuple(fused8.w.shape)} s{tuple(fused8.s.shape)}", flush=True)

c_qa = [quantize_fp8_to_fp4(qa) for _ in range(COPIES)]
c_kv = [quantize_fp8_to_fp4(kv) for _ in range(COPIES)]
c_fu = [quantize_fp8_to_fp4(fused8) for _ in range(COPIES)]

# correctness: the fused output's two halves must match the separate calls
ref_q = fp4_linear(x, c_qa[0])
ref_k = fp4_linear(x, c_kv[0])
fu = fp4_linear(x, c_fu[0])
Nq = qa.w.shape[0]
eq = float((fu[:, :Nq] - ref_q).abs().max())
ek = float((fu[:, Nq:] - ref_k).abs().max())
print(f"max|fused - separate|: q {eq:.3e}  kv {ek:.3e}", flush=True)


def ev_rot(objs, call, n=96):
    for o in objs:
        call(o)
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for i, (a, b) in enumerate(e):
        a.record(); call(objs[i % len(objs)]); b.record()
    torch.cuda.synchronize()
    v = sorted(y0.elapsed_time(y1) for y0, y1 in e)
    return v[len(v) // 2] * 1000


def best_of(objs, name):
    out = []
    for bn in (8, 16, 32, 64):
        for nw in (1, 2, 4):
            for ns in (3, 4):
                if objs[0].N % bn:
                    continue
                try:
                    t = ev_rot(objs, lambda w: fp4_linear(x, w, block_n=bn, num_warps=nw,
                                                          num_stages=ns))
                except Exception:
                    continue
                out.append((t, bn, nw, ns))
    out.sort()
    d = ev_rot(objs, lambda w: fp4_linear(x, w))
    print(f"  {name:12s} default {d:7.1f} us   best {out[0][0]:7.1f} us "
          f"(bn={out[0][1]}, warps={out[0][2]}, stages={out[0][3]}, grid={-(-objs[0].N//out[0][1])})")
    return d, out[0][0]


print("\nseparate:")
dq, bq = best_of(c_qa, "wq_a")
dk, bk = best_of(c_kv, "wkv")
print("fused:")
df, bf = best_of(c_fu, "wq_a+wkv")
print(f"\nper layer:  separate default {dq+dk:7.1f} us   separate tuned {bq+bk:7.1f} us   "
      f"fused tuned {bf:7.1f} us")
print(f"  fused vs separate-default: {bf-(dq+dk):+7.1f} us/layer = "
      f"{(bf-(dq+dk))*40/1000:+6.3f} ms per verify step")
print(f"  fused vs separate-tuned:   {bf-(bq+bk):+7.1f} us/layer = "
      f"{(bf-(bq+bk))*40/1000:+6.3f} ms per verify step")
