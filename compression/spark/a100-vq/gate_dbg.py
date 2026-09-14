"""Why does gate_linear raise? Print the first failure instead of swallowing it."""
from __future__ import annotations

import os
import sys
import traceback

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from gate_gemm import gate_linear, plan                        # noqa: E402

N, K, M = 384, 5120, 2
w16 = torch.randn(N, K, device="cuda").to(torch.bfloat16).contiguous()
x16 = torch.randn(M, K, device="cuda").to(torch.bfloat16)
ref = (x16.float() @ w16.float().T)
for bn in (32, 64, 128):
    for tgt in (48, 96):
        for nw in (2, 4, 8):
            b_, sk = plan(N, K, bn, tgt)
            try:
                y = gate_linear(x16, w16, block_n=bn, target=tgt, num_warps=nw)
                d = float((y - ref).abs().max())
                print(f"bn={bn:4d} tgt={tgt:3d} warps={nw} -> split_k={sk:3d}  ok  max|d| {d:.3e}")
            except Exception as e:
                print(f"bn={bn:4d} tgt={tgt:3d} warps={nw} -> split_k={sk:3d}  FAIL "
                      f"{type(e).__name__}: {str(e)[:160]}")
                if bn == 32 and tgt == 48 and nw == 2:
                    traceback.print_exc()
