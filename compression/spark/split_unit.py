"""Is a masked two-pass MoE call exactly the one-pass call?

The end-to-end check cannot answer this: forty layers amplify a last-bit difference into percents,
and the shipped path rounds its routed sum to bf16 while a split one rounds two partial sums. This
compares the single MoE call directly, which is where the masking logic either is or is not right.
"""
import os, sys
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4
from engine.codebook_sim import CodebookSim

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 24
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
for i in range(S):
    cb.load_slot(i, w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER),
                 w(INTER, DIM), sc(INTER, DIM))

print(f"SPLIT flag in cb3_moe: {C3.SPLIT}")
for T in (6, 512):
    x = torch.randn(T, 6 if False else DIM, dtype=torch.bfloat16, device=dev) * 0.3
    slots = torch.randint(0, S, (T, 6), dtype=torch.int32, device=dev)
    wts = torch.rand(T, 6, device=dev)
    full = C3.moe_forward_v3(x, slots, wts, cb, 10.0).float()
    # split the SLOTS the way resolve() does: half the experts "resident", half "pending"
    resident = torch.arange(S, device=dev) % 2 == 0
    a = torch.where(resident[slots.long()], slots, torch.full_like(slots, -1))
    b = torch.where(resident[slots.long()], torch.full_like(slots, -1), slots)
    two = (C3.moe_forward_v3(x, a, wts, cb, 10.0).float()
           + C3.moe_forward_v3(x, b, wts, cb, 10.0).float())
    d = (two - full).abs()
    print(f"T={T:4d}  one pass vs two masked passes: max abs {float(d.max()):.6f}  "
          f"rel {float(d.norm() / full.norm()):.3e}  "
          f"{'MATCH' if float(d.norm() / full.norm()) < 3e-3 else 'DIFFER'}")
