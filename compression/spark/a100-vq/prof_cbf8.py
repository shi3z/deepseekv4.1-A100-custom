"""One up-kernel launch of CB3 v3 and CBF8, for ncu."""
import os, sys
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, cbf8_moe as F8
from engine.codebook_sim import CodebookSim
from cbf8 import CBF8
dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 16; T = 6; K = 6
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
f8 = F8.CBF8Arena(S, dev).attach(CBF8(dev))
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    cb.load_slot(i, *args); f8.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
for _ in range(3):
    C3.moe_forward_v3(x, slots, wts, cb, 10.0)
    F8.moe_forward_cbf8(x, slots, wts, f8, 10.0)
torch.cuda.synchronize()
C3.moe_forward_v3(x, slots, wts, cb, 10.0)
F8.moe_forward_cbf8(x, slots, wts, f8, 10.0)
torch.cuda.synchronize()
