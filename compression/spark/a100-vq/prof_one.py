"""One up-kernel launch of each format, for ncu. Warmups first so only the last two are profiled."""
import os, sys
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, vq12_moe as VQM
from engine.codebook_sim import CodebookSim
from vq12 import VQ12

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 16; T = 6; K = 6
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
vqa = VQM.VQ12Arena(S, dev).attach(vq)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    vqa.load_slot(i, *args); cb.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
for _ in range(3):
    C3.moe_forward_v3(x, slots, wts, cb, 10.0)
    VQM.moe_forward_vq(x, slots, wts, vqa, 10.0)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("profiled")
C3.moe_forward_v3(x, slots, wts, cb, 10.0)
VQM.moe_forward_vq(x, slots, wts, vqa, 10.0)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
