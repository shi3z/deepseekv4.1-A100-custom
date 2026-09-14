"""Isolate _grp_packed_vq: does it rebuild exactly the packed-FP4 bytes vq12.unpack produces?"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
from vq12 import VQ12
import vq12_moe as VQM

@triton.jit
def probe(lo_ptr, hi_ptr, lut_ptr, out_ptr, BN: tl.constexpr):
    n = tl.arange(0, BN)
    L = tl.load(lo_ptr + n[:, None] * 64 + tl.arange(0, 64)[None, :])
    H = tl.load(hi_ptr + n[:, None] * 32 + tl.arange(0, 32)[None, :])
    L0, L1, L2, L3, L4, L5, L6, L7 = VQM._split8w(L, BN, 8)
    H0, H1, H2, H3, H4, H5, H6, H7 = VQM._split8w(H, BN, 4)
    for i in tl.static_range(8):
        Lk = L0 if i == 0 else (L1 if i == 1 else (L2 if i == 2 else (L3 if i == 3 else (
             L4 if i == 4 else (L5 if i == 5 else (L6 if i == 6 else L7))))))
        Hk = H0 if i == 0 else (H1 if i == 1 else (H2 if i == 2 else (H3 if i == 3 else (
             H4 if i == 4 else (H5 if i == 5 else (H6 if i == 6 else H7))))))
        t = VQM._grp_packed_vq(Lk, Hk, lut_ptr, BN)
        tl.store(out_ptr + n[:, None] * 128 + i * 16 + tl.arange(0, 16)[None, :], t)

dev = "cuda"
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
torch.manual_seed(0)
import sys as _s
BN = int(_s.argv[1]) if len(_s.argv) > 1 else 16
w = torch.randint(0, 256, (BN, 128), dtype=torch.uint8, device=dev)     # 256 weights per row
s = torch.randint(120, 126, (BN, 8), dtype=torch.uint8, device=dev)
lo, hi, _ = vq.pack(w, s)
print("lo", tuple(lo.shape), "hi", tuple(hi.shape))
ref = vq.unpack(lo, hi)                                                # [BN, 128] packed FP4
out = torch.zeros(BN, 128, dtype=torch.uint8, device=dev)
probe[(1,)](lo.contiguous(), hi.contiguous(), vq.lut.to(dev).contiguous(), out, BN=BN)
torch.cuda.synchronize()
eq = (out == ref)
print(f"decode matches vq12.unpack: {bool(eq.all())}   ({float(eq.float().mean())*100:.1f}% of bytes)")
if not bool(eq.all()):
    bad = (~eq).nonzero()[:6]
    for r, c in bad.tolist():
        print(f"  row {r} byte {c}: kernel {int(out[r,c]):3d} ref {int(ref[r,c]):3d}")
