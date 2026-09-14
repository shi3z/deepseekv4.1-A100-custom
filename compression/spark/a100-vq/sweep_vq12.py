"""(BN, num_warps, num_stages) sweep for the VQ12 up and down kernels, timed separately."""
import os, sys, time, itertools, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4, cb3_moe as C3
import vq12_moe as VQM
from vq12 import VQ12

dev = "cuda"; torch.manual_seed(0); DIM, INTER = F4.DIM, F4.INTER
T = int(os.environ.get("T", 6)); K = int(os.environ.get("K", 6)); S = 16
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
ar = VQM.VQ12Arena(S, dev).attach(vq)
for s in range(S):
    ar.load_slot(s, torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                    torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                    torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
P = T * K; BM = F4._pick_bm(P)
bs, bp, NB = F4.build_routing(slots, ar.slots, BM)
wgt = wts.reshape(-1).float().contiguous()
h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)

def bench(fn, it=40):
    try:
        for _ in range(6): fn()
    except Exception as e:
        return None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e3

def up(bn, nw, ns):
    return lambda: VQM._vq_up_kernel[(NB, INTER // bn)](
        x, ar.w1_lo, ar.w1_hi, ar.s1, ar.w3_lo, ar.w3_hi, ar.s3, h, ar.lut, ar.tab,
        wgt, bs, bp, x.stride(0), h.stride(0), 10.0,
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn, num_warps=nw, num_stages=ns)

def down(bn, nw, ns):
    return lambda: VQM._vq_down_kernel[(NB, DIM // bn)](
        h, ar.w2_lo, ar.w2_hi, ar.s2, parts, ar.lut, ar.tab, bs, bp,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn, NTOK=T,
        num_warps=nw, num_stages=ns)

print(f"T={T} K={K} P={P} BM={BM}")
for name, mk, N in (("up", up, INTER), ("down", down, DIM)):
    res = []
    for bn, nw, ns in itertools.product((16, 32, 48, 64, 96, 128), (1, 2, 4, 8), (1, 2, 3, 4)):
        if N % bn: continue
        t = bench(mk(bn, nw, ns))
        if t: res.append((t, bn, nw, ns))
    res.sort()
    base = VQM._VQ_UP_CFG[BM] if name == "up" else VQM._VQ_DOWN_CFG[BM]
    bt = bench(mk(*base))
    print(f"  {name}: current {base} -> {bt:.3f} ms")
    for t, bn, nw, ns in res[:6]:
        print(f"      ({bn:3d}, {nw}, {ns}) {t:.3f} ms   {bt/t:.2f}x")
