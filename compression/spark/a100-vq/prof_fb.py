import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
os.environ["EXPERT_FORMAT"] = "vq12"
os.environ["DSV41_CB3_STORE"] = os.path.expanduser("~/dsv41-spark/models/cb3_store3")
import cb3_store as CS, vq12_moe as VQM
from cb3 import unpack_cb3_v2
from vq12 import VQ12

dev = "cuda"
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
st = CS.maybe_open(dev); st.convert = None
ar = VQM.VQ12Arena(1, dev).attach(vq)
key = list(st.records)[0]
st.load(ar, 0, key, torch.cuda.Stream())

def t(fn, n=5):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): r = fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3, r

for lo_n, hi_n, cb_n, s_n in (("w1_lo","w1_hi","w1_cb","s1"), ("w2_lo","w2_hi","w2_cb","s2")):
    lo_t, hi_t, cb_t, s_t = (getattr(ar, x) for x in (lo_n, hi_n, cb_n, s_n))
    ms1, codes = t(lambda: unpack_cb3_v2(lo_t[0], hi_t[0], cb_t[0]))
    ms2, cu = t(lambda: codes.to(torch.uint8))
    ms3, packed = t(lambda: (cu[:, 0::2] | (cu[:, 1::2] << 4)).contiguous())
    ms4, _ = t(lambda: vq.pack(packed, s_t[0].view(torch.uint8)))
    print(f"{lo_n[:2]}: unpack {ms1:6.2f}  to_u8 {ms2:5.2f}  bytepack {ms3:5.2f}  vq.pack {ms4:5.2f}  "
          f"= {ms1+ms2+ms3+ms4:6.2f} ms   codes {tuple(codes.shape)} {codes.dtype}")
