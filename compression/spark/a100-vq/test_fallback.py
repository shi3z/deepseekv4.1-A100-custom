"""Does a CB3 record loaded into a VQ12 arena come out as the VQ12 of that record, and what does
the re-encode cost on the miss path?"""
import os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
os.environ["EXPERT_FORMAT"] = "vq12"
os.environ["DSV41_CB3_STORE"] = os.path.expanduser("~/dsv41-spark/models/cb3_store3")

import cb3_store as CS
import cb3_moe as C3
import vq12_moe as VQM
from cb3 import dequant_cb3_v2
from vq12 import VQ12
from engine.codebook_sim import CodebookSim

dev = "cuda"
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
st = CS.maybe_open(dev)
print("store format:", st.format, "records:", len(st.records))
keys = list(st.records)[:6]
stream = torch.cuda.Stream()

cbar = C3.CB3ArenaV2(1, dev)
cbar.sim = CodebookSim(3, dev)
ar = VQM.VQ12Arena(1, dev).attach(vq)

conv = st.convert
for i, key in enumerate(keys):
    st.convert = None                      # reference: the record as CB3
    st.load(cbar, 0, key, stream)
    ref = [dequant_cb3_v2(lo[0], hi[0], cb[0], s[0]).float() for lo, hi, cb, s in
           ((cbar.w1_lo, cbar.w1_hi, cbar.w1_cb, cbar.s1),
            (cbar.w2_lo, cbar.w2_hi, cbar.w2_cb, cbar.s2),
            (cbar.w3_lo, cbar.w3_hi, cbar.w3_cb, cbar.s3))]
    st.convert = conv                      # the same record through the fallback
    torch.cuda.synchronize(); t0 = time.perf_counter()
    st.load(ar, 0, key, stream)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
    got = [w.float() for w in ar.dequant_slot(0)]
    rel = [float((g - r).norm() / r.norm()) for g, r in zip(got, ref)]
    print(f"  {key}  load+convert {ms:7.1f} ms   rel err vs the CB3 record "
          f"w1 {rel[0]:.4f} w2 {rel[1]:.4f} w3 {rel[2]:.4f}")
print(f"converter: {conv.n} calls, {conv.ms / max(conv.n,1):.1f} ms each")
