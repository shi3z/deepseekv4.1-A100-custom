"""Is the CB3 -> codes -> VQ12 chain sane? Compare the VQ12 record against the CB3 one it came from."""
import os, sys, json, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3
from cb3 import dequant_cb3_v2
from engine.codebook_sim import CodebookSim
import vq12_moe as VQM
from vq12 import VQ12
from repack_vq12 import PIECES, to_codes
dev="cuda"
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
m = json.load(open("/tmp/vq12_probe.json")); stride = m["stride"]
off = {k: (int(o), int(n)) for k, (o, n, _) in m["offsets"].items()}
src = json.load(open(os.path.expanduser("~/dsv41-spark/models/cb3_store2.json")))
key, rec_v = next(iter(m["records"].items()))
rec_c = src["records"][key]
cb = C3.CB3ArenaV2(1, dev); cb.sim = CodebookSim(3, dev)
ar = VQM.VQ12Arena(1, dev).attach(vq)
def load(arena, path, rec):
    with open(path, "rb") as f:
        f.seek(rec * stride); raw = torch.frombuffer(bytearray(f.read(stride)), dtype=torch.uint8)
    for p in PIECES:
        o, n = off[p]; getattr(arena, p)[0].view(-1).copy_(raw[o:o+n])
load(cb, os.path.expanduser("~/dsv41-spark/models/cb3_store2.bin"), rec_c)
load(ar, "/tmp/vq12_probe.bin", rec_v)
w1c = dequant_cb3_v2(cb.w1_lo[0], cb.w1_hi[0], cb.w1_cb[0], cb.s1[0]).float()
codes = to_codes(w1c, cb.s1[0])
# codes -> bf16 must reproduce the CB3 values exactly
import v41_ref as R
back = R.dequant_fp4_packed(codes, cb.s1[0]).float()
print(f"CB3 -> codes -> bf16 exact: {bool((back == w1c).all())}")
w1v = ar.vq.dequant(ar.w1_lo[0], ar.w1_hi[0], ar.s1[0]).float()
print(f"VQ12 record vs its CB3 source: rel err {float((w1v-w1c).norm()/w1c.norm()):.4f} "
      f"(a second quantisation on top of CB3, so a gap is expected)")
print(f"scales preserved: {bool((ar.s1[0]==cb.s1[0]).all())}")
