"""Build a VQ12 store from the CB3 store, for a SPEED measurement only.

The routed experts' FP4 bytes were freed out of the checkpoint by punch_fp4.py once the CB3 store
existed, so the CB3 records are now the only copy of those weights on this box. Re-encoding them as
VQ12 therefore quantises twice, and the result is **not** the quality VQ12 would have: the honest
quality numbers are the A100 ones measured against the untouched checkpoint (+4.23 % wikitext PPL
for VQ12, +8.54 % for CB3). What this store is good for is tok/s -- the kernel does the same work
whatever the weight values are. For a real VQ12 deployment, stream the FP4 experts from a machine
that still has them and pack from those.

Record layout is CB3's, unchanged (same 12 slot tensors, same 14,454,784 B stride), so cb3_store.py
reads it as is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cb3_moe as C3                              # noqa: E402
from cb3 import dequant_cb3_v2                    # noqa: E402
from engine.experts import rank_from_trace        # noqa: E402
import vq12_moe as VQM                            # noqa: E402
from vq12 import VQ12                             # noqa: E402

PIECES = ("w1_lo", "w1_hi", "w1_cb", "s1", "w3_lo", "w3_hi", "w3_cb", "s3",
          "w2_lo", "w2_hi", "w2_cb", "s2")
# |value| * 2 on the E2M1 grid -> magnitude code
MAG = torch.zeros(13, dtype=torch.uint8)
for v, c in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 5), (8, 6), (12, 7)):
    MAG[v] = c


def to_codes(vals: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CB3-dequantised bf16 [N, K] + UE8M0 [N, K/32] -> packed FP4 bytes [N, K/2].

    The values sit exactly on the grid times a power of two, so the round trip is exact.
    """
    s = torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)
    t = (vals.float() / s * 2).round().to(torch.int32)
    codes = MAG.to(vals.device)[t.abs().clamp(max=12)] | ((t < 0).to(torch.uint8) << 3)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=":".join(os.path.expanduser(f"~/dsv41-spark/models/{n}")
                                              for n in ("cb3_store", "cb3_store2", "cb3_store3")))
    ap.add_argument("--out", default=os.path.expanduser("~/dsv41-spark/models/vq12_store"))
    ap.add_argument("--trace", default=os.path.expanduser(
        "~/dsv41-spark/work/results/trace-full-20260910/stats/coverage.json"))
    ap.add_argument("--budget-gb", type=float, default=165.0)
    a = ap.parse_args()

    dev = "cuda"
    vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
    src = {}
    meta = None
    for path in a.src.split(":"):
        m = json.load(open(path + ".json"))
        meta = meta or m
        fh = open(path + ".bin", "rb")
        for k, rec in m["records"].items():
            src[tuple(int(x) for x in k.split(","))] = (fh, int(rec))
    stride = int(meta["stride"])
    offsets = {k: (int(o), int(n)) for k, (o, n, _) in meta["offsets"].items()}
    print(f"source: {len(src)} CB3 records, stride {stride:,} B", flush=True)

    ranked = [tuple(int(x) for x in k) for k in rank_from_trace(a.trace)]
    todo = [k for k in ranked if k in src][: int(a.budget_gb * 1e9 // stride)]
    print(f"packing {len(todo)} experts as VQ12 -> {len(todo)*stride/1e9:.1f} GB", flush=True)

    cb = C3.CB3ArenaV2(1, dev)
    from engine.codebook_sim import CodebookSim
    cb.sim = CodebookSim(3, dev)
    ar = VQM.VQ12Arena(1, dev).attach(vq)
    buf = torch.empty(stride, dtype=torch.uint8, pin_memory=True)
    out = open(a.out + ".bin", "wb", buffering=0)
    recmap, t0 = {}, time.time()
    for i, key in enumerate(todo):
        fh, rec = src[key]
        fh.seek(rec * stride)
        raw = torch.frombuffer(bytearray(fh.read(stride)), dtype=torch.uint8)
        for p in PIECES:
            o, n = offsets[p]
            getattr(cb, p)[0].view(-1).copy_(raw[o:o + n])
        w1, w2, w3 = (dequant_cb3_v2(cb.w1_lo[0], cb.w1_hi[0], cb.w1_cb[0], cb.s1[0]),
                      dequant_cb3_v2(cb.w2_lo[0], cb.w2_hi[0], cb.w2_cb[0], cb.s2[0]),
                      dequant_cb3_v2(cb.w3_lo[0], cb.w3_hi[0], cb.w3_cb[0], cb.s3[0]))
        ar.load_slot(0, to_codes(w1, cb.s1[0]), cb.s1[0], to_codes(w2, cb.s2[0]), cb.s2[0],
                     to_codes(w3, cb.s3[0]), cb.s3[0])
        for p in PIECES:
            o, n = offsets[p]
            buf[o:o + n].copy_(getattr(ar, p)[0].reshape(-1).cpu())
        out.write(buf.numpy().tobytes())
        recmap[f"{key[0]},{key[1]}"] = i
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  {(i+1)*stride/1e9:.1f} GB  {el:.0f}s  "
                  f"eta {(len(todo)-i-1)/((i+1)/el)/60:.1f} min", flush=True)
    out.close()
    json.dump({"stride": stride, "offsets": meta["offsets"], "records": recmap,
               "n": len(todo), "format": "vq12_in_cb3_slots",
               "note": "re-encoded from the CB3 store: speed measurements only"},
              open(a.out + ".json", "w"))
    print(f"done: {len(todo)} experts, {len(todo)*stride/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
