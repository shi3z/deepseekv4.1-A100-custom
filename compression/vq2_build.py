"""Train a dim-2 vector quantizer and save it in vq_build.py's byte-pair LUT format.

Why dim 2 when dim 4 halves CB3's quantisation damage: measured on the Spark's GB10, the decode
kernel is bound by the *size of the codebook table*, not by the number of gathers. Swapping only
the table size in the VQ12 up kernel, with every instruction otherwise identical:

    4096 entries (16 kB, dim 4)   1.444 ms        64 entries (256 B, dim 2)   1.063 ms
     256 entries  (1 kB)          1.081 ms        no gather at all            0.927 ms
                                                  CB3 v3's up kernel          0.924 ms

A 64-entry table recovers 27 of the 36 points the gather costs. dim 2 at K=64 is 6 bits per two
weights -- the same 3 bit/weight as dim 4 at K=4096 and as CB3 -- and a dim-2 group is exactly one
packed-FP4 byte, so quantising is a 256-entry byte lookup and the byte-pair LUT the rest of this
repo already speaks is just that table applied twice.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lossy import FP4, vq_codebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--k", type=int, default=64, help="codebook entries (64 = 3 bit/weight)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bits = np.log2(a.k) / 2.0
    nib = np.load(a.sample)["nib"]
    print(f"training VQ dim 2, K={a.k} ({bits:.2f} bit/weight) on {nib.size/1e6:.1f}M weights ...",
          flush=True)
    enc, cb, D, pw, encg, cbg, Dg = vq_codebook(nib, 2, a.k)
    print(f"  float codebook rel-RMSE {np.sqrt(D/pw)*100:.2f}%   E2M1-snapped {np.sqrt(Dg/pw)*100:.2f}%")

    codes = np.abs(cbg[:, :, None] - FP4[None, None, :]).argmin(2).astype(np.uint8)   # [K, 2]
    # a byte is one group: low nibble = the even k, high nibble = the odd one
    b = np.arange(256, dtype=np.int64)
    ti = (b & 15) * 16 + (b >> 4)                      # vq_codebook's tuple index (c0 major)
    ent = encg[ti]                                      # byte -> codebook entry
    c = codes[ent]
    lut8 = (c[:, 0].astype(np.int64) | (c[:, 1].astype(np.int64) << 4)).astype(np.uint8)
    u = np.arange(65536, dtype=np.int64)
    out = lut8[u & 0xFF].astype(np.int64) | (lut8[(u >> 8) & 0xFF].astype(np.int64) << 8)
    ident = float((out == u).mean())
    path = a.out or f"results/vq2_{bits:g}.npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, lut_lo=(out & 0xFF).astype(np.uint8), lut_hi=(out >> 8).astype(np.uint8),
             enc=ent.astype(np.int32), codebook=codes, lut8=lut8, bits=bits, K=a.k, dim=2,
             rel_rmse_float=float(np.sqrt(D / pw)), rel_rmse_snapped=float(np.sqrt(Dg / pw)))
    print(f"  {ident*100:.2f}% of the 65536 byte pairs are left unchanged; wrote {path}")
    json.dump({"dim": 2, "bits": float(bits), "K": a.k, "rel_rmse_float": float(np.sqrt(D / pw)),
               "rel_rmse_snapped": float(np.sqrt(Dg / pw)), "identity_fraction": ident},
              open(path.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
