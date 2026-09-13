"""Residual (two stage) vector quantiser: the codebook the Spark's decode kernel actually wants.

Measured on GB10, swapping only the table in the VQ12 up kernel: a 4096-entry table costs 1.44 ms,
a 64-entry one 1.05, two 64-entry fp16 tables summed 0.99, and CB3 v3's own up kernel is 0.92. The
table's SIZE is what binds, not the number of gathers -- so a codebook with 4096 effective entries
has to be stored as something small. A residual quantiser does that: w ~ C1[i1] + C2[i2] with 64
entries each, 6 + 6 bits per four weights, the same 3 bit/weight and the same plane sizes as the
single-stage dim-4 VQ.

Because the kernel reads fp16 straight out of the two tables, the entries no longer have to sit on
the E2M1 grid -- the snap cost 14.85 % -> 16.01 % rel-RMSE in the single-stage codebook.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from lossy import FP4, vq_codebook


def kmeans(pts, w, K, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), K, replace=False, p=w / w.sum())].copy()
    for _ in range(iters):
        a = (((c * c).sum(1))[None, :] - 2.0 * (pts @ c.T)).argmin(1)
        num = np.zeros_like(c)
        den = np.zeros(K)
        np.add.at(num, a, pts * w[:, None])
        np.add.at(den, a, w)
        keep = den > 0
        c[keep] = num[keep] / den[keep, None]
    a = (((c * c).sum(1))[None, :] - 2.0 * (pts @ c.T)).argmin(1)
    return c, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--k2", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4, help="alternations between the two stages")
    ap.add_argument("--out", default="results/vq_res_64x64.npz")
    a = ap.parse_args()

    nib = np.load(a.sample)["nib"]
    t = nib.reshape(-1, 4).astype(np.int64)
    idx = ((t[:, 0] * 16 + t[:, 1]) * 16 + t[:, 2]) * 16 + t[:, 3]
    hist = np.bincount(idx, minlength=65536).astype(np.float64)
    used = np.flatnonzero(hist)
    w = hist[used]
    pts = np.stack([FP4[(used >> (4 * (3 - j))) & 15] for j in range(4)], 1)
    pw = float((w[:, None] * pts * pts).sum())
    print(f"{nib.size/1e6:.1f}M weights, {len(used)} distinct 4-tuples", flush=True)

    c1, a1 = kmeans(pts, w, a.k1)
    r = pts - c1[a1]
    c2, a2 = kmeans(r, w, a.k2, seed=1)
    for _ in range(a.rounds):                       # re-fit each stage against the other's residual
        c1, a1 = kmeans(pts - c2[a2], w, a.k1)
        c2, a2 = kmeans(pts - c1[a1], w, a.k2, seed=1)
    err = pts - (c1[a1] + c2[a2])
    rel = np.sqrt(float((w[:, None] * err * err).sum()) / pw)

    # the same budget spent on one flat codebook, snapped and not, for the comparison
    _, _, D, pw4, _, _, Dg = vq_codebook(nib, 4, 4096)
    print(f"  residual {a.k1} + {a.k2}, fp16 entries      rel-RMSE {rel*100:.2f}%")
    print(f"  single stage 4096, float                  rel-RMSE {np.sqrt(D/pw4)*100:.2f}%")
    print(f"  single stage 4096, snapped to E2M1        rel-RMSE {np.sqrt(Dg/pw4)*100:.2f}%")

    ent = np.zeros(65536, np.int32)
    ent[used] = (a1 * a.k2 + a2).astype(np.int32)   # the 12-bit index a record would store
    np.savez(a.out, c1=c1.astype(np.float32), c2=c2.astype(np.float32), enc=ent,
             k1=a.k1, k2=a.k2, bits=3.0, rel_rmse=rel)
    json.dump({"k1": a.k1, "k2": a.k2, "bits": 3.0, "rel_rmse_residual": float(rel),
               "rel_rmse_flat4096_float": float(np.sqrt(D / pw4)),
               "rel_rmse_flat4096_snapped": float(np.sqrt(Dg / pw4))},
              open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
