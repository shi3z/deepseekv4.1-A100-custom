"""How much quality can a register-only (<= 8 entry, prmt-able) codebook actually buy?

CB3 spends its three bits on eight of the sixteen E2M1 grid values, chosen per row. The kernel can
afford that because `prmt.b32` is an eight-byte table lookup in one instruction and
`cvt.rn.f16x2.e2m1x2` turns the resulting code into fp16 for free. Everything richer than eight
entries has to come out of memory, and on GB10 that gather is a third of the MoE step.

So the question is what else fits in eight entries. Measured here on real expert weights, all in
the "E2M1 unit" domain the format quantises in (w divided by its group's UE8M0 scale), with the
same scale^2 weighting CB3's own subset search uses:

  cb3         eight of the sixteen grid values, per row                       (what ships)
  free8       eight arbitrary fp16 levels, per row                            (candidate A)
  free8_blk   eight arbitrary fp16 levels, per row and 512-weight block       (candidate B)
  affine      cb3's subset, then a per-row-and-block a*x + b                  (candidate C)
  pair        w0 = A[i] + B[j], w1 = A[i] - B[j], two eight-entry tables      (candidate D)
  vq6 / vq12  the memory-table formats, for the ceiling
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from st import Checkpoint

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def rows_of(ck, name_w, name_s, rows):
    """-> u [rows, K] in E2M1 units (codes as values) and w2 [rows, K] the scale^2 weight."""
    w = ck.bytes(name_w)[:rows]
    s = ck.bytes(name_s)[:rows]
    codes = np.stack([w & 0xF, w >> 4], -1).reshape(w.shape[0], -1)
    u = FP4[codes]
    sc = np.exp2(s.astype(np.float64) - 127.0).repeat(32, axis=1)
    return u, sc * sc


def err(u, w2, q):
    return float((w2 * (u - q) ** 2).sum())


def quant_to(u, levels):
    """nearest level, per row: levels [R, L]."""
    d = np.abs(u[:, :, None] - levels[:, None, :])
    return np.take_along_axis(levels, d.argmin(2), axis=1)


def lloyd8(u, w2, L=8, iters=30):
    """Weighted Lloyd-Max over the sixteen grid values, per row -> levels [R, L]."""
    R = u.shape[0]
    hist = np.zeros((R, 16))
    idx = np.abs(u[:, :, None] - FP4[None, None, :]).argmin(2)
    np.add.at(hist, (np.repeat(np.arange(R), u.shape[1]), idx.reshape(-1)), w2.reshape(-1))
    g = FP4[None, :]
    grid = np.unique(FP4)                                  # 15 distinct values (+0 == -0)
    lev = np.tile(grid[np.linspace(0, len(grid) - 1, L).astype(int)][None, :], (R, 1))
    for _ in range(iters):
        a = np.abs(g[:, :, None] - lev[:, None, :]).argmin(2)          # [R, 16] -> level
        num = np.zeros((R, L)); den = np.zeros((R, L))
        np.add.at(num, (np.repeat(np.arange(R), 16), a.reshape(-1)),
                  (hist * g).reshape(-1))
        np.add.at(den, (np.repeat(np.arange(R), 16), a.reshape(-1)), hist.reshape(-1))
        keep = den > 0
        lev = np.where(keep, np.divide(num, np.maximum(den, 1e-30)), lev)
    return lev


def cb3_levels(u, w2):
    """CB3's own choice: the eight grid values minimising the scale^2 error, exhaustively."""
    from itertools import combinations
    R = u.shape[0]
    hist = np.zeros((R, 16))
    idx = np.abs(u[:, :, None] - FP4[None, None, :]).argmin(2)
    np.add.at(hist, (np.repeat(np.arange(R), u.shape[1]), idx.reshape(-1)), w2.reshape(-1))
    subs = np.array(list(combinations(range(16), 8)))                  # [12870, 8]
    vals = FP4[subs]                                                   # [12870, 8]
    # cost[s, g] = (g - nearest value in subset s)^2
    cost = ((FP4[None, :, None] - vals[:, None, :]) ** 2).min(2)       # [12870, 16]
    best = (hist @ cost.T).argmin(1)
    return vals[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--experts", default="29:75,15:177,3:12,37:200")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--out", default="results/cb_designs.json")
    a = ap.parse_args()
    ck = Checkpoint(a.ckpt)
    tot = {}
    for spec in a.experts.split(","):
        L, E = spec.split(":")
        for m in ("w1", "w2"):
            q = f"layers.{L}.ffn.experts.{E}.{m}"
            u, w2 = rows_of(ck, q + ".weight", q + ".scale", a.rows)
            den = float((w2 * u * u).sum())
            tot.setdefault("denom", 0.0)
            tot["denom"] += den

            lev = cb3_levels(u, w2)
            tot["cb3"] = tot.get("cb3", 0.0) + err(u, w2, quant_to(u, lev))

            lev8 = lloyd8(u, w2)
            tot["free8"] = tot.get("free8", 0.0) + err(u, w2, quant_to(u, lev8))

            # the levels have to be stored. One byte each keeps CB3's record layout byte for byte
            # (the cb plane is already 8 B a row), so score the level codecs that fit in a byte.
            def e4m3(v):
                sg = np.sign(v); x = np.abs(v)
                e = np.floor(np.log2(np.maximum(x, 1e-9)))
                m = np.round(x / 2.0 ** e * 8.0) / 8.0
                return np.where(x < 2.0 ** -9, 0.0, sg * m * 2.0 ** e)
            for tag, f in (("free8_int8", lambda v: np.round(v * 127.0 / 6.0) * 6.0 / 127.0),
                           ("free8_e4m3", e4m3)):
                lq = f(lev8)
                tot[tag] = tot.get(tag, 0.0) + err(u, w2, quant_to(u, lq))

            # per 512-weight block, same eight-level search
            K = u.shape[1]
            e_blk = 0.0
            for b in range(0, K, 512):
                ub, wb = u[:, b:b + 512], w2[:, b:b + 512]
                e_blk += err(ub, wb, quant_to(ub, lloyd8(ub, wb)))
            tot["free8_blk512"] = tot.get("free8_blk512", 0.0) + e_blk

            # CB3's subset plus a free per-row affine
            qq = quant_to(u, lev)
            sxy = (w2 * qq * u).sum(1); sxx = (w2 * qq * qq).sum(1)
            sx = (w2 * qq).sum(1); sy = (w2 * u).sum(1); sw = w2.sum(1)
            det = sxx * sw - sx * sx
            al = np.where(det > 0, (sxy * sw - sx * sy) / np.maximum(det, 1e-30), 1.0)
            be = np.where(det > 0, (sxx * sy - sx * sxy) / np.maximum(det, 1e-30), 0.0)
            tot["cb3_affine"] = tot.get("cb3_affine", 0.0) + err(u, w2, al[:, None] * qq + be[:, None])

            # the memory-table formats, scored the same way: their LUTs map packed bytes to packed
            # bytes, so applying them to the raw payload gives the quantised grid values directly
            raw = ck.bytes(q + ".weight")[:a.rows]
            for tag, path in (("vq6_dim2_64", "results/vq2_3.npz"), ("vq12_dim4_4096", "results/vq_3.0.npz")):
                d = np.load(path)
                pair = raw[:, 0::2].astype(np.int64) | (raw[:, 1::2].astype(np.int64) << 8)
                o = d["lut_lo"][pair].astype(np.int64) | (d["lut_hi"][pair].astype(np.int64) << 8)
                b0 = (o & 0xFF).astype(np.uint8); b1 = (o >> 8).astype(np.uint8)
                oq = np.empty_like(raw)
                oq[:, 0::2] = b0; oq[:, 1::2] = b1
                cq = np.stack([oq & 0xF, oq >> 4], -1).reshape(raw.shape[0], -1)
                tot[tag] = tot.get(tag, 0.0) + err(u, w2, FP4[cq])

            # candidate D: w0 = A[i] + B[j], w1 = A[i] - B[j], two eight-entry free tables per row
            R, K = u.shape
            p0, p1 = u[:, 0::2], u[:, 1::2]
            wp = 0.5 * (w2[:, 0::2] + w2[:, 1::2])
            m, dd = 0.5 * (p0 + p1), 0.5 * (p0 - p1)          # the rotated coordinates
            A = np.tile(np.quantile(m, np.linspace(0.06, 0.94, 8), axis=1).T, (1, 1))
            B = np.tile(np.quantile(dd, np.linspace(0.06, 0.94, 8), axis=1).T, (1, 1))
            for _ in range(25):
                ia = np.abs(m[:, :, None] - A[:, None, :]).argmin(2)
                ib = np.abs(dd[:, :, None] - B[:, None, :]).argmin(2)
                for T, idx, src in ((A, ia, m), (B, ib, dd)):
                    num = np.zeros_like(T); den = np.zeros_like(T)
                    rr = np.repeat(np.arange(R), idx.shape[1])
                    np.add.at(num, (rr, idx.reshape(-1)), (wp * src).reshape(-1))
                    np.add.at(den, (rr, idx.reshape(-1)), wp.reshape(-1))
                    T[:] = np.where(den > 0, num / np.maximum(den, 1e-30), T)
            qa = np.take_along_axis(A, ia, 1); qb = np.take_along_axis(B, ib, 1)
            e = ((w2[:, 0::2] * (p0 - (qa + qb)) ** 2).sum() +
                 (w2[:, 1::2] * (p1 - (qa - qb)) ** 2).sum())
            tot["pair_sumdiff"] = tot.get("pair_sumdiff", 0.0) + float(e)
    print(f"{a.rows} rows x {len(a.experts.split(',')) * 2} matrices, "
          f"scale^2-weighted relative RMSE in E2M1 units:")
    out = {}
    for k, v in tot.items():
        if k == "denom":
            continue
        out[k] = float(np.sqrt(v / tot["denom"]))
        print(f"  {k:16s} {out[k]*100:6.2f} %")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
