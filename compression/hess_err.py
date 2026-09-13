"""Activation-weighted reconstruction error of each 3-bit format, on real experts.

Plain weight RMSE says nothing about how much a format costs the model: what matters is the error
along the directions the activations actually occupy. calib_collect.py already recorded the diagonal
second moment of every expert's input over a calibration set, so

    err(L, E) = sum_j H[L, E, j] * sum_i (w[i, j] - q[i, j])^2

is the quantity a GPTQ-style method minimises, and it is comparable across formats. Three of the
four formats here have a measured wikitext perplexity, so the fourth can be placed on that axis
rather than guessed at.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark"))
from st import Checkpoint                                       # noqa: E402

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    c = torch.stack([w & 0xF, w >> 4], -1).reshape(w.shape[0], -1).long()
    return FP4.to(w.device)[c] * torch.exp2(s.float() - 127.0).repeat_interleave(32, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--hess", default="results/calib_hess.pt")
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--device", type=int, default=5)
    ap.add_argument("--out", default="results/hess_err.json")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)

    import cb3 as CB3
    from cb3_moe import CB3_BYTES_PER_SLOT                       # noqa: F401  (import check)
    from codebook_sim import CodebookSim
    from cbf8 import CBF8
    from vq12 import VQ12
    from vq6 import VQ6

    sim = CodebookSim(3, dev)
    f8 = CBF8(dev)
    v12 = VQ12("results/vq_3.0.npz", dev)
    v6 = VQ6("results/vq2_3.npz", dev)
    H = torch.load(a.hess, map_location="cpu")
    ck = Checkpoint(a.ckpt)
    rng = np.random.default_rng(0)
    picks = [(int(rng.integers(0, 40)), int(rng.integers(0, 384))) for _ in range(a.experts)]

    tot = {k: 0.0 for k in ("cb3", "cbf8", "vq6", "vq12")}
    den = 0.0
    for n, (L, E) in enumerate(picks):
        for m, hk in (("w1", "H13"), ("w3", "H13"), ("w2", "H2")):
            q = f"layers.{L}.ffn.experts.{E}.{m}"
            w = torch.from_numpy(ck.bytes(q + ".weight")).to(dev)
            s = torch.from_numpy(ck.bytes(q + ".scale")).to(dev)
            ex = dq(w, s)
            h = H[hk][L, E].to(dev).float()
            den += float(((ex * ex) * h[None, :]).sum())

            lo, hi, cb = CB3.fp4_to_cb3_v2(w, s, sim)
            got = {"cb3": CB3.dequant_cb3_v2(lo, hi, cb, s).float()}
            lo, hi, cb, sq = f8.pack(w, s)
            got["cbf8"] = f8.dequant(lo, hi, cb, sq).float()
            lo, hi, cb = v6.pack(w, s)
            got["vq6"] = v6.dequant(lo, hi, s).float()
            lo, hi, cb = v12.pack(w, s)
            got["vq12"] = v12.dequant(lo, hi, s).float()
            for k, v in got.items():
                d = ex - v
                tot[k] += float(((d * d) * h[None, :]).sum())
        if (n + 1) % 8 == 0:
            print(f"  {n+1}/{len(picks)} experts", flush=True)

    print(f"activation-weighted relative error over {len(picks)} experts "
          f"(sum_j H_j sum_i (w-q)^2 / same with w^2):")
    out = {}
    for k, v in tot.items():
        out[k] = float(np.sqrt(v / den))
        print(f"  {k:6s} {out[k]*100:6.2f} %")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
