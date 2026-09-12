"""Pack VQ12 expert records straight from the untouched FP4 checkpoint on the A100, and stream them.

The Spark's own copy of the routed experts is gone -- `punch_fp4.py` freed those blocks once the CB3
store existed -- so a VQ12 store built there can only be a re-encoding of CB3, which quantises twice
and says nothing about VQ12's real quality. This box still has the original checkpoint, so it packs
the records here and streams them over; neither machine ever has to hold both formats.

Records are the CB3 slot layout byte for byte (the 12 tensors concatenated, 14,454,784 B stride), so
`cb3_store.py` on the Spark reads them with no change:

    w1_lo [2304, 1280]  w1_hi [2304, 640]  w1_cb [2304, 8] zeros  s1 [2304, 160]
    w3_*  same
    w2_lo [5120,  576]  w2_hi [5120, 288]  w2_cb [5120, 8] zeros  s2 [5120,  72]

usage (records to stdout, index to --json):
  python a100_pack_vq12.py --count 11000 --json /tmp/vq12_real.json \\
    | sshpass -e ssh shi3z@<spark> 'cat > ~/dsv41-spark/models/vq12_real.bin'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st import Checkpoint                                    # noqa: E402

DIM, INTER = 5120, 2304
PIECES = (("w1_lo", (INTER, DIM // 4)), ("w1_hi", (INTER, DIM // 8)), ("w1_cb", (INTER, 8)),
          ("s1", (INTER, DIM // 32)),
          ("w3_lo", (INTER, DIM // 4)), ("w3_hi", (INTER, DIM // 8)), ("w3_cb", (INTER, 8)),
          ("s3", (INTER, DIM // 32)),
          ("w2_lo", (DIM, INTER // 4)), ("w2_hi", (DIM, INTER // 8)), ("w2_cb", (DIM, 8)),
          ("s2", (DIM, INTER // 32)))
STRIDE = sum(int(np.prod(s)) for _, s in PIECES)


def pack(enc: torch.Tensor, w: torch.Tensor):
    """packed FP4 [N, K/2] -> (lo [N, K/4], hi [N, K/8]); a group of 4 k is exactly 2 bytes."""
    b0 = w[:, 0::2].long()
    b1 = w[:, 1::2].long()
    idx = enc[b0 | (b1 << 8)]
    lo = (idx & 0xFF).to(torch.uint8)
    hn = (idx >> 8).to(torch.uint8)
    return lo.contiguous(), (hn[:, 0::2] | (hn[:, 1::2] << 4)).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--codebook", default="results/vq_3.0.npz")
    ap.add_argument("--ranks", default="results/rank.json", help="the engine's trace ranking")
    ap.add_argument("--count", type=int, default=11000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--device", type=int, default=5)
    ap.add_argument("--json", default="results/vq12_real.json")
    ap.add_argument("--out", default="-", help="- for stdout (pipe it), or a file path")
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.device}")
    d = np.load(a.codebook)
    enc = torch.from_numpy(d["enc"]).to(dev).long()
    ck = Checkpoint(a.ckpt)
    ranked = json.load(open(a.ranks))[a.skip:a.skip + a.count]
    out = sys.stdout.buffer if a.out == "-" else open(a.out, "wb", buffering=0)
    log = sys.stderr
    print(f"{len(ranked)} experts, {len(ranked) * STRIDE / 1e9:.1f} GB, stride {STRIDE:,}", file=log, flush=True)

    zeros = {"w1_cb": torch.zeros(INTER * 8, dtype=torch.uint8),
             "w3_cb": torch.zeros(INTER * 8, dtype=torch.uint8),
             "w2_cb": torch.zeros(DIM * 8, dtype=torch.uint8)}
    recmap, t0 = {}, time.time()
    for i, (L, E) in enumerate(ranked):
        q = f"layers.{L}.ffn.experts.{E}."
        got = {}
        for m in ("w1", "w2", "w3"):
            w = torch.from_numpy(ck.bytes(q + f"{m}.weight")).to(dev)
            s = torch.from_numpy(ck.bytes(q + f"{m}.scale"))
            lo, hi = pack(enc, w)
            got[m] = (lo.cpu(), hi.cpu(), s)
        for name, shape in PIECES:
            if name in zeros:
                out.write(zeros[name].numpy().tobytes())
                continue
            m = "w" + name[1]
            lo, hi, s = got[m]
            t = lo if name.endswith("_lo") else (hi if name.endswith("_hi") else s)
            assert t.numel() == int(np.prod(shape)), (name, t.shape, shape)
            out.write(t.reshape(-1).numpy().tobytes())
        recmap[f"{L},{E}"] = i
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(ranked)}  {(i+1)*STRIDE/1e9:.1f} GB  {el:.0f}s  "
                  f"{(i+1)*STRIDE/1e6/el:.0f} MB/s  eta {(len(ranked)-i-1)/((i+1)/el)/60:.1f} min",
                  file=log, flush=True)
    if a.out != "-":
        out.close()
    offsets, off = {}, 0
    for name, shape in PIECES:
        n = int(np.prod(shape))
        offsets[name] = [off, n, list(shape)]
        off += n
    json.dump({"stride": STRIDE, "offsets": offsets, "records": recmap, "n": len(ranked),
               "format": "vq12_from_fp4", "codebook": os.path.basename(a.codebook)},
              open(a.json, "w"))
    print(f"done: {len(ranked)} experts in {(time.time()-t0)/60:.1f} min; index -> {a.json}",
          file=log, flush=True)


if __name__ == "__main__":
    main()
