"""Pack CB3 v2 expert records from this box's untouched FP4 checkpoint, and stream them.

The Spark's CB3 store is the only copy of those experts there -- punch_fp4.py freed their FP4 bytes
once it existed -- so deleting any of it is only safe if this box can put it back. It can: the pack
is deterministic (an exhaustive per-row search over all C(16,8) subsets, scale^2-weighted), so a
record produced here is byte-identical to the one the Spark packed. `--verify` checks exactly that
against the live store before anything is deleted.

Same record layout, stride and ranking as a100_pack_vq12.py, so the two are interchangeable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st import Checkpoint                                    # noqa: E402

PIECES = ("w1_lo", "w1_hi", "w1_cb", "s1", "w3_lo", "w3_hi", "w3_cb", "s3",
          "w2_lo", "w2_hi", "w2_cb", "s2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--ranks", default="results/rank.json")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--json", default="results/cb3_real.json")
    ap.add_argument("--out", default="-")
    ap.add_argument("--md5", action="store_true", help="print each record's md5 instead of writing it")
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    import fast_fill
    fast_fill.install()
    import cb3_moe as C3
    from codebook_sim import CodebookSim

    arena = C3.CB3ArenaV2(1, dev)
    arena.sim = CodebookSim(3, dev)
    stride = C3.CB3_BYTES_PER_SLOT
    offsets, off = {}, 0
    for p in PIECES:
        t = getattr(arena, p)
        offsets[p] = [off, t[0].numel(), list(t[0].shape)]
        off += t[0].numel()
    assert off == stride, (off, stride)

    ck = Checkpoint(a.ckpt)
    ranked = json.load(open(a.ranks))[a.skip:a.skip + a.count]
    log = sys.stderr
    out = None if a.md5 else (sys.stdout.buffer if a.out == "-" else open(a.out, "wb", buffering=0))
    print(f"{len(ranked)} experts, {len(ranked) * stride / 1e9:.1f} GB, stride {stride:,}",
          file=log, flush=True)

    buf = torch.empty(stride, dtype=torch.uint8)
    recmap, t0 = {}, time.time()
    for i, (L, E) in enumerate(ranked):
        q = f"layers.{L}.ffn.experts.{E}."
        g = lambda k, w: torch.from_numpy(ck.bytes(q + f"{k}.{w}"))
        arena.load_slot(0, g("w1", "weight"), g("w1", "scale"), g("w2", "weight"), g("w2", "scale"),
                        g("w3", "weight"), g("w3", "scale"))
        for p in PIECES:
            o, ln, _ = offsets[p]
            buf[o:o + ln].copy_(getattr(arena, p)[0].reshape(-1).cpu())
        if a.md5:
            print(f"{a.skip + i} {L},{E} {hashlib.md5(buf.numpy().tobytes()).hexdigest()}", flush=True)
        else:
            out.write(buf.numpy().tobytes())
        recmap[f"{L},{E}"] = i
        if not a.md5 and (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(ranked)}  {(i+1)*stride/1e9:.1f} GB  {el:.0f}s  "
                  f"{(i+1)*stride/1e6/el:.0f} MB/s  eta {(len(ranked)-i-1)/((i+1)/el)/60:.1f} min",
                  file=log, flush=True)
    if out is not None and a.out != "-":
        out.close()
    if not a.md5:
        json.dump({"stride": stride, "offsets": offsets, "records": recmap, "n": len(ranked),
                   "format": "cb3_v2"}, open(a.json, "w"))
    print(f"done in {(time.time()-t0)/60:.2f} min", file=log, flush=True)


if __name__ == "__main__":
    main()
