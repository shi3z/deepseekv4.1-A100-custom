"""Build a pre-packed CB3 expert store on the Spark's own NVMe.

Why: with EXPERT_FORMAT=cb3 the arena holds ~31 % more experts, and an unpruned stream measured
NVMe 0.069 GB/tok at hit 0.9872 (against FP4's 0.233 / 0.9585) -- but it LOST, 148.5 ms/tok against
108.7, because every miss runs the FP4 -> CB3 pack (measured 20.8 ms/expert, 13.8 with the faster
fill in a100-vq/fast_fill.py) on top of the read. Pre-packing moves that cost off the miss path and
shrinks the read at the same time: 3.67 misses/token x 14.45 MB instead of x 18.80 MB, and no fill.

Layout: one fixed-stride record per expert, the 12 slot tensors of cb3_moe.CB3ArenaV2 concatenated
in the order (w1_lo, w1_hi, w1_cb, s1, w3_lo, w3_hi, w3_cb, s3, w2_lo, w2_hi, w2_cb, s2) so a miss
is one pread into a pinned buffer and 12 slices copied into the slot. A sidecar JSON carries the
record stride, the piece offsets and the (layer, expert) -> record map.

Packs the experts the arena does NOT warm-start with -- those are the ones that actually stream --
in the engine's own trace ranking order, as many as the given byte budget allows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cb3_moe as C3                      # noqa: E402
from engine.codebook_sim import CodebookSim   # noqa: E402
from engine.experts import rank_from_trace    # noqa: E402
import fast_fill                              # noqa: E402

PIECES = ("w1_lo", "w1_hi", "w1_cb", "s1", "w3_lo", "w3_hi", "w3_cb", "s3",
          "w2_lo", "w2_hi", "w2_cb", "s2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--trace", default=os.path.expanduser(
        "~/dsv41-spark/work/results/trace-full-20260910/stats/coverage.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/dsv41-spark/models/cb3_store"))
    ap.add_argument("--skip", type=int, default=6500, help="experts the arena warm-starts with (not packed)")
    ap.add_argument("--rank-hi", type=int, default=0, help="stop at this trace rank (0 = use --budget-gb)")
    ap.add_argument("--budget-gb", type=float, default=88.0)
    ap.add_argument("--fast", action="store_true", default=True)
    a = ap.parse_args()

    from safetensors import safe_open
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    dev = "cuda"
    sim = CodebookSim(3, dev)
    if a.fast:
        fast_fill.install()
        import cb3 as CB3
        print(f"using the fast fill ({CB3.fp4_to_cb3_v2.__name__})", flush=True)

    ranked = rank_from_trace(a.trace)
    print(f"trace ranking: {len(ranked)} experts", flush=True)
    todo = ranked[a.skip:a.rank_hi] if a.rank_hi else ranked[a.skip:]
    stride = C3.CB3_BYTES_PER_SLOT
    n = min(len(todo), int(a.budget_gb * 1e9 // stride))
    todo = todo[:n]
    print(f"packing ranks {a.skip}..{a.skip + n} = {n} experts, {n * stride / 1e9:.1f} GB, "
          f"record stride {stride:,} B", flush=True)

    arena = C3.CB3ArenaV2(1, dev)
    arena.sim = sim
    offsets, off = {}, 0
    for p in PIECES:
        t = getattr(arena, p)
        offsets[p] = [off, t[0].numel(), list(t[0].shape)]
        off += t[0].numel()
    assert off == stride, (off, stride)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # resume: the file is a fixed-stride record array, so a partial run continues at its end
    done = 0
    if os.path.exists(a.out + ".bin"):
        sz = os.path.getsize(a.out + ".bin")
        done = sz // stride
        if sz % stride:
            print(f"  truncating a partial record ({sz % stride} B)", flush=True)
            with open(a.out + ".bin", "r+b") as t:
                t.truncate(done * stride)
        print(f"  resuming: {done} records already written ({done * stride / 1e9:.1f} GB)", flush=True)
    fh = open(a.out + ".bin", "r+b" if done else "wb", buffering=0)
    fh.seek(done * stride)
    recmap, t0, files = {f"{int(k[0])},{int(k[1])}": j for j, k in enumerate(todo[:done])}, time.time(), {}

    def get(name):
        fn = idx[name]
        if fn not in files:
            files[fn] = safe_open(os.path.join(a.model, fn), "pt", device="cpu")
        return files[fn].get_tensor(name)

    buf = torch.empty(stride, dtype=torch.uint8, device="cpu", pin_memory=True)
    for i, key in enumerate(todo):
        if i < done:
            continue
        L, E = int(key[0]), int(key[1])
        q = f"layers.{L}.ffn.experts.{E}."
        ws = {k: get(q + f"{k}.weight") for k in ("w1", "w2", "w3")}
        ss = {k: get(q + f"{k}.scale") for k in ("w1", "w2", "w3")}
        arena.load_slot(0, ws["w1"], ss["w1"], ws["w2"], ss["w2"], ws["w3"], ss["w3"])
        for p in PIECES:
            o, ln, _ = offsets[p]
            buf[o:o + ln].copy_(getattr(arena, p)[0].reshape(-1).cpu())
        fh.write(buf.numpy().tobytes())
        recmap[f"{L},{E}"] = i
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            rate = (i + 1 - done) / el
            print(f"  {i+1}/{n}  {(i+1)*stride/1e9:.1f} GB  {el:.0f}s  "
                  f"({rate:.1f} experts/s, eta {(n-i-1)/max(rate,1e-9)/60:.1f} min)", flush=True)
    fh.close()
    json.dump({"stride": stride, "offsets": offsets, "records": recmap,
               "n": n, "skip": a.skip, "format": "cb3_v2"}, open(a.out + ".json", "w"))
    print(f"done: {n} experts, {n*stride/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min -> {a.out}.bin")


if __name__ == "__main__":
    main()
