"""A drop-in faster CodebookSim.requant_packed for the DGX Spark engine.

Same output, bit for bit. The original unpacks the FP4 nibbles into an int64 [N, K] tensor (189 MB
for one w13), builds the histogram with a scatter_add over it and then remaps with a second [N, K]
advanced-index gather. Three int64 passes over 23.6 M elements per matrix is what makes a miss cost
~40 ms; the exhaustive 12,870-subset search it is blamed on is a 950 MFLOP GEMM and is not the cost.

This version never leaves the packed byte domain:
  * histogram: scatter_add over the [N, K/2] PACKED bytes (256 bins), then fold 256 -> 16 with a
    constant [256, 16] nibble-count matrix, so the per-weight pass is half the elements and uint8;
  * remap: build a per-row 256-entry byte table from the chosen subset's code map and gather once
    over [N, K/2].
Scale weighting is unchanged: both nibbles of a byte share one 32-weight group scale.
"""

from __future__ import annotations

import torch


def _nibble_count_matrix(device) -> torch.Tensor:
    """[256, 16] float32: how many times code c appears in the byte value b (0, 1 or 2)."""
    b = torch.arange(256, device=device)
    lo, hi = b & 0xF, b >> 4
    m = torch.zeros(256, 16, device=device, dtype=torch.float32)
    m.scatter_add_(1, lo[:, None], torch.ones(256, 1, device=device))
    m.scatter_add_(1, hi[:, None], torch.ones(256, 1, device=device))
    return m


def install(sim):
    """Attach `requant_packed_fast` to a CodebookSim instance and return it."""
    dev = sim.cost.device
    sim._nib_m = _nibble_count_matrix(dev)
    sim._byte_lo = (torch.arange(256, device=dev) & 0xF)
    sim._byte_hi = (torch.arange(256, device=dev) >> 4)

    @torch.no_grad()
    def requant_packed_fast(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        N, K2 = w.shape
        wl = w.to(torch.int64)                                   # [N, K/2] (gather needs int64)
        # per-byte weight = the 32-weight group's scale^2 (16 bytes per group)
        scale2 = torch.exp2(2.0 * (s.float() - 127.0))           # [N, K/32]
        wgt = scale2.repeat_interleave(16, dim=1)                # [N, K/2]
        bhist = torch.zeros(N, 256, device=w.device, dtype=torch.float32)
        bhist.scatter_add_(1, wl, wgt)
        hist = bhist @ sim._nib_m                                # [N, 16]
        best = (hist @ sim.cost.T).argmin(dim=1)                 # [N]
        cmap = sim.near[best]                                    # [N, 16] code -> code
        byte_lut = cmap[:, sim._byte_lo] | (cmap[:, sim._byte_hi] << 4)   # [N, 256]
        return torch.gather(byte_lut, 1, wl).to(torch.uint8)

    sim.requant_packed_fast = requant_packed_fast
    return sim


if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
    from engine.codebook_sim import CodebookSim

    dev = "cuda"
    sim = install(CodebookSim(3, dev))
    torch.manual_seed(0)
    for (N, K) in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(118, 126, (N, K // 32), dtype=torch.uint8, device=dev)
        a = sim.requant_packed(w, s)
        b = sim.requant_packed_fast(w, s)
        same = bool((a == b).all())

        def t(fn, it=5):
            for _ in range(2):
                fn(w, s)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(it):
                fn(w, s)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / it * 1e3

        print(f"[{N}, {K}] identical: {same}   original {t(sim.requant_packed):7.2f} ms   "
              f"fast {t(sim.requant_packed_fast):7.2f} ms")
