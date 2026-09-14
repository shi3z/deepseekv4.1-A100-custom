"""Faster `cb3.fp4_to_cb3_v2` (the CB3 arena fill path) -- bit-identical output.

Why it matters: with EXPERT_FORMAT=cb3 and streaming, every expert miss runs this function, and it
is what makes a CB3 miss cost ~25-40 ms instead of the 4.6 ms the NVMe read takes. Measured on
gx10-b872: an unpruned CB3 stream reaches NVMe 0.069 GB/tok and hit 0.9872 (against FP4's 0.233 and
0.9585) and still loses -- 148.5 ms/tok vs 108.7 -- purely on fill cost.

The cost is not the 12,870-subset search (a 950 MFLOP GEMM). It is that the original materialises
the [N, K] nibble array as int64 TWICE (189 MB per w13) for the scatter_add and the codebook gather.
This version keeps both passes in the packed byte domain, [N, K/2]:

  histogram   scatter_add over the packed bytes into 256 bins, folded to 16 by a constant
              [256, 16] nibble-count matrix (both nibbles of a byte share a group scale, so the
              weight is the same for each);
  index map   a per-row [N, 256] byte table that maps a packed byte straight to its two 3-bit
              codebook indices, gathered once and split with uint8 shifts.
"""

from __future__ import annotations

import torch


def _nibble_count_matrix(device):
    b = torch.arange(256, device=device)
    m = torch.zeros(256, 16, device=device, dtype=torch.float32)
    m.scatter_add_(1, (b & 0xF)[:, None], torch.ones(256, 1, device=device))
    m.scatter_add_(1, (b >> 4)[:, None], torch.ones(256, 1, device=device))
    return m


_CACHE = {}


def _tables(device):
    if device not in _CACHE:
        b = torch.arange(256, device=device)
        _CACHE[device] = (_nibble_count_matrix(device), (b & 0xF), (b >> 4))
    return _CACHE[device]


def fp4_to_cb3_v2_fast(w_packed: torch.Tensor, scale: torch.Tensor, sim) -> tuple:
    import cb3 as CB3
    N, K2 = w_packed.shape
    x = w_packed.view(torch.uint8)
    nib_m, blo, bhi = _tables(x.device)
    xl = x.to(torch.int64)                                         # [N, K/2]
    scale2 = torch.exp2(2.0 * (scale.view(torch.uint8).float() - 127.0))
    wgt = scale2.repeat_interleave(16, dim=1)                      # one scale per 16 packed bytes
    bhist = torch.zeros(N, 256, device=x.device, dtype=torch.float32).scatter_add_(1, xl, wgt)
    hist = bhist @ nib_m                                           # [N, 16]
    best = (hist @ sim.cost.T).argmin(dim=1)                       # [N]
    pos = sim.pos[best]                                            # [N, 16] uint8, 0..7
    byte_idx = pos[:, blo] | (pos[:, bhi] << 4)                    # [N, 256] uint8
    packed_idx = torch.gather(byte_idx, 1, xl).to(torch.uint8)     # [N, K/2]
    idx = torch.empty(N, K2 * 2, dtype=torch.uint8, device=x.device)
    idx[:, 0::2] = packed_idx & 0x0F
    idx[:, 1::2] = packed_idx >> 4
    codebook = CB3._pad_codebook(sim.subsets_t[best].long())
    return CB3.pack_idx_v2(idx, codebook)


def install():
    """Replace cb3.fp4_to_cb3_v2 (and the name already imported into cb3_moe) with the fast one."""
    import cb3 as CB3
    CB3._fp4_to_cb3_v2_orig = CB3.fp4_to_cb3_v2
    CB3.fp4_to_cb3_v2 = fp4_to_cb3_v2_fast
    try:
        import cb3_moe as C3
        C3.fp4_to_cb3_v2 = fp4_to_cb3_v2_fast
    except Exception:
        pass


if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
    import cb3 as CB3
    from engine.codebook_sim import CodebookSim

    dev = "cuda"
    sim = CodebookSim(3, dev)
    torch.manual_seed(0)
    for (N, K) in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(118, 126, (N, K // 32), dtype=torch.uint8, device=dev)
        a = CB3.fp4_to_cb3_v2(w, s, sim)
        b = fp4_to_cb3_v2_fast(w, s, sim)
        same = all(bool((x == y).all()) for x, y in zip(a, b))

        def t(fn, it=5):
            for _ in range(2):
                fn(w, s, sim)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(it):
                fn(w, s, sim)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / it * 1e3

        print(f"[{N}, {K}] identical: {same}   original {t(CB3.fp4_to_cb3_v2):7.2f} ms   "
              f"fast {t(fp4_to_cb3_v2_fast):7.2f} ms")
