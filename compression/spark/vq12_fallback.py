"""Serve an expert the VQ12 store does not hold by re-encoding its CB3 record on the miss path.

The A100-packed VQ12 store covers the top 12,600 trace ranks (182 GB) because that is all the room
the Spark's disk has beside the 222 GB CB3 store. An expert past that rank cannot fall back to FP4
-- `punch_fp4.py` freed those bytes -- so `guard()` would raise, which is what stopped the first
VQ12 benchmark. Its CB3 record is still there, though, and CB3 and VQ12 share the slot geometry, so
the record can be dequantised and re-packed in place.

The values that come out are CB3-quality (quantised twice), which is exactly why this is a speed
path only: it is counted and timed so the contamination is visible rather than assumed away.
"""

from __future__ import annotations

import time

import torch

# (lo, hi, cb, scale) plane names of one matrix in the CB3 / VQ12 arena
MATS = (("w1_lo", "w1_hi", "w1_cb", "s1"),
        ("w2_lo", "w2_hi", "w2_cb", "s2"),
        ("w3_lo", "w3_hi", "w3_cb", "s3"))


def cb3_to_packed(lo: torch.Tensor, hi: torch.Tensor, cb: torch.Tensor,
                  block_w: int | None = None) -> torch.Tensor:
    """CB3 v2 planes -> packed FP4 [N, K/2], in 8 and 32 bit only.

    `cb3.unpack_cb3_v2` does the same thing through an int64 [N, NB, G, 16, 2] index tensor and an
    int64 `gather`, which is 5.8-6.8 ms a matrix -- the whole conversion has to be cheaper than the
    20.8 ms FP4->CB3 pack the store exists to avoid. Two changes: the index stays uint8, and the
    row's eight E2M1 codes are read out of one 32-bit word by a shift instead of a gather.
    """
    from cb3 import _v2_fields, block_plan

    N, K = lo.size(0), lo.size(1) * 4
    if block_w is None:
        n512, n256 = block_plan(K)
        if n512 and n256:
            return torch.cat([cb3_to_packed(lo[:, :n512 * 128], hi[:, :n512 * 64], cb, 512),
                              cb3_to_packed(lo[:, n512 * 128:], hi[:, n512 * 64:], cb, 256)], 1)
        block_w = 512 if n512 else 256
    NB, G = K // block_w, block_w // 32
    lv = lo.view(N, NB, G // 2, 16)
    hv = hi.view(N, NB, G // 4, 16)
    idx = torch.empty(N, NB, G, 16, 2, dtype=torch.uint8, device=lo.device)
    for g, r, k, sh, m, hb in _v2_fields(block_w):
        idx[:, :, g, :, r] = ((lv[:, :, k, :] >> sh) & 3) | (((hv[:, :, m, :] >> hb) & 1) << 2)
    word = torch.zeros(N, dtype=torch.int32, device=lo.device)
    for i in range(8):
        word |= cb[:, i].to(torch.int32) << (4 * i)
    codes = ((word[:, None] >> (idx.reshape(N, K).to(torch.int32) * 4)) & 0xF).to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


class Converter:
    """Rewrites one arena slot from CB3 planes to VQ12 planes, in place.

    Nothing is dequantised: CB3's per-row codebook holds E2M1 codes, so its planes already carry
    the FP4 code of every weight and the VQ encoder wants exactly those bytes. Going through bf16
    values and back (dequant -> divide by the scale -> round onto the grid) cost 45 ms an expert.
    The scales are untouched, so they stay where the record put them.
    """

    def __init__(self, device):
        self.n = 0
        self.ms = 0.0

    def __call__(self, arena, slot: int) -> None:
        t0 = time.perf_counter()
        for lo_n, hi_n, cb_n, s_n in MATS:
            lo_t, hi_t, cb_t, s_t = (getattr(arena, x) for x in (lo_n, hi_n, cb_n, s_n))
            packed = cb3_to_packed(lo_t[slot], hi_t[slot], cb_t[slot])
            lo, hi, cb = arena.vq.pack(packed, s_t[slot].view(torch.uint8))
            lo_t[slot].copy_(lo)
            hi_t[slot].copy_(hi)
            cb_t[slot].copy_(cb)
        torch.cuda.current_stream().synchronize()
        self.n += 1
        self.ms += (time.perf_counter() - t0) * 1e3
        if self.n % 50 == 0:
            print(f"[vq12] {self.n} CB3->VQ12 fallback conversions, "
                  f"{self.ms / self.n:.1f} ms each, {self.ms / 1e3:.1f} s total", flush=True)
