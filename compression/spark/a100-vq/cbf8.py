"""CBF8: CB3's three bits and CB3's record, with the E2M1 grid taken off the reconstruction levels.

CB3 spends 3 bit/weight on eight of the sixteen E2M1 grid values, chosen per row. Measured on real
expert rows (cb_designs.py, scale^2-weighted relative RMSE in E2M1 units):

    CB3, eight of the sixteen grid values         21.31 %
    eight ARBITRARY levels, per row               17.77 %      <- this format
    eight arbitrary levels, per 512-weight block  17.63 %
    CB3 plus a free per-row affine                21.13 %
    VQ6  (dim 2, 64 free pairs, needs a gather)   20.72 %
    VQ12 (dim 4, 4096 free quads, needs a gather) 16.30 %

So most of CB3's damage is not the three bits: it is that 0, 0.5, 1, 1.5, 2, 3, 4, 6 is a poor set
of levels for a block of weights. Freeing them beats the dim-2 vector quantiser outright and comes
within 1.5 points of the dim-4 one -- while staying a single eight-entry lookup, which is the only
kind of lookup a GB10 does for free (`prmt.b32`, one instruction, no memory access).

Storing a level costs one byte and nothing else: rounding the levels to int8 with a fixed 6/127
scale measures 17.78 %, indistinguishable from exact. CB3's `cb` plane is already eight bytes a
row, so **the record layout, its stride and the store are byte-for-byte unchanged**; only the
meaning of those eight bytes and the kernel's decode differ.

    lo [N, K/4]   two index bits per weight, CB3 v2's bit layout, unchanged
    hi [N, K/8]   the third bit, unchanged
    cb [N, 8]     eight int8 level codes, level = code * 6/127   (CB3: eight E2M1 codes)
    s  [N, K/32]  the UE8M0 group scales, unchanged
"""

from __future__ import annotations

import torch

LEV_SCALE = 6.0 / 127.0
FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@torch.no_grad()
def choose_levels(hist: torch.Tensor, grid: torch.Tensor, n: int = 8) -> torch.Tensor:
    """The optimal eight levels for a row, exactly, as int8 codes.

    The quantiser only ever sees the sixteen grid values and on a line the optimal partition into
    eight cells is contiguous, so this is the classic 1-D k-means dynamic program over the sorted
    points -- O(16^2 * 8) a row, vectorised across rows, with no local minimum to fall into.
    Lloyd-Max with an asymmetric initialisation lands 0.8 points WORSE than CB3 here; the exact
    partition cannot, because CB3's own choice (eight grid values, nearest-neighbour cells) is one
    of the partitions this searches.
    """
    dev = hist.device
    g, order = grid.sort()
    h = hist[:, order]
    N, M = h.shape
    z = torch.zeros(N, 1, device=dev)
    cw = torch.cat([z, h.cumsum(1)], 1)
    cx = torch.cat([z, (h * g).cumsum(1)], 1)
    cxx = torch.cat([z, (h * g * g).cumsum(1)], 1)
    # sse[:, a, b] = the cost of putting points a..b-1 in one cell; a >= b is forbidden
    w = cw[:, None, :] - cw[:, :, None]
    sx = cx[:, None, :] - cx[:, :, None]
    sse = (cxx[:, None, :] - cxx[:, :, None]) - sx * sx / w.clamp_min(1e-30)
    ab = torch.arange(M + 1, device=dev)
    sse = torch.where(ab[None, :, None] < ab[None, None, :], sse.clamp_min(0.0),
                      torch.full_like(sse, 1e30))
    cost = sse[:, 0, :].clone()
    cost[:, 0] = 0.0
    cut = torch.zeros(N, M + 1, n, dtype=torch.long, device=dev)
    for j in range(1, n):
        cand = cost[:, :, None] + sse                      # [N, k, b]
        cost, cut[:, :, j] = cand.min(1)
    lev = torch.zeros(N, n, device=dev)
    b = torch.full((N,), M, dtype=torch.long, device=dev)
    rows = torch.arange(N, device=dev)
    for j in range(n - 1, -1, -1):
        a = cut[rows, b, j] if j > 0 else torch.zeros_like(b)
        ww = cw[rows, b] - cw[rows, a]
        lev[:, j] = torch.where(ww > 0, (cx[rows, b] - cx[rows, a]) / ww.clamp_min(1e-30),
                                torch.zeros_like(ww))
        b = a
    return (lev / LEV_SCALE).round().clamp(-127, 127).to(torch.int8)


SHIFTS = (-1, 0, 1)


class CBF8:
    """Stateless: the codebook is per row, so there is nothing global to carry.

    `pack` also returns a NEW scale plane. The UE8M0 exponent of a 32-weight block was chosen for a
    sixteen-level E2M1 grid; eight free levels want a different one, and moving it is free -- the
    byte is already in the record and the kernel reads it either way. Measured on real experts it
    is worth two points:

        CB3                                21.3 %
        CBF8, the checkpoint's scales      17.7 %
        CBF8, scales re-chosen             15.6 %      (VQ12, which needs a 16 kB gather: 16.3 %)
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.grid = FP4_VALS.to(device).float()
        # the value a code takes under each candidate exponent shift: u = g * 2^-d
        vals = torch.cat([self.grid * (2.0 ** -d) for d in SHIFTS])
        self.uvals, inv = torch.unique(vals, sorted=True, return_inverse=True)
        self.utab = inv.view(len(SHIFTS), 16).T.contiguous()          # [16, n_shift] -> alphabet
        self.uval_of = self.uvals

    @torch.no_grad()
    def pack(self, w_packed: torch.Tensor, scale: torch.Tensor, rounds: int = 3):
        """packed FP4 [N, K/2] + UE8M0 [N, K/32] -> (lo, hi, cb [N, 8], s [N, K/32]).

        Every quantity here is a function of the weight's four-bit code and its block's exponent
        shift, and there are only 16 x 3 of those, so the search never builds an [N, K, 8]
        tensor: the nearest level is tabulated on the alphabet and then gathered.
        """
        from cb3 import pack_idx_v2

        x = w_packed.view(torch.uint8).to(self.device)
        s0 = scale.view(torch.uint8).to(self.device).int()
        N, K2 = x.shape
        K, nS = K2 * 2, len(SHIFTS)
        G = K // 32
        codes = torch.stack([x & 0xF, x >> 4], -1).reshape(N, K).long()
        # the value a code takes under each shift, and the exponent that goes with it
        uv = torch.stack([self.grid * (2.0 ** -d) for d in SHIFTS])                  # [nS, 16]
        sidx = torch.full((N, G), SHIFTS.index(0), dtype=torch.long, device=self.device)
        cb = lev = None
        for _ in range(rounds):
            shw = sidx.repeat_interleave(32, dim=1)                                  # [N, K]
            e = (s0 + torch.tensor(SHIFTS, device=self.device)[sidx]).clamp(0, 255)
            w2 = torch.exp2(2.0 * (e.float() - 127.0)).repeat_interleave(32, dim=1)
            ui = self.utab[codes, shw]
            hist = torch.zeros(N, len(self.uvals), device=self.device).scatter_add_(1, ui, w2)
            cb = choose_levels(hist, self.uvals)
            lev = cb.float() * LEV_SCALE
            # nearest level per (shift, code): [N, nS, 16] -> flat, then one gather per weight
            pos = (uv[None, :, :, None] - lev[:, None, None, :]).abs().argmin(3)     # [N, nS, 16]
            qv = torch.gather(lev, 1, pos.reshape(N, -1)).view(N, nS, 16)
            err = []
            for j in range(nS):
                d2 = (uv[j][None, :] - qv[:, j]) ** 2                                # [N, 16]
                ej = (s0 + SHIFTS[j]).clamp(0, 255)
                sj = torch.exp2(2.0 * (ej.float() - 127.0))                          # [N, G]
                per = torch.gather(d2, 1, codes).view(N, G, 32).sum(2) * sj
                err.append(per)
            sidx = torch.stack(err, 0).argmin(0)
        shw = sidx.repeat_interleave(32, dim=1)
        s_new = (s0 + torch.tensor(SHIFTS, device=self.device)[sidx]).clamp(0, 255).to(torch.uint8)
        idx = torch.gather(pos.reshape(N, -1), 1, shw * 16 + codes).to(torch.uint8)
        lo, hi, _ = pack_idx_v2(idx, cb.view(torch.uint8))
        return lo, hi, cb.view(torch.uint8).contiguous(), s_new.contiguous()

    @torch.no_grad()
    def dequant(self, lo: torch.Tensor, hi: torch.Tensor, cb: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        """The reference the kernel must reproduce. `scale` is the plane `pack` returned."""
        from cb3 import unpack_cb3_v2

        idx = unpack_cb3_v2(lo, hi, torch.arange(8, dtype=torch.uint8,
                                                 device=lo.device).repeat(lo.shape[0], 1))
        lev = cb.view(torch.int8).float() * LEV_SCALE
        v = torch.gather(lev, 1, idx)
        s = torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
        return (v * s).to(torch.bfloat16)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
    import v41_ref as R

    f = CBF8()
    torch.manual_seed(0)
    for (N, K) in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(118, 126, (N, K // 32), dtype=torch.uint8, device="cuda")
        lo, hi, cb, s2 = f.pack(w, s)
        assert lo.shape == (N, K // 4) and hi.shape == (N, K // 8) and cb.shape == (N, 8)
        ref = R.dequant_fp4_packed(w, s).float()
        q = f.dequant(lo, hi, cb, s2).float()
        bits = (lo.numel() + hi.numel() + cb.numel()) * 8 / (N * K)
        print(f"[{N}, {K}] planes ok, rel err vs the original {float((q-ref).norm()/ref.norm()):.4f}, "
              f"{bits:.3f} bit/weight + {s.numel()*8/(N*K):.3f} scale")
