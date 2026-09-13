"""VQ6: a dim-2 vector quantiser at 3 bit/weight, shaped for what the GB10 can actually decode.

VQ12 (dim 4, 4096 entries) halves CB3's quantisation damage and loses 27.5 % of the tokens per
second doing it. Measured with ncu and by ablation on the Spark, the whole gap is one thing: a
random gather into a 16 kB table. What binds is how many distinct L1 sectors a warp's 32 lanes
touch, not the table's bytes or the number of gathers --

    entries   4096      256       64      no gather      CB3 v3's own up kernel
    up kernel 1.444 ms  1.081 ms  1.063 ms  0.927 ms      0.924 ms
    same 4096 entries as uint16 (8 kB) 1.498 ms, as two uint8 planes (4 kB/lane) 1.532 ms

-- so narrowing the entry does nothing and shrinking the codebook does everything. At 3 bit/weight
a 64-entry codebook means dim 2: six bits per two weights, which is exactly one packed-FP4 byte.

Quality, all 40 layers unpruned, wikitext-2 against the FP4 checkpoint (ppl.py):

    CB3 (per-row scalar, 8 of 16)   3.3391  +8.54 %
    VQ6 (this, dim 2, K=64)         3.2693  +6.27 %
    VQ12 (dim 4, K=4096)            3.2067  +4.23 %

The plane sizes and the record stride are CB3's and VQ12's, unchanged, so the store, the arena and
the engine patch do not move:

    lo [N, K/4]  the low 4 bits of two lanes per byte   (lane j -> byte j >> 1, shift (j & 1) * 4)
    hi [N, K/8]  the high 2 bits of four lanes per byte (lane j -> byte j >> 2, shift (j & 3) * 2)
    cb [N, 8]    unused, the codebook is global
    s  [N, K/32] the UE8M0 scales, untouched

One lane is one byte is one entry, so the decode never expands a tile: the kernel reads two
64-entry fp16 tables (the entry's even-k and odd-k value) and feeds tl.dot directly. No 4096-entry
gather, no 16-entry E2M1 table, no duplicated index fetch.
"""

from __future__ import annotations

import numpy as np
import torch

FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class VQ6:
    """The 64 x 2-code codebook and the 256-entry byte encoder."""

    def __init__(self, npz_path: str, device="cuda"):
        d = np.load(npz_path)
        self.codes = torch.from_numpy(d["codebook"]).to(device)      # [64, 2] E2M1 codes
        self.enc = torch.from_numpy(d["enc"]).to(device).long()      # packed byte -> entry 0..63
        assert self.codes.shape == (64, 2), self.codes.shape
        v = FP4_VALS.to(device)
        self.te = v[self.codes[:, 0].long()].to(torch.float16).contiguous()   # even k
        self.to = v[self.codes[:, 1].long()].to(torch.float16).contiguous()   # odd k
        self.device = device

    @torch.no_grad()
    def pack(self, w_packed: torch.Tensor, scale: torch.Tensor):
        """packed FP4 [N, K/2] -> (lo [N, K/4], hi [N, K/8], cb [N, 8] zeros).

        The bit layout is CB3 v3's, not the obvious one: within a 128-weight block the 64 byte
        lanes are four scale groups of 16, and lane m of EVERY group lands in byte m of the tile
        the kernel loads. A group is then one constant shift away, so the decode never needs a
        per-lane shift amount -- writing it the obvious way (lane j reads byte j >> 1 and shifts by
        (j & 1) * 4) measured 2.29 ms against VQ12's 2.14, because a variable shift per element
        costs more than the 16 kB gather it was meant to replace.
        """
        x = w_packed.view(torch.uint8).to(self.device)
        N, K2 = x.shape
        assert K2 % 64 == 0, K2
        e = self.enc[x.long()].view(N, K2 // 64, 4, 16)               # [N, blocks, group, lane]
        lo = torch.cat([(e[:, :, 0] & 0xF) | ((e[:, :, 1] & 0xF) << 4),
                        (e[:, :, 2] & 0xF) | ((e[:, :, 3] & 0xF) << 4)], dim=2)
        h = (e >> 4) & 3
        hi = h[:, :, 0] | (h[:, :, 1] << 2) | (h[:, :, 2] << 4) | (h[:, :, 3] << 6)
        cb = torch.zeros(N, 8, dtype=torch.uint8, device=self.device)
        return (lo.reshape(N, -1).to(torch.uint8).contiguous(),
                hi.reshape(N, -1).to(torch.uint8).contiguous(), cb)

    @torch.no_grad()
    def unpack(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """-> the entry of every byte lane, [N, K/2], in k order."""
        N = lo.shape[0]
        nb = hi.shape[1] // 16
        l = lo.long().view(N, nb, 2, 16)
        h = hi.long().view(N, nb, 16)
        e = torch.stack([((l[:, :, g // 2] >> ((g % 2) * 4)) & 0xF) | (((h >> (2 * g)) & 3) << 4)
                         for g in range(4)], dim=2)                   # [N, blocks, group, lane]
        return e.reshape(N, -1)

    @torch.no_grad()
    def dequant(self, lo, hi, scale) -> torch.Tensor:
        e = self.unpack(lo, hi)
        c = self.codes.to(lo.device).long()[e]                        # [N, K/2, 2]
        v = FP4_VALS.to(lo.device)[c].reshape(lo.shape[0], -1)        # [N, K] in E2M1 units
        s = torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
        return (v * s).to(torch.bfloat16)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
    import v41_ref as R

    vq = VQ6(os.path.expanduser("~/dsv41-spark/a100-vq/vq2_3.npz"))
    torch.manual_seed(0)
    for (N, K) in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(118, 126, (N, K // 32), dtype=torch.uint8, device="cuda")
        lo, hi, cb = vq.pack(w, s)
        assert lo.shape == (N, K // 4) and hi.shape == (N, K // 8)
        e = vq.unpack(lo, hi)
        idem = bool((e == vq.enc[w.long()]).all())
        ref = R.dequant_fp4_packed(w, s).float()
        q = vq.dequant(lo, hi, s).float()
        bits = (lo.numel() + hi.numel()) * 8 / (N * K)
        print(f"[{N}, {K}] planes ok, entry round-trip {idem}, "
              f"rel err vs the original {float((q - ref).norm() / ref.norm()):.4f}, "
              f"{bits:.3f} bit/weight + {s.numel() * 8 / (N * K):.3f} scale")
