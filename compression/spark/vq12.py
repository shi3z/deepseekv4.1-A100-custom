"""VQ12: a 3-bit expert format that keeps CB3's slot geometry and doubles its accuracy.

CB3 spends its 3 bits on a per-row codebook of 8 of the 16 E2M1 levels -- a scalar quantiser, and
scalar is what costs it: measured over all 40 layers unpruned, CB3 is +8.54 % wikitext PPL against
the FP4 checkpoint while a dim-4 vector quantiser at the same 3 bits is +4.23 % (code: +2.17 % vs
+0.97 %). Same bytes, half the damage. The subset search CB3 does is already optimal for its
format; the gap is scalar vs vector, not the search.

VQ12 stores one 12-bit index per group of 4 consecutive k into a GLOBAL 4096-entry codebook whose
entries are four E2M1 codes. 12 bits / 4 weights = 3 bit/weight, exactly CB3's rate, and the plane
sizes are the same too:

    lo  [N, K/4]  uint8   the low 8 bits of group g's index, one byte per group
    hi  [N, K/8]  uint8   the high 4 bits, two groups per byte (even g low nibble, odd g high)
    cb  [N, 8]            unused -- the codebook is global; kept so the slot stride is unchanged
    s   [N, K/32]         the UE8M0 group scales, untouched

Because a codebook entry is four E2M1 codes, a VQ12 group expands to exactly two packed-FP4 bytes
(entry u16 -> byte0 = u16 & 0xFF holding k0,k1 and byte1 = u16 >> 8 holding k2,k3), which is the
byte tile `_chunk_dot` already consumes. The kernel change is one line of CB3's `_grp_packed`: a
per-row 32-bit codebook word indexed by a 3-bit index becomes one load from an 8 kB table.

The codebook is trained off-line (k-means over the 65536 possible 4-code tuples, weighted by their
frequency in the checkpoint, centroids snapped back onto the E2M1 grid); see
shi3z/deepseekv4.1-A100-custom, branch compression-study, `vq_build.py`.
"""

from __future__ import annotations

import numpy as np
import torch

FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class VQ12:
    """Holds the 4096 x 4-code codebook and the 65536-entry encoder table."""

    def __init__(self, npz_path: str, device="cuda"):
        d = np.load(npz_path)
        self.codes = torch.from_numpy(d["codebook"]).to(device)          # [4096, 4] uint8
        self.enc = torch.from_numpy(d["enc"]).to(device).long()          # u16 tuple -> entry
        # the kernel's table: entry -> the two packed-FP4 bytes, as one uint16
        c = self.codes.long()
        self.lut = (c[:, 0] | (c[:, 1] << 4) | (c[:, 2] << 8) | (c[:, 3] << 12)).to(torch.int32)
        self.device = device
        assert self.codes.shape == (4096, 4), self.codes.shape

    @torch.no_grad()
    def pack(self, w_packed: torch.Tensor, scale: torch.Tensor):
        """packed FP4 [N, K/2] + UE8M0 [N, K/32] -> (lo [N, K/4], hi [N, K/8], cb [N, 8] zeros).

        A group of 4 consecutive k is exactly 2 packed bytes, so the tuple index is a uint16 view of
        the payload -- the encoder is one gather, not a search.
        """
        x = w_packed.view(torch.uint8).to(self.device)
        N, K2 = x.shape
        assert K2 % 2 == 0
        b0 = x[:, 0::2].long()
        b1 = x[:, 1::2].long()
        idx = self.enc[b0 | (b1 << 8)]                                   # [N, K/4] entry 0..4095
        lo = (idx & 0xFF).to(torch.uint8)
        hi_n = (idx >> 8).to(torch.uint8)                                # 4 bits each
        hi = (hi_n[:, 0::2] | (hi_n[:, 1::2] << 4)).contiguous()
        cb = torch.zeros(N, 8, dtype=torch.uint8, device=self.device)
        return lo.contiguous(), hi, cb

    @torch.no_grad()
    def unpack(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """-> packed FP4 [N, K/2], the reference the kernel must reproduce."""
        N = lo.shape[0]
        hn = torch.empty(N, lo.shape[1], dtype=torch.uint8, device=lo.device)
        hn[:, 0::2] = hi & 0x0F
        hn[:, 1::2] = hi >> 4
        idx = lo.long() | (hn.long() << 8)
        v = self.lut.to(lo.device)[idx]                                  # [N, K/4] uint16 value
        out = torch.empty(N, lo.shape[1] * 2, dtype=torch.uint8, device=lo.device)
        out[:, 0::2] = (v & 0xFF).to(torch.uint8)
        out[:, 1::2] = ((v >> 8) & 0xFF).to(torch.uint8)
        return out

    @torch.no_grad()
    def dequant(self, lo, hi, scale) -> torch.Tensor:
        w = self.unpack(lo, hi)
        codes = torch.empty(w.shape[0], w.shape[1] * 2, dtype=torch.uint8, device=w.device)
        codes[:, 0::2] = w & 0x0F
        codes[:, 1::2] = w >> 4
        s = torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
        return (FP4_VALS.to(w.device)[codes.long()] * s).to(torch.bfloat16)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
    sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
    import v41_ref as R

    vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
    torch.manual_seed(0)
    for (N, K) in ((2304, 5120), (5120, 2304)):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(118, 126, (N, K // 32), dtype=torch.uint8, device="cuda")
        lo, hi, cb = vq.pack(w, s)
        assert lo.shape == (N, K // 4) and hi.shape == (N, K // 8) and cb.shape == (N, 8)
        rt = vq.unpack(lo, hi)
        lo2, hi2, _ = vq.pack(rt, s)
        idem = bool((lo == lo2).all() and (hi == hi2).all())
        ref = R.dequant_fp4_packed(w, s).float()
        q = vq.dequant(lo, hi, s).float()
        # the same reconstruction, reached the other way: unpack then the FP4 dequant
        q2 = R.dequant_fp4_packed(rt, s).float()
        bits = (lo.numel() + hi.numel()) * 8 / (N * K)
        print(f"[{N}, {K}] planes ok, round-trip idempotent {idem}, "
              f"dequant == FP4(unpack) {bool((q == q2).all())}, "
              f"rel err vs the original {float((q - ref).norm() / ref.norm()):.4f}, "
              f"{bits:.3f} bit/weight + {s.numel() * 8 / (N * K):.3f} scale")
