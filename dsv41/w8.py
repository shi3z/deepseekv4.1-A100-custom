"""Dense weights kept as FP8 (e4m3 bytes + E8M0 [32x32] block scales) instead of bf16.

Decode (M <= 16 rows) runs them through the tensor-core kernel in cuda/fp8_tc.cu, which reads half
the bytes of the bf16 cuBLAS path (1.1-1.2 TB/s effective on A100 vs 1.4 TB/s of bf16, i.e. ~1.6x
faster per GEMV) and computes exactly the same numbers (exact dequantization, fp32 accumulation).
Prefill (M > 16) dequantizes to a temporary bf16 matrix and uses cuBLAS.
Set DSV41_W8=0 to keep bf16 copies (old behaviour)."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .quant import dequant_fp8_block

ENABLED = os.environ.get("DSV41_W8", "1") == "1"
MAX_TC_ROWS = 512  # rows handled by the tensor-core kernels (fp8_tc <= 16, fp8_tcw <= 64, fp8_tcg beyond); then dequantize + cuBLAS


# Byte order of every 16-k group in memory (cuda/fp8_tcg.cu): position 4t + j holds k = 2t + j (j < 2) or 2t + 8 + j - 2,
# so the 32-bit word ldmatrix gives lane t is exactly its mma A fragment; fp8_tc.cu / fp8_tcw.cu read x accordingly.
PERM_K = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
_INV_PERM_K = [PERM_K.index(i) for i in range(16)]


def permute_k(w8: torch.Tensor) -> torch.Tensor:
    """uint8 [N, K] natural k order -> the in-memory order of the fp8 kernels."""
    N, K = w8.shape
    assert K % 16 == 0
    return w8.view(N, K // 16, 16)[:, :, PERM_K].reshape(N, K).contiguous()


def unpermute_k(w8: torch.Tensor) -> torch.Tensor:
    N, K = w8.shape
    return w8.view(N, K // 16, 16)[:, :, _INV_PERM_K].reshape(N, K).contiguous()


class W8:
    __slots__ = ("w8", "s8", "shape", "device")

    def __init__(self, w8: torch.Tensor, s8: torch.Tensor, permuted: bool = False):
        """w8: uint8 e4m3 bits [N, K] (natural k order unless permuted=True); s8: E8M0 [ceil(N/32), K/32]."""
        assert w8.dtype == torch.uint8 and s8.dtype == torch.uint8 and w8.dim() == 2
        self.w8 = w8.contiguous() if permuted else permute_k(w8)
        self.s8 = s8.contiguous()
        self.shape = tuple(w8.shape)
        self.device = w8.device

    @property
    def dtype(self):
        return torch.bfloat16

    def bf16(self) -> torch.Tensor:
        return dequant_fp8_block(unpermute_k(self.w8).view(torch.float8_e4m3fn), self.s8)

    @staticmethod
    def cat(ws: list) -> "W8":
        """Concatenate along N (row counts must be multiples of 32)."""
        assert all(w.shape[0] % 32 == 0 for w in ws)
        return W8(torch.cat([w.w8 for w in ws], dim=0), torch.cat([w.s8 for w in ws], dim=0), permuted=True)


def linear_w(x: torch.Tensor, w) -> torch.Tensor:
    """F.linear for a bf16 tensor weight or a W8 weight (bf16 x, bf16 result)."""
    if not isinstance(w, W8):
        return F.linear(x, w)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= MAX_TC_ROWS:
        from .cukern import fp8_gemm_tc
        y = fp8_gemm_tc(x2.contiguous().to(torch.bfloat16), w.w8, w.s8)
    else:
        try:
            from .cukern import fp8_gemm_tc
            chunks = []
            for i in range(0, x2.shape[0], MAX_TC_ROWS):
                cx = x2[i:i + MAX_TC_ROWS].contiguous().to(torch.bfloat16)
                chunks.append(fp8_gemm_tc(cx, w.w8, w.s8))
            y = torch.cat(chunks, dim=0)
        except Exception:
            y = F.linear(x2, w.bf16())
    return y.view(*lead, w.shape[0])


def oproj_a(o: torch.Tensor, wo_a, n_groups: int, rank: int) -> torch.Tensor:
    """The block-diagonal o-projection: o [b, s, g, d] x wo_a (rows g*rank..(g+1)*rank use only group g) -> [b, s, g*rank]."""
    b, s, g, d = o.shape
    if not isinstance(wo_a, W8):
        return torch.einsum("bsgd,grd->bsgr", o, wo_a.view(n_groups, rank, -1)).flatten(2)
    if b * s <= MAX_TC_ROWS:
        from .cukern import fp8_gemm_tc
        return fp8_gemm_tc(o.reshape(b * s * g, d).contiguous(), wo_a.w8, wo_a.s8, group_cols=rank).view(b, s, -1)
    return torch.einsum("bsgd,grd->bsgr", o, wo_a.bf16().view(n_groups, rank, -1)).flatten(2)
