"""Fused decode indexer (results/dense_m1_report.md, rank 3): the score over the compressed-key cache and the top-k
selection of DecodeRuntime.indexer in two launches instead of ~60 torch ops.

  score[b, t] = sum_h w[b, h] * relu(q[b, h, :] . k[seq[b], t, :])     (fp32; -inf for t >= cl[b] or masked candidates)
  idxs[b] = the index_topk largest t, sorted ascending, -1 where t >= cl[b]

Enabled in decode.py with DSV41_INDEX_FAST=1; the torch path stays the default."""
from __future__ import annotations

import ctypes
import os

import torch
import triton
import triton.language as tl

from . import cukern


@triton.jit
def _index_score_kernel(Q, K, W, SEQ, CL, CAND, OUT, n_pos, stride_kb, stride_cand, cand_off,
                        H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, HAS_CAND: tl.constexpr):
    b = tl.program_id(0)
    tb = tl.program_id(1)
    t0 = tb * BT
    t = t0 + tl.arange(0, BT)
    tmask = t < n_pos
    hs = tl.arange(0, H)
    ds = tl.arange(0, D)
    q = tl.load(Q + b * H * D + hs[:, None] * D + ds[None, :])  # [H, D] bf16 (fp4-quantized values)
    seq = tl.load(SEQ + b)
    k = tl.load(K + seq * stride_kb + t[:, None] * D + ds[None, :], mask=tmask[:, None], other=0.0)  # [BT, D]
    s = tl.dot(q, tl.trans(k))  # [H, BT] fp32
    s = tl.maximum(s, 0.0)
    w = tl.load(W + b * H + hs).to(tl.float32)
    s = tl.sum(s * w[:, None], axis=0)  # [BT]
    cl = tl.load(CL + b)
    valid = tmask & (t < cl)
    if HAS_CAND:
        c = tl.load(CAND + b * stride_cand + cand_off + t, mask=tmask, other=0)
        valid = valid & (c != 0)
    s = tl.where(valid, s, float("-inf"))
    tl.store(OUT + b * n_pos + t, s, mask=tmask)


def index_score(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor, seq: torch.Tensor, cl: torch.Tensor,
                cand: torch.Tensor | None = None) -> torch.Tensor:
    """q: bf16 [B, 1, H, D]; index_k: bf16 [S, n_pos, D] (the shared cache, all sequence slots); weights: [B, 1, H] or [B, H]
    (already scaled); seq: int64 [B]; cl: int64 [B] (compress_len); cand: bool [B, 1, >= n_pos] or None -> fp32 [B, n_pos]."""
    B, _, H, D = q.shape
    n_pos = index_k.shape[1]
    out = torch.empty(B, n_pos, device=q.device, dtype=torch.float32)
    BT = 64
    w = weights.reshape(B, H).contiguous()
    cl = cl.reshape(-1)
    if cl.dtype != torch.int64:
        cl = cl.to(torch.int64)
    with torch.cuda.device(q.device):
        _index_score_kernel[(B, triton.cdiv(n_pos, BT))](
            q.contiguous(), index_k, w, seq, cl, cand if cand is not None else out, out, n_pos, index_k.stride(0),
            cand.stride(0) if cand is not None else 0, 0, H=H, D=D, BT=BT, HAS_CAND=cand is not None, num_warps=4)
    return out


def topk_indices(score: torch.Tensor, k: int, cl: torch.Tensor) -> torch.Tensor:
    """score: fp32 [B, n] -> int32 [B, 1, k]: the k largest per row, sorted ascending, -1 where the index >= cl[b]
    (the same result as score.topk(k).indices.sort().values with the compress_len mask)."""
    B, n = score.shape
    assert k <= 1024 and score.is_contiguous()
    out = torch.empty(B, k, device=score.device, dtype=torch.int32)
    f = cukern.get_function("topk_select.cu", "topk_select", score.device)
    cukern.launch(f, (B, 1, 1), (1024, 1, 1), [ctypes.c_void_p(score.data_ptr()), ctypes.c_int(n), ctypes.c_int(k),
                                                 ctypes.c_int(score.stride(0)), ctypes.c_void_p(out.data_ptr())], score.device)
    idx = out.view(B, 1, k)
    return torch.where(idx < cl.view(B, 1, 1).to(torch.int32), idx, -1)
