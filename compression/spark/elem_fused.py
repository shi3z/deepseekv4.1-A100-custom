"""The two elementwise chains of the decode step, one kernel each instead of seven and four.

expert_ffn spends 34.3 us a layer turning two [2, 2304] bf16 GEMM results into one, and _layer_b
spends 20.5 us adding two [2, 5120] bf16 tensors -- 18 kB and 40 kB of data. Neither is bandwidth:
it is seven and four launches per layer, 440 per step between them.

Both keep the arithmetic exactly as torch had it -- widen to fp32, clamp, silu, multiply or add,
round once to bf16 -- and both were measured bit-identical to the torch sequence at T = 1..8
(a100-vq/elem_fuse.py). Set DSV41_ELEM_FUSED=0 to fall back.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(G, U, O, n, limit, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit), limit)      # clamp(up, -limit, limit)
    g = tl.minimum(g, limit)                          # clamp(gate, max=limit)
    h = (g * tl.sigmoid(g)) * u                       # silu(gate) * up
    tl.store(O + off, h.to(tl.bfloat16), mask=m)


def fused_swiglu(g, u, limit):
    o = torch.empty_like(g)
    n = g.numel()
    _swiglu_kernel[(triton.cdiv(n, 1024),)](g, u, o, n, float(limit), BLOCK=1024, num_warps=4)
    return o


@triton.jit
def _merge_kernel(R, S, O, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + off, mask=m, other=0.0).to(tl.float32)
    tl.store(O + off, (r + s).to(tl.bfloat16), mask=m)


def fused_merge(r, s):
    o = torch.empty_like(r)
    n = r.numel()
    _merge_kernel[(triton.cdiv(n, 1024),)](r, s, o, n, BLOCK=1024, num_warps=4)
    return o



SWIGLU = os.environ.get("DSV41_FUSE_SWIGLU", "1") == "1"
MERGE = os.environ.get("DSV41_FUSE_MERGE", "1") == "1"


@triton.jit
def _rmsnorm_kernel(X, W, Y, n_cols, sx, sy, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(X + row * sx + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.sum(x * x, axis=0) / n_cols
    x = x * tl.rsqrt(v + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * sy + cols, (w * x).to(tl.bfloat16), mask=m)


def fused_rmsnorm(x, w, eps):
    """bf16 [M, N] -> bf16 [M, N]; fp32 square-mean, rsqrt, scale, one rounding, one launch.

    x may be row-strided (a column slice of a wider tensor, which is what the fused qkv projection
    produces); only the last dimension has to be unit-stride.
    """
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.shape[-1]
    _rmsnorm_kernel[(x.shape[0],)](x, w, y, n, x.stride(0), y.stride(0), float(eps),
                                   BLOCK=triton.next_power_of_2(n), num_warps=8)
    return y


RMSNORM = os.environ.get("DSV41_FUSE_RMSNORM", "1") == "1"
RMSNORM_MAX_M = 16      # decode-sized calls only; prefill keeps the tiled torch reduction
RMSNORM_MAX_N = 8192


@triton.jit
def _rope_kernel(X, F, Y, D, RD, sx, sy, HEADS, INV: tl.constexpr,
                 BLOCK: tl.constexpr, BLOCK_RD: tl.constexpr):
    row = tl.program_id(0)
    keep = D - RD
    c = tl.arange(0, BLOCK)
    mk = c < keep
    tl.store(Y + row * sy + c, tl.load(X + row * sx + c, mask=mk, other=0), mask=mk)
    k = tl.arange(0, BLOCK_RD)
    mr = k < RD
    t = tl.load(X + row * sx + keep + k, mask=mr, other=0.0).to(tl.float32)
    a, b = tl.split(tl.reshape(t, [BLOCK_RD // 2, 2]))
    j = tl.arange(0, BLOCK_RD // 2)
    mj = j < (RD // 2)
    base = (row // HEADS) * RD          # [T, RD/2, 2] fp32 view of freqs_cis
    fr = tl.load(F + base + 2 * j, mask=mj, other=0.0)
    fi = tl.load(F + base + 2 * j + 1, mask=mj, other=0.0)
    if INV:
        fi = -fi
    out = tl.reshape(tl.join(a * fr - b * fi, a * fi + b * fr), [BLOCK_RD])
    tl.store(Y + row * sy + keep + k, out.to(tl.bfloat16), mask=mr)


def fused_rope(x, fq, rd, inverse=False):
    """x [..., D] bf16 -> bf16 [..., D]; rotary on the last `rd`, the rest copied through."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    M, T = x2.shape[0], fq.shape[0]
    y = torch.empty_like(x2)
    _rope_kernel[(M,)](x2, torch.view_as_real(fq), y, D, rd, x2.stride(0), y.stride(0),
                       M // T, bool(inverse), BLOCK=triton.next_power_of_2(D - rd),
                       BLOCK_RD=triton.next_power_of_2(rd), num_warps=4)
    return y.view(x.shape)


ROPE = os.environ.get("DSV41_FUSE_ROPE", "1") == "1"


@triton.jit
def _hc_post_kernel(X, RES, POST, COMB, Y, D, HC: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    md = d < D
    xv = tl.load(X + s * D + d, mask=md, other=0.0).to(tl.float32)
    for j in tl.static_range(HC):
        acc = tl.load(POST + s * HC + j).to(tl.float32) * xv
        for i in tl.static_range(HC):
            r = tl.load(RES + (s * HC + i) * D + d, mask=md, other=0.0).to(tl.float32)
            acc += tl.load(COMB + (s * HC + i) * HC + j).to(tl.float32) * r
        tl.store(Y + (s * HC + j) * D + d, acc.to(tl.bfloat16), mask=md)


def _f32c(t):
    return t if (t.dtype == torch.float32 and t.is_contiguous()) else t.float().contiguous()


def fused_hc_post(x, residual, post, comb):
    s, hc, d = residual.shape
    y = torch.empty((s, hc, d), dtype=x.dtype, device=x.device)
    _hc_post_kernel[(s, triton.cdiv(d, 1024))](x, residual, _f32c(post), _f32c(comb), y, d,
                                               HC=hc, BLOCK=1024, num_warps=4)
    return y


HC_POST = os.environ.get("DSV41_FUSE_HC", "1") == "1"
