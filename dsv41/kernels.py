import os
"""A100 (sm80) kernels for DeepSeek-V4.1: FP4 expert GEMM in Triton (weights stay packed in VRAM and
are expanded to bf16 inside the kernel), plus torch implementations of the sparse attention and the
hyper-connection Sinkhorn split that the reference does in tilelang."""
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- FP4 x BF16 GEMM
@triton.jit
def _fp4_gemm_kernel(
    A, B, S, C,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_sn, stride_sk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """C[M, N] (+)= A[M, K] (bf16) @ W[N, K]^T where W is E2M1 packed two per byte along K (low nibble
    = even k) with one E8M0 scale per (n, 32 k). With SPLIT_K > 1 each program handles a K slice and
    accumulates into C with atomics (C must be zeroed fp32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rkh = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, BLOCK_K // GROUP)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        a = tl.load(A + rm[:, None] * stride_am + (k0 + rk)[None, :] * stride_ak, mask=m_mask[:, None], other=0.0)
        packed = tl.load(B + rn[:, None] * stride_bn + (k0 // 2 + rkh)[None, :] * stride_bk, mask=n_mask[:, None], other=0)
        packed = packed.to(tl.uint8)
        codes = tl.interleave(packed & 0x0F, packed >> 4)  # [BLOCK_N, BLOCK_K] in k order
        sign = (codes >> 3) & 1
        e = (codes >> 1) & 3
        mant = (codes & 1).to(tl.float32)
        mag = tl.where(e == 0, 0.5 * mant, (1.0 + 0.5 * mant) * tl.exp2((e - 1).to(tl.float32)))
        val = tl.where(sign == 1, -mag, mag)
        sb = tl.load(S + rn[:, None] * stride_sn + (k0 // GROUP + rg)[None, :] * stride_sk, mask=n_mask[:, None], other=127)
        scale = tl.exp2(sb.to(tl.float32) - 127.0)
        scale = tl.reshape(scale, (BLOCK_N, BLOCK_K // GROUP, 1))
        scale = tl.broadcast_to(scale, (BLOCK_N, BLOCK_K // GROUP, GROUP))
        scale = tl.reshape(scale, (BLOCK_N, BLOCK_K))
        w = (val * scale).to(tl.bfloat16)  # exact: <=2 significant bits times a power of two
        acc = tl.dot(a, tl.trans(w), acc)
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def _decode_e2m1(code, sb):
    """E2M1 code (int32 0..15) and E8M0 scale byte -> fp32 value, via bf16 bit construction:
    sign | (e + sb - 1) << 7 | m << 6 for normal codes (e > 0); e == 0 is the subnormal 0.5*m."""
    sign = (code >> 3) & 1
    e = (code >> 1) & 3
    m = code & 1
    nz = ((code & 7) != 0) & (sb != 0)
    mant = tl.where(e == 0, 0, m << 6)
    bits = (sign << 15) | ((e + sb - 1) << 7) | mant
    return tl.where(nz, bits, 0).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _fp4_gemv_kernel(
    A, B, S, C,
    M, N, K,
    stride_am, stride_bn, stride_sn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """Bandwidth-bound path for few tokens (M <= BLOCK_M): each program streams BLOCK_N weight rows
    over a K slice with wide coalesced loads, decodes E2M1 -> bf16 by bit arithmetic with the E8M0
    scale folded into the exponent, and accumulates dot products with FMAs. C is zeroed fp32."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rkh = tl.arange(0, BLOCK_K // 2)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        packed = tl.load(B + rn[:, None] * stride_bn + (k0 // 2 + rkh)[None, :], mask=n_mask[:, None], other=0).to(tl.uint8)
        # scale: BLOCK_K is a multiple of 32, one E8M0 byte per 32 k -> expand over 16 packed bytes
        rg = tl.arange(0, BLOCK_K // 32)
        sb = tl.load(S + rn[:, None] * stride_sn + (k0 // 32 + rg)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.reshape(tl.broadcast_to(tl.reshape(sb, (BLOCK_N, BLOCK_K // 32, 1)), (BLOCK_N, BLOCK_K // 32, 16)), (BLOCK_N, BLOCK_K // 2))
        lo = (packed & 0x0F).to(tl.int32)
        hi = (packed >> 4).to(tl.int32)
        w_lo = _decode_e2m1(lo, sb)  # [BLOCK_N, BLOCK_K/2] values at even k
        w_hi = _decode_e2m1(hi, sb)  # odd k
        a_lo = tl.load(A + rm[:, None] * stride_am + (k0 + 2 * rkh)[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        a_hi = tl.load(A + rm[:, None] * stride_am + (k0 + 2 * rkh + 1)[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(a_lo[:, None, :] * w_lo[None, :, :], axis=2) + tl.sum(a_hi[:, None, :] * w_hi[None, :, :], axis=2)
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


def fp4_gemv(a: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, block_n: int = 8, block_k: int = 256) -> torch.Tensor:
    M, K = a.shape
    N = w_packed.shape[0]
    bm = 1 if M == 1 else (2 if M <= 2 else (4 if M <= 4 else (8 if M <= 8 else 16)))
    programs_n = triton.cdiv(N, block_n)
    split = 1
    while programs_n * split < 864 and split < 8 and (K // (split * 2)) % block_k == 0:
        split *= 2
    c = torch.zeros(M, N, device=a.device, dtype=torch.float32) if split > 1 else torch.empty(M, N, device=a.device, dtype=torch.float32)
    _fp4_gemv_kernel[(programs_n, split)](
        a, w_packed, w_scale, c, M, N, K, a.stride(0), w_packed.stride(0), w_scale.stride(0), c.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split, num_warps=4, num_stages=2,
    )
    return c


def _config(M: int, N: int, K: int):
    """Tile / split choice: small M is bandwidth bound, so spread N and K over many programs."""
    if M <= 16:
        bm, bn, bk = 16, 32, 128
    elif M <= 64:
        bm, bn, bk = 64, 64, 128
    else:
        bm, bn, bk = 64, 128, 64
    programs = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    split = 1
    while programs * split < 432 and split < 16 and (K // (split * 2)) % bk == 0:
        split *= 2
    return bm, bn, bk, split


def fp4_gemm(a: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, out_dtype=torch.float32) -> torch.Tensor:
    """a: bf16 [M, K]; w_packed: uint8/int8 [N, K/2]; w_scale: uint8 E8M0 [N, K/32]. Returns [M, N]."""
    assert a.dtype == torch.bfloat16 and a.is_contiguous()
    M, K = a.shape
    N = w_packed.shape[0]
    assert w_packed.shape[1] * 2 == K and w_scale.shape == (N, K // 32), (w_packed.shape, w_scale.shape, K)
    if M <= 16:
        c = fp4_gemv(a, w_packed, w_scale)
        return c if out_dtype == torch.float32 else c.to(out_dtype)
    bm, bn, bk, split = _config(M, N, K)
    if split > 1:
        c = torch.zeros(M, N, device=a.device, dtype=torch.float32)
    else:
        c = torch.empty(M, N, device=a.device, dtype=out_dtype)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn), split)
    _fp4_gemm_kernel[grid](
        a, w_packed, w_scale, c, M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1), w_scale.stride(0), w_scale.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP=32, SPLIT_K=split, num_warps=4, num_stages=3,
    )
    if split > 1 and out_dtype != torch.float32:
        c = c.to(out_dtype)
    return c


# --------------------------------------------------------------------------- sparse attention
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float,
                q_chunk: int | None = None) -> torch.Tensor:
    """q: [b, s, h, d] bf16; kv: [b, n, d] bf16 (one shared K=V latent for all heads);
    attn_sink: [h] fp32; topk_idxs: [b, s, t] int32 with -1 = empty slot. Returns [b, s, h, d] bf16.
    Same math as the reference kernel: softmax over the selected slots plus a per-head sink term."""
    if q_chunk is None:
        q_chunk = int(os.environ.get("DSV41_SPARSE_ATTN_CHUNK", "128"))
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for s0 in range(0, s, q_chunk):
        s1 = min(s, s0 + q_chunk)
        idx = topk_idxs[:, s0:s1].long()  # [b, sc, t]

        # A selected index is usable only when it lies inside the KV
        # tensor actually supplied to this attention call.
        #
        # Previously only negative sentinel values were filtered:
        #
        #     valid = idx >= 0
        #     idx.clamp_min(0)
        #
        # so any stale / offset index >= kv.size(1) reached
        # torch.gather() and triggered a CUDA IndexKernel device assert,
        # poisoning the whole CUDA context.
        n_kv = kv.size(1)

        if n_kv <= 0:
            raise RuntimeError(
                "sparse_attn received empty KV with non-empty queries"
            )

        valid = (
            (idx >= 0)
            & (idx < n_kv)
        )

        if os.environ.get(
            "DSV41_DEBUG_ATTN_BOUNDS",
            "0",
        ) == "1":
            bad_hi = idx >= n_kv

            if bad_hi.any().item():
                n_bad = int(
                    bad_hi.sum().item()
                )
                max_idx = int(
                    idx.max().item()
                )

                print(
                    f"[attn-bounds] "
                    f"OOB={n_bad:,} "
                    f"max_idx={max_idx:,} "
                    f"kv_rows={n_kv:,} "
                    f"q={s0:,}:{s1:,}",
                    flush=True,
                )

        safe_idx = idx.clamp(
            min=0,
            max=n_kv - 1,
        )

        g = torch.gather(
            kv,
            1,
            safe_idx
            .reshape(b, -1, 1)
            .expand(-1, -1, d),
        ).view(
            b,
            s1 - s0,
            -1,
            d,
        )  # [b, sc, t, d]
        scores = torch.einsum("bshd,bstd->bsht", q[:, s0:s1], g).float() * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(scores - mx)  # [b, sc, h, t]
        denom = p.sum(dim=-1) + torch.exp(sink - mx.squeeze(-1))
        o = torch.einsum("bsht,bstd->bshd", p.to(g.dtype), g).float() / denom.unsqueeze(-1)
        out[:, s0:s1] = o.to(q.dtype)
    return out


# --------------------------------------------------------------------------- hyper-connection mixes
from .quant import maybe_compile  # noqa: E402


@maybe_compile
def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes: [b, s, (2+hc)*hc] fp32 -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc] (doubly stochastic)."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
