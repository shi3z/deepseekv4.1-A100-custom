"""Fused Triton kernels for the small per-token operations that dominate decode when run as chains of
torch ops: RMSNorm, FP8/FP4 activation fake-quantization, the hyper-connection Sinkhorn split and the
hc pre/post mixes. Each replaces 5-60 tiny kernel launches with one."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _ceil_log2(a):
    """ceil(log2(a)) for a > 0 (fp32), exact, via the IEEE bit pattern (mirrors the reference kernel)."""
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    return e + tl.where(mant != 0, 1, 0)


@triton.jit
def _pow2(e):
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _round_e4m3(v):
    """Round fp32 v (|v| <= 448) to the nearest float8_e4m3fn value (ties to even), returned as fp32."""
    a = tl.abs(v)
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127  # floor(log2 a) for normal fp32
    e = tl.maximum(e, -6)  # e4m3 subnormals: fixed ulp 2^-9
    ulp = _pow2(e - 3)
    r = libdevice.rint(libdevice.div_rn(a, ulp)) * ulp  # exact division: Triton's / is approximate
    r = tl.minimum(r, 448.0)
    return tl.where(v < 0, -r, r)


@triton.jit
def _round_e2m1(v):
    """Round fp32 v (|v| <= 6) to the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even."""
    a = tl.abs(v)
    r = tl.where(a < 2.0, libdevice.rint(a * 2.0) * 0.5, tl.where(a < 4.0, libdevice.rint(a), libdevice.rint(a * 0.5) * 2.0))
    r = tl.minimum(r, 6.0)
    return tl.where(v < 0, -r, r)


# --------------------------------------------------------------------------- fake quant
@triton.jit(do_not_specialize=["n_groups"])
def _fake_quant_kernel(X, Y, n_groups, GROUP: tl.constexpr, MODE: tl.constexpr, ROWS: tl.constexpr):
    """MODE 0: fp8 e4m3 with pow2 scale (amax>=1e-4); 1: fp4 e2m1 with pow2 scale; 2: fp4 e2m1 with e4m3 scale."""
    pid = tl.program_id(0)
    g = pid * ROWS + tl.arange(0, ROWS)
    mask = g < n_groups
    offs = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if MODE == 0:
        amax = tl.maximum(amax, 1e-4)
        s = _pow2(_ceil_log2(libdevice.div_rn(amax, 448.0)))
        q = _round_e4m3(tl.minimum(tl.maximum(libdevice.div_rn(x, s[:, None]), -448.0), 448.0))
    elif MODE == 1:
        amax = tl.maximum(amax, 6.0 * 1.1754944e-38)
        s = _pow2(_ceil_log2(libdevice.div_rn(amax, 6.0)))
        q = _round_e2m1(tl.minimum(tl.maximum(libdevice.div_rn(x, s[:, None]), -6.0), 6.0))
    else:
        amax = tl.maximum(amax, 6.0 * 0.001953125)
        s = _round_e4m3(libdevice.div_rn(amax, 6.0))
        q = _round_e2m1(tl.minimum(tl.maximum(libdevice.div_rn(x, s[:, None]), -6.0), 6.0))
    y = q * s[:, None]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask[:, None])


def fake_quant(x: torch.Tensor, group: int, mode: int) -> torch.Tensor:
    xc = x.contiguous()
    n_groups = xc.numel() // group
    y = torch.empty_like(xc)
    ROWS = 8
    with torch.cuda.device(x.device):
        _fake_quant_kernel[(triton.cdiv(n_groups, ROWS),)](xc, y, n_groups, GROUP=group, MODE=mode, ROWS=ROWS, num_warps=4)
    return y


def fake_quant_fp8(x, block=32):
    return fake_quant(x, block, 0)


def fake_quant_fp4(x, block=32, scale_e4m3=False):
    return fake_quant(x, block, 2 if scale_e4m3 else 1)


# --------------------------------------------------------------------------- rmsnorm
@triton.jit(do_not_specialize=["n_rows"])
def _rmsnorm_kernel(X, W, Y, n_rows, eps, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * (1.0 / tl.sqrt(var + eps)) * tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xc = x.contiguous()
    D = xc.shape[-1]
    n_rows = xc.numel() // D
    y = torch.empty_like(xc)
    with torch.cuda.device(x.device):
        _rmsnorm_kernel[(n_rows,)](xc, w, y, n_rows, eps, D=D, BLOCK_D=triton.next_power_of_2(D), num_warps=8 if D >= 4096 else 4)
    return y


# --------------------------------------------------------------------------- sinkhorn split
@triton.jit(do_not_specialize=["n"])
def _sinkhorn_kernel(MIX, SCALE, BASE, PRE, POST, COMB, n, eps, iters: tl.constexpr, HC: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, HC)
    jj = tl.arange(0, HC)[:, None]
    kk = tl.arange(0, HC)[None, :]
    base_ptr = MIX + row * (2 + HC) * HC
    s0 = tl.load(SCALE)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    pre = tl.sigmoid(tl.load(base_ptr + j) * s0 + tl.load(BASE + j)) + eps
    post = 2.0 * tl.sigmoid(tl.load(base_ptr + HC + j) * s1 + tl.load(BASE + HC + j))
    idx = 2 * HC + jj * HC + kk
    comb = tl.load(base_ptr + idx) * s2 + tl.load(BASE + idx)
    rmax = tl.max(comb, axis=1)
    comb = tl.exp(comb - rmax[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in range(iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    tl.store(PRE + row * HC + j, pre)
    tl.store(POST + row * HC + j, post)
    tl.store(COMB + row * HC * HC + jj * HC + kk, comb)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    b, s, _ = mixes.shape
    n = b * s
    m = mixes.contiguous().float()
    pre = torch.empty(b, s, hc_mult, device=m.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty(b, s, hc_mult, hc_mult, device=m.device, dtype=torch.float32)
    with torch.cuda.device(m.device):
        _sinkhorn_kernel[(n,)](m, hc_scale, hc_base, pre, post, comb, n, eps, iters=sinkhorn_iters, HC=hc_mult, num_warps=1)
    return pre, post, comb


# --------------------------------------------------------------------------- hc pre / post
@triton.jit
def _hc_pre_kernel(X, PRE, Y, D: tl.constexpr, HC: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for h in tl.static_range(HC):
        p = tl.load(PRE + row * HC + h)
        acc += p * tl.load(X + (row * HC + h) * D + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + cols, acc.to(Y.dtype.element_ty), mask=mask)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """x: [b, s, hc, d], pre_mix: [b, s, hc] -> [b, s, d]"""
    b, s, hc, d = x.shape
    xc = x.contiguous()
    y = torch.empty(b, s, d, device=x.device, dtype=x.dtype)
    BLOCK_D = 1024
    with torch.cuda.device(x.device):
        _hc_pre_kernel[(b * s, triton.cdiv(d, BLOCK_D))](xc, pre_mix.contiguous().float(), y, D=d, HC=hc, BLOCK_D=BLOCK_D, num_warps=4)
    return y


@triton.jit
def _hc_post_kernel(X, R, POST, COMB, Y, D: tl.constexpr, HC: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    for i in tl.static_range(HC):
        acc = tl.load(POST + row * HC + i) * x
        for j in tl.static_range(HC):
            c = tl.load(COMB + (row * HC + j) * HC + i)  # output copy i mixes residual copy j with comb[j, i]
            acc += c * tl.load(R + (row * HC + j) * D + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + (row * HC + i) * D + cols, acc.to(Y.dtype.element_ty), mask=mask)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """x: [b, s, d], residual: [b, s, hc, d], post: [b, s, hc], comb: [b, s, hc, hc] -> [b, s, hc, d]"""
    b, s, hc, d = residual.shape
    y = torch.empty_like(residual)
    BLOCK_D = 1024
    with torch.cuda.device(x.device):
        _hc_post_kernel[(b * s, triton.cdiv(d, BLOCK_D))](x.contiguous(), residual.contiguous(), post.contiguous().float(),
                                                        comb.contiguous().float(), y, D=d, HC=hc, BLOCK_D=BLOCK_D, num_warps=4)
    return y


# --------------------------------------------------------------------------- rotary (in place on the last RD columns)
@triton.jit(do_not_specialize=["n_rows", "pos0", "pos_stride", "H", "S"])
def _rope_kernel(X, COS, SIN, n_rows, pos0, pos_stride, H, S, D: tl.constexpr, RD: tl.constexpr, INVERSE: tl.constexpr):
    row = tl.program_id(0)
    pos = pos0 + ((row // H) % S) * pos_stride
    j = tl.arange(0, RD // 2)
    base = X + row * D + (D - RD)
    xr = tl.load(base + 2 * j).to(tl.float32)
    xi = tl.load(base + 2 * j + 1).to(tl.float32)
    c = tl.load(COS + pos * (RD // 2) + j)
    s = tl.load(SIN + pos * (RD // 2) + j)
    if INVERSE:
        s = -s
    yr = xr * c - xi * s
    yi = xr * s + xi * c
    tl.store(base + 2 * j, yr.to(X.dtype.element_ty))
    tl.store(base + 2 * j + 1, yi.to(X.dtype.element_ty))


def rope_(x: torch.Tensor, rd: int, cos: torch.Tensor, sin: torch.Tensor, pos0: int, inverse: bool = False, pos_stride: int = 1) -> torch.Tensor:
    """Rotate the last `rd` elements of every row of x ([b, s, d] or [b, s, h, d], contiguous) in place,
    using position pos0 + s * pos_stride for each row. cos/sin: fp32 [max_seq, rd/2] from freqs_cis."""
    assert x.is_contiguous()
    if x.ndim == 3:
        b, s, d = x.shape
        h = 1
    else:
        b, s, h, d = x.shape
    n_rows = b * s * h
    with torch.cuda.device(x.device):
        _rope_kernel[(n_rows,)](x, cos, sin, n_rows, pos0, pos_stride, h, s, D=d, RD=rd, INVERSE=inverse, num_warps=1)
    return x


# --------------------------------------------------------------------------- sparse attention, one query per batch row
@triton.jit(do_not_specialize=["n_kv", "topk"])
def _sparse_attn_decode_kernel(Q, KV, SINK, IDX, O, n_kv, topk, scale,
                               H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    b = tl.program_id(0)
    hb = tl.program_id(1)
    hs = hb * HB + tl.arange(0, HB)
    dd = tl.arange(0, D)
    q = tl.load(Q + (b * H + hs)[:, None] * D + dd[None, :])  # [HB, D] bf16
    m_i = tl.full((HB,), -1e30, tl.float32)
    l_i = tl.zeros((HB,), tl.float32)
    acc = tl.zeros((HB, D), tl.float32)
    for t0 in range(0, topk, BLOCK_T):
        tt = t0 + tl.arange(0, BLOCK_T)
        idx = tl.load(IDX + b * topk + tt, mask=tt < topk, other=-1)
        valid = idx >= 0
        kv = tl.load(KV + (b * n_kv + tl.maximum(idx, 0))[:, None] * D + dd[None, :], mask=valid[:, None], other=0.0)  # [BLOCK_T, D]
        s = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale  # [HB, BLOCK_T]
        s = tl.where(valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new
    sink = tl.load(SINK + hs)
    l_i += tl.exp(sink - m_i)
    o = acc / l_i[:, None]
    tl.store(O + (b * H + hs)[:, None] * D + dd[None, :], o.to(O.dtype.element_ty))


def sparse_attn_decode(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """q: [b, 1, h, d] bf16; kv: [b, n, d] bf16; topk_idxs: [b, 1, t] int32 (-1 = empty). Returns [b, 1, h, d]."""
    b, s, h, d = q.shape
    assert s == 1
    n = kv.shape[1]
    t = topk_idxs.shape[-1]
    o = torch.empty_like(q)
    HB = 16
    with torch.cuda.device(q.device):
        _sparse_attn_decode_kernel[(b, h // HB)](q.contiguous(), kv.contiguous(), attn_sink.float().contiguous(), topk_idxs.contiguous(), o,
                                                 n, t, softmax_scale, H=h, HB=HB, D=d, BLOCK_T=64, num_warps=4)
    return o


# --------------------------------------------------------------------------- SwiGLU (+ routing weight) + FP8 fake quant
@triton.jit(do_not_specialize=["n_rows"])
def _swiglu_quant_kernel(GU, W, Y, n_rows, limit, INTER: tl.constexpr, GROUP: tl.constexpr, HAS_W: tl.constexpr, PERMUTE: tl.constexpr):
    """One program per (row, group of GROUP inter columns): h = w * silu(clamp(gate)) * clamp(up), then the
    per-32 FP8 fake quantization the reference applies before w2. GU: fp32 [rows, 2*INTER]; Y: bf16 [rows, INTER]."""
    row = tl.program_id(0)
    gb = tl.program_id(1)
    cols = gb * GROUP + tl.arange(0, GROUP)
    gate = tl.load(GU + row * (2 * INTER) + cols).to(tl.float32)
    up = tl.load(GU + row * (2 * INTER) + INTER + cols).to(tl.float32)
    if limit > 0:
        up = tl.minimum(tl.maximum(up, -limit), limit)
        gate = tl.minimum(gate, limit)
    h = gate * tl.sigmoid(gate) * up
    if HAS_W:
        h = h * tl.load(W + row)
    h = h.to(tl.bfloat16).to(tl.float32)  # the reference casts to bf16 before quantizing
    amax = tl.maximum(tl.max(tl.abs(h), axis=0), 1e-4)
    s = _pow2(_ceil_log2(libdevice.div_rn(amax, 448.0)))
    q = _round_e4m3(tl.minimum(tl.maximum(libdevice.div_rn(h, s), -448.0), 448.0))
    if PERMUTE:  # the 8-k permuted layout of cuda/fp4_tc.cu (bit reversal of the low 3 bits of the column)
        j = cols & 7
        cols = (cols & ~7) | ((j & 1) << 2) | (j & 2) | (j >> 2)
    tl.store(Y + row * INTER + cols, (q * s).to(tl.bfloat16))


def swiglu_quant(gu: torch.Tensor, weights: torch.Tensor | None, inter: int, limit: float, permute: bool = False) -> torch.Tensor:
    """gu: fp32 or bf16 [rows, 2*inter] (gate | up); weights: fp32 [rows] or None -> bf16 [rows, inter], FP8-rounded per 32."""
    rows = gu.shape[0]
    y = torch.empty(rows, inter, device=gu.device, dtype=torch.bfloat16)
    with torch.cuda.device(gu.device):
        _swiglu_quant_kernel[(rows, inter // 32)](gu.contiguous(), weights if weights is not None else gu, y, rows, float(limit),
                                                 INTER=inter, GROUP=32, HAS_W=weights is not None, PERMUTE=permute, num_warps=1)
    return y


# --------------------------------------------------------------------------- device-position variants for CUDA-graph decode
@triton.jit(do_not_specialize=["n_rows", "add", "pos_stride", "H", "S"])
def _rope_dev_kernel(X, COS, SIN, POS, n_rows, add, pos_stride, H, S, D: tl.constexpr, RD: tl.constexpr, INVERSE: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.load(POS + row // (H * S)) + add + ((row // H) % S) * pos_stride  # POS: one position per batch row
    pos = tl.maximum(pos, 0)
    j = tl.arange(0, RD // 2)
    base = X + row * D + (D - RD)
    xr = tl.load(base + 2 * j).to(tl.float32)
    xi = tl.load(base + 2 * j + 1).to(tl.float32)
    c = tl.load(COS + pos * (RD // 2) + j)
    s = tl.load(SIN + pos * (RD // 2) + j)
    if INVERSE:
        s = -s
    tl.store(base + 2 * j, (xr * c - xi * s).to(X.dtype.element_ty))
    tl.store(base + 2 * j + 1, (xr * s + xi * c).to(X.dtype.element_ty))


def rope_dev_(x: torch.Tensor, rd: int, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, add: int = 0,
              inverse: bool = False, pos_stride: int = 1) -> torch.Tensor:
    """Like rope_, but the base position is read from the int64 device tensor `pos` (+ add)."""
    assert x.is_contiguous()
    if x.ndim == 3:
        b, s, d = x.shape
        h = 1
    else:
        b, s, h, d = x.shape
    n_rows = b * s * h
    with torch.cuda.device(x.device):
        _rope_dev_kernel[(n_rows,)](x, cos, sin, pos, n_rows, add, pos_stride, h, s, D=d, RD=rd, INVERSE=inverse, num_warps=1)
    return x


@triton.jit(do_not_specialize=["n_kv1", "n_kv2", "topk"])
def _sparse_attn_decode2_kernel(Q, KV1, KV2, SINK, IDX, O, n_kv1, n_kv2, topk, scale,
                                H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    """Decode attention over two KV sources: slot idx < n_kv1 reads KV1[idx], otherwise KV2[idx - n_kv1]."""
    b = tl.program_id(0)
    hb = tl.program_id(1)
    hs = hb * HB + tl.arange(0, HB)
    dd = tl.arange(0, D)
    q = tl.load(Q + (b * H + hs)[:, None] * D + dd[None, :])
    m_i = tl.full((HB,), -1e30, tl.float32)
    l_i = tl.zeros((HB,), tl.float32)
    acc = tl.zeros((HB, D), tl.float32)
    for t0 in range(0, topk, BLOCK_T):
        tt = t0 + tl.arange(0, BLOCK_T)
        idx = tl.load(IDX + b * topk + tt, mask=tt < topk, other=-1)
        valid = idx >= 0
        in1 = valid & (idx < n_kv1)
        in2 = valid & (idx >= n_kv1)
        i1 = tl.maximum(idx, 0)
        i2 = tl.maximum(idx - n_kv1, 0)
        kv = tl.load(KV1 + (b * n_kv1 + i1)[:, None] * D + dd[None, :], mask=in1[:, None], other=0.0)
        kv += tl.load(KV2 + (b * n_kv2 + i2)[:, None] * D + dd[None, :], mask=in2[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
        s = tl.where(valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new
    sink = tl.load(SINK + hs)
    l_i += tl.exp(sink - m_i)
    tl.store(O + (b * H + hs)[:, None] * D + dd[None, :], (acc / l_i[:, None]).to(O.dtype.element_ty))


def sparse_attn_decode2(q, kv1, kv2, attn_sink, topk_idxs, softmax_scale):
    """q: [b, 1, h, d]; kv1: [b, n1, d] (window ring); kv2: [b, n2, d] (compressed cache) or None; idx: [b, 1, t]."""
    b, s, h, d = q.shape
    assert s == 1
    n1 = kv1.shape[1]
    if kv2 is None:
        kv2, n2 = kv1, 0
    else:
        n2 = kv2.shape[1]
    t = topk_idxs.shape[-1]
    o = torch.empty_like(q)
    HB = 16
    with torch.cuda.device(q.device):
        _sparse_attn_decode2_kernel[(b, h // HB)](q.contiguous(), kv1.contiguous(), kv2.contiguous(), attn_sink, topk_idxs.contiguous(), o,
                                                  n1, n2, t, softmax_scale, H=h, HB=HB, D=d, BLOCK_T=64, num_warps=4)
    return o


# --------------------------------------------------------------------------- split-slot decode attention (flash-decoding style)
@triton.jit(do_not_specialize=["n_kv1", "n_kv2", "topk"])
def _sattn_split_kernel(Q, KV1, KV2, IDX, PM, PL, PACC, n_kv1, n_kv2, topk, scale,
                        H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    hs = hb * HB + tl.arange(0, HB)
    dd = tl.arange(0, D)
    q = tl.load(Q + (b * H + hs)[:, None] * D + dd[None, :])
    tt = sp * BLOCK_T + tl.arange(0, BLOCK_T)
    idx = tl.load(IDX + b * topk + tt, mask=tt < topk, other=-1)
    valid = idx >= 0
    in1 = valid & (idx < n_kv1)
    in2 = valid & (idx >= n_kv1)
    kv = tl.load(KV1 + (b * n_kv1 + tl.maximum(idx, 0))[:, None] * D + dd[None, :], mask=in1[:, None], other=0.0)
    kv += tl.load(KV2 + (b * n_kv2 + tl.maximum(idx - n_kv1, 0))[:, None] * D + dd[None, :], mask=in2[:, None], other=0.0)
    s = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
    s = tl.where(valid[None, :], s, -1e30)
    m = tl.max(s, axis=1)
    p = tl.exp(s - m[:, None])
    p = tl.where(valid[None, :], p, 0.0)
    l = tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
    base = (b * H + hs) * NSPLIT + sp
    tl.store(PM + base, m)
    tl.store(PL + base, l)
    tl.store(PACC + base[:, None] * D + dd[None, :], acc)


@triton.jit
def _sattn_combine_kernel(PM, PL, PACC, SINK, O, H: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    dd = tl.arange(0, D)
    base = (b * H + h) * NSPLIT + sp
    m = tl.load(PM + base)
    l = tl.load(PL + base)
    mx = tl.max(m, axis=0)
    w = tl.exp(m - mx)
    acc = tl.load(PACC + base[:, None] * D + dd[None, :])  # [NSPLIT, D]
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0) + tl.exp(tl.load(SINK + h) - mx)
    tl.store(O + (b * H + h) * D + dd, (num / den).to(O.dtype.element_ty))


def sparse_attn_decode_split(q, kv1, kv2, attn_sink, topk_idxs, softmax_scale, block_t: int = 64):
    b, s, h, d = q.shape
    assert s == 1
    n1 = kv1.shape[1]
    if kv2 is None:
        kv2, n2 = kv1, 0
    else:
        n2 = kv2.shape[1]
    t = topk_idxs.shape[-1]
    nsplit = triton.next_power_of_2(triton.cdiv(t, block_t))
    HB = 16
    pm = torch.empty(b * h * nsplit, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    pacc = torch.empty(b * h * nsplit, d, device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _sattn_split_kernel[(b, h // HB, nsplit)](q.contiguous(), kv1.contiguous(), kv2.contiguous(), topk_idxs.contiguous(), pm, pl, pacc,
                                                  n1, n2, t, softmax_scale, H=h, HB=HB, D=d, BLOCK_T=block_t, NSPLIT=nsplit, num_warps=4)
        _sattn_combine_kernel[(b, h)](pm, pl, pacc, attn_sink, o, H=h, D=d, NSPLIT=nsplit, num_warps=4)
    return o
