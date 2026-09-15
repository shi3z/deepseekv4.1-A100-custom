"""Second round of fusion for the static decode path: one layer in ~25 launches instead of ~100.

The decode step is one CUDA graph per GPU, so nothing here is about CPU launch cost: it is the
device time of thousands of 2-5 us kernels executed back to back. Every kernel below replaces a
chain of 5-15 torch ops on a single token and keeps the reference rounding points (bf16 stores
between stages, per-32 FP8 fake quantization) so the results match the unfused path.

  hc_mix              : 24 mixing logits of the hyper-connection block (rmsnorm-scaled dot products)
  hc_pre_norm_quant   : sinkhorn split of the mixes + hc_pre + rmsnorm + fp8 fake quant (+ fp32 copy)
  norm_quant          : rmsnorm + fp8 fake quant (q_norm)
  kv_write            : kv_norm + rope + fp8 fake quant + write into the sliding-window ring
  sattn2              : split-K sparse attention over the window ring (validity from pos) and the
                        compressed cache (top-k indices), combine fused with the inverse rope
  gate_topk           : sqrt(softplus) / bias / top-k / weight normalisation of the MoE gate
  hc_post2            : (sum of routed experts + shared expert ->) bf16 -> hc_post, in place
"""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fused import _ceil_log2, _pow2, _round_e4m3


@triton.jit
def _fq8(x):
    """Per-row FP8 fake quantization of a 2D tile [rows, 32] (fp32 in, fp32 out)."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = _pow2(_ceil_log2(libdevice.div_rn(amax, 448.0)))
    q = _round_e4m3(tl.minimum(tl.maximum(libdevice.div_rn(x, s[:, None]), -448.0), 448.0))
    return q * s[:, None]


# --------------------------------------------------------------------------- hyper-connection mixes
@triton.jit
def _hc_mix_kernel(X, FN, OUT, eps, N: tl.constexpr, NMIX: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    row = tl.program_id(1)
    acc = tl.zeros((BLOCK,), tl.float32)
    ss = tl.zeros((BLOCK,), tl.float32)
    for i in range(0, N, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        x = tl.load(X + row * N + offs).to(tl.float32)
        f = tl.load(FN + k * N + offs).to(tl.float32)
        acc += x * f
        ss += x * x
    dot = tl.sum(acc, axis=0)
    var = tl.sum(ss, axis=0) / N
    tl.store(OUT + row * NMIX + k, dot * (1.0 / tl.sqrt(var + eps)))


@triton.jit
def _sinkhorn_only_kernel(MIX, SCALE, BASE, PRE, POST, COMB, eps, iters: tl.constexpr, HC: tl.constexpr):
    row = tl.program_id(0)
    _sinkhorn_part(MIX + row * (2 + HC) * HC, SCALE, BASE, PRE + row * HC, POST + row * HC, COMB + row * HC * HC, eps, iters, HC)


def hc_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, hc_eps: float):
    """(pre [B, hc], post [B, hc], comb [B, hc, hc]) from the mixing logits [B, (2+hc)*hc] (the split part alone);
    without the leading B when the input is one row."""
    dev = mixes.device
    B = mixes.shape[0] if mixes.dim() == 2 else 1
    pre = torch.empty(B, hc, device=dev, dtype=torch.float32)
    post = torch.empty(B, hc, device=dev, dtype=torch.float32)
    comb = torch.empty(B, hc, hc, device=dev, dtype=torch.float32)
    with torch.cuda.device(dev):
        _sinkhorn_only_kernel[(B,)](mixes, scale, base, pre, post, comb, hc_eps, iters=iters, HC=hc, num_warps=1)
    if mixes.dim() == 1:
        return pre[0], post[0], comb[0]
    return pre, post, comb


def hc_mix(x: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [B, 1, hc, d] bf16 (contiguous), fn: [n_mix, hc*d] fp32 -> [B, n_mix] fp32 ([n_mix] for B = 1)."""
    n_mix, N = fn.shape
    B = x.numel() // N
    assert x.numel() == B * N and x.is_contiguous()
    if B > 32:  # the per-(row, mix) kernel re-reads every row per mix: cuBLAS fp32 is 2x faster from 64 rows on
        xf = x.view(B, N).float()
        return torch.mm(xf, fn.t()) * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = torch.empty(B, n_mix, device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        _hc_mix_kernel[(n_mix, B)](x, fn, out, eps, N=N, NMIX=n_mix, BLOCK=4096, num_warps=8)
    return out.view(-1) if B == 1 else out



@triton.jit
def _sinkhorn_part(MIX, SCALE, BASE, PRE, POST, COMB, eps, iters: tl.constexpr, HC: tl.constexpr):
    j = tl.arange(0, HC)
    jj = tl.arange(0, HC)[:, None]
    kk = tl.arange(0, HC)[None, :]
    s0 = tl.load(SCALE)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    pre = tl.sigmoid(tl.load(MIX + j) * s0 + tl.load(BASE + j)) + eps
    post = 2.0 * tl.sigmoid(tl.load(MIX + HC + j) * s1 + tl.load(BASE + HC + j))
    idx = 2 * HC + jj * HC + kk
    comb = tl.load(MIX + idx) * s2 + tl.load(BASE + idx)
    rmax = tl.max(comb, axis=1)
    comb = tl.exp(comb - rmax[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in range(iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    tl.store(PRE + j, pre)
    tl.store(POST + j, post)
    tl.store(COMB + jj * HC + kk, comb)


@triton.jit
def _perm8(offs):
    """Index map of the 8-k permuted activation layout used by cuda/fp4_tc.cu (bit reversal of the low 3 bits)."""
    j = offs & 7
    return (offs & ~7) | ((j & 1) << 2) | (j & 2) | (j >> 2)


@triton.jit
def _hc_prenq_kernel(X, PREIN, MIX, SCALE, BASE, W, PRE, POST, COMB, Y, YQ, YF, YQP, eps, hc_eps,
                     iters: tl.constexpr, HC: tl.constexpr, D: tl.constexpr, R: tl.constexpr, WRITE_YF: tl.constexpr, SINKHORN: tl.constexpr,
                     WRITE_YQP: tl.constexpr):
    if SINKHORN:
        _sinkhorn_part(MIX, SCALE, BASE, PRE, POST, COMB, hc_eps, iters, HC)
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, 32)[None, :]
    offs = r * 32 + c
    mask = offs < D
    acc = tl.zeros((R, 32), tl.float32)
    for h in tl.static_range(HC):
        p = tl.load(PREIN + h)
        acc += p * tl.load(X + h * D + offs, mask=mask, other=0.0).to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)  # hc_pre output is stored in bf16
    var = tl.sum(tl.sum(acc * acc, axis=1), axis=0) / D
    y = acc * (1.0 / tl.sqrt(var + eps)) * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = y.to(tl.bfloat16).to(tl.float32)  # rmsnorm output is stored in bf16
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)
    if WRITE_YF:
        tl.store(YF + offs, y, mask=mask)
    yq = _fq8(y).to(tl.bfloat16)
    tl.store(YQ + offs, yq, mask=mask)
    if WRITE_YQP:
        tl.store(YQP + _perm8(offs), yq, mask=mask)


def hc_pre_norm_quant(x, pre_in, mixes, scale, base, w, eps, hc_eps, iters, want_f32=False, out=None, sinkhorn=True, want_perm=False):
    """x: [1, 1, hc, d] bf16; pre_in: [hc] fp32 (previous sub-block's pre mix); mixes: [(2+hc)*hc] fp32.
    Returns (pre, post, comb, y, yq, yf): the sinkhorn split of `mixes`, the hc_pre+rmsnorm output y (bf16 [1, d]),
    its fp8 fake-quantized copy yq, and (optionally) y as fp32. `out`: optional dict of persistent output buffers
    (keys pre, post, comb, y, yq, yf) for static-buffer runtimes."""
    hc, d = x.shape[-2], x.shape[-1]
    dev = x.device
    out = out or {}
    if not sinkhorn:  # the split runs elsewhere: the kernel only reads PREIN
        mixes = scale = base = x
    pre = out.get("pre") if out.get("pre") is not None else torch.empty(hc, device=dev, dtype=torch.float32)
    post = out.get("post") if out.get("post") is not None else torch.empty(hc, device=dev, dtype=torch.float32)
    comb = out.get("comb") if out.get("comb") is not None else torch.empty(hc, hc, device=dev, dtype=torch.float32)
    y = out.get("y") if out.get("y") is not None else torch.empty(1, d, device=dev, dtype=torch.bfloat16)
    yq = out.get("yq") if out.get("yq") is not None else torch.empty(1, d, device=dev, dtype=torch.bfloat16)
    yf = (out.get("yf") if out.get("yf") is not None else torch.empty(1, d, device=dev, dtype=torch.float32)) if want_f32 else y
    yqp = (out.get("yqp") if out.get("yqp") is not None else torch.empty(1, d, device=dev, dtype=torch.bfloat16)) if want_perm else y
    R = triton.next_power_of_2(d // 32)
    with torch.cuda.device(dev):
        _hc_prenq_kernel[(1,)](x, pre_in, mixes, scale, base, w, pre, post, comb, y, yq, yf, yqp, eps, hc_eps,
                               iters=iters, HC=hc, D=d, R=R, WRITE_YF=want_f32, SINKHORN=sinkhorn, WRITE_YQP=want_perm, num_warps=8)
    if want_perm:
        return pre, post, comb, y, yq, (yf if want_f32 else None), yqp
    return pre, post, comb, y, yq, (yf if want_f32 else None)


# --------------------------------------------------------------------------- hc_pre + rmsnorm + quant in two multi-CTA passes
@triton.jit
def _hc_pre2_kernel(X, PREIN, Y, SS, D: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, NB: tl.constexpr):
    """pass 1: y = bf16(sum_h pre[h] * x[h]) for one column block of one row, plus its sum of squares."""
    cb = tl.program_id(0)
    row = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    acc = tl.zeros((BLOCK,), tl.float32)
    for h in tl.static_range(HC):
        acc += tl.load(PREIN + row * HC + h) * tl.load(X + (row * HC + h) * D + cols, mask=mask, other=0.0).to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + cols, acc.to(tl.bfloat16), mask=mask)
    tl.store(SS + row * NB + cb, tl.sum(acc * acc, axis=0))


@triton.jit
def _norm_quant2_kernel(Y, SS, W, YN, YQ, YF, YQP, eps, D: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr,
                        WRITE_YF: tl.constexpr, WRITE_YQP: tl.constexpr):
    """pass 2: rmsnorm of y (variance from the pass-1 partials) and the per-32 fp8 fake quant, one column block of one row."""
    cb = tl.program_id(0)
    row = tl.program_id(1)
    var = tl.sum(tl.load(SS + row * NB + tl.arange(0, NB)), axis=0) / D
    r = tl.arange(0, BLOCK // 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    col = cb * BLOCK + r * 32 + c
    offs = row * D + col
    mask = col < D
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    y = y * (1.0 / tl.sqrt(var + eps)) * tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    y = y.to(tl.bfloat16).to(tl.float32)
    tl.store(YN + offs, y.to(tl.bfloat16), mask=mask)
    if WRITE_YF:
        tl.store(YF + offs, y, mask=mask)
    yq = _fq8(y).to(tl.bfloat16)
    tl.store(YQ + offs, yq, mask=mask)
    if WRITE_YQP:
        tl.store(YQP + _perm8(offs), yq, mask=mask)


def hc_pre_norm_quant2(x, pre_in, w, eps, want_f32=False, out=None, want_perm=False):
    """Same outputs as hc_pre_norm_quant without the split (y, yq, yf, yqp: [B, d]), as two multi-CTA kernels
    (~3 us each instead of one ~13 us single-CTA kernel). x: [B, 1, hc, d]; pre_in: [B, hc] (or [hc] for B = 1)."""
    hc, d = x.shape[-2], x.shape[-1]
    B = x.shape[0]
    dev = x.device
    out = out or {}
    BLOCK = 1024
    nb = triton.cdiv(d, BLOCK)
    NB = triton.next_power_of_2(nb)
    ytmp = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
    ss = torch.empty(B, NB, device=dev, dtype=torch.float32)
    y = out.get("y") if out.get("y") is not None else torch.empty(B, d, device=dev, dtype=torch.bfloat16)
    yq = out.get("yq") if out.get("yq") is not None else torch.empty(B, d, device=dev, dtype=torch.bfloat16)
    yf = (out.get("yf") if out.get("yf") is not None else torch.empty(B, d, device=dev, dtype=torch.float32)) if want_f32 else y
    yqp = (out.get("yqp") if out.get("yqp") is not None else torch.empty(B, d, device=dev, dtype=torch.bfloat16)) if want_perm else y
    with torch.cuda.device(dev):
        if NB > nb:
            ss.zero_()
        _hc_pre2_kernel[(nb, B)](x, pre_in, ytmp, ss, D=d, HC=hc, BLOCK=BLOCK, NB=NB, num_warps=4)
        _norm_quant2_kernel[(nb, B)](ytmp, ss, w, y, yq, yf, yqp, eps, D=d, NB=NB, BLOCK=BLOCK, WRITE_YF=want_f32, WRITE_YQP=want_perm, num_warps=4)
    if want_perm:
        return y, yq, (yf if want_f32 else None), yqp
    return y, yq, (yf if want_f32 else None)


# --------------------------------------------------------------------------- rmsnorm + fp8 quant
@triton.jit
def _norm_quant_kernel(X, W, YQ, eps, D: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, 32)[None, :]
    offs = r * 32 + c
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(tl.sum(x * x, axis=1), axis=0) / D
    y = x * (1.0 / tl.sqrt(var + eps)) * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = y.to(tl.bfloat16).to(tl.float32)
    tl.store(YQ + row * D + offs, _fq8(y).to(tl.bfloat16), mask=mask)


def norm_quant(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """rmsnorm of each row (bf16 [B, d]) followed by the per-32 FP8 fake quantization -> bf16 [B, d]."""
    d = x.shape[-1]
    xc = x.contiguous()
    B = xc.numel() // d
    yq = torch.empty_like(xc)
    with torch.cuda.device(x.device):
        _norm_quant_kernel[(B,)](xc, w, yq, eps, D=d, R=triton.next_power_of_2(d // 32), num_warps=4)
    return yq


# --------------------------------------------------------------------------- kv: norm + rope + quant + ring write
@triton.jit
def _kv_write_kernel(X, W, COS, SIN, POS, SEQ, CACHE, eps, WIN: tl.constexpr, D: tl.constexpr, RD: tl.constexpr):
    row = tl.program_id(0)
    seq = tl.load(SEQ + row)
    offs = tl.arange(0, D)
    x = tl.load(X + row * D + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * (1.0 / tl.sqrt(var + eps)) * tl.load(W + offs).to(tl.float32)
    y = y.to(tl.bfloat16).to(tl.float32)
    # rope on the last RD elements, interleaved (real, imag) pairs
    pos = tl.load(POS + row)
    re, im = tl.split(tl.reshape(y, (D // 2, 2)))
    j = tl.arange(0, D // 2)
    rj = j - (D - RD) // 2
    rm = rj >= 0
    c = tl.load(COS + pos * (RD // 2) + rj, mask=rm, other=1.0)
    s = tl.load(SIN + pos * (RD // 2) + rj, mask=rm, other=0.0)
    yr = (re * c - im * s).to(tl.bfloat16).to(tl.float32)
    yi = (re * s + im * c).to(tl.bfloat16).to(tl.float32)
    y2 = tl.reshape(tl.join(yr, yi), (D // 32, 32))
    q = _fq8(y2).to(tl.bfloat16)
    slot = pos % WIN
    r = tl.arange(0, D // 32)[:, None]
    cc = tl.arange(0, 32)[None, :]
    tl.store(CACHE + (seq * WIN + slot) * D + r * 32 + cc, q)


def kv_write(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, cache: torch.Tensor, rd: int, eps: float, seq: torch.Tensor):
    """x: bf16 [B, d]; pos, seq: int64 [B].

    Normalize index tensors onto the launch device. Triton pointer
    arguments cannot point at CPU memory.
    """
    dev = x.device

    if pos.device != dev or pos.dtype != torch.int64 or not pos.is_contiguous():
        pos = pos.to(
            device=dev,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

    if seq.device != dev or seq.dtype != torch.int64 or not seq.is_contiguous():
        seq = seq.to(
            device=dev,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

    d = x.shape[-1]
    win = cache.shape[1]
    B = x.numel() // d

    with torch.cuda.device(dev):
        _kv_write_kernel[(B,)](x.contiguous(), w, cos, sin, pos, seq, cache, eps, WIN=win, D=d, RD=rd, num_warps=4)


# --------------------------------------------------------------------------- sparse attention v2
@triton.jit(do_not_specialize=["n_kvc", "topk"])
def _sattn2_split_kernel(Q, KVW, KVC, IDX, POS, SEQ, PMAX, PLIM, PM, PL, PACC, n_kvc, topk, scale,
                         H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
                         NSPLIT: tl.constexpr, NB: tl.constexpr, NWIN: tl.constexpr, WIN: tl.constexpr, HAS_C: tl.constexpr):
    """Row b (a query at position POS[b] of sequence SEQ[b]) over the sequence's window ring (slot = position % WIN;
    the ring holds the positions up to PMAX[b], the newest written for that sequence, so slots holding positions
    beyond POS[b] are masked) and the compressed cache rows IDX[b]. Split sp handles NB consecutive key blocks
    (window blocks first, then compressed blocks) with an online softmax; the combine kernel merges the splits."""
    b = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    hs = hb * HB + tl.arange(0, HB)
    dd = tl.arange(0, D)
    seq = tl.load(SEQ + b)
    q = tl.load(Q + (b * H + hs)[:, None] * D + dd[None, :])
    pos = tl.load(POS + b)
    pmax = tl.load(PMAX + b)
    plim = tl.load(PLIM + b)  # newest ring position this row may see (== pos for ordinary decode)
    m_i = tl.full((HB,), -1e30, tl.float32)
    l_i = tl.zeros((HB,), tl.float32)
    acc = tl.zeros((HB, D), tl.float32)
    for j in range(NB):
        blk = sp * NB + j
        if blk < NWIN:
            tt = (blk * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int32)
            held = pmax - (((pmax - tt) % WIN + WIN) % WIN)  # the position slot tt holds
            valid = (held >= 0) & (held <= plim) & (pos - held < WIN)
            kv = tl.load(KVW + (seq * WIN + tt)[:, None] * D + dd[None, :], mask=valid[:, None], other=0.0)
        else:
            tt = (blk - NWIN) * BLOCK_T + tl.arange(0, BLOCK_T)
            if HAS_C:
                idx = tl.load(IDX + b * topk + tt, mask=tt < topk, other=-1)
            else:
                idx = tl.full((BLOCK_T,), -1, tl.int32)
            valid = idx >= 0
            kv = tl.load(KVC + (seq * n_kvc + tl.maximum(idx, 0))[:, None] * D + dd[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
        s = tl.where(valid[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new
    base = (b * H + hs) * NSPLIT + sp
    tl.store(PM + base, m_i)
    tl.store(PL + base, l_i)
    tl.store(PACC + base[:, None] * D + dd[None, :], acc)


@triton.jit
def _sattn2_combine_kernel(PM, PL, PACC, SINK, COS, SIN, POS, O, H: tl.constexpr, D: tl.constexpr, RD: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    dd = tl.arange(0, D)
    base = (b * H + h) * NSPLIT + sp
    m = tl.load(PM + base)
    l = tl.load(PL + base)
    mx = tl.max(m, axis=0)
    w = tl.exp(m - mx)
    acc = tl.load(PACC + base[:, None] * D + dd[None, :])
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0) + tl.exp(tl.load(SINK + h) - mx)
    o = (num / den).to(tl.bfloat16).to(tl.float32)
    # inverse rope on the last RD elements
    pos = tl.load(POS + b)
    re, im = tl.split(tl.reshape(o, (D // 2, 2)))
    j = tl.arange(0, D // 2)
    rj = j - (D - RD) // 2
    rm = rj >= 0
    c = tl.load(COS + pos * (RD // 2) + rj, mask=rm, other=1.0)
    s = -tl.load(SIN + pos * (RD // 2) + rj, mask=rm, other=0.0)
    yr = (re * c - im * s).to(tl.bfloat16)
    yi = (re * s + im * c).to(tl.bfloat16)
    tl.store(O + (b * H + h) * D + dd, tl.reshape(tl.join(yr, yi), (D,)))


SATTN_NB = int(os.environ.get("DSV41_SATTN_NB", "0"))  # experiment: key blocks per split (0 = by row count)
SATTN_HB = int(os.environ.get("DSV41_SATTN_HB", "0"))  # heads per program (0 = by row count: 16 for <= 8 rows, else 32)


def sattn2(q, kv_win, kv_c, idx, pos, attn_sink, cos, sin, rd, softmax_scale, seq, pmax, block_t: int = 64, plim=None):
    """q: [B, 1, h, d] bf16 (B query rows at positions pos [B] of sequences seq [B]); kv_win: [S, win, d] rings;
    pmax [B]: newest position written to the row's ring; kv_c: [S, n, d] compressed caches or None;
    idx: [B, 1, t] int32 rows of kv_c (-1 = none) or None. Returns the attention output with the inverse rope: [B, 1, h, d]."""
    b, s, h, d = q.shape
    assert s == 1
    win = kv_win.shape[1]
    nwin = win // block_t
    if kv_c is None:
        kv_c, n_c, t = kv_win, 0, 0
        idx = pos  # unused
        nblk = nwin
    else:
        n_c = kv_c.shape[1]
        t = idx.shape[-1]
        nblk = nwin + triton.cdiv(t, block_t)
    # key blocks per split: one row needs many splits for parallelism, many rows do not (the fp32 partials
    # [B, h, nsplit, d] then cost more than the attention itself)
    nb = SATTN_NB if SATTN_NB else (1 if b <= 8 else 4)
    nsplit = triton.next_power_of_2(triton.cdiv(nblk, nb))
    HB = SATTN_HB if SATTN_HB else (16 if b <= 8 else 32)
    pm = torch.empty(b * h * nsplit, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    pacc = torch.empty(b * h * nsplit, d, device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _sattn2_split_kernel[(b, h // HB, nsplit)](q, kv_win, kv_c, idx, pos, seq, pmax, plim if plim is not None else pos, pm, pl, pacc, n_c, t, softmax_scale,
                                                   H=h, HB=HB, D=d, BLOCK_T=block_t, NSPLIT=nsplit, NB=nb, NWIN=nwin, WIN=win, HAS_C=t > 0, num_warps=HB // 4)
        _sattn2_combine_kernel[(b, h)](pm, pl, pacc, attn_sink, cos, sin, pos, o, H=h, D=d, RD=rd, NSPLIT=nsplit, num_warps=4)
    return o


# --------------------------------------------------------------------------- MoE gate
@triton.jit
def _gate_topk_kernel(S, BIAS, EID, WT, temp, route_scale, E: tl.constexpr, EP: tl.constexpr, TOPK: tl.constexpr,
                      MODE: tl.constexpr, NORM: tl.constexpr):
    row = tl.program_id(0)
    S += row * E
    EID += row * TOPK
    WT += row * TOPK
    e = tl.arange(0, EP)
    mask = e < E
    s = tl.load(S + e, mask=mask, other=0.0) / temp
    if MODE == 0:  # sqrt(softplus)
        s = tl.sqrt(tl.where(s > 20.0, s, libdevice.log1p(tl.exp(s))))
    elif MODE == 1:  # sigmoid
        s = tl.sigmoid(s)
    else:  # softmax
        mx = tl.max(tl.where(mask, s, -1e30), axis=0)
        ex = tl.where(mask, tl.exp(s - mx), 0.0)
        s = ex / tl.sum(ex, axis=0)
    b = tl.where(mask, s + tl.load(BIAS + e, mask=mask, other=0.0), -1e30)
    wsum = 0.0
    for k in tl.static_range(TOPK):
        m = tl.max(b, axis=0)
        i = tl.min(tl.where(b == m, e, EP), axis=0)
        w = tl.sum(tl.where(e == i, s, 0.0), axis=0)
        tl.store(EID + k, i.to(tl.int32))
        tl.store(WT + k, w)
        wsum += w
        b = tl.where(e == i, -1e30, b)
    if NORM:
        for k in tl.static_range(TOPK):
            tl.store(WT + k, tl.load(WT + k) / (wsum + 1e-20) * route_scale)
    else:
        for k in tl.static_range(TOPK):
            tl.store(WT + k, tl.load(WT + k) * route_scale)


def gate_topk(scores: torch.Tensor, bias: torch.Tensor, temp: float, topk: int, route_scale: float, score_func: str, norm: bool, eid=None, wt=None):
    """scores: fp32 [B, E] (gate GEMV output) -> (eid int32 [B, topk], wt fp32 [B, topk]) like MoE.gate ([topk] when B = 1
    and no output buffers are given)."""
    E = scores.shape[-1]
    B = scores.numel() // E
    squeeze = B == 1 and eid is None and wt is None
    if eid is None:
        eid = torch.empty(B, topk, device=scores.device, dtype=torch.int32)
    if wt is None:
        wt = torch.empty(B, topk, device=scores.device, dtype=torch.float32)
    mode = {"sqrtsoftplus": 0, "sigmoid": 1, "softmax": 2}[score_func]
    with torch.cuda.device(scores.device):
        _gate_topk_kernel[(B,)](scores.contiguous(), bias, eid, wt, float(temp), float(route_scale), E=E, EP=triton.next_power_of_2(E),
                                TOPK=topk, MODE=mode, NORM=norm, num_warps=4)
    if squeeze:
        return eid.view(-1), wt.view(-1)
    return eid, wt


# --------------------------------------------------------------------------- hc_post (in place), optionally summing the MoE outputs
@triton.jit
def _hc_post2_kernel(X, Y2, YS, R, POST, COMB, y2_row_stride, y2_sum_stride, D: tl.constexpr, HC: tl.constexpr, BLOCK_D: tl.constexpr, NSUM: tl.constexpr):
    cb = tl.program_id(0)
    row = tl.program_id(1)
    X += row * D
    Y2 += row * y2_row_stride
    YS += row * D
    R += row * HC * D
    POST += row * HC
    COMB += row * HC * HC
    cols = cb * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    if NSUM > 0:
        x = tl.zeros((BLOCK_D,), tl.float32)
        for k in tl.static_range(NSUM):
            x += tl.load(Y2 + k * y2_sum_stride + cols, mask=mask, other=0.0).to(tl.float32)
        x += tl.load(YS + cols, mask=mask, other=0.0).to(tl.float32)
        x = x.to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(R + 0 * D + cols, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(R + 1 * D + cols, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(R + 2 * D + cols, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(R + 3 * D + cols, mask=mask, other=0.0).to(tl.float32)
    for i in tl.static_range(HC):
        acc = tl.load(POST + i) * x
        acc += tl.load(COMB + 0 * HC + i) * r0
        acc += tl.load(COMB + 1 * HC + i) * r1
        acc += tl.load(COMB + 2 * HC + i) * r2
        acc += tl.load(COMB + 3 * HC + i) * r3
        tl.store(R + i * D + cols, acc.to(tl.bfloat16), mask=mask)


def hc_post2_(x, residual, post, comb, y2=None, ys=None, y2_sum_first=False):
    """In-place hyper-connection post mix: residual [B, 1, hc, d] bf16 <- post * x + comb^T residual.
    Either x (bf16 [B, d]) or the MoE pieces y2 (fp32 [B, nsum, d], or [nsum, d] for B = 1, or [nsum, B, d] with
    y2_sum_first) + ys (bf16 [B, d], shared expert). post [B, hc], comb [B, hc, hc] (leading B optional when B = 1)."""
    hc, d = residual.shape[-2], residual.shape[-1]
    B = residual.shape[0]
    assert hc == 4 and residual.is_contiguous()
    if y2 is None:
        nsum, rs, ss = 0, 0, 0
    elif y2_sum_first:
        y2 = y2.contiguous()
        nsum, rs, ss = y2.shape[0], d, y2.shape[1] * d
    else:
        y2 = y2.contiguous()
        nsum, rs, ss = y2.shape[-2], y2.shape[-2] * d, d
    with torch.cuda.device(residual.device):
        _hc_post2_kernel[(triton.cdiv(d, 1024), B)](x if x is not None else residual, y2 if y2 is not None else residual,
                                                     ys.contiguous() if ys is not None else residual,
                                                     residual, post.contiguous(), comb.contiguous(), rs, ss, D=d, HC=hc, BLOCK_D=1024, NSUM=nsum, num_warps=4)
    return residual
