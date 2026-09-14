"""Grouped MoE kernel for the VQ12 expert format (a100-vq/vq12.py).

Same slot geometry, same rate and the same [BN, 64] + [BN, 32] loads per 256 K as CB3 v2
(tools/cb3_moe.py), so the arena, the store and the routing are unchanged. The only thing that
differs is how a scale group's packed-FP4 byte tile is rebuilt:

    CB3   a 3-bit index per weight, looked up in the ROW's 8-entry codebook held in a 32-bit
          register:            ne = (cw >> (ie * 4)) & 15
    VQ12  a 12-bit index per GROUP OF FOUR weights, looked up in a global 4096-entry table:
                               v  = tl.load(lut + idx)       -> two packed bytes at once

That is the whole change, and it is worth halving the quantisation damage: over all 40 layers,
unpruned, CB3 is +8.54 % wikitext PPL against the FP4 checkpoint and VQ12 is +4.23 % (code +2.17 %
vs +0.97 %) at identical bytes. CB3's per-row subset search is already optimal for its format; the
gap is scalar vs vector quantisation.

Built on the v2 path rather than v3 (whose inline-PTX decode is specific to the 3-bit-index
arithmetic), so it starts from v2's speed, not v3's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import cb3_moe as C3
from cb3_moe import _split2, _split8, CB3Arena
from fp4_moe import DIM, INTER, _chunk_dot, _pick_bm, _ue8m0, build_routing


@triton.jit
def _grp_packed_vq(lo_ptr, hi_ptr, lut_ptr, BN: tl.constexpr):
    """One scale group's packed-FP4 byte tile [BN, 16], built only from loads and elementwise ops.

    The obvious construction -- look the 8 group indices up once and `tl.interleave` the entry's two
    bytes into 16 -- produces the right VALUES (storing the tile proves it) but a register layout
    `tl.dot` then reads wrongly: the same tile is exact after a round trip through memory and wrong
    straight from the registers. Loads always produce a layout the dot accepts, so each output byte
    fetches its own group's index instead: byte j belongs to group j // 2 and takes the entry's low
    byte for even j, the high byte for odd. The redundant fetches are L1 hits.
    """
    j = tl.arange(0, 16)[None, :]
    g = j // 2                                     # the VQ group this output byte belongs to
    lo = tl.load(lo_ptr + g).to(tl.int32) & 0xFF   # [BN, 16] group g's low 8 index bits
    hb = tl.load(hi_ptr + (g // 2)).to(tl.int32) & 0xFF
    hn = tl.where(g % 2 == 0, hb & 0xF, (hb >> 4) & 0xF)
    v = tl.load(lut_ptr + (lo | (hn << 8)))        # [BN, 16] the entry's two packed bytes
    return tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF).to(tl.uint8)


@triton.jit
def _e2m1_f16(c):
    """E2M1 nibble -> fp16, by building the bit pattern: no table, no inline asm.

    code = s ee m.  e == 0 is the subnormal 0.5m; otherwise 2^(e-1) (1 + m/2), which in fp16 is
    exponent field e + 14 and mantissa field m << 9.
    """
    e = (c >> 1) & 3
    m = c & 1
    bits = tl.where(e == 0, m * 0x3800, ((e + 14) << 10) | (m << 9)) | ((c & 8) << 12)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _chunk_dot_lut(x_base, xk, mask_m, packed, scale_u8, tab_ptr):
    """`fp4_moe._chunk_dot` without its inline-asm decode.

    `_fp4_decode` is `tl.inline_asm_elementwise(..., pack=4)`, whose four-elements-per-register PTX
    depends on the tile's register layout; retried once this kernel's layout was made normal it was
    still slower here, so the nibble goes through a 16-entry fp16 table instead.
    """
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p8 = packed.to(tl.int32) & 0xFF
    # a 16-entry fp16 table beats both alternatives here: fp4_moe's packed inline asm measured
    # 1.385 ms and building the fp16 bit pattern arithmetically (`_e2m1_f16`, kept below for the
    # record) 2.220 ms, against this version's 0.677
    p = tl.dot(xe, tl.trans(tl.load(tab_ptr + (p8 & 0xF))))
    p = tl.dot(xo, tl.trans(tl.load(tab_ptr + (p8 >> 4))), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _vq_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, lut_ptr, tab_ptr, BN: tl.constexpr):
    """256 logical K = 8 scale groups: 8 index bytes + 4 nibble bytes + 1 scale byte per group."""
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    acc = _chunk_dot_lut(x_base, xk, mask_m, _grp_packed_vq(lo_ptr, hi_ptr, lut_ptr, BN), s0, tab_ptr)
    acc += _chunk_dot_lut(x_base + 32, xk, mask_m, _grp_packed_vq(lo_ptr + 8, hi_ptr + 4, lut_ptr, BN), s1, tab_ptr)
    acc += _chunk_dot_lut(x_base + 64, xk, mask_m, _grp_packed_vq(lo_ptr + 16, hi_ptr + 8, lut_ptr, BN), s2, tab_ptr)
    acc += _chunk_dot_lut(x_base + 96, xk, mask_m, _grp_packed_vq(lo_ptr + 24, hi_ptr + 12, lut_ptr, BN), s3, tab_ptr)
    acc += _chunk_dot_lut(x_base + 128, xk, mask_m, _grp_packed_vq(lo_ptr + 32, hi_ptr + 16, lut_ptr, BN), s4, tab_ptr)
    acc += _chunk_dot_lut(x_base + 160, xk, mask_m, _grp_packed_vq(lo_ptr + 40, hi_ptr + 20, lut_ptr, BN), s5, tab_ptr)
    acc += _chunk_dot_lut(x_base + 192, xk, mask_m, _grp_packed_vq(lo_ptr + 48, hi_ptr + 24, lut_ptr, BN), s6, tab_ptr)
    acc += _chunk_dot_lut(x_base + 224, xk, mask_m, _grp_packed_vq(lo_ptr + 56, hi_ptr + 28, lut_ptr, BN), s7, tab_ptr)
    return acc


@triton.jit
def _vq_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, s1_ptr, lo3_ptr, hi3_ptr, s3_ptr, h_ptr, lut_ptr, tab_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s1t = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    s3t = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc_g += _vq_block_dot(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32,
                               s1t + b * 8, lut_ptr, tab_ptr, BN)
        acc_u += _vq_block_dot(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32,
                               s3t + b * 8, lut_ptr, tab_ptr, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _vq_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, s2_ptr, y_ptr, lut_ptr, tab_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s2t = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc += _vq_block_dot(h_base + b * 256, xk, mask_m[:, None], lo2 + b * 64, hi2 + b * 32,
                             s2t + b * 8, lut_ptr, tab_ptr, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


# ---------------------------------------------------------------------------- v2: one gather
# v1 takes three dependent loads to turn a weight into an fp16 number: the group's 12-bit index out
# of the lo/hi planes, the entry out of the 4096-entry packed-byte table, then each nibble out of
# the 16-entry E2M1 table. The last two are both pure functions of (entry, position) and position
# is fixed per lane, so they collapse into one table of 8192 fp16 values indexed by
# (entry << 1 | which byte) -- one for the even-k nibble, one for the odd. That removes a gather
# and the whole byte-extraction chain (`& 0xFF`, `>> 8`, `where(j % 2)`, `& 0xF`, `>> 4`).


@triton.jit
def _grp_wf16_vq(lo_ptr, hi_ptr, te_ptr, to_ptr, BN: tl.constexpr):
    """One scale group (32 K) as two fp16 [BN, 16] weight tiles: even k and odd k."""
    j = tl.arange(0, 16)[None, :]
    g = j // 2
    lo = tl.load(lo_ptr + g).to(tl.int32) & 0xFF
    hb = tl.load(hi_ptr + (g // 2)).to(tl.int32) & 0xFF
    hn = tl.where(g % 2 == 0, hb & 0xF, (hb >> 4) & 0xF)
    idx = ((lo | (hn << 8)) << 1) | (j & 1)
    v = tl.load(te_ptr + idx)                      # both fp16 bit patterns in one int32
    we = (v & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    wo = ((v >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    return we, wo


@triton.jit
def _chunk_dot_wf16(x_base, xk, mask_m, we, wo, scale_u8):
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _vq2_grp(x_base, xk, mask_m, lo_ptr, hi_ptr, te_ptr, to_ptr, s, BN: tl.constexpr):
    we, wo = _grp_wf16_vq(lo_ptr, hi_ptr, te_ptr, to_ptr, BN)
    return _chunk_dot_wf16(x_base, xk, mask_m, we, wo, s)


@triton.jit
def _vq2_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, te_ptr, to_ptr, BN: tl.constexpr):
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    acc = _vq2_grp(x_base, xk, mask_m, lo_ptr, hi_ptr, te_ptr, to_ptr, s0, BN)
    acc += _vq2_grp(x_base + 32, xk, mask_m, lo_ptr + 8, hi_ptr + 4, te_ptr, to_ptr, s1, BN)
    acc += _vq2_grp(x_base + 64, xk, mask_m, lo_ptr + 16, hi_ptr + 8, te_ptr, to_ptr, s2, BN)
    acc += _vq2_grp(x_base + 96, xk, mask_m, lo_ptr + 24, hi_ptr + 12, te_ptr, to_ptr, s3, BN)
    acc += _vq2_grp(x_base + 128, xk, mask_m, lo_ptr + 32, hi_ptr + 16, te_ptr, to_ptr, s4, BN)
    acc += _vq2_grp(x_base + 160, xk, mask_m, lo_ptr + 40, hi_ptr + 20, te_ptr, to_ptr, s5, BN)
    acc += _vq2_grp(x_base + 192, xk, mask_m, lo_ptr + 48, hi_ptr + 24, te_ptr, to_ptr, s6, BN)
    acc += _vq2_grp(x_base + 224, xk, mask_m, lo_ptr + 56, hi_ptr + 28, te_ptr, to_ptr, s7, BN)
    return acc


@triton.jit
def _vq2_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, s1_ptr, lo3_ptr, hi3_ptr, s3_ptr, h_ptr, te_ptr, to_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s1t = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    s3t = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc_g += _vq2_block_dot(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32,
                                s1t + b * 8, te_ptr, to_ptr, BN)
        acc_u += _vq2_block_dot(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32,
                                s3t + b * 8, te_ptr, to_ptr, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _vq2_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, s2_ptr, y_ptr, te_ptr, to_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s2t = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc += _vq2_block_dot(h_base + b * 256, xk, mask_m[:, None], lo2 + b * 64, hi2 + b * 32,
                              s2t + b * 8, te_ptr, to_ptr, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


def moe_forward_vq2(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                    swiglu_limit: float = 10.0, block_m: int | None = None,
                    cfg_up=None, cfg_down=None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    assert arena.wte is not None, "VQ12Arena.attach(vq) was never called"
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or _VQ2_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or _VQ2_DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=x.device)
    _vq2_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.s1, arena.w3_lo, arena.w3_hi, arena.s3, h,
        arena.wte, arena.wto, wgt, block_slot, block_pair, x.stride(0), h.stride(0),
        float(swiglu_limit), TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _vq2_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.s2, parts, arena.wte, arena.wto, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


_VQ2_UP_CFG = {16: (16, 1, 3), 32: (16, 1, 3), 64: (16, 1, 3)}
_VQ2_DOWN_CFG = {16: (64, 4, 2), 32: (64, 4, 2), 64: (64, 4, 2)}


# Tuned for this kernel rather than inherited from CB3: the tile now comes from narrow per-byte
# fetches instead of one wide load, which moves the optimum to a smaller BN and fewer warps.
# sweep_vq12.py, 6x6 decode: up (128, 4, 1) -> (16, 1, 3) is 2.10x, down (128, 8, 2) -> (64, 4, 2)
# is 1.36x. Inheriting CB3's table would have left the kernel 2x slower than it needs to be.
_VQ_UP_CFG = {16: (16, 1, 3), 32: (16, 1, 3), 64: (16, 1, 3)}
_VQ_DOWN_CFG = {16: (64, 4, 2), 32: (64, 4, 2), 64: (64, 4, 2)}


class VQ12Arena(CB3Arena):
    """CB3's tensors exactly (the `*_cb` planes go unused: the codebook is global)."""

    def __init__(self, slots: int, device="cuda"):
        super().__init__(slots, device)
        self.vq = None      # a100-vq/vq12.py VQ12, set by the caller
        self.lut = None     # int32 [4096], entry -> the two packed-FP4 bytes
        self.tab = None     # fp16 [16], the E2M1 grid
        self.wte = None     # int32 [8192], (entry, byte) -> both nibbles' fp16 bits packed
        self.wto = None     # an alias of wte: the kernel takes one pointer

    def attach(self, vq):
        self.vq = vq
        self.lut = vq.lut.to(self.device).contiguous()
        from vq12 import FP4_VALS
        v = FP4_VALS.to(self.device).to(torch.float16)
        self.tab = v.contiguous()
        # moe_forward_vq2's fused tables: entry idx, byte h -> the fp16 weight of the even nibble
        # (wte) and of the odd one (wto). Byte 0 of an entry is codes 0, 1 and byte 1 is codes 2, 3.
        c = vq.codes.to(self.device).long()
        e = v[c[:, ::2]].reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        o = v[c[:, 1::2]].reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        self.wte = (e | (o << 16)).contiguous()
        self.wto = self.wte
        return self

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                              (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                              (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb = self.vq.pack(wg, sg)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb); s_t[slot].copy_(sg)

    def dequant_slot(self, slot: int):
        return tuple(self.vq.dequant(lo[slot], hi[slot], s[slot]) for lo, hi, s in (
            (self.w1_lo, self.w1_hi, self.s1), (self.w2_lo, self.w2_hi, self.s2),
            (self.w3_lo, self.w3_hi, self.s3)))


def moe_forward_vq(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: VQ12Arena,
                   swiglu_limit: float = 10.0, block_m: int | None = None,
                   cfg_up=None, cfg_down=None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    assert arena.lut is not None, "VQ12Arena.attach(vq) was never called"
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or _VQ_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or _VQ_DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=x.device)
    _vq_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.s1, arena.w3_lo, arena.w3_hi, arena.s3, h, arena.lut, arena.tab,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _vq_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.s2, parts, arena.lut, arena.tab, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
