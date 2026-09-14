"""MoE kernel for CBF8 (a100-vq/cbf8.py): CB3's index layout, eight arbitrary fp16 levels.

`prmt.b32` is a lookup into an eight-BYTE table, so the obvious reading is that a register-only
codebook can only hold byte-sized things -- which is why CB3 stores E2M1 codes and leans on
`cvt.rn.f16x2.e2m1x2` to turn them into numbers. But an fp16 value is two bytes, and two prmt with
the SAME selector, one into the levels' low bytes and one into their high bytes, fetch both halves;
two more prmt interleave them into f16x2 registers. Eight arbitrary fp16 levels, four instructions,
no memory access, and the E2M1 conversion disappears with it:

    per 8 weights          int   prmt   cvt   memory
    CB3 v3                  18     6     4      0      = 29 instructions
    CBF8 (this)             16     8     0      0      = 24

The index extraction is CB3 v3's, unchanged, including its byte-lane-to-nibble compaction -- prmt
wants a four-nibble selector, and those four ops are what buys it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from cb3_moe import CB3Arena, _split2, _split4, _split8, _split16
from fp4_moe import DIM, INTER, _pick_bm, _ue8m0, build_routing, build_routing_small


def _cbf8_asm(sh: int, hb: int) -> str:
    """12 instructions per four weights, 24 per invocation (eight weights).

    $4 = the lo plane's 32 bits, $5 = the hi plane's, $6/$7 = the levels' low bytes (entries 0-3 /
    4-7), $8/$9 = their high bytes. Outputs $0,$1 = the even-k weights as f16x2, $2,$3 = the odd-k.
    """
    def half(shift, bit, o0, o1):
        mv = f"shr.b32 b, $5, {bit - 2};" if bit >= 2 else f"shl.b32 b, $5, {2 - bit};"
        return f"""
shr.b32 a, $4, {shift};
and.b32 a, a, 0x03030303;
{mv}
lop3.b32 ie, a, b, 0x04040404, 0xF8;
shr.b32 t, ie, 4;
lop3.b32 r, ie, t, 0x00FF00FF, 0xA8;
shr.b32 t, r, 8;
or.b32  r, r, t;
prmt.b32 pl, $6, $7, r;
prmt.b32 ph, $8, $9, r;
prmt.b32 {o0}, pl, ph, 0x5140;
prmt.b32 {o1}, pl, ph, 0x7362;"""
    return ("{\n.reg .b32 a, b, ie, t, r, pl, ph;" + half(sh, hb, "$0", "$1")
            + half(sh + 2, hb + 1, "$2", "$3") + "\n}\n")


_F8_00 = tl.constexpr(_cbf8_asm(0, 0))
_F8_42 = tl.constexpr(_cbf8_asm(4, 2))
_F8_04 = tl.constexpr(_cbf8_asm(0, 4))
_F8_46 = tl.constexpr(_cbf8_asm(4, 6))


@triton.jit
def _grp_wf16(Lk, Hm, Alo, Blo, Ahi, Bhi, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(ASM, "=r,=r,=r,=r,r,r,r,r,r,r",
                                     [Lk, Hm, Alo, Blo, Ahi, Bhi],
                                     dtype=(tl.float16, tl.float16), is_pure=True, pack=4)


@triton.jit
def _cdot(x_base, xk, mask_m, we, wo, scale_u8):
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _lev_bytes(c):
    """Eight int8 level codes -> the low and the high byte of each level's fp16 value.

    Once per program, not per weight: the bitcast that would be ruinous inside the K loop costs
    nothing here."""
    i = c.to(tl.int32)
    i = tl.where(i > 127, i - 256, i)
    v = (i.to(tl.float32) * (6.0 / 127.0)).to(tl.float16)
    b = v.to(tl.uint16, bitcast=True).to(tl.int32)
    return (b & 0xFF).to(tl.uint8), ((b >> 8) & 0xFF).to(tl.uint8)


@triton.jit
def _cbf8_tabs(cb_ptr, BN: tl.constexpr):
    """The row's eight levels as four [BN, 16] byte tiles, laid out so pack=4 hands prmt entries
    0-3 in one register and 4-7 in the other (cf. cb3_moe._cb_ab)."""
    j = tl.arange(0, 16)[None, :]
    alo, ahi = _lev_bytes(tl.load(cb_ptr + (j % 4)))
    blo, bhi = _lev_bytes(tl.load(cb_ptr + 4 + (j % 4)))
    return alo, blo, ahi, bhi


@triton.jit
def _pair_dot(x_base, xk, mask_m, Lk, Hm, Alo, Blo, Ahi, Bhi, sA, sB, KPAR: tl.constexpr):
    """The two scale groups sharing one 16-byte lo sub-tile."""
    if KPAR == 0:
        we0, wo0 = _grp_wf16(Lk, Hm, Alo, Blo, Ahi, Bhi, _F8_00)
        we1, wo1 = _grp_wf16(Lk, Hm, Alo, Blo, Ahi, Bhi, _F8_42)
    else:
        we0, wo0 = _grp_wf16(Lk, Hm, Alo, Blo, Ahi, Bhi, _F8_04)
        we1, wo1 = _grp_wf16(Lk, Hm, Alo, Blo, Ahi, Bhi, _F8_46)
    acc = _cdot(x_base, xk, mask_m, we0, wo0, sA)
    acc += _cdot(x_base + 32, xk, mask_m, we1, wo1, sB)
    return acc


@triton.jit
def _blk(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, Alo, Blo, Ahi, Bhi,
         BN: tl.constexpr, BW: tl.constexpr):
    if BW == 512:
        L = tl.load(lo_ptr)
        H = tl.load(hi_ptr)
        La, Lb, Lc, Ld = _split4(L, BN, 32)
        L0, L1 = _split2(La, BN, 16)
        L2, L3 = _split2(Lb, BN, 16)
        L4, L5 = _split2(Lc, BN, 16)
        L6, L7 = _split2(Ld, BN, 16)
        H0, H1, H2, H3 = _split4(H, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15 = _split16(tl.load(s_ptr), BN)
        acc = _pair_dot(x_base, xk, mask_m, L0, H0, Alo, Blo, Ahi, Bhi, s0, s1, 0)
        acc += _pair_dot(x_base + 64, xk, mask_m, L1, H0, Alo, Blo, Ahi, Bhi, s2, s3, 1)
        acc += _pair_dot(x_base + 128, xk, mask_m, L2, H1, Alo, Blo, Ahi, Bhi, s4, s5, 0)
        acc += _pair_dot(x_base + 192, xk, mask_m, L3, H1, Alo, Blo, Ahi, Bhi, s6, s7, 1)
        acc += _pair_dot(x_base + 256, xk, mask_m, L4, H2, Alo, Blo, Ahi, Bhi, s8, s9, 0)
        acc += _pair_dot(x_base + 320, xk, mask_m, L5, H2, Alo, Blo, Ahi, Bhi, s10, s11, 1)
        acc += _pair_dot(x_base + 384, xk, mask_m, L6, H3, Alo, Blo, Ahi, Bhi, s12, s13, 0)
        acc += _pair_dot(x_base + 448, xk, mask_m, L7, H3, Alo, Blo, Ahi, Bhi, s14, s15, 1)
    else:
        L = tl.load(lo_ptr)
        H = tl.load(hi_ptr)
        L0, L1, L2, L3 = _split4(L, BN, 16)
        H0, H1 = _split2(H, BN, 16)
        s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
        acc = _pair_dot(x_base, xk, mask_m, L0, H0, Alo, Blo, Ahi, Bhi, s0, s1, 0)
        acc += _pair_dot(x_base + 64, xk, mask_m, L1, H0, Alo, Blo, Ahi, Bhi, s2, s3, 1)
        acc += _pair_dot(x_base + 128, xk, mask_m, L2, H1, Alo, Blo, Ahi, Bhi, s4, s5, 0)
        acc += _pair_dot(x_base + 192, xk, mask_m, L3, H1, Alo, Blo, Ahi, Bhi, s6, s7, 1)
    return acc


@triton.jit
def _cbf8_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, cb1_ptr, s1_ptr, lo3_ptr, hi3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr,
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
    s1t = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG
    s3t = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG
    A1l, B1l, A1h, B1h = _cbf8_tabs(cb1_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    A3l, B3l, A3h, B3h = _cbf8_tabs(cb3_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    l1a = lo1 + tl.arange(0, 128)[None, :]; h1a = hi1 + tl.arange(0, 64)[None, :]; s1a = s1t + tl.arange(0, 16)[None, :]
    l3a = lo3 + tl.arange(0, 128)[None, :]; h3a = hi3 + tl.arange(0, 64)[None, :]; s3a = s3t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc_g += _blk(x_base + b * 512, xk, mask_m[:, None], l1a + b * 128, h1a + b * 64, s1a + b * 16, A1l, B1l, A1h, B1h, BN, 512)
        acc_u += _blk(x_base + b * 512, xk, mask_m[:, None], l3a + b * 128, h3a + b * 64, s3a + b * 16, A3l, B3l, A3h, B3h, BN, 512)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l1b = lo1 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h1b = hi1 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s1b = s1t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        l3b = lo3 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h3b = hi3 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s3b = s3t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc_g += _blk(x_base + o + b * 256, xk, mask_m[:, None], l1b + b * 64, h1b + b * 32, s1b + b * 8, A1l, B1l, A1h, B1h, BN, 256)
            acc_u += _blk(x_base + o + b * 256, xk, mask_m[:, None], l3b + b * 64, h3b + b * 32, s3b + b * 8, A3l, B3l, A3h, B3h, BN, 256)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cbf8_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr, NB512: tl.constexpr, NB256: tl.constexpr,
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
    s2t = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG
    A2l, B2l, A2h, B2h = _cbf8_tabs(cb2_ptr + slot * (N * 8) + offs_n[:, None] * 8, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    l2a = lo2 + tl.arange(0, 128)[None, :]; h2a = hi2 + tl.arange(0, 64)[None, :]; s2a = s2t + tl.arange(0, 16)[None, :]
    for b in range(0, NB512):
        acc += _blk(h_base + b * 512, xk, mask_m[:, None], l2a + b * 128, h2a + b * 64, s2a + b * 16, A2l, B2l, A2h, B2h, BN, 512)
    if NB256 > 0:
        o: tl.constexpr = NB512 * 512
        l2b = lo2 + (NB512 * 128 + tl.arange(0, 64))[None, :]; h2b = hi2 + (NB512 * 64 + tl.arange(0, 32))[None, :]
        s2b = s2t + (NB512 * 16 + tl.arange(0, 8))[None, :]
        for b in range(0, NB256):
            acc += _blk(h_base + o + b * 256, xk, mask_m[:, None], l2b + b * 64, h2b + b * 32, s2b + b * 8, A2l, B2l, A2h, B2h, BN, 256)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


# Decode-sized calls (BM 16/32) and prefill-sized ones (BM 64) want opposite shapes: one warp on a
# 16-wide output tile for a handful of tokens, four warps on 32 for hundreds. Leaving the decode
# config in place for BM=64 measured 457 ms on a 512-token prefill against 64 with this one.
CBF8_UP_CFG = {16: (16, 1, 1), 32: (16, 1, 1), 64: (32, 4, 2)}
CBF8_DOWN_CFG = {16: (64, 4, 1), 32: (64, 4, 1), 64: (32, 4, 2)}


class CBF8Arena(CB3Arena):
    """CB3's tensors, byte for byte; `cb` holds int8 level codes instead of E2M1 codes."""

    def __init__(self, slots: int, device="cuda"):
        super().__init__(slots, device)
        self.fmt = None

    def attach(self, fmt):
        self.fmt = fmt
        return self

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                              (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                              (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb, sq = self.fmt.pack(wg, sg)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb); s_t[slot].copy_(sq)

    def dequant_slot(self, slot: int):
        return tuple(self.fmt.dequant(lo[slot], hi[slot], cb[slot], s[slot]) for lo, hi, cb, s in (
            (self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
            (self.w2_lo, self.w2_hi, self.w2_cb, self.s2),
            (self.w3_lo, self.w3_hi, self.w3_cb, self.s3)))


def moe_forward_cbf8(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CBF8Arena,
                     swiglu_limit: float = 10.0, block_m: int | None = None,
                     cfg_up=None, cfg_down=None) -> torch.Tensor:
    import cb3 as CB3
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or CBF8_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or CBF8_DOWN_CFG[BM]
    # the same routing CB3 v3 uses, so the two are comparable and the decode path stays
    # graph-capturable: below 64 pairs the block table is built in torch with static shapes
    if P <= 64:
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=x.device)
    u512, u256 = CB3.block_plan(DIM)
    d512, d256 = CB3.block_plan(INTER)
    _cbf8_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1,
        arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, NB512=u512, NB256=u256,
        num_warps=nw1, num_stages=ns1)
    _cbf8_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        NB512=d512, NB256=d256, num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
