"""Grouped MoE kernel for VQ6 (a100-vq/vq6.py): dim-2 vector quantisation at 3 bit/weight.

The whole point is the decode, and it is four instructions and two tiny loads per byte lane:

    lo, hb = load(lo_ptr + j), load(hi_ptr + j)    16 distinct bytes each, one load per tile
    idx    = (lo >> SH) & 15 | ((hb >> HB) & 3) << 4       SH, HB constant per scale group
    we, wo = load(te + idx), load(to + idx)        128 B tables, fp16 straight into tl.dot

One lane is one packed-FP4 byte is one codebook entry, so nothing is expanded, nothing is
interleaved, and the tables are 128 B each -- a warp's gather stays inside one or two L1 sectors,
which is the property VQ12's 16 kB table does not have and the reason it is 56 % slower per up
kernel than CB3 v3. No E2M1 table either: the entries are fp16 already.

Slot geometry, record stride, store and engine patch are CB3's, unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from cb3_moe import _split8, CB3Arena
from fp4_moe import DIM, INTER, _pick_bm, _ue8m0, build_routing


@triton.jit
def _grp_dot(x_base, xk, mask_m, lo, hb, te_ptr, to_ptr, scale_u8, BN: tl.constexpr,
             SH: tl.constexpr, HB: tl.constexpr):
    """One scale group out of already-loaded lo/hi tiles: two constant shifts and two tiny loads."""
    idx = ((lo >> SH) & 0xF) | (((hb >> HB) & 3) << 4)
    we = tl.load(te_ptr + idx)
    wo = tl.load(to_ptr + idx)
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _vq6_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, te_ptr, to_ptr, BN: tl.constexpr):
    """256 logical K as two 128-weight packing blocks: six wide loads, eight groups, and no gather
    into anything bigger than 128 B. Lane m of every group is byte m of its tile, which is what
    makes the shifts below constants (cf. cb3_moe._cb3v3_block_dot)."""
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    j = tl.arange(0, 16)[None, :]
    l0 = tl.load(lo_ptr + j).to(tl.int32) & 0xFF
    l1 = tl.load(lo_ptr + 16 + j).to(tl.int32) & 0xFF
    h0 = tl.load(hi_ptr + j).to(tl.int32) & 0xFF
    acc = _grp_dot(x_base, xk, mask_m, l0, h0, te_ptr, to_ptr, s0, BN, 0, 0)
    acc += _grp_dot(x_base + 32, xk, mask_m, l0, h0, te_ptr, to_ptr, s1, BN, 4, 2)
    acc += _grp_dot(x_base + 64, xk, mask_m, l1, h0, te_ptr, to_ptr, s2, BN, 0, 4)
    acc += _grp_dot(x_base + 96, xk, mask_m, l1, h0, te_ptr, to_ptr, s3, BN, 4, 6)
    l2 = tl.load(lo_ptr + 32 + j).to(tl.int32) & 0xFF
    l3 = tl.load(lo_ptr + 48 + j).to(tl.int32) & 0xFF
    h1 = tl.load(hi_ptr + 16 + j).to(tl.int32) & 0xFF
    acc += _grp_dot(x_base + 128, xk, mask_m, l2, h1, te_ptr, to_ptr, s4, BN, 0, 0)
    acc += _grp_dot(x_base + 160, xk, mask_m, l2, h1, te_ptr, to_ptr, s5, BN, 4, 2)
    acc += _grp_dot(x_base + 192, xk, mask_m, l3, h1, te_ptr, to_ptr, s6, BN, 0, 4)
    acc += _grp_dot(x_base + 224, xk, mask_m, l3, h1, te_ptr, to_ptr, s7, BN, 4, 6)
    return acc


@triton.jit
def _vq6_up_kernel(
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
        acc_g += _vq6_block_dot(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32,
                                s1t + b * 8, te_ptr, to_ptr, BN)
        acc_u += _vq6_block_dot(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32,
                                s3t + b * 8, te_ptr, to_ptr, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _vq6_down_kernel(
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
        acc += _vq6_block_dot(h_base + b * 256, xk, mask_m[:, None], lo2 + b * 64, hi2 + b * 32,
                              s2t + b * 8, te_ptr, to_ptr, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


_VQ6_UP_CFG = {16: (16, 1, 3), 32: (16, 1, 3), 64: (16, 1, 3)}
_VQ6_DOWN_CFG = {16: (128, 4, 2), 32: (128, 4, 2), 64: (128, 4, 2)}


class VQ6Arena(CB3Arena):
    """CB3's tensors exactly; the `*_cb` planes go unused because the codebook is global."""

    def __init__(self, slots: int, device="cuda"):
        super().__init__(slots, device)
        self.vq = None
        self.te = None      # fp16 [64], the entry's even-k weight
        self.to = None      # fp16 [64], its odd-k weight

    def attach(self, vq):
        self.vq = vq
        self.te = vq.te.to(self.device).contiguous()
        self.to = vq.to.to(self.device).contiguous()
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


def moe_forward_vq6(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: VQ6Arena,
                    swiglu_limit: float = 10.0, block_m: int | None = None,
                    cfg_up=None, cfg_down=None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    assert arena.te is not None, "VQ6Arena.attach(vq) was never called"
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or _VQ6_UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or _VQ6_DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=x.device)
    _vq6_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.s1, arena.w3_lo, arena.w3_hi, arena.s3, h,
        arena.te, arena.to, wgt, block_slot, block_pair, x.stride(0), h.stride(0),
        float(swiglu_limit), TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _vq6_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.s2, parts, arena.te, arena.to, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
