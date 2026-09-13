"""Where VQ12's up kernel spends its load instructions, by deleting one stage at a time.

The variants produce wrong numbers on purpose: each keeps every other instruction and only drops
one stage's loads, so the time difference is that stage's budget.

  MODE 0  full decode                        lo 16 + hi 16 + LUT 16 + E2M1 32  = 80 loads / 32 K
  MODE 1  E2M1 table -> a constant tile      lo 16 + hi 16 + LUT 16            = 48
  MODE 2  LUT -> the lo byte itself          lo 16 + hi 16 +          E2M1 32  = 64
  MODE 3  lo/hi -> constant index 0          LUT 16 + E2M1 32                  = 48
  MODE 4  no decode at all, dots only                                          = 0
"""
import os, sys, time
import torch
import triton
import triton.language as tl
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, vq12_moe as VQM
from fp4_moe import DIM, INTER, _pick_bm, _ue8m0, build_routing
from cb3_moe import _split8
from engine.codebook_sim import CodebookSim
from vq12 import VQ12


@triton.jit
def _grp(lo_ptr, hi_ptr, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN: tl.constexpr, MODE: tl.constexpr):
    j = tl.arange(0, 16)[None, :]
    g = j // 2
    if MODE == 3:
        lo = tl.zeros([BN, 16], dtype=tl.int32) + 7
        hn = tl.zeros([BN, 16], dtype=tl.int32) + 1
    else:
        lo = tl.load(lo_ptr + g).to(tl.int32) & 0xFF
        hb = tl.load(hi_ptr + (g // 2)).to(tl.int32) & 0xFF
        hn = tl.where(g % 2 == 0, hb & 0xF, (hb >> 4) & 0xF)
    if MODE == 2 or MODE == 8:
        p8 = lo | (hn << 8)
    elif MODE == 9:
        # the same 4096 entries, but as two uint8 planes: a lane reads only the byte it needs, so
        # each warp's gather spans 4 kB instead of 16
        p8 = tl.load(lut8_ptr + (j & 1) * 4096 + (lo | (hn << 8))).to(tl.int32) & 0xFF
    elif MODE == 10:
        v = tl.load(lut16_ptr + (lo | (hn << 8))).to(tl.int32) & 0xFFFF   # uint16: 8 kB
        p8 = tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF)
    elif MODE == 5:
        v = tl.load(lut_ptr + ((lo | (hn << 8)) & 63))          # a 64-entry table, same gathers
        p8 = tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF)
    elif MODE == 6:
        v = tl.load(lut_ptr + ((lo | (hn << 8)) & 255))         # 256 entries
        p8 = tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF)
    elif MODE == 7:
        v = tl.load(lut_ptr + ((lo | (hn << 8)) & 1023))        # 1024 entries
        p8 = tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF)
    else:
        v = tl.load(lut_ptr + (lo | (hn << 8)))
        p8 = tl.where(j % 2 == 0, v & 0xFF, (v >> 8) & 0xFF)
    return p8.to(tl.int32) & 0xFF


@triton.jit
def _cdot(x_base, xk, mask_m, p8, scale_u8, tab_ptr, BN: tl.constexpr, MODE: tl.constexpr):
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    if MODE == 8:
        # a residual (two stage) codebook: two 64-entry fp16 tables whose sum is the weight, so
        # 4096 combinations come out of 512 B of table and the E2M1 step disappears
        i1 = p8 & 63
        i2 = (p8 >> 6) & 63
        we = tl.load(tab_ptr + i1) + tl.load(tab_ptr + 64 + i2)
        wo = tl.load(tab_ptr + 128 + i1) + tl.load(tab_ptr + 192 + i2)
    elif MODE == 1 or MODE == 4:
        we = (p8 & 0xF).to(tl.float16)
        wo = (p8 >> 4).to(tl.float16)
    else:
        we = tl.load(tab_ptr + (p8 & 0xF))
        wo = tl.load(tab_ptr + (p8 >> 4))
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _blk(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN: tl.constexpr,
         MODE: tl.constexpr):
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    acc = tl.zeros([16, BN], dtype=tl.float32) * 0.0
    acc = _cdot(x_base, xk, mask_m, _grp(lo_ptr, hi_ptr, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s0, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 32, xk, mask_m, _grp(lo_ptr + 8, hi_ptr + 4, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s1, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 64, xk, mask_m, _grp(lo_ptr + 16, hi_ptr + 8, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s2, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 96, xk, mask_m, _grp(lo_ptr + 24, hi_ptr + 12, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s3, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 128, xk, mask_m, _grp(lo_ptr + 32, hi_ptr + 16, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s4, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 160, xk, mask_m, _grp(lo_ptr + 40, hi_ptr + 20, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s5, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 192, xk, mask_m, _grp(lo_ptr + 48, hi_ptr + 24, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s6, tab_ptr, BN, MODE)
    acc += _cdot(x_base + 224, xk, mask_m, _grp(lo_ptr + 56, hi_ptr + 28, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE), s7, tab_ptr, BN, MODE)
    return acc


@triton.jit
def _up(x_ptr, lo1_ptr, hi1_ptr, s1_ptr, lo3_ptr, hi3_ptr, s3_ptr, h_ptr, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr,
        wgt_ptr, block_slot_ptr, block_pair_ptr, stride_x, stride_h, limit,
        TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        MODE: tl.constexpr):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0); nb = tl.program_id(1)
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
        acc_g += _blk(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32, s1t + b * 8, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE)
        acc_u += _blk(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32, s3t + b * 8, lut_ptr, lut8_ptr, lut16_ptr, tab_ptr, BN, MODE)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


dev = "cuda"; torch.manual_seed(0)
S = 16; T = 6; K = 6
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
ar = VQM.VQ12Arena(S, dev).attach(vq)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    ar.load_slot(i, *args); cb.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev).float().contiguous()
BM = _pick_bm(T * K)
bn, nw, ns = VQM._VQ_UP_CFG[BM]
bs, bp, NB = build_routing(slots, ar.slots, BM)
lut16 = ar.lut.to(torch.int32).cpu().numpy()
import numpy as _np
lut8 = torch.from_numpy(_np.concatenate([(lut16 & 0xFF).astype(_np.uint8), ((lut16 >> 8) & 0xFF).astype(_np.uint8)])).to(dev)
lut16 = torch.from_numpy(lut16.astype(_np.uint16)).to(dev)
tab4 = torch.zeros(256, dtype=torch.float16, device=dev)
tab4[:16] = ar.tab; tab4[64:80] = ar.tab; tab4[128:144] = ar.tab; tab4[192:208] = ar.tab
h = torch.empty((T * K, INTER), dtype=torch.bfloat16, device=dev)

def run(mode):
    _up[(NB, INTER // bn)](x, ar.w1_lo, ar.w1_hi, ar.s1, ar.w3_lo, ar.w3_hi, ar.s3, h, ar.lut, lut8, lut16, tab4,
                           wts, bs, bp, x.stride(0), h.stride(0), 10.0, TOPK=K, N=INTER, K=DIM,
                           BM=BM, BN=bn, MODE=mode, num_warps=nw, num_stages=ns)

def bench(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

names = {9: "4096 entries as two uint8 planes", 10: "4096 entries as uint16 (8 kB)", 8: "residual, two 64-entry fp16", 0: "full, 4096-entry LUT", 1: "no E2M1 table (48)", 2: "no LUT gather (64)",
         3: "no lo/hi loads (48)", 4: "dots only (0)", 5: "LUT 64 entries (256 B)",
         6: "LUT 256 entries (1 kB)", 7: "LUT 1024 entries (4 kB)"}
base = bench(lambda: run(0))
print(f"up kernel, BM={BM} BN={bn} warps={nw} stages={ns}")
for m in (0, 10, 9, 8, 2):
    t = bench(lambda: run(m))
    print(f"  MODE {m} {names[m]:24s} {t:.3f} ms   {(t-base)/base*100:+6.1f} %")
