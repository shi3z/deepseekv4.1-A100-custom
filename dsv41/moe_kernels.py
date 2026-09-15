"""Grouped FP4 expert GEMM for A100: one launch computes, for a list of (input row, expert, output row,
weight) pairs, out[row_out] (+)= weight * in[row_in] @ W[expert]^T with W stored packed E2M1 + E8M0.

Pairs are sorted by expert on the host and cut into tiles of BLOCK_M rows; the grid is
(tiles, N blocks, K splits). Weights are decoded to bf16 bit patterns with the scale folded into
the exponent, then fed to mma (tensor cores) with the bf16 activations."""
import os

import torch
import triton
import triton.language as tl

DETERMINISTIC = os.environ.get("DSV41_DETERMINISTIC", "0") == "1"  # no split-K: fixed summation order


@triton.jit
def _decode_e2m1_bf16(code, sb):
    """E2M1 code (int32 0..15) + E8M0 scale byte -> bf16 (bit construction, exact)."""
    sign = (code >> 3) & 1
    e = (code >> 1) & 3
    m = code & 1
    nz = ((code & 7) != 0) & (sb != 0)
    mant = tl.where(e == 0, 0, m << 6)
    bits = (sign << 15) | ((e + sb - 1) << 7) | mant
    return tl.where(nz, bits, 0).to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _grouped_fp4_kernel(
    A, W, S, OUT, ROW_IN, ROW_OUT, WEIGHT, TILE_EXPERT, TILE_START, N_PAIRS,
    N, K,
    stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, ATOMIC: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert = tl.load(TILE_EXPERT + tile)
    start = tl.load(TILE_START + tile)
    pr = start + tl.arange(0, BLOCK_M)  # pair indices handled by this tile
    p_mask = pr < N_PAIRS
    # a pair beyond this expert's range belongs to the next expert: mask it out
    same = tl.load(TILE_EXPERT + tile + 0 * pr, mask=p_mask, other=-1) == expert  # placeholder, refined below
    rin = tl.load(ROW_IN + pr, mask=p_mask, other=0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rkh = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, BLOCK_K // 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kps = K // SPLIT_K
    w_base = W + expert * stride_we
    s_base = S + expert * stride_se
    for k0 in range(pid_k * kps, (pid_k + 1) * kps, BLOCK_K):
        a = tl.load(A + rin[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :], mask=p_mask[:, None], other=0.0)
        packed = tl.load(w_base + rn[:, None] * stride_wn + (k0 // 2 + rkh)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.load(s_base + rn[:, None] * stride_sn + (k0 // 32 + rg)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.reshape(tl.broadcast_to(tl.reshape(sb, (BLOCK_N, BLOCK_K // 32, 1)), (BLOCK_N, BLOCK_K // 32, 16)), (BLOCK_N, BLOCK_K // 2))
        w_lo = _decode_e2m1_bf16(packed & 0x0F, sb)
        w_hi = _decode_e2m1_bf16(packed >> 4, sb)
        w = tl.interleave(w_lo, w_hi)  # [BLOCK_N, BLOCK_K] bf16 in k order
        acc = tl.dot(a, tl.trans(w), acc)
    rout = tl.load(ROW_OUT + pr, mask=p_mask, other=0)
    wt = tl.load(WEIGHT + pr, mask=p_mask, other=0.0)
    acc = acc * wt[:, None]
    o_ptrs = OUT + rout[:, None] * stride_om + rn[None, :]
    o_mask = p_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(o_ptrs, acc, mask=o_mask)
    else:
        tl.store(o_ptrs, acc.to(OUT.dtype.element_ty), mask=o_mask)


_TILE_CACHE: dict = {}


class GroupedPairs:
    """Host-side bookkeeping for one MoE dispatch: pairs sorted by expert + tile table."""

    def __init__(self, expert_ids: torch.Tensor, row_in: torch.Tensor, row_out: torch.Tensor, weight: torch.Tensor, block_m: int):
        self.n_pairs = int(expert_ids.numel())
        self.block_m = block_m
        if self.n_pairs <= 64:
            # decode: one tile per pair, no sorting and no host<->device sync
            self.expert = expert_ids.to(torch.int32).contiguous()
            self.row_in = row_in.to(torch.int32).contiguous()
            self.row_out = row_out.to(torch.int32).contiguous()
            self.weight = weight.to(torch.float32).contiguous()
            self.tile_expert = self.expert
            key = (self.n_pairs, expert_ids.device)
            t = _TILE_CACHE.get(key)
            if t is None:
                t = _TILE_CACHE[key] = (torch.arange(self.n_pairs, dtype=torch.int32, device=expert_ids.device),
                                       torch.ones(self.n_pairs, dtype=torch.int32, device=expert_ids.device))
            self.tile_start, self.tile_count = t
            return
        order = torch.argsort(expert_ids, stable=True)
        self.expert = expert_ids[order].to(torch.int32).contiguous()
        self.row_in = row_in[order].to(torch.int32).contiguous()
        self.row_out = row_out[order].to(torch.int32).contiguous()
        self.weight = weight[order].to(torch.float32).contiguous()
        # tiles: for each expert, ceil(count / block_m) tiles of consecutive pairs
        counts = torch.bincount(self.expert.long(), minlength=int(expert_ids.max().item()) + 1 if self.n_pairs else 1)
        starts = torch.cumsum(counts, 0) - counts
        te, ts = [], []
        counts_l, starts_l = counts.tolist(), starts.tolist()
        for e, (c, s0) in enumerate(zip(counts_l, starts_l)):
            for t in range(0, c, block_m):
                te.append(e)
                ts.append(s0 + t)
        dev = expert_ids.device
        self.tile_expert = torch.tensor(te, dtype=torch.int32, device=dev)
        self.tile_start = torch.tensor(ts, dtype=torch.int32, device=dev)
        # rows past an expert's last pair must not leak into the next expert: build a per-pair mask
        # by clamping each tile's rows to its expert's range (done in kernel via p_mask + count)
        self.tile_count = torch.tensor([min(block_m, c - t) for c, s0 in zip(counts_l, starts_l) for t in range(0, c, block_m)],
                                       dtype=torch.int32, device=dev)


def grouped_fp4_gemm(a: torch.Tensor, w: torch.Tensor, s: torch.Tensor, pairs: GroupedPairs, n_out_rows: int,
                     out: torch.Tensor | None = None, block_m: int | None = None) -> torch.Tensor:
    """a: bf16 [rows_in, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]. Returns fp32 [n_out_rows, N]
    with out[row_out] += weight * a[row_in] @ W[expert]^T accumulated over pairs (atomic)."""
    K = a.shape[1]
    E, N = w.shape[0], w.shape[1]

    # Triton pointer arguments must all be CUDA tensors on the device
    # on which this GEMM is launched.  Some paths (notably DSpark/MTP)
    # can construct GroupedPairs bookkeeping on CPU even though A/W/S
    # live on CUDA.
    #
    # Normalize the complete pair table here rather than relying on
    # every caller to construct it on the correct device.
    dev = a.device

    _pair_fields = (
        ("row_in", torch.int32),
        ("row_out", torch.int32),
        ("weight", torch.float32),
        ("tile_expert", torch.int32),
        ("tile_start", torch.int32),
        ("tile_count", torch.int32),
    )

    moved = []

    for _name, _dtype in _pair_fields:
        _t = getattr(pairs, _name)

        if _t.device != dev or _t.dtype != _dtype or not _t.is_contiguous():
            moved.append(
                f"{_name}:{_t.device}->{dev}"
            )
            setattr(
                pairs,
                _name,
                _t.to(
                    device=dev,
                    dtype=_dtype,
                    non_blocking=True,
                ).contiguous(),
            )

    if moved:
        print(
            "[grouped-fp4] normalized pair tensors: "
            + ", ".join(moved),
            flush=True,
        )

    # Give a useful error here instead of an opaque Triton pointer error.
    if w.device != dev or s.device != dev:
        raise RuntimeError(
            "grouped_fp4_gemm device mismatch: "
            f"a={a.device} w={w.device} s={s.device}"
        )

    # Triton pointer arguments must all be CUDA tensors on the device
    # on which this GEMM is launched.  Some paths (notably DSpark/MTP)
    # can construct GroupedPairs bookkeeping on CPU even though A/W/S
    # live on CUDA.
    #
    # Normalize the complete pair table here rather than relying on
    # every caller to construct it on the correct device.
    dev = a.device

    _pair_fields = (
        ("row_in", torch.int32),
        ("row_out", torch.int32),
        ("weight", torch.float32),
        ("tile_expert", torch.int32),
        ("tile_start", torch.int32),
        ("tile_count", torch.int32),
    )

    moved = []

    for _name, _dtype in _pair_fields:
        _t = getattr(pairs, _name)

        if _t.device != dev or _t.dtype != _dtype or not _t.is_contiguous():
            moved.append(
                f"{_name}:{_t.device}->{dev}"
            )
            setattr(
                pairs,
                _name,
                _t.to(
                    device=dev,
                    dtype=_dtype,
                    non_blocking=True,
                ).contiguous(),
            )

    if moved:
        print(
            "[grouped-fp4] normalized pair tensors: "
            + ", ".join(moved),
            flush=True,
        )

    # Give a useful error here instead of an opaque Triton pointer error.
    if w.device != dev or s.device != dev:
        raise RuntimeError(
            "grouped_fp4_gemm device mismatch: "
            f"a={a.device} w={w.device} s={s.device}"
        )
    assert s.shape == (E, N, K // 32)
    if out is None:
        # Long prefill can request an enormous fp32 [n_out_rows, N]
        # temporary.  Keep the normal behavior for small outputs, but
        # refuse one giant allocation for large prefill workloads.
        #
        # Caller-visible shape is unchanged.  The tensor is still full
        # sized here, so this guard only helps when an existing output
        # buffer is supplied.  Large callers should use
        # grouped_fp4_gemm_chunked() below.
        out = torch.zeros(
            n_out_rows,
            N,
            device=a.device,
            dtype=torch.float32,
        )
    bm = pairs.block_m
    tiles = pairs.tile_expert.numel()
    if tiles == 0:
        return out
    bn = 64 if bm <= 16 else 128
    bk = 128 if bm <= 16 else 64
    split = 1
    while not DETERMINISTIC and tiles * triton.cdiv(N, bn) * split < 432 and split < 8 and (K // (split * 2)) % bk == 0:
        split *= 2
    print(
        "[GEMM-DEV] "
        f"a={a.device} "
        f"w={w.device} "
        f"s={s.device} "
        f"out={out.device} "
        f"row_in={pairs.row_in.device} "
        f"row_out={pairs.row_out.device} "
        f"weight={pairs.weight.device} "
        f"tile_expert={pairs.tile_expert.device} "
        f"tile_start={pairs.tile_start.device} "
        f"tile_count={pairs.tile_count.device}",
        flush=True,
    )

    with torch.cuda.device(a.device):  # Triton launches on the current device; our layers live on many
        _grouped_fp4_kernel_masked[(tiles, triton.cdiv(N, bn), split)](
            a, w, s, out, pairs.row_in, pairs.row_out, pairs.weight, pairs.tile_expert, pairs.tile_start, pairs.tile_count,
            N, K, a.stride(0), w.stride(0), w.stride(1), s.stride(0), s.stride(1), out.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, num_warps=4, num_stages=3,
        )
    return out



def grouped_fp4_gemm_chunked(
    a: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    pairs: GroupedPairs,
    n_out_rows: int,
    *,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Memory-bounded grouped FP4 GEMM.

    Splits the output-row dimension into chunks so long-prefill workloads
    do not allocate one multi-GiB fp32 [n_out_rows, N] tensor at once.

    This helper is intended for pair outputs where row_out is dense in
    [0, n_out_rows), which is the normal MoE prefill path.
    """
    if chunk_rows is None:
        chunk_rows = int(
            os.environ.get("DSV41_MOE_OUT_CHUNK_ROWS", "4096")
        )

    if n_out_rows <= chunk_rows:
        return grouped_fp4_gemm(
            a,
            w,
            s,
            pairs,
            n_out_rows,
        )

    N = w.shape[1]
    outputs = []

    # Work from the already sorted pair representation.
    row_out = pairs.row_out
    row_in = pairs.row_in
    expert = pairs.expert
    weight = pairs.weight

    for r0 in range(0, n_out_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_out_rows)

        mask = (row_out >= r0) & (row_out < r1)

        if not mask.any():
            outputs.append(
                torch.zeros(
                    r1 - r0,
                    N,
                    device=a.device,
                    dtype=torch.float32,
                )
            )
            continue

        # Triton requires every pointer argument to live on the same
        # CUDA device as the activation.  Some dispatch bookkeeping can
        # originate on CPU, so normalize explicitly here.
        dev = a.device

        e = expert[mask].to(
            device=dev,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()

        ri = row_in[mask].to(
            device=dev,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()

        ro = (row_out[mask] - r0).to(
            device=dev,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()

        wt = weight[mask].to(
            device=dev,
            dtype=torch.float32,
            non_blocking=True,
        ).contiguous()

        cp = GroupedPairs(
            e,
            ri,
            ro,
            wt,
            pairs.block_m,
        )

        out_c = grouped_fp4_gemm(
            a,
            w,
            s,
            cp,
            r1 - r0,
        )

        outputs.append(out_c)

    return torch.cat(outputs, dim=0)


@triton.jit(do_not_specialize=["N", "K"])
def _grouped_fp4_kernel_masked(
    A, W, S, OUT, ROW_IN, ROW_OUT, WEIGHT, TILE_EXPERT, TILE_START, TILE_COUNT,
    N, K,
    stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert = tl.load(TILE_EXPERT + tile)
    start = tl.load(TILE_START + tile)
    count = tl.load(TILE_COUNT + tile)
    lm = tl.arange(0, BLOCK_M)
    p_mask = lm < count
    pr = start + lm
    rin = tl.load(ROW_IN + pr, mask=p_mask, other=0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rkh = tl.arange(0, BLOCK_K // 2)
    rg = tl.arange(0, BLOCK_K // 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kps = K // SPLIT_K
    w_base = W + expert.to(tl.int64) * stride_we
    s_base = S + expert.to(tl.int64) * stride_se
    for k0 in range(pid_k * kps, (pid_k + 1) * kps, BLOCK_K):
        a = tl.load(A + rin[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :], mask=p_mask[:, None], other=0.0)
        packed = tl.load(w_base + rn[:, None] * stride_wn + (k0 // 2 + rkh)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.load(s_base + rn[:, None] * stride_sn + (k0 // 32 + rg)[None, :], mask=n_mask[:, None], other=0).to(tl.int32)
        sb = tl.reshape(tl.broadcast_to(tl.reshape(sb, (BLOCK_N, BLOCK_K // 32, 1)), (BLOCK_N, BLOCK_K // 32, 16)), (BLOCK_N, BLOCK_K // 2))
        w_lo = _decode_e2m1_bf16(packed & 0x0F, sb)
        w_hi = _decode_e2m1_bf16(packed >> 4, sb)
        w = tl.interleave(w_lo, w_hi)
        acc = tl.dot(a, tl.trans(w), acc)
    rout = tl.load(ROW_OUT + pr, mask=p_mask, other=0)
    wt = tl.load(WEIGHT + pr, mask=p_mask, other=0.0)
    acc = acc * wt[:, None]
    o_ptrs = OUT + rout[:, None].to(tl.int64) * stride_om + rn[None, :]
    tl.atomic_add(o_ptrs, acc, mask=p_mask[:, None] & n_mask[None, :])
