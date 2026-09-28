"""Our CUDA C kernels, compiled with nvcc to cubin and launched through the driver API (ctypes), so
they work with any torch build and inside CUDA-graph capture."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("DSV41_NVCC", "/usr/local/cuda-12.8/bin/nvcc")
_cuda = ctypes.CDLL("libcuda.so.1")
_modules: dict[tuple[str, int], ctypes.c_void_p] = {}
_funcs: dict[tuple[str, int], ctypes.c_void_p] = {}


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with CUDA error {err}")


def _cubin(src_name: str) -> bytes:
    src = os.path.join(HERE, "cuda", src_name)
    code = open(src, "rb").read()
    tag = hashlib.sha1(code).hexdigest()[:12]
    out = os.path.join(HERE, "cuda", f".{src_name}.{tag}.sm80.cubin")
    if not os.path.exists(out):
        subprocess.run([NVCC, "-cubin", "-arch=sm_80", "-O3", "-o", out, src], check=True)
    return open(out, "rb").read()


def get_function(src_name: str, func: str, device: torch.device) -> ctypes.c_void_p:
    key = (src_name, device.index)
    if key not in _modules:
        with torch.cuda.device(device):
            torch.cuda.current_stream()  # make sure the context exists
            image = _cubin(src_name)
            mod = ctypes.c_void_p()
            _check(_cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
            _modules[key] = mod
    fkey = (src_name + ":" + func, device.index)
    if fkey not in _funcs:
        f = ctypes.c_void_p()
        _check(_cuda.cuModuleGetFunction(ctypes.byref(f), _modules[key], func.encode()), "cuModuleGetFunction")
        _funcs[fkey] = f
    return _funcs[fkey]


def launch(func, grid, block, args, device: torch.device, shared: int = 0):
    """args: list of ctypes values (c_void_p for pointers)."""
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    with torch.cuda.device(device):  # the module handle belongs to this device's primary context
        stream = torch.cuda.current_stream(device).cuda_stream
        _check(_cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                                    ctypes.c_void_p(stream), ptrs, None), "cuLaunchKernel")


def fp4_gemv_pairs(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, row_in: torch.Tensor, expert: torch.Tensor,
                   wt: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """x: bf16 [rows_in, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; row_in/expert: int32 [pairs];
    wt: fp32 [pairs]. Returns fp32 [pairs, N] = wt * x[row_in] @ W[expert]^T (one program per row: no atomics)."""
    E, N, Kh = w.shape
    K = Kh * 2
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and K <= 5120
    out = torch.empty(n_pairs, N, device=x.device, dtype=torch.float32)
    f = get_function("fp4_gemv.cu", "fp4_gemv_pairs", x.device)
    rows_per_block = 8 * 4
    grid = ((N + rows_per_block - 1) // rows_per_block, n_pairs, 1)
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_int(w.stride(1)),
            ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)), ctypes.c_int(s.stride(1)),
            ctypes.c_void_p(row_in.data_ptr()), ctypes.c_void_p(expert.data_ptr()), ctypes.c_void_p(wt.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(out.stride(0)), ctypes.c_int(N), ctypes.c_int(K)]
    launch(f, grid, (256, 1, 1), args, x.device)
    return out


def fp8_gemv(x: torch.Tensor, w_fp8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """x: bf16 [M<=16, K]; w_fp8: float8_e4m3fn/uint8 [N, K]; s_u8: uint8 [ceil(N/32), K/32] (E8M0). fp32 [M, N]."""
    M, K = x.shape
    N = w_fp8.shape[0]
    assert M == 1 and K % 32 == 0 and K <= 8192 and x.stride(1) == 1
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    f = get_function("fp8_gemv.cu", "fp8_gemv", x.device)
    rows_per_block = 8 * 4
    grid = ((N + rows_per_block - 1) // rows_per_block, 1, 1)
    smem = 4 * (256 + 16 * M * (K // 16))
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(M),
            ctypes.c_void_p(w_fp8.data_ptr()), ctypes.c_int(w_fp8.stride(0)),
            ctypes.c_void_p(s_u8.data_ptr()), ctypes.c_int(s_u8.stride(0)),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(out.stride(0)), ctypes.c_int(N), ctypes.c_int(K)]
    launch(f, grid, (256, 1, 1), args, x.device, shared=smem)
    return out


# --------------------------------------------------------------------------- FP8-weight tensor-core GEMM (M <= 16)
def _splits_for(N: int, K: int) -> int:
    """Split-K factor so that at least ~1024 warps stream the weights (each warp owns 8 columns)."""
    warps = N // 8
    s = 1
    while warps * s < 4096 and (K // (s * 2)) >= 256:
        s *= 2
    return s


FP8_W_LAYOUT = os.environ.get("DSV41_FP8_W", "1") == "1"
FP8_G_LAYOUT = os.environ.get("DSV41_FP8_G", "1") == "1"  # tiled kernel for > 64 rows
FP8_W_SPLITS = int(os.environ.get("DSV41_FP8_W_SPLITS", "0"))
FP8_W_STAGES = int(os.environ.get("DSV41_FP8_W_STAGES", "3"))  # x stages in shared memory
FP8_W_MW = int(os.environ.get("DSV41_FP8_W_MW", "1"))  # warps along M per block (1, 2 or 4 for 64 rows; 1 or 2 for 32)
FP8_M1 = os.environ.get("DSV41_FP8_M1", "0") == "1"  # dedicated M=1 GEMV (cuda/fp8_gemv_m1.cu) for single-row calls


def fp8_gemv_m1(x: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, group_cols: int = 0, out: torch.Tensor | None = None) -> torch.Tensor:
    """One-row FP8 GEMV (results/dense_m1_report.md, Strategy A): x bf16 [1, K] (or [N/group_cols, K] block-diagonal),
    w8 / s8 as fp8_gemm_tc -> bf16 [1, N]. 8-warp blocks, KW warps per output row, U 16-byte loads in flight per lane."""
    K = x.shape[1]
    N = w8.shape[0]
    KW = 1 if N >= 4096 else 2 if N >= 1024 else 4
    while K % (16 * KW) != 0 and KW > 1:
        KW //= 2
    U = 4 if K <= 2048 else 2
    RB = 8 // KW
    if out is None:
        out = torch.empty(1, N, device=x.device, dtype=torch.bfloat16)
    f = get_function("fp8_gemv_m1.cu", f"fp8_gemv_m1_k{KW}u{U}", x.device)
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()),
            ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]), ctypes.c_int(group_cols),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0)]
    launch(f, ((N + RB - 1) // RB, 1, 1), (256, 1, 1), args, x.device)
    return out
FUSED_DENSE_CHAIN = int(os.environ.get("DSV41_FUSED_DENSE_CHAIN", "0"))  # 1: [wq_a; wkv] concatenated GEMV; 2: + fake-quant fused into wo_b; 3: + q_norm fused into wq_b (slower, record only)


def fp8_gemv_m1_pre(x_raw: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, norm_w: torch.Tensor | None, eps: float, mode: int,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """One-row FP8 GEMV with the activation prologue fused (cuda/fp8_gemv_m1.cu, results/dense_chain_report.md):
    mode 1 = rmsnorm(x_raw) * norm_w then per-32 fp8 fake quant (replaces fused2.norm_quant), mode 2 = fake quant
    only (replaces fused.fake_quant_fp8); x_raw bf16 [1, K] (K <= 2048 / 8192; a row prefix view is fine), result bf16 [1, N]."""
    K = x_raw.shape[-1]
    N = w8.shape[0]
    assert K % 32 == 0 and K <= (2048 if mode == 1 else 8192) and x_raw.stride(-1) == 1 and x_raw.data_ptr() % 16 == 0
    KW = 1 if N >= 4096 else 2 if N >= 1024 else 4
    while K % (16 * KW) != 0 and KW > 1:
        KW //= 2
    U = 4 if K <= 2048 else 2
    if (KW, U) not in ((1, 4), (1, 2), (2, 2), (4, 4)):
        KW, U = (1, 2)
    RB = 8 // KW
    if out is None:
        out = torch.empty(1, N, device=x_raw.device, dtype=torch.bfloat16)
    f = get_function("fp8_gemv_m1.cu", f"fp8_gemv_m1_pre{mode}_k{KW}u{U}", x_raw.device)
    args = [ctypes.c_void_p(x_raw.data_ptr()), ctypes.c_void_p(norm_w.data_ptr() if norm_w is not None else 0), ctypes.c_float(float(eps)),
            ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0)]
    launch(f, ((N + RB - 1) // RB, 1, 1), (256, 1, 1), args, x_raw.device, shared=K * 2)
    return out


_attr_done: set = set()  # experiment: force the split-K factor


def fp8_gemm_tc(x: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, group_cols: int = 0, out_dtype=torch.bfloat16) -> torch.Tensor:
    """x: bf16 [M, K] (contiguous); w8: uint8 (e4m3 bits) [N, K] in the w8.PERM_K byte order; s8: uint8 (E8M0) [ceil(N/32), K/32].
    Returns x @ dequant(w8, s8)^T as [M, N] (fp32 accumulation, rounded to out_dtype).
    group_cols > 0: block-diagonal use (x: [B * N/group_cols, K]; output row b, column n uses x row
    b * (N/group_cols) + n // group_cols) -> [B, N]."""
    M, K = x.shape
    N = w8.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and w8.is_contiguous() and s8.is_contiguous()
    assert K % 64 == 0 and N % 8 == 0 and (group_cols == 0 or group_cols % 8 == 0 and N % group_cols == 0)
    Mo = M // (N // group_cols) if group_cols else M
    if FP8_M1 and Mo == 1 and out_dtype == torch.bfloat16 and K % 16 == 0 and N % 8 == 0:
        return fp8_gemv_m1(x, w8, s8, group_cols)
    tiled = FP8_G_LAYOUT and Mo > 64 and N % 128 == 0 and K % 128 == 0 and (group_cols == 0 or group_cols % 128 == 0)
    if tiled:  # many rows: CUTLASS-style tiles (fp8_tcg.cu), any M
        return _fp8_gemm_tcg(x, w8, s8, Mo, N, K, group_cols, out_dtype)
    wlayout = FP8_W_LAYOUT and Mo > 16 and N % 64 == 0 and (group_cols == 0 or group_cols % 64 == 0)
    rows_per = 64 if wlayout else 16
    if Mo > rows_per:  # chunk
        per = (N // group_cols) if group_cols else 1
        return torch.cat([fp8_gemm_tc(x[i * per : (i + rows_per) * per], w8, s8, group_cols, out_dtype) for i in range(0, Mo, rows_per)], dim=0)
    WARPS = 4
    if wlayout:  # weights as the mma A operand (16 columns per warp), up to 64 rows per pass, weights read once
        MT = 2 if Mo <= 16 else 4 if Mo <= 32 else 8
        splits = 1  # ~3 blocks per SM (>= 256 blocks) with the fp32 partials kept under half the weight bytes
        while (N // 64) * splits < 256 and splits * 2 * Mo * 8 <= K and K // (splits * 2) >= 256:
            splits *= 2
        if FP8_W_SPLITS:
            splits = FP8_W_SPLITS
    else:
        splits = _splits_for(N, K)
    kps = -(-K // splits)
    kps = -(-kps // 128) * 128
    splits = -(-K // kps)
    part = torch.empty(splits, Mo, N, device=x.device, dtype=torch.float32)
    if wlayout:
        MW = FP8_W_MW if MT >= 4 else 1  # warps along M (each MT/MW tiles)
        f = get_function("fp8_tcw.cu", f"fp8_gemm_tcw{MT}s{FP8_W_STAGES}" if MW == 1 else f"fp8_gemm_tcw{MT}m{MW}s{FP8_W_STAGES}", x.device)
        grid = (N // 64, splits, 1)
        shared = FP8_W_STAGES * 8 * MT * 256
        if (f.value, x.device.index) not in _attr_done:  # dynamic shared memory above 48 KB needs the attribute
            _check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(shared)), "cuFuncSetAttribute")  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _attr_done.add((f.value, x.device.index))
    else:
        f = get_function("fp8_tc.cu", "fp8_gemm_tc8" if Mo <= 8 else "fp8_gemm_tc16", x.device)
        grid = ((N // 8 + WARPS - 1) // WARPS, splits, 1)
    fused_epilogue = out_dtype == torch.bfloat16
    y = torch.empty(Mo, N, device=x.device, dtype=torch.bfloat16) if fused_epilogue else None
    counters = _tile_counters(x.device, N // 8) if fused_epilogue else None
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(Mo),
            ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr() if y is not None else 0), ctypes.c_void_p(counters.data_ptr() if counters is not None else 0), ctypes.c_int(splits)]
    launch(f, grid, (WARPS * 32 * (MW if wlayout else 1), 1, 1), args, x.device, shared=shared if wlayout else 0)
    if fused_epilogue:
        return y
    y = part[0] if splits == 1 else part.sum(dim=0)
    return y.to(out_dtype)


def _fp8_gemm_tcg(x, w8, s8, Mo, N, K, group_cols, out_dtype):
    """fp8_tcg.cu: 256-thread blocks, tile 64 rows x 128 columns x 128 k, split-K with the fused epilogue."""
    mblocks = (Mo + 63) // 64
    splits = 1  # ~256 blocks, fp32 partials at most the weight bytes
    while (N // 128) * mblocks * splits < 256 and splits * 2 * Mo * 4 <= K and K // (splits * 2) >= 256:
        splits *= 2
    if FP8_W_SPLITS:
        splits = FP8_W_SPLITS
    kps = -(-K // splits)
    kps = -(-kps // 128) * 128
    splits = -(-K // kps)
    part = torch.empty(splits, Mo, N, device=x.device, dtype=torch.float32)
    f = get_function("fp8_tcg.cu", "fp8_gemm_tcg", x.device)
    shared = 2 * (128 * 128 + 64 * 256)
    if (f.value, x.device.index) not in _attr_done:
        _check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(shared)), "cuFuncSetAttribute")
        _attr_done.add((f.value, x.device.index))
    fused_epilogue = out_dtype == torch.bfloat16
    y = torch.empty(Mo, N, device=x.device, dtype=torch.bfloat16) if fused_epilogue else None
    counters = _tile_counters(x.device, (N // 128) * mblocks) if fused_epilogue else None
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(Mo),
            ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr() if y is not None else 0), ctypes.c_void_p(counters.data_ptr() if counters is not None else 0), ctypes.c_int(splits)]
    launch(f, (N // 128, mblocks, splits), (256, 1, 1), args, x.device, shared=shared)
    if fused_epilogue:
        return y
    y = part[0] if splits == 1 else part.sum(dim=0)
    return y.to(out_dtype)


_counters: dict = {}


def _tile_counters(device, n_tiles: int) -> torch.Tensor:
    """Zeroed per-tile counters for the split-K epilogue (self-resetting; one buffer per device, grown on demand)."""
    c = _counters.get(device)
    if c is None or c.numel() < n_tiles:
        c = _counters[device] = torch.zeros(max(n_tiles, 8192), dtype=torch.int32, device=device)
    return c


# --------------------------------------------------------------------------- FP4 expert GEMM on tensor cores (grouped by expert)
_PERM8 = [0, 4, 2, 6, 1, 5, 3, 7]


def permute_x(x: torch.Tensor) -> torch.Tensor:
    """bf16 [M, K] -> the 8-k permuted layout fp4_gemm_tc expects (within every 8 k: 0,4,2,6,1,5,3,7)."""
    M, K = x.shape
    return x.view(M, K // 8, 8)[:, :, _PERM8].reshape(M, K).contiguous()


FP4_W_LAYOUT = os.environ.get("DSV41_FP4_W", "1") == "1"
FP4_W_MAX = 64  # tokens per group the second layout handles (groups are split at this size)


FP4_M1 = os.environ.get("DSV41_FP4_M1", "0") == "1"  # one-token groups on cuda/fp4_gemv_m1.cu (results/dense_m1_report.md rank 1)
FP4_M1_KW = int(os.environ.get("DSV41_FP4_M1_KW", "1"))
FP4_M1_U = int(os.environ.get("DSV41_FP4_M1_U", "2"))
FP4_M1_MIXED = os.environ.get("DSV41_FP4_M1_MIXED", "0") == "1"  # also split bucketed (B > 1) steps between the two kernels


def fp4_gemv_m1(xp: torch.Tensor, w: torch.Tensor, s: torch.Tensor, grp_expert: torch.Tensor, grp_start: torch.Tensor,
                pair_tok: torch.Tensor, n_pairs: int, shard_start: int = 0, shard_n: int = 1 << 30,
                zero_out: bool = False, out: torch.Tensor | None = None, kw: int | None = None, u: int | None = None) -> torch.Tensor:
    """The one-token groups of a grouped expert GEMM (same arguments as fp4_gemm_tc; groups with != 1 pair are ignored)."""
    E, N, Kh = w.shape
    K = Kh * 2
    G = grp_expert.numel()
    KW, U = kw or FP4_M1_KW, u or FP4_M1_U
    while K % (32 * KW) != 0 and KW > 1:
        KW //= 2
    RB = 8 // KW
    if out is None:
        out = torch.empty(n_pairs, N, device=xp.device, dtype=torch.float32)
    f = get_function("fp4_gemv_m1.cu", f"fp4_gemv_m1_k{KW}u{U}", xp.device)
    args = [ctypes.c_void_p(xp.data_ptr()), ctypes.c_int(xp.stride(0)),
            ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)),
            ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr()), ctypes.c_void_p(pair_tok.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_int(shard_start), ctypes.c_int(min(shard_n, E)), ctypes.c_int(1 if zero_out else 0)]
    launch(f, ((N + RB - 1) // RB, G, 1), (256, 1, 1), args, xp.device)
    return out


def fp4_gemm_tc(xp: torch.Tensor, w: torch.Tensor, s: torch.Tensor, grp_expert: torch.Tensor, grp_start: torch.Tensor,
                pair_tok: torch.Tensor, n_pairs: int, max_tokens: int, shard_start: int = 0, shard_n: int = 1 << 30,
                zero_out: bool = False, out: torch.Tensor | None = None, min_tokens: int = 0) -> torch.Tensor:
    """xp: permuted bf16 [rows, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; groups g: expert grp_expert[g] with pairs
    grp_start[g]..grp_start[g+1]-1 (<= max_tokens each), pair p uses x row pair_tok[p]. Returns fp32 [n_pairs, N].
    Groups of <= 8 tokens run on fp4_tc.cu (x as the mma A operand); larger groups (up to 64) on fp4_tcw.cu, which
    reads the expert once whatever the token count (fp4_tc.cu's 16-token variant handles 9..16 when disabled).
    Expert parallelism: ids are global, this GPU holds [shard_start, shard_start + shard_n); other groups are skipped
    (zero_out: their output rows are zeroed)."""
    E, N, Kh = w.shape
    K = Kh * 2
    assert xp.dtype == torch.bfloat16 and xp.is_contiguous() and K % 128 == 0 and N % 8 == 0
    G = grp_expert.numel()
    if out is None:
        out = torch.empty(n_pairs, N, device=xp.device, dtype=torch.float32)
    WARPS = 4
    big = max_tokens > 8 and (FP4_W_LAYOUT or max_tokens > 16)
    assert max_tokens <= (FP4_W_MAX if big else 16)
    assert not big or N % 64 == 0
    common = [ctypes.c_void_p(xp.data_ptr()), ctypes.c_int(xp.stride(0)),
              ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)),
              ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr()), ctypes.c_void_p(pair_tok.data_ptr()),
              ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(K),
              ctypes.c_int(shard_start), ctypes.c_int(min(shard_n, E)), ctypes.c_int(1 if zero_out else 0)]
    small_max = 8 if (big or max_tokens <= 8) else 16
    if min_tokens <= small_max:  # (min_tokens > 0: the one-token groups were handled by fp4_gemv_m1)
        f = get_function("fp4_tc.cu", "fp4_gemm_tc8" if small_max <= 8 else "fp4_gemm_tc16", xp.device)
        launch(f, ((N // 8 + WARPS - 1) // WARPS, G, 1), (WARPS * 32, 1, 1), common + [ctypes.c_int(min_tokens), ctypes.c_int(small_max)], xp.device)
    if big:
        f = get_function("fp4_tcw.cu", "fp4_gemm_tcw", xp.device)
        shared = 3 * 64 * 256
        if (f.value, xp.device.index) not in _attr_done:
            _check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(shared)), "cuFuncSetAttribute")
            _attr_done.add((f.value, xp.device.index))
        launch(f, (N // 64, G, 1), (WARPS * 32, 1, 1), common + [ctypes.c_int(9), ctypes.c_int(FP4_W_MAX)], xp.device, shared=shared)
    return out


# --------------------------------------------------------------------------- device-side GPU messaging (expert parallelism)
def p2p_copy(dst: torch.Tensor, src: torch.Tensor, device: torch.device):
    """Copy src (contiguous, size multiple of 16 B) into dst (possibly on another GPU) with a kernel on `device`."""
    n = src.numel() * src.element_size()
    assert n % 16 == 0 and dst.numel() * dst.element_size() >= n
    f = get_function("p2p.cu", "p2p_copy", device)
    n16 = n // 16
    launch(f, ((n16 + 255) // 256, 1, 1), (256, 1, 1), [ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(n16)], device)


def p2p_copy_row(dst_base: torch.Tensor, row_idx: torch.Tensor, src: torch.Tensor, device: torch.device, seq: torch.Tensor):
    """dst_base[seq[b], row_idx[b]] = src[b] for every row b (dst_base: [S, rows, D]; src: [B, 1, D]; row_idx, seq: int64 [B])."""
    B = src.shape[0]
    n = src.numel() * src.element_size() // B
    assert n % 16 == 0 and src.shape[-1] == dst_base.shape[-1]
    f = get_function("p2p.cu", "p2p_copy_row", device)
    row16 = n // 16
    bstride16 = dst_base.stride(0) * dst_base.element_size() // 16
    launch(f, ((row16 * B + 255) // 256, 1, 1), (256, 1, 1), [ctypes.c_void_p(dst_base.data_ptr()), ctypes.c_void_p(row_idx.data_ptr()), ctypes.c_void_p(src.data_ptr()),
                                                            ctypes.c_int(row16), ctypes.c_int(B), ctypes.c_longlong(bstride16), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_sum_rows(dst: torch.Tensor, src: torch.Tensor, device: torch.device, groups: int = 1, dst_stride: int | None = None):
    """dst[g] = sum of the `rows` rows of group g of src [groups * rows, n] (dst rows `dst_stride` elements apart)."""
    total, n = src.shape
    rows = total // groups
    if dst_stride is None:
        dst_stride = dst.stride(0) if dst.dim() > 1 else n
    f = get_function("p2p.cu", "p2p_sum_rows", device)
    launch(f, ((n + 255) // 256, groups, 1), (256, 1, 1), [ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(rows), ctypes.c_int(n),
                                                          ctypes.c_int(groups), ctypes.c_longlong(dst_stride)], device)


def p2p_signal(flag_ptrs: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Set the flags at the addresses in flag_ptrs (int64 device tensor on `device`) to the value of seq (int32 device scalar)."""
    f = get_function("p2p.cu", "p2p_signal", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flag_ptrs.data_ptr()), ctypes.c_int(flag_ptrs.numel()), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_wait(flags: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Spin (one thread) until all flags (int32 [n] on `device`) >= seq."""
    f = get_function("p2p.cu", "p2p_wait", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flags.data_ptr()), ctypes.c_int(flags.numel()), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_wait_masked(flags: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor, device: torch.device):
    """Spin until every flag whose mask entry (int32 [n] on `device`, device-computed) is non-zero has reached seq."""
    f = get_function("p2p.cu", "p2p_wait_masked", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flags.data_ptr()), ctypes.c_int(flags.numel()), ctypes.c_void_p(seq.data_ptr()),
                                    ctypes.c_void_p(mask.data_ptr())], device)


def p2p_seq_bump(seq: torch.Tensor, device: torch.device):
    f = get_function("p2p.cu", "p2p_seq_bump", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(seq.data_ptr())], device)


def p2p_multicast(dst_ptrs: torch.Tensor, src: torch.Tensor, flag_ptrs: torch.Tensor | None, seq: torch.Tensor, device: torch.device, counter: torch.Tensor | None = None):
    """One kernel: src (size multiple of 16) into every destination address in dst_ptrs (int64 on `device`), then the last
    block sets the flags at flag_ptrs to seq (counter: an int32 device scalar, zero at first use, self-resetting)."""
    n = src.numel() * src.element_size()
    assert n % 16 == 0
    f = get_function("p2p.cu", "p2p_multicast", device)
    n16 = n // 16
    blocks = (n16 + 1023) // 1024
    if flag_ptrs is not None and counter is None:
        counter = _tile_counters(device, 8192)[-1:]
    launch(f, (blocks, 1, 1), (1024, 1, 1), [ctypes.c_void_p(dst_ptrs.data_ptr()), ctypes.c_int(dst_ptrs.numel()), ctypes.c_void_p(src.data_ptr()), ctypes.c_int(n16),
                                            ctypes.c_void_p(flag_ptrs.data_ptr() if flag_ptrs is not None else 0), ctypes.c_void_p(seq.data_ptr()),
                                            ctypes.c_void_p(counter.data_ptr() if counter is not None else 0)], device)


def p2p_stamp(dst: torch.Tensor, device: torch.device):
    """Write the GPU global timer (ns) into dst (int64 scalar on `device`), stream-ordered."""
    f = get_function("p2p.cu", "p2p_stamp", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(dst.data_ptr())], device)


_cuda.cuMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def memcpy_async(dst: torch.Tensor, src: torch.Tensor, device: torch.device, nbytes: int | None = None):
    """cuMemcpyAsync(dst, src) on `device`'s current stream (unified addressing: dst may live on a peer GPU; the copy
    engines do the transfer, which matters across sockets where kernel-initiated P2P stores crawl)."""
    n = src.numel() * src.element_size() if nbytes is None else nbytes
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _check(_cuda.cuMemcpyAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), n, ctypes.c_void_p(stream)), "cuMemcpyAsync")


def hit_mask_select(eid: torch.Tensor, eid_rem: torch.Tensor, spec_gu: torch.Tensor, gu: torch.Tensor,
                    e_pred: int, hit_flag: torch.Tensor | None = None):
    """GPU-side Hit/Miss resolution for speculative expert execution (no host roundtrips)."""
    topk_e = eid.numel()
    N = spec_gu.shape[-1]
    dev = eid.device
    fn = get_function("hit_select.cu", "hit_mask_select", dev)
    flag_ptr = ctypes.c_void_p(hit_flag.data_ptr()) if hit_flag is not None else ctypes.c_void_p(0)
    args = [
        ctypes.c_void_p(eid.data_ptr()),
        ctypes.c_void_p(eid_rem.data_ptr()),
        ctypes.c_void_p(spec_gu.data_ptr()),
        ctypes.c_void_p(gu.data_ptr()),
        ctypes.c_int(e_pred),
        flag_ptr,
        ctypes.c_int(topk_e),
        ctypes.c_int(N),
    ]
    launch(fn, (1, 1, 1), (256, 1, 1), args, dev)



# --------------------------------------------------------------------------- persistent fused expert chain (S = 1)
PERSISTENT_EXPERT = int(os.environ.get("DSV41_PERSISTENT_EXPERT", "0"))  # 1: two-launch fused chain, 2: single persistent launch


class _ChainArgs(ctypes.Structure):
    _fields_ = [("xqp", ctypes.c_void_p), ("xq", ctypes.c_void_p), ("eid", ctypes.c_void_p), ("wt", ctypes.c_void_p),
                ("topk", ctypes.c_int), ("shard_start", ctypes.c_int), ("shard_n", ctypes.c_int),
                ("w13", ctypes.c_void_p), ("w13_stride", ctypes.c_longlong), ("s13", ctypes.c_void_p), ("s13_stride", ctypes.c_longlong),
                ("w2", ctypes.c_void_p), ("w2_stride", ctypes.c_longlong), ("s2", ctypes.c_void_p), ("s2_stride", ctypes.c_longlong),
                ("sh_w13", ctypes.c_void_p), ("sh_s13", ctypes.c_void_p), ("sh_s13_cols", ctypes.c_int),
                ("sh_w2", ctypes.c_void_p), ("sh_s2", ctypes.c_void_p), ("sh_s2_cols", ctypes.c_int),
                ("has_shared", ctypes.c_int), ("limit", ctypes.c_float),
                ("gu", ctypes.c_void_p), ("part", ctypes.c_void_p), ("ys", ctypes.c_void_p), ("ctrs", ctypes.c_void_p),
                ("dim", ctypes.c_int), ("inter", ctypes.c_int)]


class _ChainArgs2(ctypes.Structure):
    _fields_ = [("a", _ChainArgs), ("h", ctypes.c_void_p), ("done", ctypes.c_void_p)]


_chain_state: dict = {}  # per device: scratch buffers and the resident grid size


def _chain_state_for(device: torch.device, inter: int):
    st = _chain_state.get(device.index)
    if st is None:
        f = get_function("expert_chain.cu", "expert_chain_5120_2304", device)
        nb = ctypes.c_int()
        _check(_cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(nb), f, ctypes.c_int(256), ctypes.c_size_t(0)),
               "cuOccupancyMaxActiveBlocksPerMultiprocessor")
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        grid = int(os.environ.get("DSV41_PERSISTENT_GRID", "0")) or max(1, nb.value) * sms
        st = _chain_state[device.index] = {
            "f": f, "grid": grid,
            "f1": get_function("expert_chain.cu", "expert_chain_p1_5120_2304", device),
            "f2": get_function("expert_chain.cu", "expert_chain_p2_5120_2304", device),
            "gu": torch.zeros(7, 2 * inter, device=device, dtype=torch.float32),
            "h": torch.zeros(7, inter, device=device, dtype=torch.bfloat16),
            "ctrs": torch.zeros(5, device=device, dtype=torch.int32),
            "done": torch.zeros(8, device=device, dtype=torch.int32),
        }
        print(f"[expert-chain] {device}: {nb.value} resident blocks/SM x {sms} SMs -> grid {grid}", flush=True)
    return st


def expert_chain(xqp: torch.Tensor, xq: torch.Tensor, eid: torch.Tensor, wt: torch.Tensor, shard: tuple, sh: dict, moe,
                 part: torch.Tensor, ys: torch.Tensor | None, mode: int | None = None) -> None:
    """One launch for the whole expert phase of one token on this GPU (cuda/expert_chain.cu): the routed experts of
    the shard [shard_start, shard_start + n) among the topk (eid, wt) pairs, and, when ys is given, the shared expert
    (moe.sh_w13 / moe.sh_w2 as W8). part: fp32 [dim] (routed sum, routing weights applied), ys: bf16 [dim]."""
    dim = xqp.shape[-1]
    inter = moe.inter
    mode = mode or PERSISTENT_EXPERT or 1
    assert dim == 5120 and inter == 2304, (dim, inter)  # the compiled instantiation
    st = _chain_state_for(xqp.device, inter)
    a = _ChainArgs()
    a.xqp, a.xq = xqp.data_ptr(), xq.data_ptr()
    a.eid, a.wt = eid.data_ptr(), wt.data_ptr()
    a.topk, a.shard_start, a.shard_n = eid.numel(), shard[0], shard[1]
    a.w13, a.w13_stride = sh["w13"].data_ptr(), sh["w13"].stride(0)
    a.s13, a.s13_stride = sh["s13"].data_ptr(), sh["s13"].stride(0)
    a.w2, a.w2_stride = sh["w2"].data_ptr(), sh["w2"].stride(0)
    a.s2, a.s2_stride = sh["s2"].data_ptr(), sh["s2"].stride(0)
    if ys is not None:
        w13, w2 = moe.sh_w13, moe.sh_w2
        a.sh_w13, a.sh_s13, a.sh_s13_cols = w13.w8.data_ptr(), w13.s8.data_ptr(), w13.s8.shape[1]
        a.sh_w2, a.sh_s2, a.sh_s2_cols = w2.w8.data_ptr(), w2.s8.data_ptr(), w2.s8.shape[1]
        a.has_shared = 1
        a.ys = ys.data_ptr()
    else:
        a.sh_w13 = a.sh_s13 = a.sh_w2 = a.sh_s2 = 0
        a.sh_s13_cols = a.sh_s2_cols = 0
        a.has_shared = 0
        a.ys = 0
    a.limit = float(moe.swiglu_limit)
    a.gu, a.part, a.ctrs = st["gu"].data_ptr(), part.data_ptr(), st["ctrs"].data_ptr()
    a.dim, a.inter = dim, inter
    if mode == 2:  # single persistent launch (grid barrier)
        launch(st["f"], (st["grid"], 1, 1), (256, 1, 1), [a], xqp.device)
        return
    b = _ChainArgs2()
    b.a = a
    b.h, b.done = st["h"].data_ptr(), st["done"].data_ptr()
    # phase 1: one block per 8-row item; the number of local experts is only known on the device, so the grid covers
    # the maximum (topk routed slots + shared) and surplus blocks exit at once
    n1 = (eid.numel() + (1 if ys is not None else 0)) * (2 * inter // 8)
    n2 = (dim // 8) * (2 if ys is not None else 1)
    launch(st["f1"], (n1, 1, 1), (256, 1, 1), [b], xqp.device)
    launch(st["f2"], (n2, 1, 1), (256, 1, 1), [b], xqp.device)
