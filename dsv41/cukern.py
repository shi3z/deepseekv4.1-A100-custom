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


# nvcc -arch for cubin builds. Default sm_80 (A100); set via set_cuda_arch() / serve --cuda-arch.
CUDA_ARCH = "sm_80"


def set_cuda_arch(arch: str) -> None:
    """Set nvcc -arch (e.g. sm_86 for A6000). Call before the first kernel load."""
    global CUDA_ARCH
    arch = arch.strip()
    if not arch.startswith("sm_"):
        arch = f"sm_{arch.replace('.', '')}"
    CUDA_ARCH = arch


def _cubin(src_name: str, arch: str | None = None) -> bytes:
    arch = arch or CUDA_ARCH
    src = os.path.join(HERE, "cuda", src_name)
    code = open(src, "rb").read()
    tag = hashlib.sha1(code).hexdigest()[:12]
    out = os.path.join(HERE, "cuda", f".{src_name}.{tag}.{arch}.cubin")
    if not os.path.exists(out):
        subprocess.run([NVCC, "-cubin", f"-arch={arch}", "-O3", "-o", out, src], check=True)
    return open(out, "rb").read()


def get_function(src_name: str, func: str, device: torch.device) -> ctypes.c_void_p:
    key = (src_name, device.index, CUDA_ARCH)
    if key not in _modules:
        with torch.cuda.device(device):
            torch.cuda.current_stream()  # make sure the context exists
            image = _cubin(src_name)
            mod = ctypes.c_void_p()
            _check(_cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
            _modules[key] = mod
    fkey = (src_name + ":" + func, device.index, CUDA_ARCH)
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
FP8_G_BM128 = os.environ.get("DSV41_FP8_G_BM128", "0") == "1"  # 128-row blocks (1 block/SM): measured slower, off
FP8_W_SPLITS = int(os.environ.get("DSV41_FP8_W_SPLITS", "0"))
FP8_W_STAGES = int(os.environ.get("DSV41_FP8_W_STAGES", "3"))  # x stages in shared memory
FP8_W_MW = int(os.environ.get("DSV41_FP8_W_MW", "1"))  # warps along M per block (1, 2 or 4 for 64 rows; 1 or 2 for 32)
_attr_done: set = set()  # experiment: force the split-K factor


def fp8_gemm_tc(x: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, group_cols: int = 0, out_dtype=torch.bfloat16, tiled: bool = False) -> torch.Tensor:
    """x: bf16 [M, K] (contiguous); w8: uint8 (e4m3 bits) [N, K] in the w8.PERM_K byte order (tiled=True: also w8.tile);
    s8: uint8 (E8M0) [ceil(N/32), K/32].
    Returns x @ dequant(w8, s8)^T as [M, N] (fp32 accumulation, rounded to out_dtype).
    group_cols > 0: block-diagonal use (x: [B * N/group_cols, K]; output row b, column n uses x row
    b * (N/group_cols) + n // group_cols) -> [B, N]."""
    M, K = x.shape
    N = w8.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and w8.is_contiguous() and s8.is_contiguous()
    assert K % 64 == 0 and N % 8 == 0 and (group_cols == 0 or group_cols % 8 == 0 and N % group_cols == 0)
    Mo = M // (N // group_cols) if group_cols else M
    use_tcg = FP8_G_LAYOUT and Mo > 64 and N % 128 == 0 and K % 128 == 0 and (group_cols == 0 or group_cols % 128 == 0)
    if use_tcg:  # many rows: CUTLASS-style tiles (fp8_tcg.cu), any M
        return _fp8_gemm_tcg(x, w8, s8, Mo, N, K, group_cols, out_dtype, tiled)
    wlayout = FP8_W_LAYOUT and Mo > 16 and N % 64 == 0 and (group_cols == 0 or group_cols % 64 == 0)
    rows_per = 64 if wlayout else 16
    if Mo > rows_per:  # chunk
        per = (N // group_cols) if group_cols else 1
        return torch.cat([fp8_gemm_tc(x[i * per : (i + rows_per) * per], w8, s8, group_cols, out_dtype, tiled) for i in range(0, Mo, rows_per)], dim=0)
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
            ctypes.c_void_p(y.data_ptr() if y is not None else 0), ctypes.c_void_p(counters.data_ptr() if counters is not None else 0), ctypes.c_int(splits),
            ctypes.c_int(1 if tiled else 0)]
    launch(f, grid, (WARPS * 32 * (MW if wlayout else 1), 1, 1), args, x.device, shared=shared if wlayout else 0)
    if fused_epilogue:
        return y
    y = part[0] if splits == 1 else part.sum(dim=0)
    return y.to(out_dtype)


def _fp8_gemm_tcg(x, w8, s8, Mo, N, K, group_cols, out_dtype, tiled=False):
    """fp8_tcg.cu: 256-thread blocks, tile 64 rows x 128 columns x 128 k, split-K with the fused epilogue."""
    BM = 128 if (FP8_G_BM128 and Mo >= 128) else 64  # 128-row blocks halve the weight decode per mma (1 block/SM)
    mblocks = (Mo + BM - 1) // BM
    splits = 1  # ~256 blocks (128-row blocks: ~110, one per SM), fp32 partials at most the weight bytes
    while (N // 128) * mblocks * splits < (256 if BM == 64 else 110) and splits * 2 * Mo * 4 <= K and K // (splits * 2) >= 256:
        splits *= 2
    if FP8_W_SPLITS:
        splits = FP8_W_SPLITS
    kps = -(-K // splits)
    kps = -(-kps // 128) * 128
    splits = -(-K // kps)
    part = torch.empty(splits, Mo, N, device=x.device, dtype=torch.float32)
    f = get_function("fp8_tcg.cu", "fp8_gemm_tcg" if BM == 64 else "fp8_gemm_tcg128", x.device)
    shared = 2 * (128 * 128 + BM * 256)
    if (f.value, x.device.index) not in _attr_done:
        _check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(shared)), "cuFuncSetAttribute")
        _attr_done.add((f.value, x.device.index))
    fused_epilogue = out_dtype == torch.bfloat16
    y = torch.empty(Mo, N, device=x.device, dtype=torch.bfloat16) if fused_epilogue else None
    counters = _tile_counters(x.device, (N // 128) * mblocks) if fused_epilogue else None
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(Mo),
            ctypes.c_void_p(w8.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr() if y is not None else 0), ctypes.c_void_p(counters.data_ptr() if counters is not None else 0), ctypes.c_int(splits),
            ctypes.c_int(1 if tiled else 0)]
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
FP4_TILED = os.environ.get("DSV41_FP4_TILED", "1") == "1"  # experts stored tiled (quant.tile_fp4); the loader clears it in offload mode
FP4_W_MAX = 64  # tokens per group the second layout handles (groups are split at this size)
# groups up to this many tokens use fp4_tc.cu (x as the mma A operand, from registers), larger ones fp4_tcw.cu (x staged in
# shared memory). With the tiled weight layout fp4_tcw.cu is faster at every size (1 token/expert: 1.19 -> 1.46 TB/s), so 0.
FP4_SMALL_MAX = int(os.environ.get("DSV41_FP4_SMALL_MAX", "0"))


def fp4_gemm_tc(xp: torch.Tensor, w: torch.Tensor, s: torch.Tensor, grp_expert: torch.Tensor, grp_start: torch.Tensor,
                pair_tok: torch.Tensor, n_pairs: int, max_tokens: int, shard_start: int = 0, shard_n: int = 1 << 30,
                zero_out: bool = False, out: torch.Tensor | None = None, out_col: int = 0) -> torch.Tensor:
    """xp: permuted bf16 [rows, K]; w: uint8 [E, N, K/2]; s: uint8 [E, N, K/32]; groups g: expert grp_expert[g] with pairs
    grp_start[g]..grp_start[g+1]-1 (<= max_tokens each), pair p uses x row pair_tok[p]. Returns fp32 [n_pairs, N].
    Groups of <= 8 tokens run on fp4_tc.cu (x as the mma A operand); larger groups (up to 64) on fp4_tcw.cu, which
    reads the expert once whatever the token count (fp4_tc.cu's 16-token variant handles 9..16 when disabled).
    Expert parallelism: ids are global, this GPU holds [shard_start, shard_start + shard_n); other groups are skipped
    (zero_out: their output rows are zeroed)."""
    E, N, Kh = w.shape  # w may be a column slice w_full[:, n0:n0+N] (n0 % 16 == 0): the same byte offset in both layouts
    K = Kh * 2
    assert xp.dtype == torch.bfloat16 and xp.is_contiguous() and K % 128 == 0 and N % 8 == 0
    G = grp_expert.numel()
    if out is None:
        out = torch.empty(n_pairs, N, device=xp.device, dtype=torch.float32)
    ldo = out.stride(0)
    assert out.shape[1] >= out_col + N
    WARPS = 4
    big = max_tokens > FP4_SMALL_MAX and (FP4_W_LAYOUT or max_tokens > 16 or FP4_SMALL_MAX == 0)
    assert max_tokens <= (FP4_W_MAX if big else 16)
    assert not big or N % 64 == 0
    common = [ctypes.c_void_p(xp.data_ptr()), ctypes.c_int(xp.stride(0)),
              ctypes.c_void_p(w.data_ptr()), ctypes.c_longlong(w.stride(0)), ctypes.c_void_p(s.data_ptr()), ctypes.c_longlong(s.stride(0)),
              ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr()), ctypes.c_void_p(pair_tok.data_ptr()),
              ctypes.c_void_p(out.data_ptr() + out_col * 4), ctypes.c_int(ldo), ctypes.c_int(N), ctypes.c_int(K),
              ctypes.c_int(shard_start), ctypes.c_int(min(shard_n, E)), ctypes.c_int(1 if zero_out else 0)]
    tiled = [ctypes.c_int(1 if FP4_TILED else 0)]
    small_max = FP4_SMALL_MAX if (big or max_tokens <= FP4_SMALL_MAX) else 16
    if small_max > 0:
        f = get_function("fp4_tc.cu", "fp4_gemm_tc8" if small_max <= 8 else "fp4_gemm_tc16", xp.device)
        launch(f, ((N // 8 + WARPS - 1) // WARPS, G, 1), (WARPS * 32, 1, 1), common + [ctypes.c_int(0), ctypes.c_int(small_max)] + tiled, xp.device)
    if big:
        f = get_function("fp4_tcw.cu", "fp4_gemm_tcw", xp.device)
        shared = 3 * 64 * 256
        if (f.value, xp.device.index) not in _attr_done:
            _check(_cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(shared)), "cuFuncSetAttribute")
            _attr_done.add((f.value, xp.device.index))
        launch(f, (N // 64, G, 1), (WARPS * 32, 1, 1), common + [ctypes.c_int(FP4_SMALL_MAX + 1), ctypes.c_int(FP4_W_MAX)] + tiled, xp.device, shared=shared)
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


def p2p_sum_rows_idx(dst: torch.Tensor, src: torch.Tensor, idx: torch.Tensor, device: torch.device, groups: int, col0: int = 0):
    """dst [groups, n] = per group the sum of `rows` src rows picked by idx (int32 [groups * rows]) over src columns
    [col0, col0 + n) (src: fp32 [*, ld], any row stride)."""
    rows = idx.numel() // groups
    n = dst.shape[-1]
    f = get_function("p2p.cu", "p2p_sum_rows_idx", device)
    launch(f, ((n + 255) // 256, groups, 1), (256, 1, 1), [ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(idx.data_ptr()),
                                                          ctypes.c_int(rows), ctypes.c_int(n), ctypes.c_int(groups), ctypes.c_longlong(src.stride(0)),
                                                          ctypes.c_int(col0), ctypes.c_longlong(dst.stride(0))], device)


def moe_dispatch(eid: torch.Tensor, wt: torch.Tensor, topk: int, n_experts: int, gmax: int, device: torch.device):
    """One-launch expert bucketing (cuda/p2p.cu moe_dispatch): eid int32 [n], wt fp32 [n] ->
    (tok_sorted int32 [n], wt_sorted fp32 [n], inv int32 [n], grp_expert int32 [n], grp_start int32 [n+1])."""
    n = eid.numel()
    assert n_experts <= 512
    tok_sorted = torch.empty(n, dtype=torch.int32, device=device)
    wt_sorted = torch.empty(n, dtype=torch.float32, device=device)
    inv = torch.empty(n, dtype=torch.int32, device=device)
    grp_expert = torch.empty(n, dtype=torch.int32, device=device)
    grp_start = torch.empty(n + 1, dtype=torch.int32, device=device)
    f = get_function("p2p.cu", "moe_dispatch", device)
    launch(f, (1, 1, 1), (1024, 1, 1), [ctypes.c_void_p(eid.data_ptr()), ctypes.c_void_p(wt.data_ptr()), ctypes.c_int(n), ctypes.c_int(topk),
                                       ctypes.c_int(n_experts), ctypes.c_int(gmax), ctypes.c_void_p(tok_sorted.data_ptr()), ctypes.c_void_p(wt_sorted.data_ptr()),
                                       ctypes.c_void_p(inv.data_ptr()), ctypes.c_void_p(grp_expert.data_ptr()), ctypes.c_void_p(grp_start.data_ptr())], device)
    return tok_sorted, wt_sorted, inv, grp_expert, grp_start


def p2p_signal(flag_ptrs: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Set the flags at the addresses in flag_ptrs (int64 device tensor on `device`) to the value of seq (int32 device scalar)."""
    f = get_function("p2p.cu", "p2p_signal", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flag_ptrs.data_ptr()), ctypes.c_int(flag_ptrs.numel()), ctypes.c_void_p(seq.data_ptr())], device)


def p2p_wait(flags: torch.Tensor, seq: torch.Tensor, device: torch.device):
    """Spin (one thread) until all flags (int32 [n] on `device`) >= seq."""
    f = get_function("p2p.cu", "p2p_wait", device)
    launch(f, (1, 1, 1), (1, 1, 1), [ctypes.c_void_p(flags.data_ptr()), ctypes.c_int(flags.numel()), ctypes.c_void_p(seq.data_ptr())], device)


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
