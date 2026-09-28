"""M = 1 FP8 GEMV microbenchmark on the decode shapes: current dev kernel (fp8_tc.cu, mma path with split-K),
the historical tiled-layout variant (compression-study 4e0fd85, cuda/fp8_tc_hist_tiled.cu) and the dedicated
M=1 kernel (cuda/fp8_gemv_m1.cu). Weights rotate over enough copies to defeat the 40 MB L2, so the numbers are
HBM streaming rates. usage: python -m dsv41.bench_gemv_m1 [--device cuda:6] [--iters 50]"""
import argparse, ctypes, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dsv41 import cukern
from dsv41.w8 import permute_k

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:6")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--shapes", default="")
a = ap.parse_args()
d = torch.device(a.device)
torch.cuda.set_device(d)
torch.manual_seed(0)
# (name, N, K, group_cols, x rows)
SHAPES = [("wq_a", 1280, 5120, 0, 1), ("wq_b", 32768, 1280, 0, 1), ("wkv", 512, 5120, 0, 1), ("wo_a", 8192, 4096, 1024, 8),
          ("wo_b", 5120, 8192, 0, 1), ("sh_w13", 4608, 5120, 0, 1), ("sh_w2", 5120, 2304, 0, 1), ("head", 129280, 5120, 0, 1)]
if a.shapes:
    SHAPES = [s for s in SHAPES if s[0] in a.shapes.split(",")]
L2_BYTES = 40 << 20

def tile(w8):
    N, K = w8.shape
    return w8.view(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3).reshape(N, K).contiguous()

def make(N, K):
    w = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=d)
    w[(w & 0x7F) == 0x7F] = 0x40  # no NaN codes
    s = torch.randint(118, 126, ((N + 31) // 32, K // 32), dtype=torch.uint8, device=d)
    return w, s

def reference(x, w, s, group_cols):
    wf = w.view(torch.float8_e4m3fn).float()
    N, K = w.shape
    sf = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.int() - 127)
    sf = sf.repeat_interleave(32, 0)[:N].repeat_interleave(32, 1)
    wf = wf * sf
    if group_cols:
        xg = N // group_cols
        out = torch.empty(N, device=d)
        for g in range(xg):
            out[g * group_cols:(g + 1) * group_cols] = wf[g * group_cols:(g + 1) * group_cols] @ x[g].float()
        return out
    return wf @ x[0].float()

def timeit(fn, iters):
    """GPU time per launch with the launches captured in a CUDA graph (no host launch gaps, like the decode graphs)."""
    fn(0); torch.cuda.synchronize(d)
    s = torch.cuda.Stream(d)
    with torch.cuda.stream(s):
        fn(1)
    torch.cuda.synchronize(d)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(iters):
            fn(i)
    torch.cuda.synchronize(d)
    g.replay(); torch.cuda.synchronize(d)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(3):
        g.replay()
    e1.record(); torch.cuda.synchronize(d)
    return e0.elapsed_time(e1) * 1000 / (3 * iters)

def m1_launch(x, w8p, s8, N, K, group_cols, KW, U, y, yf=None):
    f = cukern.get_function("fp8_gemv_m1.cu", f"fp8_gemv_m1_k{KW}u{U}", d)
    RB = 8 // KW
    grid = ((N + RB - 1) // RB, 1, 1)
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_void_p(w8p.data_ptr()), ctypes.c_void_p(s8.data_ptr()),
            ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(yf.data_ptr() if yf is not None else 0)]
    cukern.launch(f, grid, (256, 1, 1), args, d)

def hist_launch(x, w8t, s8, N, K, group_cols, y, part, counters, splits, kps):
    """the historical fp8_tc.cu (tiled layout arg), same launch geometry as dev's fp8_gemm_tc for M <= 8"""
    f = cukern.get_function("fp8_tc_hist_tiled.cu", "fp8_gemm_tc8", d)
    Mo = 1
    grid = ((N // 8 + 3) // 4, splits, 1)
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.stride(0)), ctypes.c_int(Mo),
            ctypes.c_void_p(w8t.data_ptr()), ctypes.c_void_p(s8.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(s8.shape[1]),
            ctypes.c_void_p(part.data_ptr()), ctypes.c_int(N), ctypes.c_int(kps), ctypes.c_int(group_cols),
            ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(counters.data_ptr()), ctypes.c_int(splits), ctypes.c_int(1)]
    cukern.launch(f, grid, (128, 1, 1), args, d)

print(f"device {d} {torch.cuda.get_device_name(d)}; {a.iters} iterations, weights rotated over copies > 40 MB L2")
print(f"{'shape':7s} {'N':>6s} {'K':>5s} {'MiB':>6s} | {'dev tc8':>9s} {'GB/s':>6s} {'err':>8s} | {'hist tiled':>10s} {'GB/s':>6s} {'err':>8s} | {'m1 best':>12s} {'us':>7s} {'GB/s':>6s} {'err':>8s} | m1 variants (us)")
for name, N, K, gc, xrows in SHAPES:
    nbytes = N * K + ((N + 31) // 32) * (K // 32)
    ncopies = max(1, min(16, math.ceil(L2_BYTES * 2.5 / nbytes)))
    ws = [make(N, K) for _ in range(ncopies)]
    x = (torch.randn(xrows, K, device=d) * 0.3).to(torch.bfloat16)
    ref = reference(x, ws[0][0], ws[0][1], gc)
    scale = ref.abs().max().item()
    # dev kernel (W8 layout: k-permuted)
    wp = [(permute_k(w), s) for w, s in ws]
    y_dev = cukern.fp8_gemm_tc(x.reshape(-1, K).contiguous(), wp[0][0], wp[0][1], group_cols=gc)
    err_dev = (y_dev.float().view(-1) - ref).abs().max().item() / scale
    t_dev = timeit(lambda i: cukern.fp8_gemm_tc(x.reshape(-1, K).contiguous(), wp[i % ncopies][0], wp[i % ncopies][1], group_cols=gc), a.iters)
    # historical tiled kernel: same split heuristic as dev
    splits = cukern._splits_for(N, K)
    kps = -(-K // splits); kps = -(-kps // 128) * 128; splits = -(-K // kps)
    part = torch.empty(splits, 1, N, device=d, dtype=torch.float32)
    counters = cukern._tile_counters(d, N // 8)
    wt = [(tile(w), s) for w, s in wp]
    y_h = torch.empty(1, N, device=d, dtype=torch.bfloat16)
    hist_launch(x.reshape(-1, K).contiguous(), wt[0][0], wt[0][1], N, K, gc, y_h, part, counters, splits, kps)
    torch.cuda.synchronize(d)
    err_h = (y_h.float().view(-1) - ref).abs().max().item() / scale
    t_h = timeit(lambda i: hist_launch(x.reshape(-1, K).contiguous(), wt[i % ncopies][0], wt[i % ncopies][1], N, K, gc, y_h, part, counters, splits, kps), a.iters)
    # dedicated M=1 kernel: KW x U sweep
    y_m = torch.empty(N, device=d, dtype=torch.bfloat16)
    res = {}
    for KW in (1, 2, 4, 8):
        if K % (16 * KW) != 0:
            continue
        for U in (2, 4, 8):
            xin = x.reshape(-1, K).contiguous()
            m1_launch(xin, wp[0][0], wp[0][1], N, K, gc, KW, U, y_m)
            torch.cuda.synchronize(d)
            err = (y_m.float() - ref).abs().max().item() / scale
            t = timeit(lambda i: m1_launch(xin, wp[i % ncopies][0], wp[i % ncopies][1], N, K, gc, KW, U, y_m), a.iters)
            res[(KW, U)] = (t, err)
    best = min(res, key=lambda k: res[k][0])
    tb, eb = res[best]
    gbs = lambda t: nbytes / t / 1e3
    print(f"{name:7s} {N:6d} {K:5d} {nbytes / 2**20:6.1f} | {t_dev:9.1f} {gbs(t_dev):6.0f} {err_dev:8.1e} | {t_h:10.1f} {gbs(t_h):6.0f} {err_h:8.1e} | k{best[0]}u{best[1]:<9d} {tb:7.1f} {gbs(tb):6.0f} {eb:8.1e} | "
          + " ".join(f"k{k[0]}u{k[1]}={v[0]:.1f}" for k, v in sorted(res.items())), flush=True)
    del ws, wp, wt
