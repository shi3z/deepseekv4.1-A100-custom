// Persistent fused expert chain for one decode token (DSV41_PERSISTENT_EXPERT=1): the routed experts of this GPU's
// shard AND (on the layer owner) the shared expert, w13 -> SwiGLU + FP8 rounding -> w2 -> sum, in ONE launch.
//
// Replaces, per layer and GPU, the dependent launches fp4_gemm_tc (w13), swiglu_quant, fp4_gemm_tc (w2),
// p2p_sum_rows and, on the owner, fp8 GEMV (shared w13), swiglu_quant, fp8 GEMV (shared w2): 7 launches -> 1.
// The intermediates (gu: fp32 [7, 2*INTER]) stay in L2; the SwiGLU outputs live in shared memory.
//
// Phase 1: rows of w13 of every local expert (FP4, K = DIM) and of the shared expert (FP8, K = DIM) -> gu[slot][*].
// Work items: 8 weight rows (tensor-core m16n8k16 fragments as in fp4_tc.cu / fp8_tc.cu, the 8 warps split K,
// partials reduced in shared memory), assigned statically round-robin (item i -> block i mod grid; no atomics: with hundreds of
// blocks claiming from one counter the serialized atomics cost more than the imbalance they remove).
// Grid barrier (sense reversing, self resetting: graph-replayable with constant arguments; every block must be
// resident, the launcher sizes the grid from the occupancy API).
// Phase 2: every block computes the SwiGLU + FP8 fake-quant of each slot once into shared memory (the exact math
// of fused.py::_swiglu_quant_kernel; routed experts scaled by their routing weight and stored in the 8-k permuted
// order the FP4 nibbles need), then rows of w2: a warp owns output row n and sums over ALL local experts in
// registers (deterministic, no atomics) -> part[n] fp32; the shared expert's rows -> ys[n] bf16.
//
// Numerics: FP4 / FP8 decodes are the exact register decodes of fp4_tc.cu / fp8_tc.cu; dot products accumulate in
// fp32 (per-lane chunk partials, warp shuffle reduce); SwiGLU uses sigmoid = 1 / (1 + expf(-g)), IEEE division
// and rint for the e4m3 rounding.
#include <cuda_bf16.h>
#include <stdint.h>

#define THREADS 256
#define WARPS 8
#define MAXSLOT 7      // up to 6 routed experts + 1 shared
#define U 4            // 16-byte loads in flight per lane

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t bf16x2_fma0(uint32_t a, uint32_t b) {
    uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u)); return r;
}
__device__ __forceinline__ float2 bf2f(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u));
}
__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t f2, uint32_t* o) {
    uint32_t a = ((w & 0x00070007u) << 6) | ((w & 0x00080008u) << 12);
    uint32_t b = ((w & 0x07000700u) >> 2) | ((w & 0x08000800u) << 4);
    uint32_t c = ((w & 0x00700070u) << 2) | ((w & 0x00800080u) << 8);
    uint32_t d = ((w & 0x70007000u) >> 6) | (w & 0x80008000u);
    o[0] = bf16x2_fma0(a, f2); o[1] = bf16x2_fma0(b, f2); o[2] = bf16x2_fma0(c, f2); o[3] = bf16x2_fma0(d, f2);
}
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    uint32_t t0 = prmt(w, 0, 0x1404), t1 = prmt(w, 0, 0x3424);
    lo = ((t0 >> 4) & 0x07F007F0u) | (t0 & 0x80008000u);
    hi = ((t1 >> 4) & 0x07F007F0u) | (t1 & 0x80008000u);
}
// FP4: one 16-byte chunk = 32 k (permuted x: four uint4), scale folded as bf16 2^(s-1)
__device__ __forceinline__ float fp4_chunk_dot(const uint4 w, const uint4* xq, int sb) {
    const uint32_t fb = (uint32_t)(sb + 126) << 7;
    const uint32_t f2 = fb | (fb << 16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t o[4]; e2m1x8_to_bf16(ww[j], f2, o);
        const uint32_t xv[4] = {xq[j].x, xq[j].y, xq[j].z, xq[j].w};
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 wv = bf2f(o[i]), xx = bf2f(xv[i]); a0 = fmaf(wv.x, xx.x, a0); a1 = fmaf(wv.y, xx.y, a1); }
    }
    return a0 + a1;
}
// FP8: one 16-byte chunk = 16 k in the PERM_K byte order (x natural: two uint4), unscaled (x 2^-120)
__device__ __forceinline__ float fp8_chunk_dot(const uint4 w, const uint4 xa, const uint4 xb) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi; e4m3x4_to_bf16(ww[j], lo, hi);
        const float2 a = bf2f(lo), b = bf2f(hi), p = bf2f(xv[j]), q = bf2f(xv[j + 4]);
        a0 = fmaf(a.x, p.x, a0); a1 = fmaf(a.y, p.y, a1); a0 = fmaf(b.x, q.x, a0); a1 = fmaf(b.y, q.y, a1);
    }
    return a0 + a1;
}
__device__ __forceinline__ float fp8_scale(int sb) {  // 2^(s-127) * 2^120
    return (sb + 120 > 0 && sb + 120 < 255) ? __uint_as_float((uint32_t)(sb + 120) << 23) : 0.f;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// ---- FP4, 8 weight rows n0..n0+7 x one token, k steps of 128 assigned round-robin to the block's warps
// (fp4_tc.cu's fragment layout: lane g = lane/4 owns row n0+g, t = lane%4 owns k = K0 + 32t .. +31 of each step).
// Returns in c[0], c[1] the outputs of columns n0 + 2t, n0 + 2t + 1 on the lanes with g == 0 (token row 0).
__device__ __forceinline__ void fp4_rows8_mma(const uint8_t* __restrict__ W8rows, int wstride, const uint8_t* __restrict__ S8rows, int sstride,
                                              const __nv_bfloat16* __restrict__ xperm, int K, int warp, float* c) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint8_t* wrow = W8rows + (long long)g * wstride;
    const uint8_t* srow = S8rows + (long long)g * sstride;
    c[0] = c[1] = c[2] = c[3] = 0.f;
    const int nsteps = K / 128;
    constexpr int PF = 5;  // steps whose weight loads are in flight together (5 x 16 B per lane)
    for (int st0 = warp; st0 < nsteps; st0 += WARPS * PF) {
        uint4 wc[PF]; int sc[PF];
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int st = st0 + u * WARPS;
            const bool ok = st < nsteps;
            const int kb = (ok ? st : 0) * 128 + 32 * t;
            wc[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (kb >> 1))) : make_uint4(0, 0, 0, 0);
            sc[u] = ok ? __ldg(srow + (kb >> 5)) : 0;
        }
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int st = st0 + u * WARPS;
            if (st >= nsteps) break;
            const int kb = st * 128 + 32 * t;
            uint4 ya[4];
            if (g == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) ya[i] = *reinterpret_cast<const uint4*>(xperm + kb + 8 * i);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) ya[i] = make_uint4(0, 0, 0, 0);
            }
            const uint32_t fb = (uint32_t)(sc[u] + 126) << 7;
            const uint32_t f2 = fb | (fb << 16);
            uint32_t b[16];
            e2m1x8_to_bf16(wc[u].x, f2, b); e2m1x8_to_bf16(wc[u].y, f2, b + 4); e2m1x8_to_bf16(wc[u].z, f2, b + 8); e2m1x8_to_bf16(wc[u].w, f2, b + 12);
            const uint32_t* xav = reinterpret_cast<const uint32_t*>(ya);
#pragma unroll
            for (int s2 = 0; s2 < 8; ++s2) {
                const uint32_t bf[2] = {b[2 * s2], b[2 * s2 + 1]};
                const uint32_t af[4] = {xav[2 * s2], 0u, xav[2 * s2 + 1], 0u};
                mma16816(c, af, bf);
            }
        }
    }
}
// ---- FP8 (PERM_K byte order), 8 weight rows x one token, 128-k steps round-robin over the warps (fp8_tc.cu layout:
// lane owns row n0+g and k = K0 + 16t .. +15 of each 64-k half). x natural order.
__device__ __forceinline__ void fp8_rows8_mma(const uint8_t* __restrict__ W8rows, int wstride, const uint8_t* __restrict__ Srow, int scols,
                                              const __nv_bfloat16* __restrict__ x, int K, int warp, float* c) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint8_t* wrow = W8rows + (long long)g * wstride;
    c[0] = c[1] = c[2] = c[3] = 0.f;
    const int nsteps = K / 128;
    constexpr int PF = 3;  // steps (2 halves each) in flight together: 6 x 16 B per lane
    for (int st0 = warp; st0 < nsteps; st0 += WARPS * PF) {
        uint4 wc[PF][2]; int sc[PF][2];
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int st = st0 + u * WARPS;
            const bool ok = st < nsteps;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kb = (ok ? st : 0) * 128 + 64 * h + 16 * t;
                wc[u][h] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + kb)) : make_uint4(0, 0, 0, 0);
                sc[u][h] = ok ? __ldg(Srow + (kb >> 5)) : 0;
            }
        }
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int st = st0 + u * WARPS;
            if (st >= nsteps) break;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kb = st * 128 + 64 * h + 16 * t;
                uint4 xa0, xa1;
                if (g == 0) { xa0 = *reinterpret_cast<const uint4*>(x + kb); xa1 = *reinterpret_cast<const uint4*>(x + kb + 8); }
                else { xa0 = make_uint4(0, 0, 0, 0); xa1 = xa0; }
                const uint32_t fb = (sc[u][h] + 120 > 0) ? ((uint32_t)(sc[u][h] + 120) << 7) : 0u;
                const uint32_t f2 = fb | (fb << 16);
                uint32_t b[8];
                e4m3x4_to_bf16(wc[u][h].x, b[0], b[1]); e4m3x4_to_bf16(wc[u][h].y, b[2], b[3]); e4m3x4_to_bf16(wc[u][h].z, b[4], b[5]); e4m3x4_to_bf16(wc[u][h].w, b[6], b[7]);
#pragma unroll
                for (int i = 0; i < 8; ++i) b[i] = bf16x2_fma0(b[i], f2);
                const uint32_t xav[8] = {xa0.x, xa0.y, xa0.z, xa0.w, xa1.x, xa1.y, xa1.z, xa1.w};
#pragma unroll
                for (int s2 = 0; s2 < 4; ++s2) {
                    const uint32_t bf[2] = {b[2 * s2], b[2 * s2 + 1]};
                    const uint32_t af[4] = {xav[s2], 0u, xav[s2 + 4], 0u};
                    mma16816(c, af, bf);
                }
            }
        }
    }
}
// block-level reduce of the 8 warps' k-split partials of an 8-row item: red[warp][8] -> out[0..7] (threads 0..7)
__device__ __forceinline__ float rows8_reduce(float* red, const float* c, int warp) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (g == 0) { red[warp * 8 + 2 * t] = c[0]; red[warp * 8 + 2 * t + 1] = c[1]; }
    __syncthreads();
    float v = 0.f;
    if (threadIdx.x < 8) {
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += red[w * 8 + threadIdx.x];
    }
    __syncthreads();
    return v;
}

// ---- FP4 row dot over chunks [c_lo, c_hi) of a row (chunk = 32 k), x permuted (global or shared)
__device__ __forceinline__ float fp4_row_dot(const uint8_t* wrow, const uint8_t* srow, const __nv_bfloat16* xperm, int c_lo, int c_hi, int lane) {
    float acc = 0.f;
    for (int c0 = c_lo + lane; c0 < c_hi; c0 += 32 * U) {
        uint4 wv[U]; int sb[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            const bool ok = c < c_hi;
            wv[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + c * 16)) : make_uint4(0, 0, 0, 0);
            sb[u] = ok ? __ldg(srow + c) : 0;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            if (c < c_hi) {
                const uint4* xq = reinterpret_cast<const uint4*>(xperm + c * 32);
                const uint4 xv[4] = {xq[0], xq[1], xq[2], xq[3]};
                acc += fp4_chunk_dot(wv[u], xv, sb[u]);
            }
        }
    }
    return acc;
}
// ---- FP8 row dot over chunks [c_lo, c_hi) (chunk = 16 k), scale row indexed by chunk / 2
__device__ __forceinline__ float fp8_row_dot(const uint8_t* wrow, const uint8_t* srow, const __nv_bfloat16* x, int c_lo, int c_hi, int lane) {
    float acc = 0.f;
    for (int c0 = c_lo + lane; c0 < c_hi; c0 += 32 * U) {
        uint4 wv[U]; int sb[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            const bool ok = c < c_hi;
            wv[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + c * 16)) : make_uint4(0, 0, 0, 0);
            sb[u] = ok ? __ldg(srow + (c >> 1)) : 0;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            if (c < c_hi) {
                const uint4 xa = *reinterpret_cast<const uint4*>(x + c * 16);
                const uint4 xb = *reinterpret_cast<const uint4*>(x + c * 16 + 8);
                acc = fmaf(fp8_chunk_dot(wv[u], xa, xb), fp8_scale(sb[u]), acc);
            }
        }
    }
    return acc;
}

// ---- SwiGLU + per-32 e4m3 fake quant (fused.py math)
__device__ __forceinline__ float round_e4m3(float v) {
    float a = fabsf(v);
    int e = ((__float_as_int(a) >> 23) & 0xFF) - 127;
    e = max(e, -6);
    const float ulp = __int_as_float((e - 3 + 127) << 23);
    const float inv_ulp = __int_as_float((3 - e + 127) << 23);  // exact reciprocal of a power of two
    float r = rintf(a * inv_ulp) * ulp;
    r = fminf(r, 448.f);
    return v < 0.f ? -r : r;
}
__device__ __forceinline__ float pow2_ceil_log2(float a) {
    const int bits = __float_as_int(a);
    int e = ((bits >> 23) & 0xFF) - 127;
    if (bits & 0x7FFFFF) e += 1;
    return __int_as_float((e + 127) << 23);
}
__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16(v)); }

// ---- grid barrier (sense reversing, self resetting)
__device__ __forceinline__ void grid_barrier(unsigned int* count, unsigned int* gen, unsigned int nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int g = *((volatile unsigned int*)gen);
        __threadfence();
        if (atomicAdd(count, 1u) == nblocks - 1u) {
            *count = 0u;
            __threadfence();
            atomicAdd(gen, 1u);
        } else {
            while (*((volatile unsigned int*)gen) == g) { }
        }
        __threadfence();
    }
    __syncthreads();
}

struct ChainArgs {
    const __nv_bfloat16* xqp;   // [DIM] 8-k permuted fp8-rounded activation (routed experts)
    const __nv_bfloat16* xq;    // [DIM] natural order (shared expert)
    const int* eid;             // [TOPK] global expert ids
    const float* wt;            // [TOPK] routing weights
    int topk, shard_start, shard_n;
    const uint8_t* w13; long long w13_stride;   // FP4 [n_shard, 2*INTER, DIM/2]
    const uint8_t* s13; long long s13_stride;   // [n_shard, 2*INTER, DIM/32]
    const uint8_t* w2;  long long w2_stride;    // FP4 [n_shard, DIM, INTER/2]
    const uint8_t* s2;  long long s2_stride;    // [n_shard, DIM, INTER/32]
    const uint8_t* sh_w13; const uint8_t* sh_s13; int sh_s13_cols;  // FP8 [2*INTER, DIM] + [ceil(2*INTER/32), DIM/32]
    const uint8_t* sh_w2;  const uint8_t* sh_s2;  int sh_s2_cols;   // FP8 [DIM, INTER] + [ceil(DIM/32), INTER/32]
    int has_shared;
    float limit;
    float* gu;                  // scratch fp32 [MAXSLOT, 2*INTER]
    float* part;                // out fp32 [DIM]: sum over the local routed experts (routing weights applied)
    __nv_bfloat16* ys;          // out bf16 [DIM]: shared expert (owner only)
    unsigned int* ctrs;         // [2]: barrier count, barrier gen
    int dim, inter;
};

template <int DIM, int INTER>
__device__ __forceinline__ void expert_chain_body(const ChainArgs& A)
{
    constexpr int N1 = 2 * INTER;           // rows of w13
    constexpr int I1_FP4 = N1 / 8;          // phase-1 items per routed slot: 8 rows, the 8 warps split K
    constexpr int I1_FP8 = N1 / 8;          // phase-1 items of the shared expert: 8 rows, the 8 warps split K
    constexpr int I2 = DIM / 8;             // phase-2 items per (routed | shared): 8 rows
    __shared__ __nv_bfloat16 h_s[MAXSLOT][INTER];  // SwiGLU outputs: routed slots permuted, shared natural
    __shared__ float red[WARPS * 8];
    __shared__ int s_loc[MAXSLOT];
    __shared__ float s_wt[MAXSLOT];
    __shared__ int s_nloc;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int i = 0; i < A.topk; ++i) {
            const int e = A.eid[i] - A.shard_start;
            if (e >= 0 && e < A.shard_n) { s_loc[n] = e; s_wt[n] = A.wt[i]; ++n; }
        }
        s_nloc = n;
    }
    __syncthreads();
    const int nloc = s_nloc;
    const int G = gridDim.x;
    // ================================================================ phase 1: w13 rows -> gu
    // items in weight-byte units (FP4 item = 1 unit of 20 KB, FP8 item = 2 units); block b owns the units
    // [b * total / G, (b + 1) * total / G) -> the tail imbalance is at most one unit
    const int n_fp4 = nloc * I1_FP4;
    const int n_items1 = n_fp4 + (A.has_shared ? I1_FP8 : 0);
    const int units1 = n_fp4 + 2 * (n_items1 - n_fp4);
    const int u_lo = (int)(((long long)blockIdx.x * units1) / G), u_hi = (int)(((long long)(blockIdx.x + 1) * units1) / G);
    const int it_lo = u_lo < n_fp4 ? u_lo : n_fp4 + (u_lo - n_fp4 + 1) / 2;
    const int it_hi = u_hi < n_fp4 ? u_hi : n_fp4 + (u_hi - n_fp4 + 1) / 2;
    for (int item = it_lo; item < it_hi; ++item) {
        float c[4];
        int slot, n0;
        if (item < n_fp4) {
            slot = item / I1_FP4;
            const int e = s_loc[slot];
            n0 = (item % I1_FP4) * 8;
            fp4_rows8_mma(A.w13 + (long long)e * A.w13_stride + (long long)n0 * (DIM / 2), DIM / 2,
                          A.s13 + (long long)e * A.s13_stride + (long long)n0 * (DIM / 32), DIM / 32, A.xqp, DIM, warp, c);
        } else {
            slot = nloc;
            n0 = (item - n_fp4) * 8;
            fp8_rows8_mma(A.sh_w13 + (long long)n0 * DIM, DIM, A.sh_s13 + (long long)(n0 >> 5) * A.sh_s13_cols, A.sh_s13_cols, A.xq, DIM, warp, c);
        }
        const float v = rows8_reduce(red, c, warp);
        if (threadIdx.x < 8) A.gu[slot * N1 + n0 + threadIdx.x] = v;
    }
    grid_barrier(&A.ctrs[0], &A.ctrs[1], G);
    // ================================================================ phase 2 prologue: SwiGLU + fake quant -> smem
    // (every block builds the vectors it needs; the loads of GB groups are issued together so the block pays one
    // L2 latency per batch instead of one per group)
    {
        constexpr int GB = 8;
        const int nslot = nloc + (A.has_shared ? 1 : 0);
        const int ngroups = nslot * (INTER / 32);
        for (int g0 = warp; g0 < ngroups; g0 += WARPS * GB) {
            float gate[GB], up[GB];
#pragma unroll
            for (int i = 0; i < GB; ++i) {
                const int g = g0 + i * WARPS;
                const bool ok = g < ngroups;
                const int slot = ok ? g / (INTER / 32) : 0, k = ok ? (g % (INTER / 32)) * 32 + lane : lane;
                const float* gu = A.gu + slot * N1;
                gate[i] = ok ? __ldg(gu + k) : 0.f;
                up[i] = ok ? __ldg(gu + INTER + k) : 0.f;
            }
#pragma unroll
            for (int i = 0; i < GB; ++i) {
                const int g = g0 + i * WARPS;
                if (g >= ngroups) break;
                const int slot = g / (INTER / 32), k = (g % (INTER / 32)) * 32 + lane;
                float gt = gate[i], u = up[i];
                if (A.limit > 0.f) { u = fminf(fmaxf(u, -A.limit), A.limit); gt = fminf(gt, A.limit); }
                float h = gt * (1.f / (1.f + expf(-gt))) * u;
                if (slot < nloc) h *= s_wt[slot];
                h = bf16_round(h);
                float amax = fabsf(h);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                amax = fmaxf(amax, 1e-4f);
                const float sc = pow2_ceil_log2(__fdiv_rn(amax, 448.f));
                const float inv = __int_as_float(254 * (1 << 23) - __float_as_int(sc));  // 1 / sc, exact for a power of two
                const float q = round_e4m3(fminf(fmaxf(h * inv, -448.f), 448.f));
                const float v = q * sc;
                int kk = k;
                if (slot < nloc) { const int j = k & 7; kk = (k & ~7) | ((j & 1) << 2) | (j & 2) | (j >> 2); }
                h_s[slot][kk] = __float2bfloat16(v);
            }
        }
    }
    __syncthreads();
    // ================================================================ phase 2: w2 rows -> part / ys
    // routed item = nloc units (of 9 KB), shared item = 2 units
    const int n_items2 = I2 + (A.has_shared ? I2 : 0);
    const int ru = nloc > 0 ? nloc : 1;
    const int units2 = ru * I2 + 2 * (n_items2 - I2);
    const int v_lo = (int)(((long long)blockIdx.x * units2) / G), v_hi = (int)(((long long)(blockIdx.x + 1) * units2) / G);
    const int jt_lo = v_lo < ru * I2 ? (v_lo + ru - 1) / ru : I2 + (v_lo - ru * I2 + 1) / 2;
    const int jt_hi = v_hi < ru * I2 ? (v_hi + ru - 1) / ru : I2 + (v_hi - ru * I2 + 1) / 2;
    for (int item = jt_lo; item < jt_hi; ++item) {
        float c[4];
        if (item < I2) {
            const int n0 = item * 8;
            float tot = 0.f;
            for (int slot = 0; slot < nloc; ++slot) {
                const int e = s_loc[slot];
                fp4_rows8_mma(A.w2 + (long long)e * A.w2_stride + (long long)n0 * (INTER / 2), INTER / 2,
                              A.s2 + (long long)e * A.s2_stride + (long long)n0 * (INTER / 32), INTER / 32, h_s[slot], INTER, warp, c);
                tot += rows8_reduce(red, c, warp);
            }
            if (threadIdx.x < 8) A.part[n0 + threadIdx.x] = tot;
        } else {
            const int n0 = (item - I2) * 8;
            fp8_rows8_mma(A.sh_w2 + (long long)n0 * INTER, INTER, A.sh_s2 + (long long)(n0 >> 5) * A.sh_s2_cols, A.sh_s2_cols, h_s[nloc], INTER, warp, c);
            const float v = rows8_reduce(red, c, warp);
            if (threadIdx.x < 8) A.ys[n0 + threadIdx.x] = __float2bfloat16(v);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS, 4)
expert_chain_5120_2304(ChainArgs A) { expert_chain_body<5120, 2304>(A); }

// ================================================================================ two-launch variant
// Phase 1: grid = one block per 8-row item (routed FP4 items, then the shared expert's FP8 items); the last block
// to finish a slot (atomic counter per slot, self resetting) computes that slot's SwiGLU + fake quant into
// A.h (bf16 [MAXSLOT][INTER], routed permuted / shared natural). Phase 2: grid = one block per 8-row output item
// (routed rows summing over the local experts, then the shared expert's rows), h read from L2. No grid barrier, no
// residency requirement, no redundant prologue; 2 launches instead of 7 (4 on the routed critical path).
struct ChainArgs2 {
    ChainArgs a;
    __nv_bfloat16* h;            // scratch bf16 [MAXSLOT, INTER]
    unsigned int* done;          // [MAXSLOT] finished-item counters (self resetting)
};

template <int DIM, int INTER>
__device__ __forceinline__ void swiglu_slot(const ChainArgs& A, int slot, int nloc, float wt, __nv_bfloat16* h_out) {
    constexpr int N1 = 2 * INTER;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int g = warp; g < INTER / 32; g += WARPS) {
        const int k = g * 32 + lane;
        const float* gu = A.gu + slot * N1;
        float gt = __ldg(gu + k), u = __ldg(gu + INTER + k);
        if (A.limit > 0.f) { u = fminf(fmaxf(u, -A.limit), A.limit); gt = fminf(gt, A.limit); }
        float hh = gt * (1.f / (1.f + expf(-gt))) * u;
        if (slot < nloc) hh *= wt;
        hh = bf16_round(hh);
        float amax = fabsf(hh);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        amax = fmaxf(amax, 1e-4f);
        const float sc = pow2_ceil_log2(__fdiv_rn(amax, 448.f));
        const float inv = __int_as_float(254 * (1 << 23) - __float_as_int(sc));
        const float q = round_e4m3(fminf(fmaxf(hh * inv, -448.f), 448.f));
        int kk = k;
        if (slot < nloc) { const int j = k & 7; kk = (k & ~7) | ((j & 1) << 2) | (j & 2) | (j >> 2); }
        h_out[kk] = __float2bfloat16(q * sc);
    }
}

template <int DIM, int INTER>
__device__ __forceinline__ void expert_chain_p1_body(const ChainArgs2& B)
{
    const ChainArgs& A = B.a;
    constexpr int N1 = 2 * INTER, I1 = N1 / 8;
    __shared__ float red[WARPS * 8];
    __shared__ int s_loc[MAXSLOT]; __shared__ float s_wt[MAXSLOT]; __shared__ int s_nloc; __shared__ int s_last;
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int i = 0; i < A.topk; ++i) { const int e = A.eid[i] - A.shard_start; if (e >= 0 && e < A.shard_n) { s_loc[n] = e; s_wt[n] = A.wt[i]; ++n; } }
        s_nloc = n;
    }
    __syncthreads();
    const int nloc = s_nloc, n_fp4 = nloc * I1;
    const int item = blockIdx.x;
    if (item >= n_fp4 + (A.has_shared ? I1 : 0)) return;
    float c[4]; int slot, n0;
    if (item < n_fp4) {
        slot = item / I1; const int e = s_loc[slot]; n0 = (item % I1) * 8;
        fp4_rows8_mma(A.w13 + (long long)e * A.w13_stride + (long long)n0 * (DIM / 2), DIM / 2,
                      A.s13 + (long long)e * A.s13_stride + (long long)n0 * (DIM / 32), DIM / 32, A.xqp, DIM, warp, c);
    } else {
        slot = nloc; n0 = (item - n_fp4) * 8;
        fp8_rows8_mma(A.sh_w13 + (long long)n0 * DIM, DIM, A.sh_s13 + (long long)(n0 >> 5) * A.sh_s13_cols, A.sh_s13_cols, A.xq, DIM, warp, c);
    }
    const float v = rows8_reduce(red, c, warp);
    if (threadIdx.x < 8) A.gu[slot * N1 + n0 + threadIdx.x] = v;
    // last block of this slot: SwiGLU of the whole slot
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(&B.done[slot], 1u) == (unsigned)(I1 - 1));
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    swiglu_slot<DIM, INTER>(A, slot, nloc, slot < nloc ? s_wt[slot] : 0.f, B.h + slot * INTER);
    __syncthreads();
    if (threadIdx.x == 0) { B.done[slot] = 0u; __threadfence(); }
}

template <int DIM, int INTER>
__device__ __forceinline__ void expert_chain_p2_body(const ChainArgs2& B)
{
    const ChainArgs& A = B.a;
    constexpr int I2 = DIM / 8;
    __shared__ float red[WARPS * 8];
    __shared__ int s_loc[MAXSLOT]; __shared__ int s_nloc;
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int i = 0; i < A.topk; ++i) { const int e = A.eid[i] - A.shard_start; if (e >= 0 && e < A.shard_n) { s_loc[n] = e; ++n; } }
        s_nloc = n;
    }
    __syncthreads();
    const int nloc = s_nloc;
    const int item = blockIdx.x;
    float c[4];
    if (item < I2) {
        const int n0 = item * 8;
        float tot = 0.f;
        for (int slot = 0; slot < nloc; ++slot) {
            const int e = s_loc[slot];
            fp4_rows8_mma(A.w2 + (long long)e * A.w2_stride + (long long)n0 * (INTER / 2), INTER / 2,
                          A.s2 + (long long)e * A.s2_stride + (long long)n0 * (INTER / 32), INTER / 32, B.h + slot * INTER, INTER, warp, c);
            tot += rows8_reduce(red, c, warp);
        }
        if (threadIdx.x < 8) A.part[n0 + threadIdx.x] = tot;
    } else if (A.has_shared) {
        const int n0 = (item - I2) * 8;
        fp8_rows8_mma(A.sh_w2 + (long long)n0 * INTER, INTER, A.sh_s2 + (long long)(n0 >> 5) * A.sh_s2_cols, A.sh_s2_cols, B.h + nloc * INTER, INTER, warp, c);
        const float v = rows8_reduce(red, c, warp);
        if (threadIdx.x < 8) A.ys[n0 + threadIdx.x] = __float2bfloat16(v);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
expert_chain_p1_5120_2304(ChainArgs2 B) { expert_chain_p1_body<5120, 2304>(B); }
extern "C" __global__ void __launch_bounds__(THREADS)
expert_chain_p2_5120_2304(ChainArgs2 B) { expert_chain_p2_body<5120, 2304>(B); }
