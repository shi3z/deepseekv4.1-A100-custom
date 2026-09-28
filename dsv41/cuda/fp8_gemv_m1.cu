// Dedicated M = 1 GEMV for FP8 (e4m3) weights with [32 x 32] E8M0 block scales (Strategy A of the dense
// decode experiment): y[n] = sum_k x[k] * W[n, k] * 2^(S[n/32, k/32] - 127), bf16 x, fp32 accumulation,
// bf16 (and optionally fp32) output. No tensor cores: at one row the mma path uses 1/16 of the tile and the
// kernel is a pure weight stream, so this kernel only cares about bytes in flight and launch/tail overhead.
//
// Weight bytes are the dev layout (w8.PERM_K: within every 16 k, stored byte 4s+j holds k 2s+j for j < 2 and
// k 2s+8+j-2 for j >= 2), so decoded word j (bytes 4j..4j+3) pairs with x words j (k 2j, 2j+1) and j+4
// (k 2j+8, 2j+9) of the same 16-k group; a dot product does not care about the order of its terms.
//
// Block = 8 warps. KW warps share one output row (each a K/KW slice, reduced through shared memory), so a
// block produces 8/KW rows; the launcher picks KW so that the grid has >= ~1000 blocks whatever N. A lane
// streams 16-byte weight chunks (k = 16 * chunk) with U loads in flight, decodes e4m3 -> bf16 exactly
// (fp8_tc.cu's bit placement) and accumulates in fp32; the block scale is applied once per 16-k chunk
// (every chunk lies inside one 32-k scale block).
//
// group_cols > 0 (block-diagonal o-projection): row n reads x row n / group_cols (x is [N/group_cols, K]).
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// 4 e4m3 bytes -> two bf16x2 words with exponent field 0000eeee (value * 2^-120 of the e4m3 number)
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    uint32_t t0 = prmt(w, 0, 0x1404);
    uint32_t t1 = prmt(w, 0, 0x3424);
    lo = ((t0 >> 4) & 0x07F007F0u) | (t0 & 0x80008000u);
    hi = ((t1 >> 4) & 0x07F007F0u) | (t1 & 0x80008000u);
}

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u));
}

// dot of 16 weight bytes (one k chunk, unscaled bf16 placement) with the matching 16 x values (two uint4 of bf16)
__device__ __forceinline__ float chunk_dot(const uint4 w, const uint4 xa, const uint4 xb) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        e4m3x4_to_bf16(ww[j], lo, hi);
        const float2 a = bf16x2_to_float2(lo), b = bf16x2_to_float2(hi);
        const float2 p = bf16x2_to_float2(xv[j]), q = bf16x2_to_float2(xv[j + 4]);
        acc0 = fmaf(a.x, p.x, acc0); acc1 = fmaf(a.y, p.y, acc1);
        acc0 = fmaf(b.x, q.x, acc0); acc1 = fmaf(b.y, q.y, acc1);
    }
    return acc0 + acc1;
}

template <int KW, int U>
__device__ __forceinline__ void fp8_gemv_m1_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N, int K, int Kc, int group_cols,
        __nv_bfloat16* __restrict__ y, float* __restrict__ yf)
{
    constexpr int WARPS = 8, RB = WARPS / KW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = blockIdx.x * RB + warp / KW;
    const int kw = warp % KW;
    __shared__ float red[WARPS];
    float acc = 0.f;
    if (n < N) {
        const int xr = group_cols > 0 ? n / group_cols : 0;
        const __nv_bfloat16* xrow = X + (long long)xr * ldx;
        const uint8_t* wrow = W + (long long)n * K;
        const uint8_t* srow = S + (long long)(n >> 5) * Kc;
        const int chunks = K / 16;                    // 16-byte chunks per row
        const int per = chunks / KW;                  // this warp's chunk range (K % (16 * KW) == 0)
        const int c0 = kw * per;
        // decoded weights carry 2^-120: fold it into the scale once
        float acc_u[U];
#pragma unroll
        for (int u = 0; u < U; ++u) acc_u[u] = 0.f;
        for (int base = c0 + lane; base < c0 + per; base += 32 * U) {
            uint4 wv[U];
            int sb[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {              // U independent 16-byte weight loads in flight per lane
                const int c = base + 32 * u;
                const bool ok = c < c0 + per;
                wv[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)c * 16)) : make_uint4(0, 0, 0, 0);
                sb[u] = ok ? __ldg(srow + (c >> 1)) : 0;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = base + 32 * u;
                if (c < c0 + per) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(xrow + c * 16);
                    const uint4 xb = *reinterpret_cast<const uint4*>(xrow + c * 16 + 8);
                    const float d = chunk_dot(wv[u], xa, xb);
                    // 2^(s - 127) * 2^120 = 2^(s - 7): fp32 exponent field s + 120 (as fp8_tc.cu's bf16 factor)
                    const float sc = (sb[u] + 120 > 0 && sb[u] + 120 < 255) ? __uint_as_float((uint32_t)(sb[u] + 120) << 23) : 0.f;
                    acc_u[u] = fmaf(d, sc, acc_u[u]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc += acc_u[u];
    }
    // warp reduce
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x < RB) {
        const int nn = blockIdx.x * RB + threadIdx.x;
        if (nn < N) {
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < KW; ++j) s += red[threadIdx.x * KW + j];
            if (y) y[nn] = __float2bfloat16(s);
            if (yf) yf[nn] = s;
        }
    }
}

#define DEF(KW, U) \
extern "C" __global__ void __launch_bounds__(256) \
fp8_gemv_m1_k##KW##u##U(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
                        int N, int K, int Kc, int group_cols, __nv_bfloat16* __restrict__ y, float* __restrict__ yf) \
{ fp8_gemv_m1_body<KW, U>(X, ldx, W, S, N, K, Kc, group_cols, y, yf); }

DEF(1, 4) DEF(2, 4) DEF(4, 4) DEF(8, 4)
DEF(1, 8) DEF(2, 8) DEF(4, 8) DEF(8, 8)
DEF(1, 2) DEF(2, 2) DEF(4, 2) DEF(8, 2)

// ---------------------------------------------------------------------------------------------------------------
// Variant with the activation prologue fused (DSV41_FUSED_DENSE_CHAIN): the raw input vector x_raw (bf16 [K], the
// previous GEMV's output) is normalized and/or fp8-fake-quantized by every block into shared memory before its
// rows are streamed, replacing the separate `norm_quant` (q_norm before wq_b) / `fake_quant` (before wo_b) launch.
//   MODE 1: y = bf16( x * rsqrt(mean(x^2) + eps) * w );  then per-32 fp8 fake quant   (fused2._norm_quant_kernel)
//   MODE 2: per-32 fp8 fake quant only                                                (fused._fake_quant_kernel, MODE 0)
// Both round exactly as the Triton kernels: amax >= 1e-4, s = 2^ceil(log2(amax / 448)), e4m3 round-to-nearest-even
// of clamp(x / s, +-448), y = q * s -> bf16 (s and the e4m3 ulp are powers of two, so x * (1/s) == x / s exactly).
// K <= 2048 for MODE 1 / 8192 for MODE 2 (K * 2 bytes of dynamic shared memory).
// The prologue is replicated by every block: every lane loads all of its elements up front (one dependent memory
// round trip) and the first U weight loads of every warp are issued before the prologue so the DRAM stream starts
// while the block is still rounding. MODE 1 spreads the work over all 8 warps (element-parallel, amax by shuffles),
// MODE 2 (K up to 8192) uses one lane per group; each layout measured fastest for its shape (wq_b / wo_b).
__device__ __forceinline__ float pow2_ceil_log2_f(float a) {
    const int bits = __float_as_int(a);
    int e = ((bits >> 23) & 0xFF) - 127;
    if (bits & 0x7FFFFF) e += 1;
    return __int_as_float((e + 127) << 23);
}
__device__ __forceinline__ float round_e4m3_f(float v) {
    float a = fabsf(v);
    int e = ((__float_as_int(a) >> 23) & 0xFF) - 127;
    e = max(e, -6);
    const float ulp = __int_as_float((e - 3 + 127) << 23), inv_ulp = __int_as_float((3 - e + 127) << 23);
    float r = rintf(a * inv_ulp) * ulp;
    r = fminf(r, 448.f);
    return v < 0.f ? -r : r;
}

template <int KW, int U, int MODE>
__device__ __forceinline__ void fp8_gemv_m1_pre_body(const __nv_bfloat16* __restrict__ Xraw, const __nv_bfloat16* __restrict__ NW, float eps,
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N, int K, int Kc, __nv_bfloat16* __restrict__ y, float* __restrict__ yf)
{
    constexpr int WARPS = 8, RB = WARPS / KW;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [K] the prologue's output
    __shared__ float red[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = blockIdx.x * RB + warp / KW;
    const int kw = warp % KW;
    const uint8_t* wrow = W + (long long)n * K;
    const uint8_t* srow = S + (long long)(n >> 5) * Kc;
    const int chunks = K / 16, per = chunks / KW, c0 = kw * per, cend = c0 + per;
    // ---- first U weight loads in flight before the prologue
    uint4 wv[U]; int sb[U];
    int base = c0 + lane;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int c = base + 32 * u;
        const bool ok = n < N && c < cend;
        wv[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)c * 16)) : make_uint4(0, 0, 0, 0);
        sb[u] = ok ? __ldg(srow + (c >> 1)) : 0;
    }
    // ---- prologue. It is replicated by every block and the per-row GEMV work is small (K/16/32 chunks per lane), so
    // it is issue-bound, not latency-bound (ncu, wq_b shape: 1.65M -> 4.27M instructions): keep it short.
    if (MODE == 1) {
        // element-parallel over all 256 lanes: warp w handles 32-groups w, w+8, ..., lane l the group's element l
        // (one coalesced 64-byte load per warp per group, all loads in flight at once, amax by 5 shuffles per group
        // with the GB chains interleaved). K <= 32 * 8 * GB = 2048. The block-wide sum of squares comes first.
        constexpr int GB = 8;
        const int ngroups = K / 32;
        float v[GB], nw[GB];
#pragma unroll
        for (int i = 0; i < GB; ++i) {
            const int g = warp + WARPS * i;
            const bool ok = g < ngroups;
            v[i] = ok ? __bfloat162float(Xraw[g * 32 + lane]) : 0.f;
            nw[i] = ok ? __bfloat162float(NW[g * 32 + lane]) : 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < GB; ++i) ss = fmaf(v[i], v[i], ss);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) tot += red[w];
        const float rs = 1.f / sqrtf(tot / (float)K + eps);
        float amax[GB];
#pragma unroll
        for (int i = 0; i < GB; ++i) { v[i] = __bfloat162float(__float2bfloat16(v[i] * rs * nw[i])); amax[i] = fabsf(v[i]); }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (int i = 0; i < GB; ++i) amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], o));
        }
#pragma unroll
        for (int i = 0; i < GB; ++i) {
            const int g = warp + WARPS * i;
            if (g < ngroups) {
                const float s = pow2_ceil_log2_f(__fdiv_rn(fmaxf(amax[i], 1e-4f), 448.f));
                const float q = round_e4m3_f(fminf(fmaxf(v[i] * (1.f / s), -448.f), 448.f));   // 1/s exact (power of two)
                xs[g * 32 + lane] = __float2bfloat16(q * s);
            }
        }
    } else {
        // one lane per 32-group (K <= 8192): four independent 16-byte loads, amax in registers, no shuffles
        const int g = threadIdx.x;
        if (g < K / 32) {
            const uint4* src = reinterpret_cast<const uint4*>(Xraw + g * 32);
            uint4 raw[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) raw[j] = __ldg(src + j);
            float v[32];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t rr[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) { const float2 f = bf16x2_to_float2(rr[i]); v[j * 8 + i * 2] = f.x; v[j * 8 + i * 2 + 1] = f.y; }
            }
            float amax = 1e-4f;
#pragma unroll
            for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
            const float s = pow2_ceil_log2_f(__fdiv_rn(amax, 448.f));
            const float inv_s = 1.f / s;   // power of two: exact
            uint4* dst = reinterpret_cast<uint4*>(xs + g * 32);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float q0 = round_e4m3_f(fminf(fmaxf(v[j * 8 + i * 2] * inv_s, -448.f), 448.f));
                    const float q1 = round_e4m3_f(fminf(fmaxf(v[j * 8 + i * 2 + 1] * inv_s, -448.f), 448.f));
                    const __nv_bfloat162 b = __floats2bfloat162_rn(q0 * s, q1 * s);
                    o[i] = *reinterpret_cast<const uint32_t*>(&b);
                }
                dst[j] = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
    }
    __syncthreads();
    // ---- GEMV from shared memory (fp8_gemv_m1_body's loop, software-pipelined by one step)
    float acc = 0.f;
    if (n < N) {
        float acc_u[U];
#pragma unroll
        for (int u = 0; u < U; ++u) acc_u[u] = 0.f;
        for (; base < cend; base += 32 * U) {
            uint4 wn[U]; int sn[U];
            const int nb = base + 32 * U;
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = nb + 32 * u;
                const bool ok = c < cend;
                wn[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)c * 16)) : make_uint4(0, 0, 0, 0);
                sn[u] = ok ? __ldg(srow + (c >> 1)) : 0;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = base + 32 * u;
                if (c < cend) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(xs + c * 16);
                    const uint4 xb = *reinterpret_cast<const uint4*>(xs + c * 16 + 8);
                    const float d = chunk_dot(wv[u], xa, xb);
                    const float sc = (sb[u] + 120 > 0 && sb[u] + 120 < 255) ? __uint_as_float((uint32_t)(sb[u] + 120) << 23) : 0.f;
                    acc_u[u] = fmaf(d, sc, acc_u[u]);
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) { wv[u] = wn[u]; sb[u] = sn[u]; }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc += acc_u[u];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __syncthreads();   // red[] was the prologue's reduction buffer
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x < RB) {
        const int nn = blockIdx.x * RB + threadIdx.x;
        if (nn < N) {
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < KW; ++j) s += red[threadIdx.x * KW + j];
            if (y) y[nn] = __float2bfloat16(s);
            if (yf) yf[nn] = s;
        }
    }
}

#define DEFP(KW, U, MODE) \
extern "C" __global__ void __launch_bounds__(256) \
fp8_gemv_m1_pre##MODE##_k##KW##u##U(const __nv_bfloat16* __restrict__ Xraw, const __nv_bfloat16* __restrict__ NW, float eps, \
                        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N, int K, int Kc, __nv_bfloat16* __restrict__ y, float* __restrict__ yf) \
{ fp8_gemv_m1_pre_body<KW, U, MODE>(Xraw, NW, eps, W, S, N, K, Kc, y, yf); }

DEFP(1, 4, 1) DEFP(1, 2, 1) DEFP(2, 2, 1) DEFP(4, 4, 1)
DEFP(1, 4, 2) DEFP(1, 2, 2) DEFP(2, 2, 2) DEFP(4, 4, 2)
