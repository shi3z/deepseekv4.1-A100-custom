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
