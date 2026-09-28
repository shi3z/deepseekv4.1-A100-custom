// Skinny GEMM with FP8 (e4m3) weights and [32 x 32] E8M0 block scales on Ampere tensor cores:
//   out[m, n] = sum_k x[m, k] * (W[n, k] * 2^(S[n/32, k/32] - 127)),   M <= 16 rows of x (bf16)
//
// A100 has no FP8 units, but e4m3 -> bf16 is exact with two integer ops per pair plus one bf16x2
// multiply: place the byte [s eeee mmm] as s<<15 | e<<7 | m<<4 (a bf16 with exponent 0000eeee and
// mantissa mmm0000) and multiply by 2^120: normals become 2^(e-7)(1+m/8), and the bf16 subnormal
// 16m * 2^-133 becomes m * 2^-9, which is exactly the e4m3 subnormal. The block scale is folded into
// the same multiply (factor 2^(s-7)), so the products fed to mma.m16n8k16 are the exactly dequantized
// weights and the accumulation is fp32, like the bf16 cuBLAS path it replaces at half the bytes.
//
// Layout trick: lane (g = lane/4, t = lane%4) of a warp owns row n = n0 + g and 16 consecutive k
// (K0 + 16t .. +15) per 64-wide k step, loaded as one 16-byte transaction. The k index inside the mma
// tile is a permutation of the physical k; x is loaded with the same permutation, and a dot product
// does not care about the order of its terms. The weight bytes are stored k-permuted within every
// 16-k group (w8.PERM_K, shared with fp8_tcg.cu's ldmatrix path): byte 4s+j of the group holds k 2s+j
// (j < 2) or 2s+8+j-2, so mma step s pairs decoded words 2s, 2s+1 with x words s and s+4.
//
// grid: (N / 8 / WARPS, splits). Each warp: 8 output columns, k range [ks, ks + k_per_split).
// out: fp32 partials [splits, M, N] (row stride ldo = N); M rows < 16 are zero-padded on the fly.
// group_cols > 0 (the block-diagonal o-projection): the x row used for column n is n / group_cols,
// out row is 0 and only that row of the tile is kept.
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 4

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// 4 e4m3 bytes -> two bf16x2 words (unscaled: exponent 0000eeee), lo = bytes 0,1; hi = bytes 2,3
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    uint32_t t0 = prmt(w, 0, 0x1404);  // [0, b0, 0, b1] -> b0 at bits 15:8, b1 at 31:24
    uint32_t t1 = prmt(w, 0, 0x3424);
    lo = ((t0 >> 4) & 0x07F007F0u) | (t0 & 0x80008000u);
    hi = ((t1 >> 4) & 0x07F007F0u) | (t1 & 0x80008000u);
}

__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));  // sm_80 has no mul.bf16x2, fma with 0 is exact
    return r;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <bool M8>
__device__ __forceinline__ void fp8_gemm_tc_body(const __nv_bfloat16* __restrict__ X, int ldx, int M,
            const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N, int K, int Kc,
            float* __restrict__ out, int ldo, int k_per_split, int group_cols,
            __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * WARPS + warp) * 8;
    if (n0 >= N) return;
    const int n = n0 + g;
    const int ks = blockIdx.y * k_per_split;
    const int ke = min(K, ks + k_per_split);
    // lane's first 16 weight bytes and the step per 64 k: row-major, or the tiled layout [N/16][K/64][16 rows][64 B]
    // (w8.tile: a warp's 8 rows x 64 B per 64-k step are one contiguous 512-byte range)
    const uint8_t* wrow = tiled ? W + (long long)(n >> 4) * (K / 64) * 1024 + (n & 15) * 64 + 16 * t : W + (long long)n * K + 16 * t;
    const int wstep = tiled ? 1024 : 64;
    const uint8_t* srow = S + (long long)(n >> 5) * Kc;
    int xr0, xr1;
    if (group_cols > 0) {  // block-diagonal: row b of the output uses x row b * xgroups + (column group)
        const int xgroups = N / group_cols;
        xr0 = g < M ? g * xgroups + n0 / group_cols : -1;
        xr1 = (!M8 && g + 8 < M) ? (g + 8) * xgroups + n0 / group_cols : -1;
    } else { xr0 = g < M ? g : -1; xr1 = (!M8 && g + 8 < M) ? g + 8 : -1; }
    const __nv_bfloat16* x0 = xr0 >= 0 ? X + (long long)xr0 * ldx : nullptr;
    const __nv_bfloat16* x1 = xr1 >= 0 ? X + (long long)xr1 * ldx : nullptr;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    // 128 k per iteration (two 64-k halves, two 16-byte weight loads per lane), next iteration's loads
    // issued before this iteration's math: 4 weight loads in flight per lane
    uint4 wv[2], xa[2][2], xb[2][2];
    int sb[2];
    const uint4 z4 = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int h = 0; h < 2; ++h) { wv[h] = z4; xa[h][0] = xa[h][1] = xb[h][0] = xb[h][1] = z4; sb[h] = 0; }
    auto load = [&](int kk) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kb = kk + 64 * h + 16 * t;
            const bool ok = kb < ke;
            wv[h] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (kb >> 6) * wstep)) : z4;
            sb[h] = ok ? __ldg(srow + (kb >> 5)) : 0;
            if (x0 && ok) { xa[h][0] = *reinterpret_cast<const uint4*>(x0 + kb); xa[h][1] = *reinterpret_cast<const uint4*>(x0 + kb + 8); }
            if (!M8) { if (x1 && ok) { xb[h][0] = *reinterpret_cast<const uint4*>(x1 + kb); xb[h][1] = *reinterpret_cast<const uint4*>(x1 + kb + 8); } }
        }
    };
    int k0 = ks;
    if (k0 < ke) load(k0);
    for (; k0 < ke; k0 += 128) {
        uint4 wc[2] = {wv[0], wv[1]};
        int sc[2] = {sb[0], sb[1]};
        uint4 ya[2][2] = {{xa[0][0], xa[0][1]}, {xa[1][0], xa[1][1]}};
        uint4 yb[2][2] = {{xb[0][0], xb[0][1]}, {xb[1][0], xb[1][1]}};
        if (k0 + 128 < ke) load(k0 + 128);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t fb = (sc[h] + 120 > 0) ? ((uint32_t)(sc[h] + 120) << 7) : 0u;
            const uint32_t f2 = fb | (fb << 16);
            uint32_t b[8];
            e4m3x4_to_bf16(wc[h].x, b[0], b[1]);
            e4m3x4_to_bf16(wc[h].y, b[2], b[3]);
            e4m3x4_to_bf16(wc[h].z, b[4], b[5]);
            e4m3x4_to_bf16(wc[h].w, b[6], b[7]);
#pragma unroll
            for (int i = 0; i < 8; ++i) b[i] = hmul2_bf16(b[i], f2);
            const uint32_t xav[8] = {ya[h][0].x, ya[h][0].y, ya[h][0].z, ya[h][0].w, ya[h][1].x, ya[h][1].y, ya[h][1].z, ya[h][1].w};
            const uint32_t xbv[8] = {yb[h][0].x, yb[h][0].y, yb[h][0].z, yb[h][0].w, yb[h][1].x, yb[h][1].y, yb[h][1].z, yb[h][1].w};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t bf[2] = {b[2 * s], b[2 * s + 1]};
                // permuted weight layout (w8.PERM_K): decoded words 2s, 2s+1 hold k (2s, 2s+1) and (2s+8, 2s+9) of the
                // lane's 16-k group = x words s and s + 4
                const uint32_t af[4] = {xav[s], M8 ? 0u : xbv[s], xav[s + 4], M8 ? 0u : xbv[s + 4]};
                mma16816(c, af, bf);
            }
        }
    }
    float* o = out + (long long)blockIdx.y * M * ldo;
    if (g < M) { o[(long long)g * ldo + n0 + 2 * t] = c[0]; o[(long long)g * ldo + n0 + 2 * t + 1] = c[1]; }
    if (!M8 && g + 8 < M) { o[(long long)(g + 8) * ldo + n0 + 2 * t] = c[2]; o[(long long)(g + 8) * ldo + n0 + 2 * t + 1] = c[3]; }
    if (y == nullptr) return;
    // epilogue: the last warp to finish this 8-column tile (over all splits) sums the partials in split
    // order and writes bf16. Counter per tile, reset for the next launch.
    __threadfence();
    __shared__ unsigned int last[WARPS];
    if (lane == 0) last[warp] = (splits == 1) ? 1u : (atomicInc(&counters[n0 >> 3], (unsigned)splits - 1) == (unsigned)(splits - 1));
    __syncwarp();
    if (!last[warp]) return;
    __threadfence();
    const int rows = M;
    for (int i = lane; i < rows * 8; i += 32) {
        const int r = i >> 3, col = n0 + (i & 7);
        float acc = 0.f;
        for (int sp = 0; sp < splits; ++sp) acc += out[((long long)sp * M + r) * ldo + col];
        y[(long long)r * N + col] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp8_gemm_tc8(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
             int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols,
             __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{ fp8_gemm_tc_body<true>(X, ldx, M, W, S, N, K, Kc, out, ldo, k_per_split, group_cols, y, counters, splits, tiled); }

extern "C" __global__ void __launch_bounds__(WARPS * 32)
fp8_gemm_tc16(const __nv_bfloat16* __restrict__ X, int ldx, int M, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
              int N, int K, int Kc, float* __restrict__ out, int ldo, int k_per_split, int group_cols,
              __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ counters, int splits, int tiled)
{ fp8_gemm_tc_body<false>(X, ldx, M, W, S, N, K, Kc, out, ldo, k_per_split, group_cols, y, counters, splits, tiled); }
