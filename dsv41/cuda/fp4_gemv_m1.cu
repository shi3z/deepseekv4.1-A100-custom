// Dedicated one-token-per-group FP4 expert GEMV (rank-1 experiment of results/dense_m1_report.md):
//   out[p, n] = sum_k x[tok, k] * W[e, n, k] * 2^(S[e, n, k/32] - 127)     for every group g with exactly ONE pair p
// Same packed E2M1 weights ([E, N, K/2] uint8, dev layout) and E8M0 scales ([E, N, K/32]) as fp4_tc.cu, same exact
// register decode (E2M1 placed as s<<15 | e<<7 | m<<6 times 2^126, scale folded as the bf16 factor 2^(s-1)), same
// 8-k permuted x (cukern.permute_x: the 4 decoded bf16x2 words of a 32-bit weight word hold the k pairs
// (0,4) (2,6) (1,5) (3,7), which is exactly the order of 8 consecutive permuted x values), fp32 accumulation.
//
// The tensor-core kernel gives a one-token group (N / 32) blocks of 4 warps: at 1-2 local experts of the 6 that
// is 144-288 blocks streaming 12-24 MiB (0.4-0.6 TB/s measured). Here a 16-byte weight chunk (32 k, one scale
// byte) is one lane's unit, KW warps share a row (K/KW each, reduced through shared memory), 8 warps per block,
// so an expert yields N * KW / 8 blocks. Groups with more than one token are left to fp4_tc.cu / fp4_tcw.cu
// (the launcher passes them min_tok = 2). zero_out: a one-token group whose expert is not in this shard has its
// output row zeroed (rows of other shards must be zero for the partial sum).
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t bf16x2_fma0(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}

__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t f2, uint32_t* o) {
    uint32_t a = ((w & 0x00070007u) << 6) | ((w & 0x00080008u) << 12);
    uint32_t b = ((w & 0x07000700u) >> 2) | ((w & 0x08000800u) << 4);
    uint32_t c = ((w & 0x00700070u) << 2) | ((w & 0x00800080u) << 8);
    uint32_t d = ((w & 0x70007000u) >> 6) | (w & 0x80008000u);
    o[0] = bf16x2_fma0(a, f2);
    o[1] = bf16x2_fma0(b, f2);
    o[2] = bf16x2_fma0(c, f2);
    o[3] = bf16x2_fma0(d, f2);
}

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u));
}

// one 16-byte chunk (32 k) against the matching 32 permuted x values (four uint4); f2 = folded scale
__device__ __forceinline__ float chunk_dot(const uint4 w, const uint4* xq, uint32_t f2) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t o[4];
        e2m1x8_to_bf16(ww[j], f2, o);
        const uint32_t xv[4] = {xq[j].x, xq[j].y, xq[j].z, xq[j].w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 wv = bf16x2_to_float2(o[i]), xx = bf16x2_to_float2(xv[i]);
            acc0 = fmaf(wv.x, xx.x, acc0);
            acc1 = fmaf(wv.y, xx.y, acc1);
        }
    }
    return acc0 + acc1;
}

template <int KW, int U>
__device__ __forceinline__ void fp4_gemv_m1_body(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint8_t* __restrict__ W, long long stride_we, const uint8_t* __restrict__ S, long long stride_se,
        const int* __restrict__ grp_expert, const int* __restrict__ grp_start, const int* __restrict__ pair_tok,
        float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out)
{
    constexpr int WARPS = 8, RB = WARPS / KW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = blockIdx.y;
    const int p0 = grp_start[grp], p1 = grp_start[grp + 1];
    if (p1 - p0 != 1) return;  // only one-token groups here
    const int n = blockIdx.x * RB + warp / KW;
    const int e = grp_expert[grp] - shard_start;
    if (e < 0 || e >= shard_n) {
        if (zero_out && threadIdx.x < RB) {
            const int nn = blockIdx.x * RB + threadIdx.x;
            if (nn < N) out[(long long)p0 * ldo + nn] = 0.f;
        }
        return;
    }
    __shared__ float red[WARPS];
    float acc = 0.f;
    if (n < N) {
        const int kw = warp % KW;
        const __nv_bfloat16* xrow = X + (long long)pair_tok[p0] * ldx;
        const uint8_t* wrow = W + (long long)e * stride_we + (long long)n * (K / 2);
        const uint8_t* srow = S + (long long)e * stride_se + (long long)n * (K / 32);
        const int chunks = K / 32;
        const int per = chunks / KW;
        const int c0 = kw * per;
        float acc_u[U];
#pragma unroll
        for (int u = 0; u < U; ++u) acc_u[u] = 0.f;
        for (int base = c0 + lane; base < c0 + per; base += 32 * U) {
            uint4 wv[U];
            int sb[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = base + 32 * u;
                const bool ok = c < c0 + per;
                wv[u] = ok ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)c * 16)) : make_uint4(0, 0, 0, 0);
                sb[u] = ok ? __ldg(srow + c) : 0;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = base + 32 * u;
                if (c < c0 + per) {
                    const uint4* xq = reinterpret_cast<const uint4*>(xrow + c * 32);
                    const uint4 xv[4] = {xq[0], xq[1], xq[2], xq[3]};
                    const uint32_t fb = (uint32_t)(sb[u] + 126) << 7;  // 2^(s-1) as bf16
                    const uint32_t f2 = fb | (fb << 16);
                    acc_u[u] += chunk_dot(wv[u], xv, f2);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc += acc_u[u];
    }
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
            out[(long long)p0 * ldo + nn] = s;
        }
    }
}

#define DEF(KW, U) \
extern "C" __global__ void __launch_bounds__(256) \
fp4_gemv_m1_k##KW##u##U(const __nv_bfloat16* __restrict__ X, int ldx, const uint8_t* __restrict__ W, long long stride_we, \
                        const uint8_t* __restrict__ S, long long stride_se, const int* __restrict__ grp_expert, const int* __restrict__ grp_start, \
                        const int* __restrict__ pair_tok, float* __restrict__ out, int ldo, int N, int K, int shard_start, int shard_n, int zero_out) \
{ fp4_gemv_m1_body<KW, U>(X, ldx, W, stride_we, S, stride_se, grp_expert, grp_start, pair_tok, out, ldo, N, K, shard_start, shard_n, zero_out); }

DEF(1, 2) DEF(2, 2) DEF(4, 2) DEF(8, 2)
DEF(1, 4) DEF(2, 4) DEF(4, 4) DEF(8, 4)
