// Row-wise top-k index selection for the decode indexer (results/dense_m1_report.md, rank 3):
//   for every row b: the indices of the k largest of score[b, 0..n), sorted ascending, as int32 (n <= 1 << 20).
// One block of 1024 threads per row, radix select on the 32-bit monotone key (4 passes of 8-bit digits, MSB first,
// block histograms in shared memory), then the selected indices are compacted into shared memory and bitonic-sorted.
// Ties at the threshold key are resolved first-come (like torch.topk: any subset). Replaces gatherTopK (43 us)
// + radixSortKVInPlace (36 us) + ~10 glue kernels by one launch; k <= 1024, n arbitrary.
#include <stdint.h>

#define TPB 1024
#define KMAX 1024

__device__ __forceinline__ uint32_t key_of(float f) {
    // monotone map: larger float -> larger uint; then invert so that the LARGEST scores have the SMALLEST keys
    uint32_t u = __float_as_uint(f);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ~u;
}

extern "C" __global__ void __launch_bounds__(TPB)
topk_select(const float* __restrict__ score, int n, int k, int ld, int* __restrict__ out)
{
    const int b = blockIdx.x;
    const float* row = score + (long long)b * ld;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_remaining, s_count, s_digit;
    __shared__ int sel[KMAX];
    const int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_remaining = (unsigned)k; }
    __syncthreads();
    // ---- radix select: find the k-th smallest key (= k-th largest score)
    uint32_t prefix = 0u, mask = 0u;  // matched high digits and their mask
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0u;
        __syncthreads();
        for (int i = tid; i < n; i += TPB) {
            const uint32_t kk = key_of(row[i]);
            if ((kk & mask) == prefix) atomicAdd(&hist[(kk >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {  // scan the 256 bins for the digit that reaches the remaining count
            unsigned int acc = 0u, rem = s_remaining;
            int dsel = 255;
            for (int dgt = 0; dgt < 256; ++dgt) {
                const unsigned int c = hist[dgt];
                if (acc + c >= rem) { dsel = dgt; s_remaining = rem - acc; break; }
                acc += c;
            }
            s_digit = (unsigned)dsel;
        }
        __syncthreads();
        prefix |= (s_digit << shift);
        mask |= (0xFFu << shift);
        __syncthreads();
    }
    const uint32_t thr = prefix;          // the k-th smallest key; s_remaining = how many == thr to take
    if (tid == 0) s_count = 0u;
    __syncthreads();
    // ---- gather: all keys < thr, plus the first s_remaining keys == thr
    for (int i = tid; i < n; i += TPB) {
        const uint32_t kk = key_of(row[i]);
        if (kk < thr) {
            const unsigned int p = atomicAdd(&s_count, 1u);
            if (p < (unsigned)k) sel[p] = i;
        }
    }
    __syncthreads();
    for (int i = tid; i < n; i += TPB) {
        const uint32_t kk = key_of(row[i]);
        if (kk == thr) {
            const unsigned int p = atomicAdd(&s_count, 1u);
            if (p < (unsigned)k) sel[p] = i;
        }
    }
    __syncthreads();
    // ---- bitonic sort ascending of k entries (k padded to KMAX with INT_MAX)
    for (int i = tid; i < KMAX; i += TPB) if (i >= k) sel[i] = 0x7FFFFFFF;
    __syncthreads();
    for (int size = 2; size <= KMAX; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < KMAX; i += TPB) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = ((i & size) == 0);
                    const int a = sel[i], c = sel[j];
                    if ((a > c) == up) { sel[i] = c; sel[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += TPB) out[(long long)b * k + i] = sel[i];
}
