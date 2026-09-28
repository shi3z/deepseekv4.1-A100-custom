// Device-side messaging between GPUs of one process (peer access enabled): copies by kernel stores into
// peer memory, flag signal / wait kernels. All capturable in CUDA graphs; used by the expert-parallel
// runtime (dsv41/ep.py) so that a whole token runs as one graph per GPU without host round trips.
#include <stdint.h>

extern "C" __global__ void p2p_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, int n16) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) dst[i] = src[i];
}

// for every row b: copy row `row16` uint4 from src[b] into dst_base + seq[b] * bstride16 + row_idx[b] * row16
extern "C" __global__ void p2p_copy_row(uint4* __restrict__ dst_base, const long long* __restrict__ row_idx, const uint4* __restrict__ src,
                                       int row16, int nb, long long bstride16, const long long* __restrict__ seq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row16 * nb) return;
    int b = i / row16, j = i - b * row16;
    dst_base[seq[b] * bstride16 + row_idx[b] * row16 + j] = src[i];
}

// per group g (< groups): dst[g * dst_stride + i] = sum over `rows` rows of src[(g * rows + r) * n + i]
extern "C" __global__ void p2p_sum_rows(float* __restrict__ dst, const float* __restrict__ src, int rows, int n, int groups, long long dst_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (i >= n || g >= groups) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += src[((long long)g * rows + r) * n + i];
    dst[(long long)g * dst_stride + i] = acc;
}

// after the preceding kernels of this stream have completed: publish `value` (read from seq_ptr) to up to 8 peer flags
extern "C" __global__ void p2p_signal(int** flags, int n, const int* seq_ptr) {
    __threadfence_system();
    int v = *seq_ptr;
    for (int i = 0; i < n; ++i) { volatile int* f = (volatile int*)flags[i]; *f = v; }
    __threadfence_system();
}

// spin until every one of n local flags has reached the value in seq_ptr
extern "C" __global__ void p2p_wait(volatile int* flags, int n, const int* seq_ptr) {
    int v = *seq_ptr;
    for (int i = 0; i < n; ++i) { while (flags[i] < v) { } }
    __threadfence_system();
}

// same, but only the flags whose mask entry is non-zero (device-computed: peers this token does not need)
extern "C" __global__ void p2p_wait_masked(volatile int* flags, int n, const int* seq_ptr, const int* mask) {
    int v = *seq_ptr;
    for (int i = 0; i < n; ++i) { if (mask[i]) { while (flags[i] < v) { } } }
    __threadfence_system();
}

extern "C" __global__ void p2p_seq_bump(int* seq_ptr) { *seq_ptr += 1; }

// one launch: copy `n16` uint4 of src into up to 8 destinations (peer inboxes); the last block to finish
// signals their flags (counter reset for the next launch)
extern "C" __global__ void p2p_multicast(uint4** dsts, int ndst, const uint4* __restrict__ src, int n16,
                                        int** flags, const int* seq_ptr, unsigned int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) {
        uint4 v = src[i];
        for (int d = 0; d < ndst; ++d) dsts[d][i] = v;
    }
    if (flags == nullptr) return;
    __threadfence_system();
    __syncthreads();
    __shared__ unsigned int last;
    if (threadIdx.x == 0) last = (atomicInc(counter, gridDim.x - 1) == gridDim.x - 1);
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence_system();
        int v = *seq_ptr;
        for (int d = 0; d < ndst; ++d) { volatile int* f = (volatile int*)flags[d]; *f = v; }
        __threadfence_system();
    }
}

// timeline stamp: the global nanosecond timer (same clock on every GPU of the node)
extern "C" __global__ void p2p_stamp(long long* dst) {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *dst = t;
}
