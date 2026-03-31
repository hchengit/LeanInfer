// LeanInfer Phase 3 — Fused FFN CUDA Kernels
//
// Cooperative fusion: RMSNorm + gate/up projections + SiLU*mul in a single
// kernel launch, keeping the normalized hidden state in shared memory to
// eliminate intermediate global-memory round-trips.
//
// This is the CUDA equivalent of metal/leaninfer-fused.metal.
//
// Target: NVIDIA GPUs with compute capability ≥ 7.0 (Volta+).
// Block = 256 threads (8 warps × 32 threads).
// Shared memory: x_norm[K] where K ≤ 8192 (32 KB for FP32, 16 KB for FP16).
//
// Kernels:
//   li_fused_rms_norm_matmul_f32   — RMSNorm(x) @ W
//   li_fused_rms_norm_swiglu_f32   — RMSNorm(x) + gate + up + SiLU*mul
//   li_fused_rms_norm_swiglu_f16   — FP16 weights, FP32 accumulation
//
// SPDX-License-Identifier: MIT

#include "leaninfer-cuda-common.cuh"

// Aliases for readability
#define warp_reduce_sum_f32 li_warp_reduce_sum
#define silu_f32 li_silu

// ---------------------------------------------------------------------------
// li_fused_rms_norm_matmul_f32
//
// Computes: dst[n] = dot(rms_norm(x, gamma, eps), W_row_n)
//
// Grid  : (ceil(N / WARPS_PER_BLOCK), 1, 1)
// Block : (256, 1, 1) — 8 warps
//
// Each block:
//   1. All 256 threads collaboratively compute RMS(x) and store
//      x_norm[0..K-1] into shared memory.
//   2. Each of the 8 warps computes one output row dot product.
//
// Constraint: K ≤ 8192 (shared float x_norm[8192] = 32 KB).
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE = 256>
__global__ void li_fused_rms_norm_matmul_f32(
        const float * __restrict__ x,       // [K]
        const float * __restrict__ gamma,   // [K]
        const float * __restrict__ W,       // [N, K] row-major
        float       * __restrict__ dst,     // [N]
        const int K,
        const int N,
        const float eps) {


    constexpr int N_WARPS = BLOCK_SIZE / WARP_SIZE;

    extern __shared__ float smem[];          // x_norm[K] + reduce_buf[N_WARPS]
    float * x_norm     = smem;
    float * reduce_buf = smem + K;

    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;

    // -- Phase 1: sum of squares (all threads) --
    float sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float v = x[k];
        sq += v * v;
    }
    sq = warp_reduce_sum_f32(sq);
    if (lane_id == 0) reduce_buf[warp_id] = sq;
    __syncthreads();

    float total = 0.0f;
    if (tid < N_WARPS) total = reduce_buf[tid];
    total = warp_reduce_sum_f32(total);
    float rms_scale = rsqrtf(total / (float)K + eps);
    __syncthreads();

    // -- Phase 2: normalize x → x_norm (all threads) --
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        x_norm[k] = x[k] * gamma[k] * rms_scale;
    }
    __syncthreads();

    // -- Phase 3: each warp computes one output-row dot product --
    int out_row = blockIdx.x * N_WARPS + warp_id;
    if (out_row < N) {
        float acc = 0.0f;
        const float * w_row = W + (int64_t)out_row * K;
        for (int k = lane_id; k < K; k += WARP_SIZE) {
            acc += x_norm[k] * w_row[k];
        }
        acc = warp_reduce_sum_f32(acc);
        if (lane_id == 0) dst[out_row] = acc;
    }
}

// ---------------------------------------------------------------------------
// li_fused_rms_norm_swiglu_f32
//
// Fused: RMSNorm(x) + SwiGLU(gate, up) in one kernel.
// Computes: hidden[n] = silu(gate_n) * up_n  where
//   gate_n = dot(rms_norm(x), W_gate_row_n)
//   up_n   = dot(rms_norm(x), W_up_row_n)
//
// Each warp computes gate[n] and up[n] for the same n, then
// immediately applies SiLU and stores hidden[n]. Eliminates:
//   • The write of x_norm to global memory
//   • The write+read of gate and up to/from global memory
//   → saves ~5 global-memory round-trips per FFN layer
//
// Grid  : (ceil(N / N_WARPS), 1, 1)
// Block : (BLOCK_SIZE, 1, 1)
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE = 256>
__global__ void li_fused_rms_norm_swiglu_f32(
        const float * __restrict__ x,       // [K]
        const float * __restrict__ gamma,   // [K]
        const float * __restrict__ W_gate,  // [N, K] row-major
        const float * __restrict__ W_up,    // [N, K] row-major
        float       * __restrict__ hidden,  // [N] — silu(gate)*up
        const int K,
        const int N,
        const float eps) {


    constexpr int N_WARPS = BLOCK_SIZE / WARP_SIZE;

    extern __shared__ float smem[];
    float * x_norm     = smem;
    float * reduce_buf = smem + K;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // -- Phase 1: RMS sum --
    float sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float v = x[k];
        sq += v * v;
    }
    sq = warp_reduce_sum_f32(sq);
    if (lane_id == 0) reduce_buf[warp_id] = sq;
    __syncthreads();

    float total = 0.0f;
    if (tid < N_WARPS) total = reduce_buf[tid];
    total = warp_reduce_sum_f32(total);
    float rms_scale = rsqrtf(total / (float)K + eps);
    __syncthreads();

    // -- Phase 2: compute x_norm --
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        x_norm[k] = x[k] * gamma[k] * rms_scale;
    }
    __syncthreads();

    // -- Phase 3: each warp: dot gate, dot up, fuse SiLU*mul --
    int out_row = blockIdx.x * N_WARPS + warp_id;
    if (out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const float * wg_row = W_gate + (int64_t)out_row * K;
        const float * wu_row = W_up   + (int64_t)out_row * K;
        for (int k = lane_id; k < K; k += WARP_SIZE) {
            float xn = x_norm[k];
            gate_acc += xn * wg_row[k];
            up_acc   += xn * wu_row[k];
        }
        gate_acc = warp_reduce_sum_f32(gate_acc);
        up_acc   = warp_reduce_sum_f32(up_acc);
        if (lane_id == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}

// ---------------------------------------------------------------------------
// li_fused_rms_norm_swiglu_f16
//
// FP16 weight variant. Accumulation in FP32. x and gamma remain FP32.
// x_norm stored as half in shared memory (16 KB for K=8192).
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE = 256>
__global__ void li_fused_rms_norm_swiglu_f16(
        const float  * __restrict__ x,
        const float  * __restrict__ gamma,
        const half   * __restrict__ W_gate,
        const half   * __restrict__ W_up,
        float        * __restrict__ hidden,
        const int K,
        const int N,
        const float eps) {


    constexpr int N_WARPS = BLOCK_SIZE / WARP_SIZE;

    extern __shared__ char smem_raw[];
    half  * x_norm     = (half *)smem_raw;                              // K halves
    float * reduce_buf = (float *)(smem_raw + K * sizeof(half));        // N_WARPS floats

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // -- Phase 1: RMS sum --
    float sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float v = x[k];
        sq += v * v;
    }
    sq = warp_reduce_sum_f32(sq);
    if (lane_id == 0) reduce_buf[warp_id] = sq;
    __syncthreads();

    float total = 0.0f;
    if (tid < N_WARPS) total = reduce_buf[tid];
    total = warp_reduce_sum_f32(total);
    float rms_scale = rsqrtf(total / (float)K + eps);
    __syncthreads();

    // -- Phase 2: compute x_norm as FP16 --
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        x_norm[k] = __float2half(x[k] * gamma[k] * rms_scale);
    }
    __syncthreads();

    // -- Phase 3: each warp: gate + up + SiLU*mul --
    int out_row = blockIdx.x * N_WARPS + warp_id;
    if (out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const half * wg_row = W_gate + (int64_t)out_row * K;
        const half * wu_row = W_up   + (int64_t)out_row * K;
        for (int k = lane_id; k < K; k += WARP_SIZE) {
            float xn = __half2float(x_norm[k]);
            gate_acc += xn * __half2float(wg_row[k]);
            up_acc   += xn * __half2float(wu_row[k]);
        }
        gate_acc = warp_reduce_sum_f32(gate_acc);
        up_acc   = warp_reduce_sum_f32(up_acc);
        if (lane_id == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers
// ---------------------------------------------------------------------------
extern "C" {

void li_launch_fused_rms_norm_matmul_f32(
        const float * x, const float * gamma, const float * W,
        float * dst, int K, int N, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / 32;
    int grid = (N + WARPS - 1) / WARPS;
    size_t smem = K * sizeof(float) + WARPS * sizeof(float);
    li_fused_rms_norm_matmul_f32<BLOCK><<<grid, BLOCK, smem, stream>>>(
        x, gamma, W, dst, K, N, eps);
}

void li_launch_fused_rms_norm_swiglu_f32(
        const float * x, const float * gamma,
        const float * W_gate, const float * W_up,
        float * hidden, int K, int N, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / 32;
    int grid = (N + WARPS - 1) / WARPS;
    size_t smem = K * sizeof(float) + WARPS * sizeof(float);
    li_fused_rms_norm_swiglu_f32<BLOCK><<<grid, BLOCK, smem, stream>>>(
        x, gamma, W_gate, W_up, hidden, K, N, eps);
}

void li_launch_fused_rms_norm_swiglu_f16(
        const float * x, const float * gamma,
        const half  * W_gate, const half  * W_up,
        float * hidden, int K, int N, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / 32;
    int grid = (N + WARPS - 1) / WARPS;
    size_t smem = K * sizeof(half) + WARPS * sizeof(float);
    li_fused_rms_norm_swiglu_f16<BLOCK><<<grid, BLOCK, smem, stream>>>(
        x, gamma, W_gate, W_up, hidden, K, N, eps);
}

}  // extern "C"
