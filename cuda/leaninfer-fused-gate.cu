// LeanInfer — Fused RMSNorm + SiLU-gate kernel for DeltaNet gated output
//
// Replaces two separate kernel launches in build_gated_output():
//   1. RMSNorm(output)           → norm_out
//   2. silu(z) * norm_out        → gated
// with a single kernel:
//   gated[i] = silu(z[i]) * (output[i] * gamma[i] * rms_scale)
//
// Eliminates the norm_out intermediate write to global memory and saves
// one kernel launch per DeltaNet layer (24 layers on Qwen 3.5-9B).
//
// Input layout (from llama-delta-net.cpp build_gated_output):
//   output: [head_v_dim, num_v_heads * n_tok]  (reshaped to 2D)
//   z:      [head_v_dim, num_v_heads * n_tok]  (reshaped to 2D)
//   gamma:  [head_v_dim]                        (ssm_norm weight)
//   dst:    [head_v_dim, num_v_heads * n_tok]  (gated output)
//
// Each row of output/z/dst has head_v_dim elements. RMSNorm is computed
// per-row (each row is one head × one token). The gating with z is
// element-wise after normalization.
//
// Grid:  (num_v_heads * n_tok, 1, 1)  — one block per row
// Block: (256, 1, 1)
//
// SPDX-License-Identifier: MIT

#include "leaninfer-cuda-common.cuh"

// ---------------------------------------------------------------------------
// li_fused_rms_norm_silu_gate_f32
//
// For each row r in [0, n_rows):
//   rms = rsqrt(sum(output_row^2) / K + eps)
//   dst[r][k] = silu(z[r][k]) * (output[r][k] * gamma[k] * rms)
//
// One block per row. Block collaboratively computes RMS, then each thread
// handles a subset of the K elements for the gated multiply.
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE = 256>
__global__ void li_fused_rms_norm_silu_gate_f32(
        const float * __restrict__ output,  // [n_rows, K]
        const float * __restrict__ z,       // [n_rows, K]
        const float * __restrict__ gamma,   // [K]
        float       * __restrict__ dst,     // [n_rows, K]
        const int K,
        const float eps) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * out_row = output + (int64_t)row * K;
    const float * z_row   = z      + (int64_t)row * K;
    float       * dst_row = dst    + (int64_t)row * K;

    // -- Phase 1: compute sum of squares for this row --
    __shared__ float reduce_buf[BLOCK_SIZE / WARP_SIZE];

    float sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float v = out_row[k];
        sq += v * v;
    }
    sq = li_warp_reduce_sum(sq);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) reduce_buf[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; i++) total += reduce_buf[i];
        reduce_buf[0] = rsqrtf(total / (float)K + eps);
    }
    __syncthreads();
    float rms_scale = reduce_buf[0];

    // -- Phase 2: fused norm + silu(z) * norm_out --
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float norm_val = out_row[k] * gamma[k] * rms_scale;
        float z_val    = z_row[k];
        float silu_z   = z_val / (1.0f + expf(-z_val));
        dst_row[k] = silu_z * norm_val;
    }
}

// ---------------------------------------------------------------------------
// Host launch wrapper
// ---------------------------------------------------------------------------
extern "C" {

void li_launch_fused_rms_norm_silu_gate_f32(
        const float * output, const float * z, const float * gamma,
        float * dst, int K, int n_rows, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    li_fused_rms_norm_silu_gate_f32<BLOCK><<<n_rows, BLOCK, 0, stream>>>(
        output, z, gamma, dst, K, eps);
}

}  // extern "C"
