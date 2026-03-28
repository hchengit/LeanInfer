//
// LeanInfer Phase 2b — Fused FFN Metal kernels
//
// Cooperative fusion: RMSNorm + gate/up projections + SiLU*mul in a single
// dispatch, keeping the normalized hidden state in threadgroup memory to
// eliminate the intermediate device-memory round-trips that the baseline
// graph incurs (5 dispatches → 2 dispatches for a full FFN block).
//
// Target: M2 (Apple Silicon, Metal 3). Threadgroup = 256 threads (8 SIMD
// groups × 32 threads). Supports hidden dims up to K=8192 (FP32) / K=16384
// (FP16) — covers all Qwen, DeepSeek-R1 variants up to 70B.
//
// Kernels:
//   kernel_fused_rms_norm_matmul_f32   — RMSNorm(x) @ W in one dispatch
//   kernel_fused_rms_norm_matmul_f16   — same for FP16 weights
//   kernel_fused_rms_norm_swiglu_f32   — RMSNorm(x) + gate+up+SiLU*mul
//   kernel_fused_rms_norm_swiglu_f16   — same for FP16 weights
//
// Note: the W layout expected here is row-major [N, K] (each row is one
// output neuron's weight vector). ik_llama.cpp stores weights transposed
// relative to standard PyTorch convention — verify strides in the .mm
// wrapper before dispatching.
//
// SPDX-License-Identifier: MIT
//

#include <metal_stdlib>
using namespace metal;

#define N_SIMD      32u   // SIMD group width on Apple Silicon
#define TILE_N      8u    // SIMD groups per threadgroup → output rows per TG
#define TG_THREADS  256u  // TILE_N * N_SIMD
// Max K for f32 path: 32768 byte TG limit / 4 bytes - headroom for rms_scale + simd_sums
#define MAX_K_F32   8160u
// Max K for f16 path: 32768 byte TG limit / 2 bytes - headroom
#define MAX_K_F16   16320u

// ---------------------------------------------------------------------------
// Helper: SiLU activation  silu(x) = x * sigmoid(x) = x / (1 + e^-x)
// ---------------------------------------------------------------------------
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}
inline half silu_f16(half x) {
    return x / (half(1.0f) + exp(-x));
}

// ---------------------------------------------------------------------------
// kernel_fused_rms_norm_matmul_f32
//
// Computes: dst[n] = dot(rms_norm(x, gamma, eps), W_row_n)
//
// Grid  : ceil(N / TILE_N) threadgroups, 1D
// TG    : TG_THREADS threads
//
// Each threadgroup:
//   1. All TG_THREADS threads collaboratively compute RMS(x) and store
//      x_norm[0..K-1] into threadgroup memory.
//   2. Each of the TILE_N SIMD groups computes one output row dot product.
//
// Constraints: K ≤ MAX_K_F32 (threadgroup float x_norm[] + overhead ≤ 32 KB).
// ---------------------------------------------------------------------------
kernel void kernel_fused_rms_norm_matmul_f32(
        device  const float * x       [[buffer(0)]],   // [K]
        device  const float * gamma   [[buffer(1)]],   // [K]
        device  const float * W       [[buffer(2)]],   // [N, K] row-major
        device        float * dst     [[buffer(3)]],   // [N]
        constant  int32_t   & K       [[buffer(4)]],
        constant  int32_t   & N       [[buffer(5)]],
        constant  float     & eps     [[buffer(6)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    threadgroup float rms_scale_tg[1];
    threadgroup float simd_sums[TILE_N];
    threadgroup float x_norm[MAX_K_F32];

    const int Ku = (int)K;

    // -- Phase 1: sum of squares (all threads) --
    float sq = 0.0f;
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        float v = x[k];
        sq += v * v;
    }
    sq = simd_sum(sq);
    if (simd_lid == 0) simd_sums[simd_gid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < TILE_N; ++s) total += simd_sums[s];
        rms_scale_tg[0] = rsqrt(total / float(Ku) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 2: normalize x → x_norm (all threads) --
    float scale = rms_scale_tg[0];
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_norm[k] = x[k] * gamma[k] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 3: each SIMD group computes one output-row dot product --
    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float acc = 0.0f;
        const device float * w_row = W + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            acc += x_norm[k] * w_row[k];
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) dst[out_row] = acc;
    }
}

// ---------------------------------------------------------------------------
// kernel_fused_rms_norm_matmul_f16
//
// FP16 variant — weights stored as half. Accumulation in float32.
// Larger K range (x_norm stored as half → 2 bytes each).
// ---------------------------------------------------------------------------
kernel void kernel_fused_rms_norm_matmul_f16(
        device  const float * x       [[buffer(0)]],
        device  const float * gamma   [[buffer(1)]],
        device  const half  * W       [[buffer(2)]],   // FP16 weights
        device        float * dst     [[buffer(3)]],
        constant  int32_t   & K       [[buffer(4)]],
        constant  int32_t   & N       [[buffer(5)]],
        constant  float     & eps     [[buffer(6)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    threadgroup float rms_scale_tg[1];
    threadgroup float simd_sums[TILE_N];
    threadgroup half  x_norm[MAX_K_F16];

    const int Ku = (int)K;

    float sq = 0.0f;
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        float v = x[k];
        sq += v * v;
    }
    sq = simd_sum(sq);
    if (simd_lid == 0) simd_sums[simd_gid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < TILE_N; ++s) total += simd_sums[s];
        rms_scale_tg[0] = rsqrt(total / float(Ku) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rms_scale_tg[0];
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_norm[k] = half(x[k] * gamma[k] * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float acc = 0.0f;
        const device half * w_row = W + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            acc += float(x_norm[k]) * float(w_row[k]);
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) dst[out_row] = acc;
    }
}

// ---------------------------------------------------------------------------
// kernel_fused_rms_norm_swiglu_f32
//
// Fused: RMSNorm(x) + SwiGLU(gate, up) in one dispatch.
// Computes: hidden[n] = silu(gate_n) * up_n  where
//   gate_n = dot(rms_norm(x), W_gate_row_n)
//   up_n   = dot(rms_norm(x), W_up_row_n)
//
// Each SIMD group computes gate[n] and up[n] for the same n, then
// immediately applies SiLU and stores hidden[n]. This eliminates:
//   • The write of x_norm to device memory
//   • The write+read of gate and up to/from device memory
//   → saves ~5 device-memory round-trips per FFN layer
//
// Grid  : ceil(N / TILE_N) threadgroups, 1D
// TG    : TG_THREADS threads
// ---------------------------------------------------------------------------
kernel void kernel_fused_rms_norm_swiglu_f32(
        device  const float * x       [[buffer(0)]],   // [K]
        device  const float * gamma   [[buffer(1)]],   // [K]
        device  const float * W_gate  [[buffer(2)]],   // [N, K] row-major
        device  const float * W_up    [[buffer(3)]],   // [N, K] row-major
        device        float * hidden  [[buffer(4)]],   // [N] — silu(gate)*up
        constant  int32_t   & K       [[buffer(5)]],
        constant  int32_t   & N       [[buffer(6)]],
        constant  float     & eps     [[buffer(7)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    threadgroup float rms_scale_tg[1];
    threadgroup float simd_sums[TILE_N];
    threadgroup float x_norm[MAX_K_F32];

    const int Ku = (int)K;

    // -- Phase 1: RMS sum --
    float sq = 0.0f;
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        float v = x[k];
        sq += v * v;
    }
    sq = simd_sum(sq);
    if (simd_lid == 0) simd_sums[simd_gid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < TILE_N; ++s) total += simd_sums[s];
        rms_scale_tg[0] = rsqrt(total / float(Ku) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 2: compute x_norm --
    float scale = rms_scale_tg[0];
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_norm[k] = x[k] * gamma[k] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 3: each SIMD group: dot gate, dot up, fuse SiLU*mul --
    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const device float * wg_row = W_gate + out_row * (uint)Ku;
        const device float * wu_row = W_up   + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            float xn = x_norm[k];
            gate_acc += xn * wg_row[k];
            up_acc   += xn * wu_row[k];
        }
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);
        if (simd_lid == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}

// ---------------------------------------------------------------------------
// kernel_fused_rms_norm_swiglu_f16
//
// FP16 weight variant of the SwiGLU kernel. Accumulation in FP32.
// ---------------------------------------------------------------------------
kernel void kernel_fused_rms_norm_swiglu_f16(
        device  const float * x       [[buffer(0)]],
        device  const float * gamma   [[buffer(1)]],
        device  const half  * W_gate  [[buffer(2)]],
        device  const half  * W_up    [[buffer(3)]],
        device        float * hidden  [[buffer(4)]],
        constant  int32_t   & K       [[buffer(5)]],
        constant  int32_t   & N       [[buffer(6)]],
        constant  float     & eps     [[buffer(7)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    threadgroup float rms_scale_tg[1];
    threadgroup float simd_sums[TILE_N];
    threadgroup half  x_norm[MAX_K_F16];

    const int Ku = (int)K;

    float sq = 0.0f;
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        float v = x[k];
        sq += v * v;
    }
    sq = simd_sum(sq);
    if (simd_lid == 0) simd_sums[simd_gid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < TILE_N; ++s) total += simd_sums[s];
        rms_scale_tg[0] = rsqrt(total / float(Ku) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rms_scale_tg[0];
    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_norm[k] = half(x[k] * gamma[k] * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const device half * wg_row = W_gate + out_row * (uint)Ku;
        const device half * wu_row = W_up   + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            float xn = float(x_norm[k]);
            gate_acc += xn * float(wg_row[k]);
            up_acc   += xn * float(wu_row[k]);
        }
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);
        if (simd_lid == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}

// ===========================================================================
// Norm-free SwiGLU kernels
//
// These skip the RMSNorm phase — the input x is assumed to be already
// normalized. Used by the eval callback to replace GGML_OP_FUSED_UP_GATE
// which receives pre-normed input from a preceding RMSNorm node.
//
// Computes: hidden[n] = silu(dot(x, W_gate[n])) * dot(x, W_up[n])
//
// The input x is loaded into threadgroup memory once, then each SIMD group
// computes the gate and up dot products for one output row.
//
// Grid  : ceil(N / TILE_N) threadgroups, 1D
// TG    : TG_THREADS threads
// ===========================================================================

kernel void kernel_swiglu_f32(
        device  const float * x       [[buffer(0)]],   // [K] — pre-normed input
        device  const float * W_gate  [[buffer(1)]],   // [N, K] row-major
        device  const float * W_up    [[buffer(2)]],   // [N, K] row-major
        device        float * hidden  [[buffer(3)]],   // [N] output
        constant  int32_t   & K       [[buffer(4)]],
        constant  int32_t   & N       [[buffer(5)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    // Cache x in threadgroup memory for reuse across TILE_N dot products
    threadgroup float x_tg[MAX_K_F32];

    const int Ku = (int)K;

    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_tg[k] = x[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group: dot gate, dot up, fuse SiLU*mul
    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const device float * wg_row = W_gate + out_row * (uint)Ku;
        const device float * wu_row = W_up   + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            float xv = x_tg[k];
            gate_acc += xv * wg_row[k];
            up_acc   += xv * wu_row[k];
        }
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);
        if (simd_lid == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}

kernel void kernel_swiglu_f16(
        device  const float * x       [[buffer(0)]],   // [K] — pre-normed input (f32)
        device  const half  * W_gate  [[buffer(1)]],   // [N, K] row-major (f16)
        device  const half  * W_up    [[buffer(2)]],   // [N, K] row-major (f16)
        device        float * hidden  [[buffer(3)]],   // [N] output (f32)
        constant  int32_t   & K       [[buffer(4)]],
        constant  int32_t   & N       [[buffer(5)]],
        uint  tid      [[thread_position_in_threadgroup]],
        uint  tgid     [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  simd_lid [[thread_index_in_simdgroup]]) {

    threadgroup float x_tg[MAX_K_F32];

    const int Ku = (int)K;

    for (int k = (int)tid; k < Ku; k += (int)TG_THREADS) {
        x_tg[k] = x[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row = tgid * TILE_N + simd_gid;
    if ((int)out_row < N) {
        float gate_acc = 0.0f;
        float up_acc   = 0.0f;
        const device half * wg_row = W_gate + out_row * (uint)Ku;
        const device half * wu_row = W_up   + out_row * (uint)Ku;
        for (int k = (int)simd_lid; k < Ku; k += (int)N_SIMD) {
            float xv = x_tg[k];
            gate_acc += xv * float(wg_row[k]);
            up_acc   += xv * float(wu_row[k]);
        }
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);
        if (simd_lid == 0) {
            hidden[out_row] = silu_f32(gate_acc) * up_acc;
        }
    }
}
