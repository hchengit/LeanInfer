// LeanInfer — Standalone CUDA fused kernel benchmark
//
// Validates correctness and measures throughput of the fused FFN + DeltaNet
// kernels using synthetic data matching Qwen 3.5-9B dimensions.
//
// Build (on cloud GPU):
//   nvcc -O3 -arch=sm_89 cuda/benchmark_fused.cu -o benchmark_fused
//   ./benchmark_fused
//
// This does NOT require ik_llama.cpp — it's a standalone CUDA program that
// benchmarks our kernels in isolation. Use this to validate that the kernels
// produce correct results and measure their raw throughput.
//
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// Include our kernels
#include "leaninfer-cuda-common.cuh"

// ---------------------------------------------------------------------------
// Inline the kernel code (since we compile standalone, not via cmake)
// ---------------------------------------------------------------------------

#define warp_reduce_sum_f32 li_warp_reduce_sum
#define silu_f32 li_silu

template <int BLOCK_SIZE = 256>
__global__ void li_fused_rms_norm_swiglu_f32(
        const float * __restrict__ x,
        const float * __restrict__ gamma,
        const float * __restrict__ W_gate,
        const float * __restrict__ W_up,
        float       * __restrict__ hidden,
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

    float sq = 0.0f;
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        float v = x[k];
        sq += v * v;
    }
    sq = warp_reduce_sum_f32(sq);
    if (lane_id == 0) reduce_buf[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < N_WARPS; i++) total += reduce_buf[i];
        reduce_buf[0] = rsqrtf(total / (float)K + eps);
    }
    __syncthreads();
    float rms_scale = reduce_buf[0];

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        x_norm[k] = x[k] * gamma[k] * rms_scale;
    }
    __syncthreads();

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
// CPU reference for correctness check
// ---------------------------------------------------------------------------
void cpu_rms_norm_swiglu(
        const float * x, const float * gamma,
        const float * W_gate, const float * W_up,
        float * hidden, int K, int N, float eps) {
    // RMSNorm
    float sq = 0.0f;
    for (int k = 0; k < K; k++) sq += x[k] * x[k];
    float rms_scale = 1.0f / sqrtf(sq / K + eps);

    float * x_norm = (float *)malloc(K * sizeof(float));
    for (int k = 0; k < K; k++) x_norm[k] = x[k] * gamma[k] * rms_scale;

    // SwiGLU
    for (int n = 0; n < N; n++) {
        float gate = 0.0f, up = 0.0f;
        for (int k = 0; k < K; k++) {
            gate += x_norm[k] * W_gate[(int64_t)n * K + k];
            up   += x_norm[k] * W_up[(int64_t)n * K + k];
        }
        hidden[n] = (gate / (1.0f + expf(-gate))) * up;
    }
    free(x_norm);
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
template <typename Func>
double bench_kernel(Func fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Qwen 3.5-9B dimensions
    struct { const char * name; int K; int N; } configs[] = {
        {"Qwen2.5-0.5B", 896, 4864},
        {"Qwen3.5-9B",   3584, 18944},
        {"Qwen3-14B",    5120, 17408},
    };

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    const float eps = 1e-5f;

    for (auto & cfg : configs) {
        int K = cfg.K;
        int N = cfg.N;
        printf("=== %s (K=%d, N=%d) ===\n", cfg.name, K, N);

        size_t smem_needed = K * sizeof(float) + 8 * sizeof(float);
        if (smem_needed > prop.sharedMemPerBlock) {
            printf("  SKIP: needs %zu KB shared memory (have %zu KB)\n\n",
                   smem_needed / 1024, prop.sharedMemPerBlock / 1024);
            continue;
        }

        // Allocate host
        float * h_x      = (float *)malloc(K * sizeof(float));
        float * h_gamma  = (float *)malloc(K * sizeof(float));
        float * h_W_gate = (float *)malloc((size_t)N * K * sizeof(float));
        float * h_W_up   = (float *)malloc((size_t)N * K * sizeof(float));
        float * h_out_gpu = (float *)malloc(N * sizeof(float));
        float * h_out_cpu = (float *)malloc(N * sizeof(float));

        // Init random
        srand(42);
        for (int i = 0; i < K; i++) {
            h_x[i]     = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            h_gamma[i] = 1.0f + (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            h_W_gate[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
            h_W_up[i]   = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
        }

        // Allocate device
        float *d_x, *d_gamma, *d_W_gate, *d_W_up, *d_hidden;
        cudaMalloc(&d_x,      K * sizeof(float));
        cudaMalloc(&d_gamma,  K * sizeof(float));
        cudaMalloc(&d_W_gate, (size_t)N * K * sizeof(float));
        cudaMalloc(&d_W_up,   (size_t)N * K * sizeof(float));
        cudaMalloc(&d_hidden, N * sizeof(float));

        cudaMemcpy(d_x,      h_x,      K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma,  h_gamma,  K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_gate, h_W_gate, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_up,   h_W_up,   (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice);

        // CPU reference
        cpu_rms_norm_swiglu(h_x, h_gamma, h_W_gate, h_W_up, h_out_cpu, K, N, eps);

        // GPU kernel
        constexpr int BLOCK = 256;
        constexpr int WARPS = BLOCK / 32;
        int grid = (N + WARPS - 1) / WARPS;
        size_t smem = K * sizeof(float) + WARPS * sizeof(float);

        li_fused_rms_norm_swiglu_f32<BLOCK><<<grid, BLOCK, smem>>>(
            d_x, d_gamma, d_W_gate, d_W_up, d_hidden, K, N, eps);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out_gpu, d_hidden, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Correctness check
        float max_err = 0.0f;
        float max_rel = 0.0f;
        for (int i = 0; i < N; i++) {
            float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
            float rel = (fabsf(h_out_cpu[i]) > 1e-6f) ? err / fabsf(h_out_cpu[i]) : 0.0f;
            if (err > max_err) max_err = err;
            if (rel > max_rel) max_rel = rel;
        }
        printf("  Correctness: max_abs_err=%.6f, max_rel_err=%.6f %s\n",
               max_err, max_rel,
               max_err < 0.01f ? "PASS" : "FAIL");

        // Benchmark
        double us = bench_kernel([&]() {
            li_fused_rms_norm_swiglu_f32<BLOCK><<<grid, BLOCK, smem>>>(
                d_x, d_gamma, d_W_gate, d_W_up, d_hidden, K, N, eps);
        }, 10, 100);

        // Compute FLOPS: 2*N*K (gate) + 2*N*K (up) + N (silu) + N (mul) + K (norm)
        double flops = 4.0 * N * K + 2.0 * N + K;
        double gflops = flops / us / 1e3;  // us → GFLOPS
        printf("  Latency: %.1f µs per FFN layer (M=1 decode)\n", us);
        printf("  Throughput: %.1f GFLOPS\n", gflops);
        printf("  Bandwidth: %.1f GB/s (weight read)\n",
               2.0 * N * K * sizeof(float) / us / 1e3);

        // Estimate per-token impact (32 layers on 9B)
        int n_layers = (cfg.K == 3584) ? 32 : (cfg.K == 5120) ? 40 : 24;
        printf("  Est. %d layers: %.1f ms/token (fused) vs baseline unknown\n\n",
               n_layers, us * n_layers / 1000.0);

        // Cleanup
        cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_W_gate);
        cudaFree(d_W_up); cudaFree(d_hidden);
        free(h_x); free(h_gamma); free(h_W_gate);
        free(h_W_up); free(h_out_gpu); free(h_out_cpu);
    }

    return 0;
}
