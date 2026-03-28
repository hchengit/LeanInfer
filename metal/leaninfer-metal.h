#pragma once

// LeanInfer Phase 2b — Metal backend extensions
//
// This header is always safe to include; all APIs are no-ops when compiled
// outside Apple platforms (GGML_USE_METAL not defined).
//
// Usage (from main.cpp or server.cpp):
//
//   #include "leaninfer-metal.h"
//
//   // After llama_new_context_with_model():
//   leaninfer_metal_context * li_metal = leaninfer_metal_init(ctx, params);
//
//   // Run inference as normal — the Metal context intercepts the scheduler.
//
//   // Cleanup:
//   leaninfer_metal_free(li_metal);
//
// The MTLHeap is sized automatically: model_file_bytes × 1.4 (weights +
// KV cache + activation headroom). Pass heap_size_bytes=0 to use this
// auto-sizing heuristic.
//
// SPDX-License-Identifier: MIT

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle returned by leaninfer_metal_init.
typedef struct leaninfer_metal_context leaninfer_metal_context;

// Runtime configuration passed to leaninfer_metal_init.
typedef struct {
    // MTLHeap size in bytes. Pass 0 to auto-size (model_file × 1.4).
    size_t heap_size_bytes;

    // Minimum tensor element count to route to GPU. Ops on smaller tensors
    // are dispatched to CPU to avoid GPU launch overhead. Default: 16384
    // (= 64 KB at FP32 on M2 — below this, CPU is faster).
    int    gpu_min_elements;

    // Enable fused RMSNorm+SwiGLU kernels for FFN blocks. Eliminates
    // 3 intermediate device-memory round-trips per FFN layer.
    // Only applies when Metal is active.
    bool   enable_fused_ffn;

    // Print Metal device info and heap allocation on init.
    bool   verbose;
} leaninfer_metal_params;

// Default params — safe to memset or value-initialize.
static inline leaninfer_metal_params leaninfer_metal_default_params(void) {
    leaninfer_metal_params p;
    p.heap_size_bytes  = 0;      // auto
    p.gpu_min_elements = 16384;
    p.enable_fused_ffn = true;
    p.verbose          = false;
    return p;
}

#if defined(GGML_USE_METAL)

// Initialize LeanInfer Metal extensions.
//
// Must be called AFTER llama_new_context_with_model() and BEFORE the first
// llama_decode() call. Allocates an MTLHeap, compiles leaninfer-fused.metal,
// and installs a scheduler eval callback that intercepts FFN computation.
//
// Returns NULL on failure (Metal not available, OOM, shader compile error).
leaninfer_metal_context * leaninfer_metal_init(
        struct llama_context      * ctx,
        leaninfer_metal_params      params);

// Free the Metal context and release the MTLHeap.
// Safe to call with NULL.
void leaninfer_metal_free(leaninfer_metal_context * li_ctx);

// Override the GPU-routing threshold at runtime (thread-safe).
void leaninfer_metal_set_gpu_min_elements(leaninfer_metal_context * li_ctx, int min_elem);

// Query the heap: returns bytes used / bytes allocated.
void leaninfer_metal_heap_stats(leaninfer_metal_context * li_ctx,
                                size_t * used_bytes_out,
                                size_t * total_bytes_out);

// Print a one-line summary of what the Metal context installed.
void leaninfer_metal_print_info(leaninfer_metal_context * li_ctx);

#else  // !GGML_USE_METAL

// Stub implementations — compiled away to nothing on non-Apple builds.
static inline leaninfer_metal_context * leaninfer_metal_init(
        struct llama_context * ctx, leaninfer_metal_params params) {
    (void)ctx; (void)params; return NULL;
}
static inline void leaninfer_metal_free(leaninfer_metal_context * p)          { (void)p; }
static inline void leaninfer_metal_set_gpu_min_elements(leaninfer_metal_context * p, int n) { (void)p; (void)n; }
static inline void leaninfer_metal_heap_stats(leaninfer_metal_context * p, size_t * u, size_t * t) { (void)p; (void)u; (void)t; }
static inline void leaninfer_metal_print_info(leaninfer_metal_context * p)    { (void)p; }

#endif  // GGML_USE_METAL

#ifdef __cplusplus
}
#endif
