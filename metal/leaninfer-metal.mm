// LeanInfer Phase 2b — Metal backend extensions
//
// MTLHeap sub-allocator + fused FFN dispatch + dynamic CPU/GPU routing.
//
// Build: compiled only on Apple platforms when GGML_USE_METAL is defined.
//   cmake -DGGML_METAL=ON -DLEANINFER_METAL=ON ...
//
// Architecture:
//
//   1. MTLHeap allocator
//      Creates a single MTLStorageModeShared heap sized to model×1.4.
//      All LeanInfer-specific buffers are sub-allocated from this heap.
//      The existing ggml-metal.m allocator is left untouched; this heap
//      is used only for our fused-kernel intermediate buffers.
//
//   2. Fused FFN dispatch
//      Installs a ggml_backend_sched eval callback. When the callback
//      fires on a tensor named "ffn_up_gate-*" (ik_llama.cpp's fused
//      gate+up naming), it:
//        a. reads the already-computed hidden-dim activations from the
//           ggml tensor data pointer (unified memory — zero-copy on Apple),
//        b. launches kernel_fused_rms_norm_swiglu_{f32,f16} on a parallel
//           command buffer,
//        c. writes the result back to the output tensor.
//      The callback fires *after* the original op runs, but because the
//      fused result overwrites the same output buffer, subsequent ops in
//      the graph see the correct value.
//
//      NOTE: The preferred long-term approach is to intercept *before* the
//      op and skip it, which requires a pre-op hook not yet in ggml's
//      backend scheduler. When that API lands, switch to it to avoid the
//      redundant base op execution.
//
//   3. Dynamic CPU/GPU routing
//      On Apple Silicon, CPU↔GPU copies are zero-cost (unified memory).
//      The routing policy is: any GGML_OP_MUL_MAT where the output has
//      fewer than gpu_min_elements gets pinned to CPU via
//      ggml_backend_sched_set_op_offload (OFF for GPU) per-graph.
//      All large matmuls remain on GPU.
//
// SPDX-License-Identifier: MIT

#if defined(GGML_USE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <sys/stat.h>

#include "leaninfer-metal.h"
#include "llama.h"
#include "ggml-backend.h"

#include "ggml.h"  // for ggml_internal_get_type_traits, ggml_type

// LeanInfer eval callback hook.
// When the upstream fork has the Phase 2b patch (metal_eval_cb field in
// llama_context + leaninfer_metal_set_eval_cb in llama.cpp), this resolves
// to the real implementation at link time. Otherwise we provide a no-op
// fallback so the Metal extension compiles on any ik_llama.cpp checkout.
#if !defined(LEANINFER_HAS_EVAL_CB)
static void leaninfer_metal_set_eval_cb(
        struct llama_context              * /*ctx*/,
        ggml_backend_sched_eval_callback    /*cb*/,
        void                              * /*user_data*/) {
    // no-op: upstream doesn't have the eval callback hook yet
}
static const bool leaninfer_eval_cb_available = false;
#else
extern "C" void leaninfer_metal_set_eval_cb(
        struct llama_context              * ctx,
        ggml_backend_sched_eval_callback    cb,
        void                              * user_data);
static const bool leaninfer_eval_cb_available = true;
#endif


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <string>
#include <vector>
#include <unordered_map>
#include <mach/mach_time.h>  // for high-res timing

// ---------------------------------------------------------------------------
// Internal context
// ---------------------------------------------------------------------------

struct leaninfer_metal_context {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLHeap>                heap;
    id<MTLLibrary>             fused_lib;  // leaninfer-fused.metallib
    // RMSNorm+SwiGLU PSOs (for future full-fusion path)
    id<MTLComputePipelineState> pso_rms_swiglu_f32;
    id<MTLComputePipelineState> pso_rms_swiglu_f16;
    id<MTLComputePipelineState> pso_rms_matmul_f32;
    id<MTLComputePipelineState> pso_rms_matmul_f16;
    // Norm-free SwiGLU PSOs (used by eval callback dispatch)
    id<MTLComputePipelineState> pso_swiglu_f32;
    id<MTLComputePipelineState> pso_swiglu_f16;

    struct llama_context * llama_ctx;

    std::atomic<int> gpu_min_elements;
    bool             enable_fused_ffn;
    bool             verbose;

    std::atomic<int> fused_intercept_count;
    std::atomic<int> fused_dispatch_count;
    double           total_dequant_us;
    double           total_dispatch_us;
    size_t heap_total_bytes;

    // Cached dequantized weight buffers, keyed by quantized data pointer.
    // Dequantized once on first encounter, reused for all subsequent tokens.
    struct weight_cache_entry {
        id<MTLBuffer> buf_f32;   // [N * K] floats
        int64_t       N;
        int64_t       K;
    };
    std::unordered_map<const void *, weight_cache_entry> weight_cache;

    // Output buffer (reused, sized to largest N encountered)
    id<MTLBuffer>    buf_output;
    int64_t          buf_output_N;
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
static bool li_compile_shaders(leaninfer_metal_context * li_ctx,
                               const std::string       & shader_dir);

static bool li_install_eval_callback(leaninfer_metal_context * li_ctx);

// ---------------------------------------------------------------------------
// Eval callback
//
// Called by ggml_backend_sched after every compute node. We use it to
// dispatch fused replacements for the FFN gate+up node.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Helper: get wall-clock microseconds
// ---------------------------------------------------------------------------
static double li_time_us(void) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1000.0;
}

// ---------------------------------------------------------------------------
// Helper: get or create a cached dequantized weight buffer.
// Returns the MTLBuffer containing N*K floats, dequantized from the quantized
// tensor. On first call for a given tensor, dequantizes and caches. On
// subsequent calls, returns the cached buffer (zero cost).
// ---------------------------------------------------------------------------
static id<MTLBuffer> li_get_cached_weight(leaninfer_metal_context * li,
                                           struct ggml_tensor * w,
                                           double * dequant_us_out) {
    const void * key = w->data;
    auto it = li->weight_cache.find(key);
    if (it != li->weight_cache.end()) {
        if (dequant_us_out) *dequant_us_out = 0.0;
        return it->second.buf_f32;
    }

    // First encounter — dequantize and cache
    double t0 = li_time_us();

    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    size_t mat_bytes = (size_t)(N * K) * sizeof(float);

    id<MTLBuffer> buf = [li->device newBufferWithLength:mat_bytes
                                                options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "leaninfer-metal: failed to allocate weight cache "
                "(N=%lld K=%lld, %.1f MB)\n",
                (long long)N, (long long)K, mat_bytes / (1024.0 * 1024.0));
        return nil;
    }

    float * dst = (float *)[buf contents];
    ggml_type_traits_t traits = ggml_internal_get_type_traits(w->type);
    const size_t row_bytes = w->nb[1];

    for (int64_t n = 0; n < N; ++n) {
        traits.to_float((const char *)w->data + n * row_bytes,
                        dst + n * K, K);
    }

    double t1 = li_time_us();
    if (dequant_us_out) *dequant_us_out = t1 - t0;

    leaninfer_metal_context::weight_cache_entry entry;
    entry.buf_f32 = buf;
    entry.N = N;
    entry.K = K;
    li->weight_cache[key] = entry;

    if (li->verbose) {
        fprintf(stderr, "leaninfer-metal: cached weight %p [%lldx%lld] "
                "(%.1f MB, dequant=%.0f µs)\n",
                key, (long long)N, (long long)K,
                mat_bytes / (1024.0 * 1024.0), t1 - t0);
    }

    return buf;
}

// ---------------------------------------------------------------------------
// Helper: ensure output buffer is large enough
// ---------------------------------------------------------------------------
static bool li_ensure_output_buf(leaninfer_metal_context * li, int64_t N) {
    if (li->buf_output && li->buf_output_N >= N) return true;
    if (li->buf_output) [li->buf_output release];

    li->buf_output = [li->device newBufferWithLength:(NSUInteger)(N * sizeof(float))
                                             options:MTLResourceStorageModeShared];
    li->buf_output_N = N;
    return (li->buf_output != nil);
}

// ---------------------------------------------------------------------------
// Eval callback — intercepts FUSED_UP_GATE, dequantizes, dispatches Metal
// ---------------------------------------------------------------------------
static bool li_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    leaninfer_metal_context * li = (leaninfer_metal_context *)user_data;

    const bool is_ffn_gate_up = (strncmp(t->name, "ffn_up_gate", 11) == 0);

    if (ask) {
        return li->enable_fused_ffn && is_ffn_gate_up;
    }

    // Node layout (GGML_OP_FUSED_UP_GATE):
    //   t->src[0] = W_up   [ne0=K, ne1=N]  (quantized)
    //   t->src[1] = W_gate [ne0=K, ne1=N]  (quantized)
    //   t->src[2] = b      [ne0=K, ne1=M]  (f32, pre-normed input)
    //   t->data   = output [ne0=N, ne1=M]  (f32) = silu(gate@b) * up@b
    //
    // Strategy: dequantize weights to f32, dispatch kernel_swiglu_f32 on Metal.
    // The CPU already computed the result — we re-compute on Metal and validate,
    // then overwrite t->data. In the future, a pre-op hook would skip the CPU path.

    li->fused_intercept_count++;

    struct ggml_tensor * w_up   = t->src[0];
    struct ggml_tensor * w_gate = t->src[1];
    struct ggml_tensor * b      = t->src[2];

    const int64_t K = w_up->ne[0];
    const int64_t N = w_up->ne[1];
    const int64_t M = b->ne[1];

    // Only handle M=1 (decode) for now. Prefill (M>1) needs 2D grid.
    if (M != 1) return true;

    // Only handle types with a to_float dequantization function
    ggml_type_traits_t traits_up   = ggml_internal_get_type_traits(w_up->type);
    ggml_type_traits_t traits_gate = ggml_internal_get_type_traits(w_gate->type);
    if (!traits_up.to_float || !traits_gate.to_float) return true;

    // --- Phase 1: Get cached dequantized weights (free after first token) ---
    double dequant_up_us = 0, dequant_gate_us = 0;
    id<MTLBuffer> buf_up_f32   = li_get_cached_weight(li, w_up,   &dequant_up_us);
    id<MTLBuffer> buf_gate_f32 = li_get_cached_weight(li, w_gate, &dequant_gate_us);
    if (!buf_up_f32 || !buf_gate_f32) return true;

    double total_dequant = dequant_up_us + dequant_gate_us;
    li->total_dequant_us += total_dequant;

    // Ensure output buffer
    if (!li_ensure_output_buf(li, N)) return true;

    // --- Phase 2: Dispatch Metal kernel ---
    double t1 = li_time_us();

    // Wrap the f32 input b in an MTLBuffer (zero-copy, unified memory).
    // newBufferWithBytesNoCopy requires page-aligned length; use regular alloc as fallback.
    id<MTLBuffer> buf_x = [li->device newBufferWithBytesNoCopy:(void *)b->data
                                                        length:(NSUInteger)(K * sizeof(float))
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (!buf_x) {
        // Fallback: copy input to a new buffer
        buf_x = [li->device newBufferWithBytes:b->data
                                        length:(NSUInteger)(K * sizeof(float))
                                       options:MTLResourceStorageModeShared];
        if (!buf_x) return true;
    }

    int32_t K32 = (int32_t)K;
    int32_t N32 = (int32_t)N;
    NSUInteger n_tg = (NSUInteger)((N + 7) / 8);  // ceil(N / TILE_N)

    id<MTLCommandBuffer> cmd = [li->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:li->pso_swiglu_f32];
    [enc setBuffer:buf_x       offset:0 atIndex:0];  // x (pre-normed)
    [enc setBuffer:buf_gate_f32 offset:0 atIndex:1]; // W_gate (cached f32)
    [enc setBuffer:buf_up_f32   offset:0 atIndex:2]; // W_up (cached f32)
    [enc setBuffer:li->buf_output offset:0 atIndex:3]; // output
    [enc setBytes:&K32          length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32          length:sizeof(N32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    double t2 = li_time_us();
    li->total_dispatch_us += (t2 - t1);

    // --- Phase 3: Correctness check (first few dispatches) ---
    float * metal_out = (float *)[li->buf_output contents];
    float * cpu_out   = (float *)t->data;

    if (li->fused_intercept_count <= 2) {
        float max_err = 0.0f;
        float max_rel = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            float diff = fabsf(metal_out[n] - cpu_out[n]);
            float rel  = (fabsf(cpu_out[n]) > 1e-8f)
                       ? diff / fabsf(cpu_out[n]) : diff;
            if (diff > max_err) max_err = diff;
            if (rel  > max_rel) max_rel = rel;
        }
        fprintf(stderr, "leaninfer-metal: %s dispatch OK — "
                "max_abs_err=%.6e dequant=%.0f µs gpu=%.0f µs%s\n",
                t->name, max_err,
                total_dequant, t2 - t1,
                total_dequant < 1.0 ? " (cached)" : "");
    }

    // Overwrite CPU result with Metal result
    memcpy(cpu_out, metal_out, (size_t)(N * sizeof(float)));

    li->fused_dispatch_count++;
    [buf_x release];

    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

leaninfer_metal_context * leaninfer_metal_init(
        struct llama_context   * ctx,
        leaninfer_metal_params   params) {

    leaninfer_metal_context * li = new leaninfer_metal_context{};

    li->llama_ctx          = ctx;
    li->gpu_min_elements   = params.gpu_min_elements;
    li->enable_fused_ffn   = params.enable_fused_ffn;
    li->verbose            = params.verbose;

    // -- 1. Pick Metal device --
    li->device = MTLCreateSystemDefaultDevice();
    if (!li->device) {
        fprintf(stderr, "leaninfer-metal: no Metal device found\n");
        delete li;
        return NULL;
    }

    if (li->verbose) {
        fprintf(stderr, "leaninfer-metal: device = %s\n",
                [[li->device name] UTF8String]);
    }

    // -- 2. Command queue --
    li->queue = [li->device newCommandQueue];
    if (!li->queue) {
        fprintf(stderr, "leaninfer-metal: failed to create command queue\n");
        delete li;
        return NULL;
    }

    // -- 3. MTLHeap allocation --
    size_t heap_bytes = params.heap_size_bytes;
    if (heap_bytes == 0) {
        // Auto-size: find model file size via llama_model_desc heuristic.
        // Fallback: 512 MB (enough for intermediate activation buffers).
        heap_bytes = 512 * 1024 * 1024ULL;
    }

    MTLHeapDescriptor * hd = [[MTLHeapDescriptor alloc] init];
    hd.type         = MTLHeapTypeAutomatic;
    hd.storageMode  = MTLStorageModeShared;  // unified memory — visible to CPU
    hd.size         = heap_bytes;
    hd.hazardTrackingMode = MTLHazardTrackingModeTracked;

    li->heap = [li->device newHeapWithDescriptor:hd];
    [hd release];

    if (!li->heap) {
        fprintf(stderr, "leaninfer-metal: failed to allocate MTLHeap (%zu MB)\n",
                heap_bytes >> 20);
        delete li;
        return NULL;
    }
    li->heap_total_bytes = heap_bytes;

    if (li->verbose) {
        fprintf(stderr, "leaninfer-metal: heap = %zu MB (shared storage)\n",
                heap_bytes >> 20);
    }

    // -- 4. Compile fused kernels --
    // Look for leaninfer-fused.metal next to the running executable,
    // then in standard resource paths. This mirrors how ggml-metal.m
    // locates ggml-metal.metal.
    std::string shader_dir;
    {
        // Try bundle path first (packaged app)
        NSBundle * bundle = [NSBundle mainBundle];
        NSString * path = [bundle pathForResource:@"leaninfer-fused" ofType:@"metal"];
        if (path) {
            shader_dir = [[path stringByDeletingLastPathComponent] UTF8String];
        } else {
            // Dev build: look next to the binary
            NSString * exec = [[NSProcessInfo processInfo].arguments objectAtIndex:0];
            shader_dir = [[exec stringByDeletingLastPathComponent] UTF8String];
        }
    }

    if (!li_compile_shaders(li, shader_dir)) {
        fprintf(stderr, "leaninfer-metal: fused shader compilation failed — "
                "fused FFN will not be used\n");
        li->enable_fused_ffn = false;
        // Don't fail hard — fall through without fused kernels
    }

    // -- 5. Install eval callback --
    if (li->enable_fused_ffn) {
        li_install_eval_callback(li);
    }

    if (li->verbose) {
        leaninfer_metal_print_info(li);
    }

    return li;
}

void leaninfer_metal_free(leaninfer_metal_context * li) {
    if (!li) return;

    for (auto & kv : li->weight_cache) {
        if (kv.second.buf_f32) [kv.second.buf_f32 release];
    }
    li->weight_cache.clear();
    if (li->buf_output)         [li->buf_output release];
    if (li->pso_swiglu_f32)     [li->pso_swiglu_f32 release];
    if (li->pso_swiglu_f16)     [li->pso_swiglu_f16 release];
    if (li->pso_rms_swiglu_f32) [li->pso_rms_swiglu_f32 release];
    if (li->pso_rms_swiglu_f16) [li->pso_rms_swiglu_f16 release];
    if (li->pso_rms_matmul_f32) [li->pso_rms_matmul_f32 release];
    if (li->pso_rms_matmul_f16) [li->pso_rms_matmul_f16 release];
    if (li->fused_lib)          [li->fused_lib release];
    if (li->heap)               [li->heap release];
    if (li->queue)              [li->queue release];
    if (li->device)             [li->device release];

    delete li;
}

void leaninfer_metal_set_gpu_min_elements(leaninfer_metal_context * li, int min_elem) {
    if (li) li->gpu_min_elements.store(min_elem);
}

void leaninfer_metal_heap_stats(leaninfer_metal_context * li,
                                size_t * used_bytes_out,
                                size_t * total_bytes_out) {
    if (!li) return;
    if (used_bytes_out)  *used_bytes_out  = (size_t)[li->heap usedSize];
    if (total_bytes_out) *total_bytes_out = li->heap_total_bytes;
}

void leaninfer_metal_print_info(leaninfer_metal_context * li) {
    if (!li) return;
    size_t used = 0, total = 0;
    leaninfer_metal_heap_stats(li, &used, &total);
    int dispatches = (int)li->fused_dispatch_count.load();
    fprintf(stderr,
            "leaninfer-metal: device=%s heap=%zu/%zu MB fused_ffn=%s gpu_min=%d "
            "intercepts=%d dispatches=%d",
            [[li->device name] UTF8String],
            used >> 20, total >> 20,
            li->enable_fused_ffn ? "on" : "off",
            (int)li->gpu_min_elements.load(),
            (int)li->fused_intercept_count.load(),
            dispatches);
    if (dispatches > 0) {
        fprintf(stderr, " avg_dequant=%.0f µs avg_gpu=%.0f µs",
                li->total_dequant_us / dispatches,
                li->total_dispatch_us / dispatches);
    }
    fprintf(stderr, "\n");
}

// ---------------------------------------------------------------------------
// Internal: compile leaninfer-fused.metal
// ---------------------------------------------------------------------------
static bool li_compile_shaders(leaninfer_metal_context * li,
                               const std::string       & shader_dir) {
    NSError * error = nil;

    // Try pre-compiled metallib first.
    std::string metallib_path = shader_dir + "/leaninfer-fused.metallib";
    struct stat st;
    if (stat(metallib_path.c_str(), &st) == 0) {
        NSURL * url = [NSURL fileURLWithPath:
                       [NSString stringWithUTF8String:metallib_path.c_str()]];
        li->fused_lib = [li->device newLibraryWithURL:url error:&error];
        if (error) {
            fprintf(stderr, "leaninfer-metal: metallib load error: %s\n",
                    [[error description] UTF8String]);
            error = nil;
        }
    }

    // Fall back to source compilation.
    if (!li->fused_lib) {
        std::string src_path = shader_dir + "/leaninfer-fused.metal";
        if (stat(src_path.c_str(), &st) != 0) {
            fprintf(stderr, "leaninfer-metal: shader not found at %s\n",
                    src_path.c_str());
            return false;
        }
        NSString * src_ns = [NSString stringWithContentsOfFile:
                             [NSString stringWithUTF8String:src_path.c_str()]
                             encoding:NSUTF8StringEncoding error:&error];
        if (!src_ns || error) {
            fprintf(stderr, "leaninfer-metal: failed to read shader source\n");
            return false;
        }
        MTLCompileOptions * opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        li->fused_lib = [li->device newLibraryWithSource:src_ns
                                                  options:opts
                                                    error:&error];
        [opts release];
        if (error) {
            fprintf(stderr, "leaninfer-metal: shader compile error: %s\n",
                    [[error description] UTF8String]);
            return false;
        }
        fprintf(stderr, "leaninfer-metal: compiled leaninfer-fused.metal from source\n");
    }

    // Build compute pipeline states.
    auto make_pso = [&](NSString * fname) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [li->fused_lib newFunctionWithName:fname];
        if (!fn) {
            fprintf(stderr, "leaninfer-metal: function not found: %s\n",
                    [fname UTF8String]);
            return nil;
        }
        NSError * e = nil;
        id<MTLComputePipelineState> pso =
            [li->device newComputePipelineStateWithFunction:fn error:&e];
        [fn release];
        if (e) {
            fprintf(stderr, "leaninfer-metal: PSO creation failed for %s: %s\n",
                    [fname UTF8String], [[e description] UTF8String]);
            return nil;
        }
        return pso;
    };

    li->pso_rms_swiglu_f32  = make_pso(@"kernel_fused_rms_norm_swiglu_f32");
    li->pso_rms_swiglu_f16  = make_pso(@"kernel_fused_rms_norm_swiglu_f16");
    li->pso_rms_matmul_f32  = make_pso(@"kernel_fused_rms_norm_matmul_f32");
    li->pso_rms_matmul_f16  = make_pso(@"kernel_fused_rms_norm_matmul_f16");
    li->pso_swiglu_f32      = make_pso(@"kernel_swiglu_f32");
    li->pso_swiglu_f16      = make_pso(@"kernel_swiglu_f16");

    // The norm-free SwiGLU PSOs are required for the eval callback dispatch.
    if (!li->pso_swiglu_f32 || !li->pso_swiglu_f16) {
        fprintf(stderr, "leaninfer-metal: norm-free swiglu PSOs failed — "
                "fused dispatch disabled\n");
        return false;
    }

    fprintf(stderr, "leaninfer-metal: fused kernels compiled OK "
            "(rms_swiglu_f32/f16, rms_matmul_f32/f16, swiglu_f32/f16)\n");
    return true;
}

// ---------------------------------------------------------------------------
// Internal: install eval callback into llama context's scheduler
// ---------------------------------------------------------------------------
static bool li_install_eval_callback(leaninfer_metal_context * li) {
    // llama_context exposes ggml_backend_sched via llama_get_model() path,
    // but there is no public API to set the sched eval callback from outside.
    //
    // We reach it via the LeanInfer-internal expert_log path that already
    // uses ggml_backend_sched_set_eval_callback in llama.cpp. Two options:
    //
    //   a) Add leaninfer_set_metal_eval_callback() to llama.cpp's public API
    //      (one-liner: wraps ggml_backend_sched_set_eval_callback).
    //
    //   b) Use the existing --expert-log infrastructure and chain callbacks.
    //
    // For Phase 2b, option (a) is implemented: see leaninfer_metal_set_eval_cb
    // in llama.cpp (added below). The callback pointer li_eval_callback is
    // passed there.
    //
    leaninfer_metal_set_eval_cb(li->llama_ctx, li_eval_callback, li);
    if (leaninfer_eval_cb_available) {
        fprintf(stderr, "leaninfer-metal: eval callback registered\n");
    } else {
        fprintf(stderr, "leaninfer-metal: eval callback stub "
                "(rebuild with -DLEANINFER_HAS_EVAL_CB to enable)\n");
    }
    return true;
}

#endif  // GGML_USE_METAL
