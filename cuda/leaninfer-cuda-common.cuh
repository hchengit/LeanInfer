#pragma once
// LeanInfer CUDA — shared device primitives
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float li_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float li_silu(float x) {
    return x / (1.0f + expf(-x));
}
