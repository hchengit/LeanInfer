# LeanInfer Phase 3 — CUDA fused kernel integration
#
# Included from upstream/src/CMakeLists.txt via:
#   include("${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/leaninfer-cuda.cmake" OPTIONAL)
#
# Enable with:  cmake -DLEANINFER_CUDA=ON -DGGML_CUDA=ON ...

cmake_minimum_required(VERSION 3.17)

option(LEANINFER_CUDA "Enable LeanInfer CUDA fused kernels (FFN + DeltaNet)" OFF)

if (LEANINFER_CUDA AND NOT GGML_CUDA)
    message(WARNING "LEANINFER_CUDA=ON requires -DGGML_CUDA=ON")
elseif (LEANINFER_CUDA)
    set(_LI_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda")

    # Add CUDA kernel source files to the llama target.
    # These compile as CUDA (.cu) — CMake handles nvcc dispatch automatically.
    target_sources(llama PRIVATE
        "${_LI_CUDA_DIR}/leaninfer-fused-ffn.cu"
        "${_LI_CUDA_DIR}/leaninfer-fused-deltanet.cu"
    )

    target_compile_definitions(llama PUBLIC LEANINFER_CUDA)

    # Allow #include "leaninfer-cuda.h" from llama and downstream targets.
    target_include_directories(llama PUBLIC "${_LI_CUDA_DIR}")

    # Ensure compute capability ≥ 7.0 (Volta+) for warp shuffle intrinsics.
    # ik_llama.cpp already sets CUDA architectures; we just verify the minimum.
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" PARENT_SCOPE)
    endif()

    message(STATUS "LeanInfer CUDA fused kernels ENABLED — FFN + DeltaNet")
endif()
