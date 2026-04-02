# LeanInfer Phase 3 — CUDA fused kernel integration
#
# Included from upstream/src/CMakeLists.txt via:
#   include("${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/leaninfer-cuda.cmake" OPTIONAL)
#
# Enable with:  cmake -DLEANINFER_CUDA=ON -DGGML_CUDA=ON ...
#
# Fix: builds CUDA kernels as a separate static library (leaninfer_cuda) and
# links it into llama. This avoids the CMake issue where target_sources with
# absolute paths outside the source tree produces .o files the linker can't find.

cmake_minimum_required(VERSION 3.17)

option(LEANINFER_CUDA "Enable LeanInfer CUDA fused kernels (FFN + DeltaNet)" OFF)

if (LEANINFER_CUDA AND NOT GGML_CUDA)
    message(WARNING "LEANINFER_CUDA=ON requires -DGGML_CUDA=ON")
elseif (LEANINFER_CUDA)
    set(_LI_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda")

    # Ensure CUDA language is enabled
    enable_language(CUDA)

    # Build fused kernels as a separate static library
    add_library(leaninfer_cuda STATIC
        "${_LI_CUDA_DIR}/leaninfer-fused-ffn.cu"
        "${_LI_CUDA_DIR}/leaninfer-fused-deltanet.cu"
        "${_LI_CUDA_DIR}/leaninfer-fused-gate.cu"
    )

    # Set CUDA architectures (inherit from parent or set defaults)
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(leaninfer_cuda PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(leaninfer_cuda PROPERTIES
            CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()

    # CUDA compilation flags
    target_compile_options(leaninfer_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    # Position-independent code (needed when linking into shared lib)
    set_target_properties(leaninfer_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF)

    # Include path for leaninfer-cuda.h
    target_include_directories(leaninfer_cuda PUBLIC "${_LI_CUDA_DIR}")

    # Link the static library into llama
    target_link_libraries(llama PRIVATE leaninfer_cuda)

    # Expose LEANINFER_CUDA define so #ifdef guards compile in
    target_compile_definitions(llama PUBLIC LEANINFER_CUDA)

    # Also expose the include path to llama consumers
    target_include_directories(llama PUBLIC "${_LI_CUDA_DIR}")

    message(STATUS "LeanInfer CUDA fused kernels ENABLED — FFN + DeltaNet (separate static lib)")
endif()
