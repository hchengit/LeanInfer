# LeanInfer Phase 2b — Metal backend extensions
#
# Included from upstream/src/CMakeLists.txt via:
#   include("${CMAKE_CURRENT_SOURCE_DIR}/../../metal/leaninfer-metal.cmake" OPTIONAL)
#
# The OPTIONAL keyword means this silently no-ops if the metal/ directory is
# absent (e.g., on a plain ik_llama.cpp checkout without LeanInfer).
# Enable with:  cmake -DLEANINFER_METAL=ON -DGGML_METAL=ON ...

cmake_minimum_required(VERSION 3.17)

option(LEANINFER_METAL "Enable LeanInfer Metal extensions (fused FFN, MTLHeap)" OFF)

if (LEANINFER_METAL AND NOT APPLE)
    message(WARNING "LEANINFER_METAL=ON has no effect on non-Apple platforms")
elseif (LEANINFER_METAL AND NOT GGML_METAL)
    message(WARNING "LEANINFER_METAL=ON requires -DGGML_METAL=ON")
elseif (LEANINFER_METAL)
    # Source lives in metal/ at the repo root (two levels above upstream/src/).
    set(_LI_METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../metal")

    # Add leaninfer-metal.mm to the llama library target.
    target_sources(llama PRIVATE "${_LI_METAL_DIR}/leaninfer-metal.mm")

    # Expose LEANINFER_METAL define so #ifdef guards compile in.
    target_compile_definitions(llama PUBLIC LEANINFER_METAL)

    # Allow leaninfer-metal.h to be found by llama and all linking targets.
    target_include_directories(llama PUBLIC "${_LI_METAL_DIR}")

    # Link Metal + Foundation frameworks (leaninfer-metal.mm uses Metal API directly).
    find_library(METAL_FRAMEWORK    Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(llama PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

    # Copy leaninfer-fused.metal next to the binary so the runtime can JIT-compile it.
    configure_file(
        "${_LI_METAL_DIR}/leaninfer-fused.metal"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/leaninfer-fused.metal"
        COPYONLY)

    message(STATUS "LeanInfer Metal extensions ENABLED — fused FFN + MTLHeap")
endif()
