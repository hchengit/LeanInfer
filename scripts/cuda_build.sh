#!/usr/bin/env bash
# LeanInfer Phase 3 — Cloud GPU CUDA build script
#
# Builds ik_llama.cpp with CUDA backend + LeanInfer fused kernels.
# Run on a cloud GPU instance (Vast.ai, RunPod, Lambda, etc.).
#
# Usage:
#   ./scripts/cuda_build.sh [--debug] [--jobs N] [--arch SM]
#
# Prerequisites:
#   - CUDA Toolkit ≥ 11.7 (nvcc in PATH)
#   - cmake ≥ 3.17
#   - upstream/ cloned (run scripts/setup_upstream.sh first)
#
# Flags:
#   --debug     Debug build
#   --jobs N    Parallel jobs (default: nproc)
#   --arch SM   CUDA architecture (default: auto-detect, e.g., 86 for A100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-cuda"
UPSTREAM_DIR="$REPO_ROOT/upstream"

BUILD_TYPE="Release"
N_JOBS=$(nproc 2>/dev/null || echo 8)
CUDA_ARCH=""

for arg in "$@"; do
    case "$arg" in
        --debug)    BUILD_TYPE="Debug" ;;
        --jobs=*)   N_JOBS="${arg#--jobs=}" ;;
        --arch=*)   CUDA_ARCH="${arg#--arch=}" ;;
    esac
done

echo "=== LeanInfer CUDA Build ==="
echo "  Build type : $BUILD_TYPE"
echo "  Jobs       : $N_JOBS"
echo "  Source     : $UPSTREAM_DIR"
echo "  Output     : $BUILD_DIR"

# ---- Verify prerequisites ----
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA Toolkit or add to PATH."
    echo "  On Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "  Or: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | head -1)
echo "  CUDA       : $NVCC_VERSION"

if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install with: sudo apt install cmake"
    exit 1
fi

# Auto-detect GPU architecture if not specified
if [[ -z "$CUDA_ARCH" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        # Extract compute capability from nvidia-smi
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "  GPU        : $GPU_NAME"
        # Try to get SM version from deviceQuery or parse from GPU name
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo "")
    fi
    if [[ -z "$CUDA_ARCH" ]]; then
        CUDA_ARCH="70;75;80;86;89;90"
        echo "  Arch       : auto (targeting $CUDA_ARCH)"
    else
        echo "  Arch       : $CUDA_ARCH (auto-detected)"
    fi
else
    echo "  Arch       : $CUDA_ARCH (manual)"
fi
echo ""

# ---- Ensure upstream exists ----
if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    echo "upstream/ not found — running setup_upstream.sh"
    bash "$REPO_ROOT/scripts/setup_upstream.sh"
fi

# ---- Apply fused gate patches to upstream (idempotent) ----
if [[ -f "$REPO_ROOT/scripts/patch_fused_gate.sh" ]]; then
    bash "$REPO_ROOT/scripts/patch_fused_gate.sh"
fi

# ---- Wire LeanInfer CUDA + Metal cmake into upstream (idempotent) ----
CMAKE_FILE="$UPSTREAM_DIR/src/CMakeLists.txt"

# CUDA cmake integration
CUDA_INCLUDE='include("${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/leaninfer-cuda.cmake" OPTIONAL)'
if ! grep -qF "leaninfer-cuda.cmake" "$CMAKE_FILE"; then
    echo "# LeanInfer CUDA integration (appended by cuda_build.sh)" >> "$CMAKE_FILE"
    echo "$CUDA_INCLUDE" >> "$CMAKE_FILE"
    echo "Patched: appended LeanInfer CUDA include"
fi

# Metal cmake integration (harmless on Linux — OPTIONAL makes it no-op)
METAL_INCLUDE='include("${CMAKE_CURRENT_SOURCE_DIR}/../../metal/leaninfer-metal.cmake" OPTIONAL)'
if ! grep -qF "leaninfer-metal.cmake" "$CMAKE_FILE"; then
    echo "# LeanInfer Metal integration (appended by cuda_build.sh)" >> "$CMAKE_FILE"
    echo "$METAL_INCLUDE" >> "$CMAKE_FILE"
fi
echo ""

# ---- Configure ----
mkdir -p "$BUILD_DIR"

cmake -S "$UPSTREAM_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_CUDA=ON \
    -DLEANINFER_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    2>&1 | grep -E "^(--|LeanInfer|CMake|CUDA|Build)" | head -40

echo ""
echo "=== Building ($N_JOBS jobs) ==="
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$N_JOBS"

# ---- Smoke test ----
echo ""
echo "=== Smoke test ==="
CLI="$BUILD_DIR/bin/llama-cli"
[[ -f "$CLI" ]] || CLI="$BUILD_DIR/bin/main"

if [[ -f "$CLI" ]]; then
    "$CLI" --help 2>&1 | head -3
    echo ""
    echo "Binary OK. Run with -ngl 99 to offload all layers to GPU."
else
    echo "WARNING: llama-cli not found at $CLI"
fi

# ---- Summary ----
echo ""
echo "=== Build complete ==="
echo ""
echo "Benchmark decode (GPU offload):"
echo "  $BUILD_DIR/bin/llama-cli \\"
echo "    --model models/qwen35-9b-instruct-q4_k_m.gguf \\"
echo "    -ngl 99 --kv-compress \\"
echo "    -n 256 -p \"Explain transformer inference\""
echo ""
echo "Benchmark with profiler:"
echo "  $BUILD_DIR/bin/llama-cli \\"
echo "    --model models/qwen35-9b-instruct-q4_k_m.gguf \\"
echo "    -ngl 99 -n 32 -p \"Hello\" \\"
echo "    --leaninfer-profile 2>&1 | grep -E 'ffn|delta'"
