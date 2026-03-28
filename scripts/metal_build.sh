#!/usr/bin/env bash
# LeanInfer Phase 2b — macOS Metal build script
#
# Builds ik_llama.cpp with Metal backend + LeanInfer extensions on an
# Apple Silicon Mac (M2+). Requires Xcode Command Line Tools.
#
# Usage:
#   ./scripts/metal_build.sh [--debug] [--no-embed] [--jobs N]
#
# Flags:
#   --debug       Debug build (no optimizations, Metal shader debug symbols)
#   --no-embed    Don't embed metallib; load .metal source at runtime instead
#   --jobs N      Parallel build jobs (default: nproc)
#
# Outputs:
#   build-metal/bin/llama-cli    — CLI binary with Metal backend
#   build-metal/bin/llama-server — server binary with Metal backend
#   build-metal/bin/ggml-metal.metal         — base ggml Metal shader
#   build-metal/bin/leaninfer-fused.metal    — LeanInfer fused FFN shader

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-metal"
UPSTREAM_DIR="$REPO_ROOT/upstream"
METAL_DIR="$REPO_ROOT/metal"

# ---- Parse flags ----
BUILD_TYPE="Release"
EMBED_METALLIB="ON"
N_JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)

for arg in "$@"; do
    case "$arg" in
        --debug)    BUILD_TYPE="Debug"; EMBED_METALLIB="OFF" ;;
        --no-embed) EMBED_METALLIB="OFF" ;;
        --jobs=*)   N_JOBS="${arg#--jobs=}" ;;
        --jobs)     shift; N_JOBS="$1" ;;
    esac
done

echo "=== LeanInfer Metal Build ==="
echo "  Build type    : $BUILD_TYPE"
echo "  Embed metallib: $EMBED_METALLIB"
echo "  Jobs          : $N_JOBS"
echo "  Source        : $UPSTREAM_DIR"
echo "  LeanInfer Metal: $METAL_DIR"
echo "  Output        : $BUILD_DIR"
echo ""

# ---- Verify prerequisites ----
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script requires macOS."
    exit 1
fi

if ! command -v xcrun &>/dev/null; then
    echo "ERROR: Xcode Command Line Tools not found."
    echo "  Install with: xcode-select --install"
    exit 1
fi

if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install with: brew install cmake"
    exit 1
fi

METAL_VERSION=$(xcrun metal --version 2>&1 | head -1) || true
if [[ "$METAL_VERSION" == *"error"* && "$EMBED_METALLIB" == "ON" ]]; then
    echo "ERROR: Metal shader compiler not found and embed mode is ON."
    echo "  Either install Xcode or use --no-embed (JIT shaders at runtime)."
    exit 1
fi
echo "Metal compiler: $METAL_VERSION"
echo ""

# ---- Wire LeanInfer Metal into upstream CMakeLists (idempotent) ----
#
# upstream/src/CMakeLists.txt is part of the ik_llama.cpp tree (gitignored
# in the LeanInfer repo). We append a single include() line so cmake picks up
# metal/leaninfer-metal.cmake, which contains all our Metal cmake logic.
# This append is guarded so it's safe to run the script multiple times.
#
CMAKE_FILE="$UPSTREAM_DIR/src/CMakeLists.txt"
INCLUDE_LINE='include("${CMAKE_CURRENT_SOURCE_DIR}/../../metal/leaninfer-metal.cmake" OPTIONAL)'

if ! grep -qF "leaninfer-metal.cmake" "$CMAKE_FILE"; then
    echo "# LeanInfer Metal integration (appended by metal_build.sh)" >> "$CMAKE_FILE"
    echo "$INCLUDE_LINE" >> "$CMAKE_FILE"
    echo "Patched: appended LeanInfer Metal include to $CMAKE_FILE"
else
    echo "CMakeLists already patched — skipping"
fi
echo ""

# ---- Configure ----
mkdir -p "$BUILD_DIR"

cmake -S "$UPSTREAM_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY="$EMBED_METALLIB" \
    -DGGML_METAL_SHADER_DEBUG="$([[ $BUILD_TYPE == Debug ]] && echo ON || echo OFF)" \
    -DLEANINFER_METAL=ON \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    2>&1 | grep -E "^(--|LeanInfer|CMake Warning|CMake Error|Build|Configure|Generating)" | head -40

echo ""
echo "=== Building ($N_JOBS jobs) ==="
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$N_JOBS"

# ---- Pre-compile leaninfer-fused.metal to metallib if not embedding ----
BIN_DIR="$BUILD_DIR/bin"
FUSED_METAL="$METAL_DIR/leaninfer-fused.metal"

if [[ "$EMBED_METALLIB" == "OFF" && -f "$FUSED_METAL" ]]; then
    echo ""
    echo "=== Pre-compiling leaninfer-fused.metallib ==="
    xcrun -sdk macosx metal \
        -O3 -ffast-math \
        -c "$FUSED_METAL" \
        -o "$BIN_DIR/leaninfer-fused.air"
    xcrun -sdk macosx metallib \
        "$BIN_DIR/leaninfer-fused.air" \
        -o "$BIN_DIR/leaninfer-fused.metallib"
    rm -f "$BIN_DIR/leaninfer-fused.air"
    echo "  → $BIN_DIR/leaninfer-fused.metallib"
fi

# ---- Smoke test ----
echo ""
echo "=== Smoke test ==="
CLI="$BIN_DIR/llama-cli"
[[ -f "$CLI" ]] || CLI="$BIN_DIR/main"

MODEL="$REPO_ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if [[ -f "$CLI" && -f "$MODEL" ]]; then
    echo "Running: llama-cli -ngl 99 -n 20 --model qwen2.5-0.5b..."
    "$CLI" \
        --model "$MODEL" \
        -ngl 99 \
        -n 20 \
        -p "Hello, Apple Silicon" \
        --no-display-prompt \
        2>&1 | tail -5
else
    echo "  Skipping smoke test (binary or model not found)"
    echo "  Binary : $CLI"
    echo "  Model  : $MODEL"
    echo "  Download model: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/"
fi

# ---- Summary ----
echo ""
echo "=== Build complete ==="
echo ""
echo "Decode with full GPU offload (-ngl 99 = all layers to Metal):"
echo "  $BIN_DIR/llama-cli \\"
echo "    --model models/qwen35-9b-instruct-q4_k_m.gguf \\"
echo "    -ngl 99 --kv-compress \\"
echo "    -n 512 -p \"<|im_start|>user\nExplain transformer inference\n<|im_end|>\n<|im_start|>assistant\n\""
echo ""
echo "Tile calibration (run after build):"
echo "  pip3 install metalcompute numpy"
echo "  python3 scripts/tile_sweep.py --model qwen35-9b --m 1   # decode"
echo "  python3 scripts/tile_sweep.py --model qwen35-9b --m 32  # prefill"
