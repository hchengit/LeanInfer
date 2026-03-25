#!/bin/bash
# LeanInfer setup script
# Clones ik_llama.cpp upstream and applies patches

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="$PROJECT_DIR/upstream"
PATCHES_DIR="$PROJECT_DIR/patches"

IK_REPO="https://github.com/ikawrakow/ik_llama.cpp.git"

echo "=== LeanInfer Setup ==="

# Clone upstream if not present
if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "Cloning ik_llama.cpp..."
    git clone --depth 1 "$IK_REPO" "$UPSTREAM_DIR"
else
    echo "Upstream already exists. Pulling latest..."
    cd "$UPSTREAM_DIR" && git pull && cd "$PROJECT_DIR"
fi

# Apply patches in order
if [ -d "$PATCHES_DIR" ] && ls "$PATCHES_DIR"/*.patch 1>/dev/null 2>&1; then
    echo "Applying patches..."
    cd "$UPSTREAM_DIR"
    for patch in "$PATCHES_DIR"/*.patch; do
        echo "  Applying $(basename "$patch")..."
        git apply "$patch" || {
            echo "  WARNING: Failed to apply $(basename "$patch"). May need rebase."
        }
    done
    cd "$PROJECT_DIR"
else
    echo "No patches to apply yet."
fi

# Build
echo ""
echo "To build:"
echo "  cd upstream"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DGGML_CUDA=ON  # or without for CPU-only"
echo "  cmake --build . -j\$(nproc)"
echo ""
echo "=== Setup Complete ==="
