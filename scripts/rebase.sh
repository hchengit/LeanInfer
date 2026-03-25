#!/bin/bash
# Rebase LeanInfer patches on latest ik_llama.cpp upstream
# Run this periodically to stay current with upstream changes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="$PROJECT_DIR/upstream"
PATCHES_DIR="$PROJECT_DIR/patches"

echo "=== LeanInfer Rebase ==="

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: upstream/ not found. Run setup.sh first."
    exit 1
fi

# Save current state
cd "$UPSTREAM_DIR"
echo "Reverting current patches..."
git checkout .

# Pull latest
echo "Pulling latest ik_llama.cpp..."
git pull

# Re-apply patches
if ls "$PATCHES_DIR"/*.patch 1>/dev/null 2>&1; then
    echo "Re-applying patches..."
    FAILED=0
    for patch in "$PATCHES_DIR"/*.patch; do
        echo -n "  $(basename "$patch")... "
        if git apply --check "$patch" 2>/dev/null; then
            git apply "$patch"
            echo "OK"
        else
            echo "CONFLICT — needs manual resolution"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "WARNING: $FAILED patch(es) failed. Review and update manually."
        exit 1
    fi
else
    echo "No patches to apply."
fi

echo ""
echo "=== Rebase Complete ==="
