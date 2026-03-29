#!/usr/bin/env bash
# LeanInfer — Clone and set up the upstream ik_llama.cpp fork.
#
# The LeanInfer repo stores its own additions (metal/, instrument/, scripts/,
# configs/, profiles/, docs/) but gitignores upstream/ — the modified
# ik_llama.cpp tree lives in a separate fork.
#
# Usage:
#   ./scripts/setup_upstream.sh                    # uses default fork URL
#   UPSTREAM_REPO=https://github.com/... ./scripts/setup_upstream.sh
#
# Environment:
#   UPSTREAM_REPO   — git clone URL (default: hchengit/ik_llama.cpp on GitHub)
#   UPSTREAM_REF    — branch or commit to check out (default: main)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPSTREAM_DIR="$REPO_ROOT/upstream"

UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/hchengit/Lean_llama.cpp.git}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"

if [[ -d "$UPSTREAM_DIR/.git" ]]; then
    echo "upstream/ already exists — pulling latest"
    cd "$UPSTREAM_DIR"
    git fetch origin
    git checkout "$UPSTREAM_REF"
    git pull origin "$UPSTREAM_REF" || true
    echo "upstream/ updated to $(git rev-parse --short HEAD)"
    exit 0
fi

echo "Cloning upstream from $UPSTREAM_REPO (ref: $UPSTREAM_REF)"
git clone --depth 50 "$UPSTREAM_REPO" "$UPSTREAM_DIR"
cd "$UPSTREAM_DIR"
git checkout "$UPSTREAM_REF"
echo "upstream/ ready at $(git rev-parse --short HEAD)"
