#!/bin/bash
# LeanInfer one-command quantization
# Converts HuggingFace GGUF models with optimized per-tensor quantization
#
# Usage:
#   ./quantize.sh <input.gguf> <preset> [output.gguf]
#
# Presets:
#   qwen35-27b-quality    IQ4_K uniform, best quality (~15 GB)
#   qwen35-27b-balanced   IQ4_K attn + IQ3_K FFN (~12 GB)
#   qwen35-27b-lean       IQ4_K attn + IQ3_K FFN (~10 GB)
#   qwen35-27b-ultra-lean IQ2_K uniform (~8 GB)
#   qwen35-9b-quality     IQ4_K uniform (~5 GB)
#   qwen35-9b-lean        IQ4_K attn + IQ3_K FFN (~3.5 GB)
#   deepseek-r1-14b       IQ4_K uniform (~8 GB)
#
# Examples:
#   ./quantize.sh Qwen3.5-27B-f16.gguf qwen35-27b-lean
#   ./quantize.sh Qwen3.5-27B-f16.gguf qwen35-27b-lean my-model-lean.gguf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QUANTIZE="$PROJECT_DIR/upstream/build/bin/llama-quantize"

if [ ! -f "$QUANTIZE" ]; then
    echo "ERROR: llama-quantize not found. Run setup.sh and build first."
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.gguf> <preset> [output.gguf]"
    echo ""
    echo "Presets: qwen35-27b-quality, qwen35-27b-balanced, qwen35-27b-lean,"
    echo "         qwen35-27b-ultra-lean, qwen35-9b-quality, qwen35-9b-lean,"
    echo "         deepseek-r1-14b"
    exit 1
fi

INPUT="$1"
PRESET="$2"
OUTPUT="${3:-${INPUT%.gguf}-${PRESET}.gguf}"
THREADS=$(nproc)

if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

# Check for imatrix (optional but recommended)
IMATRIX_FLAG=""
IMATRIX_FILE="${INPUT%.gguf}-imatrix.dat"
if [ -f "$IMATRIX_FILE" ]; then
    echo "Found importance matrix: $IMATRIX_FILE"
    IMATRIX_FLAG="--imatrix $IMATRIX_FILE"
fi

echo "=== LeanInfer Quantize ==="
echo "Input:  $INPUT"
echo "Preset: $PRESET"
echo "Output: $OUTPUT"
echo ""

case "$PRESET" in
    qwen35-27b-quality|qwen35-9b-quality|deepseek-r1-14b)
        echo "Strategy: IQ4_K uniform (best quality)"
        $QUANTIZE $IMATRIX_FLAG \
            --attn-qkv-type IQ4_K --attn-output-type IQ4_K \
            --ffn-gate-type IQ4_K --ffn-up-type IQ4_K --ffn-down-type IQ4_K \
            "$INPUT" "$OUTPUT" IQ4_K $THREADS
        ;;

    qwen35-27b-balanced)
        echo "Strategy: IQ4_K attention + IQ3_K FFN (balanced)"
        $QUANTIZE $IMATRIX_FLAG \
            --attn-qkv-type IQ4_K --attn-output-type IQ4_K \
            --ffn-gate-type IQ3_K --ffn-up-type IQ3_K --ffn-down-type IQ3_K \
            "$INPUT" "$OUTPUT" IQ3_K $THREADS
        ;;

    qwen35-27b-lean|qwen35-9b-lean)
        echo "Strategy: IQ4_K attention + IQ3_K FFN (lean)"
        $QUANTIZE $IMATRIX_FLAG \
            --attn-qkv-type IQ4_K --attn-output-type IQ4_K \
            --ffn-gate-type IQ3_K --ffn-up-type IQ3_K --ffn-down-type IQ3_K \
            "$INPUT" "$OUTPUT" IQ3_K $THREADS
        ;;

    qwen35-27b-ultra-lean)
        echo "Strategy: IQ2_K uniform (ultra-lean, quality tradeoff)"
        $QUANTIZE $IMATRIX_FLAG \
            "$INPUT" "$OUTPUT" IQ2_K $THREADS
        ;;

    *)
        echo "ERROR: Unknown preset: $PRESET"
        echo "Available: qwen35-27b-quality, qwen35-27b-balanced, qwen35-27b-lean,"
        echo "           qwen35-27b-ultra-lean, qwen35-9b-quality, qwen35-9b-lean,"
        echo "           deepseek-r1-14b"
        exit 1
        ;;
esac

echo ""
echo "=== Quantization Complete ==="
echo "Output: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"
echo ""
echo "To run with matching runtime config:"
echo "  llama-cli -m $OUTPUT --config $PROJECT_DIR/configs/presets/$PRESET.conf"
