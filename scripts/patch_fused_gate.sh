#!/usr/bin/env bash
# LeanInfer — Patch upstream ik_llama.cpp with fused RMSNorm+SiLU-gate op
#
# Adds GGML_OP_FUSED_RMS_SILU_GATE to ggml and wires it into the DeltaNet
# gated output path. Replaces 2 kernel launches (RMSNorm + fused_mul_unary)
# with 1 fused launch per DeltaNet layer.
#
# This script is idempotent — safe to run multiple times.
#
# Usage: ./scripts/patch_fused_gate.sh
#        Called automatically by cuda_build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPSTREAM="$REPO_ROOT/upstream"

MARKER="GGML_OP_FUSED_RMS_SILU_GATE"

# ---- 1. ggml.h: add op to enum ----
GGML_H="$UPSTREAM/ggml/include/ggml.h"
if ! grep -q "$MARKER" "$GGML_H"; then
    echo "Patching $GGML_H — adding $MARKER to enum"
    sed -i "/GGML_OP_DELTA_NET,/a\\
\\
        $MARKER,  // LeanInfer: fused RMSNorm + silu(z) * norm for DeltaNet gated output" "$GGML_H"

    # Add the API function declaration after ggml_fused_mul_unary_inplace
    sed -i "/ggml_fused_mul_unary_inplace/,/;/{
        /;/a\\
\\
    // LeanInfer: fused RMSNorm + SiLU-gate\\
    // dst[i] = silu(z[i]) * (output[i] * gamma[i % K] * rms_scale)\\
    // where rms_scale = rsqrt(sum(output_row^2) / K + eps)\\
    GGML_API struct ggml_tensor * ggml_fused_rms_silu_gate(\\
            struct ggml_context * ctx,\\
            struct ggml_tensor  * output,  /* [K, n_rows] */\\
            struct ggml_tensor  * z,       /* [K, n_rows] */\\
            struct ggml_tensor  * gamma,   /* [K] */\\
            float                 eps);
    }" "$GGML_H"
else
    echo "ggml.h already patched"
fi

# ---- 2. ggml.c: add op name + constructor + compute support ----
GGML_C="$UPSTREAM/ggml/src/ggml.c"
if ! grep -q "$MARKER" "$GGML_C"; then
    echo "Patching $GGML_C — adding op implementation"

    # Add op name in the string table (after DELTA_NET entry)
    sed -i '/\[GGML_OP_DELTA_NET\]/a\
    [GGML_OP_FUSED_RMS_SILU_GATE]   = "FUSED_RMS_SILU_GATE",' "$GGML_C"

    # Add constructor function before ggml_fill (which comes after delta_net)
    sed -i '/^\/\/ ggml_fill$/i\
// LeanInfer: ggml_fused_rms_silu_gate\
\
struct ggml_tensor * ggml_fused_rms_silu_gate(\
        struct ggml_context * ctx,\
        struct ggml_tensor  * output,\
        struct ggml_tensor  * z,\
        struct ggml_tensor  * gamma,\
        float                 eps) {\
    GGML_ASSERT(ggml_are_same_shape(output, z));\
    GGML_ASSERT(gamma->ne[0] == output->ne[0]);\
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, output->ne);\
    result->op     = GGML_OP_FUSED_RMS_SILU_GATE;\
    result->src[0] = output;\
    result->src[1] = z;\
    result->src[2] = gamma;\
    ggml_set_op_params_f32(result, 0, eps);\
    return result;\
}\
' "$GGML_C"

    # Add compute case (CPU fallback — does the same as separate norm+gate)
    # Insert after the DELTA_NET compute case
    sed -i '/case GGML_OP_DELTA_NET:/{
        # Find the next "break;" after this case and insert our case after it
        n
        /break;/a\
        case GGML_OP_FUSED_RMS_SILU_GATE:\
            {\
                /* CPU fallback: norm + silu(z) * norm_out */\
                const struct ggml_tensor * src_out = node->src[0];\
                const struct ggml_tensor * src_z   = node->src[1];\
                const struct ggml_tensor * src_g   = node->src[2];\
                const float eps = ggml_get_op_params_f32(node, 0);\
                const int64_t K = src_out->ne[0];\
                const int64_t n_rows = ggml_nrows(src_out);\
                for (int64_t r = 0; r < n_rows; ++r) {\
                    const float * out_row = (const float *)src_out->data + r * K;\
                    const float * z_row   = (const float *)src_z->data   + r * K;\
                    const float * gamma   = (const float *)src_g->data;\
                    float * dst_row = (float *)node->data + r * K;\
                    float sq = 0.0f;\
                    for (int64_t k = 0; k < K; ++k) sq += out_row[k] * out_row[k];\
                    float rms = 1.0f / sqrtf(sq / (float)K + eps);\
                    for (int64_t k = 0; k < K; ++k) {\
                        float norm_val = out_row[k] * gamma[k] * rms;\
                        float zv = z_row[k];\
                        dst_row[k] = (zv / (1.0f + expf(-zv))) * norm_val;\
                    }\
                }\
            } break;
    }' "$GGML_C"

    # Add to the "is_contiguous" check (supports_op)
    sed -i '/case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous/a\
        case GGML_OP_FUSED_RMS_SILU_GATE: return true;' "$GGML_C"
else
    echo "ggml.c already patched"
fi

# ---- 3. ggml-cuda.cu: add CUDA dispatch ----
GGML_CUDA="$UPSTREAM/ggml/src/ggml-cuda.cu"
if ! grep -q "$MARKER" "$GGML_CUDA"; then
    echo "Patching $GGML_CUDA — adding CUDA dispatch"

    # Add dispatch case after FUSED_MUL_UNARY
    sed -i '/case GGML_OP_FUSED_MUL_UNARY:/,/break;/{
        /break;/a\
        case GGML_OP_FUSED_RMS_SILU_GATE:\
            {\
                const float eps = ggml_get_op_params_f32(dst, 0);\
                const int K = (int)dst->src[0]->ne[0];\
                const int n_rows = (int)ggml_nrows(dst->src[0]);\
                li_launch_fused_rms_norm_silu_gate_f32(\
                    (const float *)dst->src[0]->data,\
                    (const float *)dst->src[1]->data,\
                    (const float *)dst->src[2]->data,\
                    (float *)dst->data,\
                    K, n_rows, eps, ctx.stream());\
            } break;
    }' "$GGML_CUDA"

    # Add supports_op case
    sed -i '/case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous/a\
        case GGML_OP_FUSED_RMS_SILU_GATE: return true;' "$GGML_CUDA"

    # Add include for our kernel header at the top (after existing includes)
    if ! grep -q "leaninfer-cuda.h" "$GGML_CUDA"; then
        sed -i '1,/^#include/{
            /^#include/a\
#ifdef LEANINFER_CUDA\
#include "leaninfer-cuda.h"\
#endif
        }' "$GGML_CUDA"
    fi
else
    echo "ggml-cuda.cu already patched"
fi

# ---- 4. llama-delta-net.cpp: modify build_gated_output ----
DELTA_NET="$UPSTREAM/src/llama-delta-net.cpp"
if ! grep -q "ggml_fused_rms_silu_gate" "$DELTA_NET"; then
    echo "Patching $DELTA_NET — using fused gate op in build_gated_output"

    # Replace the two separate ops (llm_build_norm + ggml_fused_mul_unary)
    # with our single fused op.
    #
    # Original (lines ~403-406):
    #   ggml_tensor * attn_out_norm = llm_build_context::llm_build_norm(ctx0, attn_out_2d, lctx.model.hparams, ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    #   cb(attn_out_norm, "attn_rms_norm", il);
    #   attn_out_norm = ggml_fused_mul_unary(ctx0, z_2d, attn_out_norm, GGML_UNARY_OP_SILU);
    #   cb(attn_out_norm, "attn_out_norm", il);
    #
    # Replacement:
    #   ggml_tensor * attn_out_norm = ggml_fused_rms_silu_gate(ctx0, attn_out_2d, z_2d, ssm_norm, lctx.model.hparams.f_norm_rms_eps);
    #   cb(attn_out_norm, "attn_out_norm", il);

    sed -i '/ggml_tensor \* attn_out_norm = llm_build_context::llm_build_norm/,/cb(attn_out_norm, "attn_out_norm", il);/{
        /ggml_tensor \* attn_out_norm = llm_build_context::llm_build_norm/c\
    ggml_tensor * attn_out_norm = ggml_fused_rms_silu_gate(ctx0, attn_out_2d, z_2d, ssm_norm, lctx.model.hparams.f_norm_rms_eps);
        /cb(attn_out_norm, "attn_rms_norm", il);/d
        /attn_out_norm = ggml_fused_mul_unary/d
    }' "$DELTA_NET"
else
    echo "llama-delta-net.cpp already patched"
fi

echo ""
echo "=== Fused gate patches applied ==="
echo "  ggml.h:              GGML_OP_FUSED_RMS_SILU_GATE enum + API"
echo "  ggml.c:              constructor + CPU fallback + supports_op"
echo "  ggml-cuda.cu:        CUDA dispatch → li_fused_rms_norm_silu_gate_f32"
echo "  llama-delta-net.cpp: build_gated_output() uses fused op"
