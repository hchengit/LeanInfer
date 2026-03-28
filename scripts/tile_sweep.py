#!/usr/bin/env python3
"""
LeanInfer Phase 2b — Metal GEMM tile size calibration for Apple Silicon.

Sweeps (TILE_M, TILE_N, TILE_K) combinations and measures FP32/FP16
matrix-multiply throughput (GFLOPS) using Metal compute shaders compiled
at runtime via the `metalcompute` package.

Run on M2 Mac:
    pip install metalcompute numpy
    python3 scripts/tile_sweep.py --model qwen35-9b

Output: scripts/tile_config.json
  {
    "device": "Apple M2",
    "f32_best": {"tile_m": 64, "tile_n": 64, "tile_k": 16, "gflops": 1240.5},
    "f16_best": {"tile_m": 128, "tile_n": 128, "tile_k": 32, "gflops": 2381.0},
    "recommended": {
      "qwen35-9b":  {"hidden": 3584, "ffn": 18944, "tile": "f16_128x128x32"},
      "qwen3-14b":  {"hidden": 5120, "ffn": 17408, "tile": "f16_128x128x32"},
      "deepseek-14b": {"hidden": 5120, "ffn": 14336, "tile": "f16_128x128x32"}
    }
  }

Model dim reference (hidden × ffn_dim):
  Qwen 3.5-9B:       3584 × 18944   (6 GFlops/layer at decode)
  Qwen3-14B:         5120 × 17408
  DeepSeek-R1-14B:   5120 × 14336
  Qwen3-72B:         8192 × 29568
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

# Check for metalcompute
try:
    import metalcompute as mc
except ImportError:
    sys.exit(
        "metalcompute is required: pip install metalcompute\n"
        "  (macOS only — run this script on the M2 Mac)"
    )

# ---------------------------------------------------------------------------
# Metal shader for tiled GEMM benchmark
# Parameterised via preprocessor macros injected at compile time.
# ---------------------------------------------------------------------------
GEMM_SHADER_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

#define TILE_M {TILE_M}
#define TILE_N {TILE_N}
#define TILE_K {TILE_K}

kernel void tiled_gemm_{dtype}(
        device  const {ctype}   * A      [[buffer(0)]],   // [M, K]
        device  const {ctype}   * B      [[buffer(1)]],   // [K, N]
        device        float     * C      [[buffer(2)]],   // [M, N]
        constant  int32_t & M [[buffer(3)]],
        constant  int32_t & N [[buffer(4)]],
        constant  int32_t & K [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2   ntg [[threads_per_threadgroup]])
{{
    threadgroup {ctype} tA[TILE_M][TILE_K];
    threadgroup {ctype} tB[TILE_K][TILE_N];

    const int bm = tgpig.y;
    const int bn = tgpig.x;
    const int tm = tpitg.y;
    const int tn = tpitg.x;

    float acc = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_K) {{
        // Load A tile
        if (bm * TILE_M + tm < M && bk + tn < K)
            tA[tm][tn] = A[(bm * TILE_M + tm) * K + bk + tn];
        else
            tA[tm][tn] = 0;

        // Load B tile
        if (bk + tm < K && bn * TILE_N + tn < N)
            tB[tm][tn] = B[(bk + tm) * N + bn * TILE_N + tn];
        else
            tB[tm][tn] = 0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < TILE_K; ++k)
            acc += float(tA[tm][k]) * float(tB[k][tn]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (bm * TILE_M + tm < M && bn * TILE_N + tn < N)
        C[(bm * TILE_M + tm) * N + bn * TILE_N + tn] = acc;
}}
"""

# Tile combinations to sweep (TILE_M, TILE_N, TILE_K)
# Constrained to: TILE_M * TILE_N * sizeof(float) ≤ 32KB threadgroup mem
TILE_CONFIGS = [
    (8,   8,   8),
    (16,  16,  8),
    (16,  16,  16),
    (32,  32,  8),
    (32,  32,  16),
    (64,  64,  8),
    (64,  64,  16),
    (128, 128, 8),
    (128, 128, 16),
    (128, 128, 32),
]

# Filter configs where threadgroup memory ≤ 32 KB (2 tiles, f32)
def tg_mem_bytes(tm, tn, tk, dtype_size):
    return (tm * tk + tk * tn) * dtype_size

TILE_CONFIGS_F32 = [(tm, tn, tk) for (tm, tn, tk) in TILE_CONFIGS
                    if tg_mem_bytes(tm, tn, tk, 4) <= 32768]
TILE_CONFIGS_F16 = [(tm, tn, tk) for (tm, tn, tk) in TILE_CONFIGS
                    if tg_mem_bytes(tm, tn, tk, 2) <= 32768]

# ---------------------------------------------------------------------------
# Model dim definitions
# ---------------------------------------------------------------------------
MODEL_DIMS = {
    "qwen25-0.5b": {"hidden": 896,  "ffn": 4864},
    "qwen35-9b":   {"hidden": 3584, "ffn": 18944},
    "qwen3-14b":   {"hidden": 5120, "ffn": 17408},
    "deepseek-14b":{"hidden": 5120, "ffn": 14336},
    "qwen3-72b":   {"hidden": 8192, "ffn": 29568},
}

# ---------------------------------------------------------------------------
# Benchmark one tile config
# ---------------------------------------------------------------------------
def benchmark_tile(dev, M, N, K, tile_m, tile_n, tile_k, dtype_str, n_iter=10):
    """Returns GFLOPS or 0.0 on compile/dispatch failure."""

    ctype = "float" if dtype_str == "f32" else "half"
    shader = GEMM_SHADER_TEMPLATE.format(
        TILE_M=tile_m, TILE_N=tile_n, TILE_K=tile_k,
        dtype=dtype_str, ctype=ctype
    )

    try:
        fn = dev.kernel(shader).function(f"tiled_gemm_{dtype_str}")
    except Exception as e:
        return 0.0, f"compile error: {e}"

    np_dtype = np.float32 if dtype_str == "f32" else np.float16
    A = np.random.randn(M, K).astype(np_dtype)
    B = np.random.randn(K, N).astype(np_dtype)
    C = np.zeros((M, N), dtype=np.float32)

    # Warm-up
    try:
        fn(A, B, C, np.int32(M), np.int32(N), np.int32(K),
           threads=(tile_n, tile_m),
           grid=(math.ceil(N / tile_n), math.ceil(M / tile_m)))
    except Exception as e:
        return 0.0, f"dispatch error: {e}"

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(A, B, C, np.int32(M), np.int32(N), np.int32(K),
           threads=(tile_n, tile_m),
           grid=(math.ceil(N / tile_n), math.ceil(M / tile_m)))
    elapsed = (time.perf_counter() - t0) / n_iter

    flops = 2.0 * M * N * K
    gflops = flops / elapsed / 1e9
    return gflops, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LeanInfer Metal tile sweep")
    parser.add_argument("--model", default="qwen35-9b",
                        choices=list(MODEL_DIMS.keys()),
                        help="Model to calibrate tile sizes for")
    parser.add_argument("--m", type=int, default=1,
                        help="Batch size M (1 = decode, 32+ = prefill)")
    parser.add_argument("--n-iter", type=int, default=10,
                        help="Iterations per config for timing")
    parser.add_argument("--out", default="scripts/tile_config.json",
                        help="Output JSON path")
    parser.add_argument("--all-models", action="store_true",
                        help="Sweep all model dims")
    args = parser.parse_args()

    # Init Metal device
    dev = mc.Device()
    device_name = dev.device_name()
    print(f"Metal device: {device_name}")

    models_to_sweep = list(MODEL_DIMS.keys()) if args.all_models else [args.model]
    M = args.m

    results = {"device": device_name, "batch_m": M, "models": {}}

    for dtype in ["f32", "f16"]:
        configs = TILE_CONFIGS_F32 if dtype == "f32" else TILE_CONFIGS_F16
        best_gflops = 0.0
        best_cfg = None

        print(f"\n=== dtype={dtype} ===")
        print(f"{'TILE_M':>8} {'TILE_N':>8} {'TILE_K':>8} {'GFLOPS':>10}")

        # Use a representative large matmul for overall tile selection
        dims = MODEL_DIMS[args.model]
        K = dims["hidden"]
        N = dims["ffn"]

        for (tile_m, tile_n, tile_k) in configs:
            if tile_m > M and M < 8:
                tile_m_actual = 1  # degenerate tile for decode
            else:
                tile_m_actual = tile_m

            gflops, err = benchmark_tile(
                dev, M, N, K,
                tile_m_actual, tile_n, tile_k,
                dtype, args.n_iter
            )
            if err:
                print(f"  {tile_m_actual:>8} {tile_n:>8} {tile_k:>8}  SKIP ({err})")
                continue

            print(f"  {tile_m_actual:>8} {tile_n:>8} {tile_k:>8} {gflops:>10.1f}")

            if gflops > best_gflops:
                best_gflops = gflops
                best_cfg = {"tile_m": tile_m_actual, "tile_n": tile_n,
                            "tile_k": tile_k, "gflops": round(gflops, 1)}

        results[f"{dtype}_best"] = best_cfg
        print(f"\n  Best {dtype}: tile={best_cfg} at {best_gflops:.1f} GFLOPS")

    # Per-model recommendations
    recommended = {}
    for model_name in models_to_sweep:
        dims = MODEL_DIMS[model_name]
        # For decode (M=1), f16 with small tile is usually best
        # For prefill (M=32+), f16 with large tile wins
        best_dtype = "f16_best" if "f16_best" in results else "f32_best"
        cfg = results.get(best_dtype, {})
        recommended[model_name] = {
            "hidden":   dims["hidden"],
            "ffn_dim":  dims["ffn"],
            "tile":     f"{best_dtype.split('_')[0]}_{cfg.get('tile_m','?')}x{cfg.get('tile_n','?')}x{cfg.get('tile_k','?')}",
            "gflops":   cfg.get("gflops", 0.0)
        }

    results["recommended"] = recommended

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Print one-liner for ASSESSMENT.md
    best = results.get("f16_best") or results.get("f32_best") or {}
    print(f"\nBest config: {best} — add to ASSESSMENT.md Phase 2b notes")


if __name__ == "__main__":
    main()
