# LeanInfer — Technical Assessment & Architecture Plan

**Date:** 2026-03-24
**Status:** Phase 0a complete, Phase 0b/0c + Phase 1 next
**Last Updated:** 2026-03-26

---

## 0. Progress Tracker

### Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Git repo initialized | ✅ Done | `/home/junc/LeanInfer/` (7 commits, main branch) |
| ik_llama.cpp cloned + built (CPU/AVX2) | ✅ Built | `upstream/` (Ryzen 7735U, 8 threads, no CUDA) |
| Test model (Qwen 2.5-0.5B Q4_K_M) | ✅ Downloaded | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (469 MB) |
| Remote repo (GitHub) | ⬜ Not started | — |
| CI pipeline (Linux + macOS) | ⬜ Not started | — |

### Phase 0: Instrument & Measure

| Component | Status | Location |
|-----------|--------|----------|
| Profiler library | ✅ Working | `instrument/leaninfer_profiler.h` + `.cpp` |
| Chrome tracing JSON output | ✅ Working | Perfetto visualization confirmed |
| CMake integration (-DLEANINFER_PROFILE=ON) | ✅ Working | `upstream/src/CMakeLists.txt` |
| Hook: llama_decode (top-level) | ✅ Hooked | `upstream/src/llama.cpp` |
| Hook: ubatch loop (token batch) | ✅ Hooked | `upstream/src/llama.cpp` |
| Hook: graph_compute (ggml dispatch) | ✅ Hooked | `upstream/src/llama.cpp` |
| Hook: per-node in ggml compute thread | ✅ Hooked | `upstream/ggml/src/ggml.c` (thread 0 only) |
| Analysis script (CLI trace summary) | ✅ Working | `instrument/analyze.py` |
| Baseline trace (Qwen 2.5-0.5B) | ✅ Captured | `traces/first_run.json` — 72.7 tok/s, 93.9% in graph_compute |
| Per-node trace (Qwen 2.5-0.5B) | ✅ Captured | `traces/per_layer_run.json` — 13,362 events, full op breakdown |
| Per-node trace (Qwen 3.5-9B hybrid) | ✅ Captured | `traces/qwen35_9b_run.json` — 45,442 events, DeltaNet/Attention breakdown |
| Per-node trace (Qwen3-14B transformer) | ✅ Captured | `traces/qwen3_14b_run.json` — 40,066 events, FFN-dominated breakdown |
| Comparative analysis (hybrid vs transformer) | ✅ Complete | See comparative findings below — FFN 30→43%, DeltaNet only 9.2% overhead |
| Expert usage tracker (MoE) | ⬜ Not started | Needs MoE model to test |
| Benchmark harness (multi-turn, long-think) | ⬜ Not started | `scripts/benchmark.sh` |

**Phase 0b Profiling Results (Qwen 2.5-0.5B Q4_K_M, Ryzen 7735U AVX2, 8 threads):**

```
Compute time breakdown by operation type (decode, 17 tokens):

  Operation        % of Compute    What It Is
  ─────────────    ────────────    ──────────────────────────────────
  ffn_up_gate         35.7%       Fused FFN gate+up projection (big matmul)
  result_output       35.0%       Output head (896 → 151,936 vocab)
  ffn_out             18.0%       FFN down projection
  Qcur                 4.9%       Attention Q projection
  kqv_out              3.3%       Attention output projection
  fa                   1.8%       Flash attention
  norms + copies       <1%        RMSNorm, K/V cache copies, embeddings

  Key findings:
  • FFN = 53.7% of all compute (up_gate + down) — primary optimization target
  • Output head = 35.0% — disproportionately large on small vocab models
  • Attention = ~10% — already fast thanks to ik_llama's flash attention
  • Layers are perfectly uniform (2.5-3.2% each, no outliers)
  • On larger models (8B+), FFN share rises to 60-70%+ as output head shrinks proportionally
```

**Phase 0b Profiling Results (Qwen 3.5-9B Q4_K_M HYBRID, Ryzen 7735U AVX2, 8 threads):**

```
Baseline: 6.08 tok/s decode, 29.5 tok/s prefill
Architecture: qwen35 — 32 layers, 24 DeltaNet + 8 Attention (3:1 pattern)
DeltaNet layers: [0,1,2], [4,5,6], [8,9,10], [12,13,14], [16,17,18], [20,21,22], [24,25,26], [28,29,30]
Attention layers: [3, 7, 11, 15, 19, 23, 27, 31]

  Category             % of Compute    Details
  ─────────────────    ────────────    ──────────────────────────────────
  DeltaNet layers         49.9%       24 layers — recurrent state + linear attn
  FFN (all layers)        30.3%       ffn_up_gate fused matmul — biggest single op
  Output head             14.0%       result_output — vocab projection
  Attention layers        14.2%       8 layers — standard softmax attention
  qkv_mixed               13.6%       Q/K/V projections (both layer types)
  delta_net_fused_raw      1.4%       Actual DeltaNet state update (cheap!)
  linear_attn_out          5.3%       Linear attention output projection

  Key findings:
  • DeltaNet state update itself is only 1.4% — the projections around it dominate
  • Cooperative tensor fusion target: qkv_mixed + delta_net_fused → single dispatch
  • FFN remains the biggest single op (30.3%) — same as standard transformers
  • 3:1 DeltaNet:Attention confirmed — exactly as Qwen 3.5 spec documents
  • ssm_a tensors show "unknown" — ik_llama.cpp has partial Qwen 3.5 support
```

**Comparative Analysis: Hybrid (Qwen 3.5-9B) vs Standard Transformer (Qwen3-14B):**

```
Model Summary:
                                   Qwen 3.5-9B       Qwen3-14B
                                      (Hybrid)   (Transformer)
  ─────────────────────────────────────────────────────────────
  Architecture                          qwen35           qwen3
  Layers                                    32              40
  DeltaNet layers                           24               0
  Attention layers                           8              40
  Hidden dim                             3,584           5,120
  FFN dim                               18,944          17,408
  Decode tok/s (Ryzen 7735U)              6.08            4.16
  Total compute (32 tokens)            5,405 ms        8,029 ms
  Events captured                       45,442          40,066

Compute Category Breakdown:
                      Category    9B Hybrid      %     14B Xfmr      %
  ────────────────────────────────────────────────────────────────────
  FFN (gate+up+down)               1,640 ms  30.3%    3,478 ms  43.3%
  Attention ops                    1,013 ms  18.7%    1,366 ms  17.0%
  DeltaNet ops                       496 ms   9.2%        0 ms   0.0%
  Output head                        759 ms  14.0%      579 ms   7.2%
  Other (norms, copies, misc)      1,497 ms  27.7%    2,605 ms  32.4%

Top Operations Per Model:
  Qwen 3.5-9B (Hybrid)              Qwen3-14B (Transformer)
  ─────────────────────              ──────────────────────────
  ffn_up_gate    1,640ms  30.3%      ffn_up_gate    3,478ms  43.3%
  result_output    759ms  14.0%      Qcur             746ms   9.3%
  qkv_mixed        733ms  13.6%      result_output    576ms   7.2%
  linear_attn_out  286ms   5.3%      attn_out         519ms   6.5%
  Qaux             184ms   3.4%      l_out            152ms   1.9%
  delta_net_fused   77ms   1.4%      (norms/copies dominate rest)

Key Findings — What The Data Proves:

  1. FFN IS KING — AND GROWS WITH MODEL SIZE
     30.3% on 9B → 43.3% on 14B → estimated 50%+ on 27B.
     Every FFN optimization (quantization, expert paging, cooperative
     tensor fusion) has outsized impact on larger models.
     This confirms Phase 2 and Metal backend priorities.

  2. DELTANET IS CHEAP
     Only 9.2% overhead to replace 24 attention layers with recurrent state.
     The hybrid architecture gives 75% fewer attention layers (8 vs 40) at a
     fraction of the cost. Result: Qwen 3.5-9B is 46% FASTER than Qwen3-14B
     (6.08 vs 4.16 tok/s) despite being a newer, more complex architecture.
     DeltaNet layers also eliminate KV cache for those 24 layers — huge RAM savings.

  3. DELTANET STATE UPDATE ITSELF IS TRIVIAL
     delta_net_fused_raw = only 1.4% of compute. The expensive parts are
     the projections feeding it (qkv_mixed 13.6%, linear_attn_out 5.3%).
     Cooperative tensor fusion target: fuse projections + state update into
     a single dispatch to eliminate intermediate device memory round-trips.

  4. OUTPUT HEAD SHRINKS PROPORTIONALLY
     14.0% on 0.5B → 14.0% on 9B → 7.2% on 14B → <5% on 27B+.
     Not worth optimizing specifically. It's a one-time cost per token.

  5. ATTENTION IS CONSTANT ~17-19% REGARDLESS OF ARCHITECTURE
     Hybrid (8 layers): 18.7%. Standard (40 layers): 17.0%.
     Each hybrid attention layer does more work (handles context that
     DeltaNet compressed). Flash attention already handles this well.

  6. "OTHER" CATEGORY IS 28-32% — THE HIDDEN OPPORTUNITY
     Norms, KV cache copies, unnamed intermediate nodes. These are the exact
     device memory round-trips that cooperative tensors would eliminate.
     On Metal with cooperative tensors: potentially 5x reduction in this traffic.

  Optimization Priority (confirmed by data):
  ──────────────────────────────────────────
  #1  FFN quantization + expert paging       (30-43% of compute, growing)
  #2  Kernel fusion — CUDA + Metal           (28-32% intermediate traffic — see §4.4b)
  #3  KV cache compression                  (attention benefits, DeltaNet doesn't need KV)
  #4  DeltaNet state management             (9.2% — fix correctness first, optimize later)
```

### Phase 1: Fix Qwen 3.5

| Component | Status | Location |
|-----------|--------|----------|
| Hybrid memory manager (dual recurrent + KV) | ✅ Working | `src/llama.cpp` (seq_rm fix) + `examples/server/server-context.cpp` (checkpoint fix) |
| Thinking control layer (--no-think) | ✅ Working | `common/common.h` + `common/common.cpp` — bans `<think>` token, sets reasoning_budget=0, presence_penalty=1.5 |
| Recurrent state quantization (FP16) | ✅ Working | `src/llama.cpp` (alloc FP16) + `src/llama-delta-net.cpp` (cast on read, auto-convert on write) — 50% state memory reduction |

### Phase 2: RAM Reduction

| Component | Status | Location |
|-----------|--------|----------|
| Tiered KV cache + CoT eviction | ⬜ Not started | — |
| Quantization presets (7 model configs + 3 sampling) | ✅ Working | `configs/presets/` (quality/balanced/lean/ultra-lean) + `configs/sampling/` + `scripts/quantize.sh` |
| Frequency-aware expert paging (MoE) | ⬜ Not started | — |

### Phase 2b: Metal Backend (M5/A19 Optimized)

| Component | Status | Location |
|-----------|--------|----------|
| Metal 4 backend skeleton (MTLDevice, ggml reg) | ⬜ Not started | — |
| TensorOps + MPP GEMM dispatch | ⬜ Not started | — |
| Cooperative tensor fusion (attn, FFN, DeltaNet) | ⬜ Not started | — |
| Unified memory allocator (MTLBuffer heaps) | ⬜ Not started | — |
| Tile size auto-tuning (tile_sweep.py) | ⬜ Not started | `scripts/tile_sweep.py` |
| M5-specific quant presets (Q8 attn + Q4 FFN) | ⬜ Not started | — |

### Phase 3: Speed & Intelligence

| Component | Status | Location |
|-----------|--------|----------|
| Reasoning-aware speculative decoding | ⬜ Not started | — |
| Fused DeltaNet block (proj + state + output) | ⬜ Not started | Novel — CUDA + Metal (see §4.4b) |
| Fused FFN block (norm + gate + SiLU + up + down) | ⬜ Not started | CUTLASS epilogues (CUDA), cooperative tensors (Metal) |
| RMSNorm + projection fusion | ⬜ Not started | Both platforms, medium complexity |
| Default runtime repacking (-rtr) | ⬜ Not started | — |
| Predictive expert prefetch | ⬜ Not started | — |
| Dynamic CPU/GPU operator routing | ⬜ Not started | — |

---

## 1. Problem Statement

Running 8-27B reasoning models on consumer hardware (16-32 GB RAM) is painful:

| Model | FP16 | Q4 | + KV Cache (reasoning) | Real RAM Needed |
|---|---|---|---|---|
| Qwen 3.5-9B | 18 GB | ~5 GB | +2-4 GB | ~7-9 GB |
| Qwen 3.5-27B | 54 GB | ~15 GB | +4-8 GB | ~19-23 GB |
| DeepSeek-R1-14B | 28 GB | ~8 GB | +3-6 GB | ~11-14 GB |

Reasoning models are especially brutal because chain-of-thought generates thousands of intermediate thinking tokens that inflate KV cache far beyond normal chat use.

New hybrid architectures (Qwen 3.5's Gated DeltaNet) break existing inference engines' memory management assumptions entirely.

---

## 2. Landscape Analysis

### 2.1 llama.cpp (mainline)

**Strengths:**
- Universal model support (any GGUF)
- 18 hardware backends (Metal, CUDA, Vulkan, SYCL, etc.)
- Massive community, active development
- 6 speculative decoding strategies
- Granular quantization (1.5-bit to FP32 with imatrix)

**Weaknesses:**
- All neurons computed uniformly — no sparsity exploitation
- Layer-level GPU offloading only (not neuron-level)
- No reasoning-specific optimizations
- Qwen 3.5 broken: full prompt reprocessing every turn (issue #20225)
- Checkpoint logic assumes SWA, incompatible with recurrent state
- `enable_thinking` passthrough broken by PR #18675 refactor

### 2.2 ik_llama.cpp (ikawrakow's fork)

**Background:** Iwan Kawrakow created k-quants, i-quants, and imatrix for mainline llama.cpp (108 commits, Apr 2023 - Mar 2024). Hard-forked June 2024. Sole primary developer.

**Strengths (over mainline):**
- **CPU performance:** 2.9x faster prompt processing (Zen4), 3.4x faster (ARM NEON)
- **IQK quants:** 2.7x lower quantization error at same bit-width
- **Trellis quants (IQ1-4_KT):** Seed-based, near-optimal compression
- **FlashMLA-3:** Fastest CPU-only DeepSeek inference
- **Fused MoE (`-fmoe`):** 26% faster prompt, 7% faster generation
- **Smart expert reduction (`-ser`):** Skip low-probability experts, ~20% speedup
- **Hadamard K-cache (`-khad`):** 3x error reduction for Q4 K-cache
- **Q8_KV:** Per-row scaling, 100% reproducible, 15% faster
- **Runtime repacking (`-rtr`):** 1.6x speedup without preprocessing
- **Tensor overrides (`--custom-q`, `-ot`):** Surgical per-tensor quantization and placement
- **Graph split:** Tensor parallelism within layers for multi-GPU (~60% faster PP)

**Weaknesses:**
- Only CPU (AVX2+, NEON) and CUDA backends fully maintained
- Single developer — bus factor of 1
- Qwen 3.5 bugs: VRAM leak with f32 SSM tensors, endless tool call loops on aarch64, tool calls producing only 1 token
- Not a GitHub fork — cannot trivially merge upstream llama.cpp changes

### 2.3 PowerInfer (Tiiny AI / SJTU-IPADS)

**Core Insight:** Neuron activations follow a power-law distribution. ~17% of neurons account for 80% of activations. "Hot" neurons preloaded on GPU, "cold" neurons computed on CPU.

**Strengths:**
- 11.7x speedup on Falcon-40B (RTX 4090 vs llama.cpp)
- Neuron-level placement via ILP solver (offline profiling)
- Adaptive activation predictors per layer (>95% accuracy)
- TurboSparse dReLU: 90% sparsity with quality improvement
- SmallThinker: 108 tok/s for 4B model on desktop CPU
- PowerInfer-2: 47B model on a phone at 11.68 tok/s

**Weaknesses:**
- **Requires ReLU-family activations** for full benefit (SiLU/SwiGLU models only get 1.5-1.7x)
- Narrow hardware support (CUDA + x86 only)
- Far fewer models supported
- No speculative decoding
- Small ecosystem

**What we can borrow:**
- Hot/cold concept applied to **MoE experts** (not neurons) — works regardless of activation function
- Offline profiling methodology for expert frequency analysis
- The architectural pattern of separating placement decisions (offline) from execution (online)

---

## 3. Qwen 3.5 — The Architecture Challenge

### 3.1 What It Is

Qwen 3.5 is a **hybrid Gated DeltaNet + Attention** model — NOT a standard transformer.

**Layer structure (27B):**
- 64 total layers
- 16 repeating blocks of: `3× (Gated DeltaNet + FFN) → 1× (Gated Attention + FFN)`
- **75% recurrent (O(n))**, 25% attention (O(n²))

**Gated DeltaNet layers:**
- Linear attention variant with delta rule for hidden state updates
- Fixed-size recurrent state matrix (not a growing KV cache)
- Depthwise causal 1D convolution with carry state
- 48 linear attention heads for V, 16 for QK, head dim 128

**Gated Attention layers:**
- Standard grouped-query attention (24 Q heads, 4 KV heads)
- Head dim 256, RoPE dim 64
- These use normal KV cache

**Model variants:**
- Dense: 0.8B, 2B, 4B, 9B, 27B
- MoE: 35B-A3B, 397B-A17B (512 experts, 10 routed + 1 shared)
- Context: 262K native, ~1M with YaRN

### 3.2 Why It Breaks Everything

**Problem 1: Full prompt reprocessing every turn (llama.cpp #20225)**

The checkpoint validation logic uses SWA semantics:
```
pos_min_thold = max(0, n_past - n_swa)
```
For recurrent models, `pos_min` returns the full sequence length (recurrent state encompasses all positions). So `pos_min > pos_min_thold` is ALWAYS true → checkpoints rejected → full reprocessing. A 15K-token conversation takes ~8 minutes per turn.

**Problem 2: Cannot partially truncate recurrent memory**

`llama_memory_seq_rm()` with a partial range fails for recurrent layers. You can't delete tokens 50-100 from a compressed state matrix. Recovery code tries checkpoint restore but position accounting doesn't sync between recurrent state and KV cache → "Invalid input batch" errors.

**Problem 3: Thinking always on**

PR #18675 replaced `chat_params.enable_thinking` with `common_chat_templates_support_enable_thinking()` which only checks capability, not the actual setting. The model thinks regardless of flags.

**Current workaround (fragile):**
```bash
--chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0
# Both flags required together. Plus:
# presence_penalty=1.5 recommended to prevent thinking loops
```

**Problem 4: Parallel slots crash (fixed)**

DeltaNet state not checkpointed properly with `--parallel 3+`. Fixed by PR #20232.

### 3.3 The Silver Lining

Qwen 3.5's architecture is actually **favorable** for RAM-constrained inference once the memory manager is fixed:
- 75% of layers use **fixed-size** recurrent state instead of growing KV cache
- Only 25% of layers contribute to KV cache growth
- The recurrent state is small and constant regardless of sequence length
- O(n) complexity for 75% of computation → faster long-context inference

---

## 4. Architecture Plan

### 4.1 Why Fork ik_llama.cpp

| Factor | Mainline llama.cpp | ik_llama.cpp |
|---|---|---|
| CPU speed | Baseline | 2-3x faster |
| Quant quality | Good | Best available |
| MoE support | Basic | FlashMLA, fused MoE, expert reduction |
| KV innovations | Standard | Hadamard, Q8_KV, per-row scaling |
| Backend breadth | 18 backends | CPU + CUDA only |
| Contributor base | Hundreds | ~1 primary developer |
| Our target HW | CPU + CUDA | CPU + CUDA ✓ |

We need depth on CPU + CUDA, not breadth across 18 backends. ik_llama.cpp is the right base.

#### Why Metal Matters (Phase 2b — Apple Silicon)

Apple Silicon's **unified memory architecture** changes the game. Unlike CUDA where you pay a PCIe tax copying between CPU RAM and VRAM, Metal GPU, CPU, and ANE all share the same physical memory pool. A 32 GB M-series Mac has 32 GB available to *everything* — no offloading needed.

**What Metal gives us (without ANE):**
- Full GPU acceleration via Metal compute shaders (GEMM, attention, fused DeltaNet)
- Zero-copy unified memory — `MTLBuffer` shared heaps, no CPU↔GPU transfers
- Hot/Warm/Cold tiering using unified heaps with async prefetch/evict
- A 27B Q4 model + full KV cache can fit entirely in a 32 GB Mac's unified pool
- Runtime flags: `--backend=metal|cuda|cpu`, `--metal-unified=true`
- macOS CI targets + Metal vs CPU microbenchmarks

**What ANE/CoreML would add on top (Phase 4 — Optional):**
- Export small fused operators (DeltaNet blocks, per-expert FFNs) as `.mlmodelc`
- Marginal latency wins on hot, low-power small kernels
- Better battery life on MacBook (ANE is more power-efficient than Metal GPU)
- Requires quant presets (int8/uint8/bfloat16/FP16) and CoreML conversion toolchain
- CoreML has strict op constraints — DeltaNet's gated delta rule may not map cleanly

**Bottom line:** Metal alone delivers ~90% of the Apple Silicon benefit. ANE is a power-efficiency optimization, not a correctness or speed requirement. Phase 4 can be deferred indefinitely without impacting functionality.

### 4.2 Phase 0: Instrument & Measure (Foundation)

> **"If you don't measure, you're guessing."** This phase runs BEFORE and ALONGSIDE all other phases. Every optimization we make must be validated by data, not intuition.

#### 0a. Latency Profiler

Instrument ik_llama.cpp's inference loop to emit per-layer timing data:

```
┌─────────────────────────────────────────────────────────┐
│                  LeanInfer Profiler                      │
├─────────────────────────────────────────────────────────┤
│  Per-layer metrics:                                     │
│  • layer_id, layer_type (DeltaNet | Attention | FFN)    │
│  • compute_time_us                                      │
│  • memory_access_bytes (weights loaded)                 │
│  • cache_hit/miss (KV or recurrent state)               │
│  • device (CPU | CUDA | Metal)                          │
│                                                         │
│  Per-token metrics:                                     │
│  • total_latency_us                                     │
│  • time_in_matmul vs time_in_attention vs time_in_io    │
│  • token_type (system | user | thinking | answer)       │
│  • kv_cache_size_bytes at this token                    │
│  • recurrent_state_size_bytes at this token             │
│                                                         │
│  Per-expert metrics (MoE only):                         │
│  • expert_id, layer_id                                  │
│  • activation_count (cumulative)                        │
│  • load_source (VRAM | RAM | SSD)                       │
│  • load_latency_us (if paged in)                        │
│  • routing_probability                                  │
│                                                         │
│  Session summary:                                       │
│  • p50/p95/p99 token latency                            │
│  • peak_memory_bytes (VRAM, RAM, total)                 │
│  • cache_eviction_count                                 │
│  • expert_page_fault_count (MoE)                        │
│  • time_breakdown: compute% vs memory% vs io%           │
└─────────────────────────────────────────────────────────┘
```

Implementation — 3 hook points in ik_llama.cpp:

```
Hook 1: Tensor Access (MOST IMPORTANT)
  Where: ggml_compute_forward_*() / ggml_graph_compute()
  What:  Wrap every weight access with runtime_get_tensor()
  Why:   This is the entry point for streaming, caching, prefetch

Hook 2: Layer Execution Loop
  Where: for (i = 0; i < n_layers; i++) { eval_layer(ctx, i); }
  Before: predictor.predict(i) → prefetch(predicted)
  After:  tracker.record(i) → cache.update()
  Why:    Prediction, learning, adaptation loop

Hook 3: KV Cache Access
  Where: kv_cache[kv_index] inside attention code
  What:  Inject compression, eviction, tiering
  Why:    Second biggest bandwidth win after weights
```

- `--profile` flag enables collection; `--profile-output <file>` writes JSON traces
- Compatible with Chrome's `chrome://tracing` format for visual flame graphs
- Zero overhead when disabled (compile-time `#ifdef LEANINFER_PROFILE`)
- **Do NOT touch:** ggml math kernels, attention logic, tokenizer — we optimize data movement, not math

#### 0b. Expert Usage Tracker (MoE)

For MoE models, track expert activation patterns across inference:
- Per-expert cumulative activation count
- Expert co-activation matrix (which experts fire together)
- Layer-to-layer expert transition probabilities
- Output: frequency profile files consumed by Phase 2c (expert paging) and Phase 3 (predictive prefetch)
- Runs as offline profiling pass on calibration corpus (~1M tokens), same approach as PowerInfer's profiler

#### 0c. Benchmark Harness

Standardized benchmarks for reasoning workloads (not generic text generation):
- **Multi-turn reasoning**: 5-turn conversation with chain-of-thought, measure latency per turn
- **Long thinking**: Single query requiring 2000+ thinking tokens, measure cache growth + eviction
- **Expert coverage**: Track what % of experts are used across a diverse prompt set
- **Memory high-water mark**: Peak RAM + VRAM across full session
- Compare against: baseline ik_llama.cpp, mainline llama.cpp, Ollama

> **Rule: No optimization lands without before/after benchmark numbers.**

### 4.3 Phase 1: Fix What's Broken (Qwen 3.5 Compatibility)

#### 1a. Hybrid Memory Manager

The root cause is that all memory is treated as KV cache. Build a dual system:

```
┌─────────────────────────────────────────────┐
│            Hybrid Memory Manager             │
├──────────────────────┬──────────────────────┤
│   Recurrent State    │      KV Cache        │
│  (DeltaNet layers)   │  (Attention layers)  │
│                      │                      │
│  • Fixed-size        │  • Growing           │
│  • Non-truncatable   │  • Supports trim     │
│  • Checkpoint =      │  • Checkpoint =      │
│    full state copy   │    standard KV save  │
│  • pos_min: N/A      │  • pos_min: normal   │
│    (skip SWA check)  │    SWA semantics     │
└──────────────────────┴──────────────────────┘
```

Implementation:
- Separate `pos_min` calculation by layer type
- Recurrent layers excluded from SWA threshold check
- Recurrent checkpoints: snapshot full state matrix (it's fixed-size, cheap to copy)
- `llama_memory_seq_rm()`: for recurrent layers, either no-op (keep state) or full reset (no partial)
- Sync mechanism: recurrent state tracks a sequence position counter alongside the attention KV cache

#### 1b. Thinking Control Layer

- Fix `enable_thinking` passthrough (revert PR #18675 logic)
- Add `--no-think` server flag combining:
  - `enable_thinking: false` in template kwargs
  - `reasoning_budget: 0`
  - `<think>` added to banned token list (fallback)
  - `presence_penalty: 1.5` auto-applied
- Per-request API toggle: `"thinking": true|false`
- Sampling presets:
  - Thinking mode: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
  - No-think mode: temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
  - Coding mode: temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0

#### 1c. Recurrent State Quantization

- Quantize DeltaNet state matrices from FP32 → Q8_KV (ik_llama's per-row scaling)
- 75% memory reduction for recurrent state
- For 27B: 48 DeltaNet layers × state matrix → significant savings at constant cost

### 4.4 Design Principles: Effective Bandwidth

> **"You don't win by widening the doorway. You win by redesigning the workflow around the doorway's limit."**

The fundamental bottleneck in autoregressive decoding is **memory bandwidth, not compute**. The hardware spends most of its time waiting for weights and KV cache data, not doing math. Every optimization in this project attacks one of these 5 levers:

```
┌─────────────────────────────────────────────────────────────────┐
│                 5 Levers of Effective Bandwidth                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MOVE FEWER BYTES                                            │
│     Tiered quantization: hot=Q6/Q8, warm=Q4, cold=Q2/Q3        │
│     KV cache quantization: Q4 + Hadamard transforms             │
│     Skip inactive experts entirely (-ser)                       │
│                                                                 │
│  2. MAKE EACH BYTE DO MORE WORK                                │
│     Weight prepacking: store in kernel-native blocked format    │
│     ik_llama -rtr: runtime repacking to optimal SIMD layout     │
│     Fused kernels: fewer intermediate reads/writes              │
│     MARLIN-style quant-specific layouts                         │
│                                                                 │
│  3. REUSE DATA BEFORE EVICTING                                 │
│     Keep attention/router/embeddings ALWAYS hot                 │
│     Hot expert cache with LRU + cooldown (prevent thrashing)    │
│     FlashAttention: tile and keep working data on-chip longer   │
│                                                                 │
│  4. REDUCE KV CACHE TRAFFIC                                    │
│     GQA (Qwen 3.5: 24 Q heads, 4 KV heads)                    │
│     DeltaNet: 75% of layers have fixed-size state, not KV      │
│     CoT eviction: dump thinking tokens after </think>           │
│     Q4 K-cache with Hadamard transforms                        │
│                                                                 │
│  5. OVERLAP MOVEMENT WITH COMPUTE                              │
│     Predictive prefetch: load layer N+1 while computing N      │
│     Async expert paging: SSD→RAM→VRAM in background thread     │
│     Speculative decoding: use idle compute during bandwidth     │
│     stalls                                                      │
│     Tensor grouping: batch load blk.N.* together (fewer seeks) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Three inviolable rules:**

1. **Never stream on the critical path.** If a weight isn't loaded, fall back to a slower path — never block compute waiting for I/O.
2. **Keep attention, embeddings, and routers always hot.** These are accessed every single token. Streaming them is always a loss.
3. **Anti-thrash cooldown on tier transitions.** A tensor must stay in its tier for a minimum duration before promotion/demotion. Without this, marginal tensors flip hot↔cold every few tokens, wasting bandwidth on movement instead of compute.

**Memory hierarchy (what we're building):**

```
┌──────────────────────────────────────────────────────────┐
│ Tier 0  GPU registers / L1 cache    (hardware, ignore)   │
├──────────────────────────────────────────────────────────┤
│ Tier 1  HOT — VRAM / RAM                                │
│         Attention, embeddings, router, hot experts        │
│         Format: Q6/Q8, kernel-native prepacked           │
│         Policy: always resident, never evict             │
├──────────────────────────────────────────────────────────┤
│ Tier 2  WARM — Compressed RAM                            │
│         Warm experts, older KV cache, secondary FFN      │
│         Format: Q3/Q4, packed                            │
│         Policy: resident, evictable under pressure       │
├──────────────────────────────────────────────────────────┤
│ Tier 3  COLD — NVMe SSD (GGUF random access)            │
│         Rare experts, evicted thinking tokens            │
│         Format: Q2/Q3 in GGUF, seek by tensor offset    │
│         Policy: async prefetch on prediction, never      │
│         block compute                                    │
└──────────────────────────────────────────────────────────┘

32 GB system budget example:
  HOT:  16 GB — attention + top experts
  WARM: 10 GB — compressed weights
  COLD: disk  — everything else
```

**GGUF enables this natively.** Each tensor has a name, offset, size, and dtype in the tensor directory. We can `seek(offset) → read(size)` for any individual tensor without loading the full file. llama.cpp already mmaps the GGUF, but with no intelligence about what to page. We add the intelligence.

**NUMA-aware placement (multi-socket systems):** On dual-socket Xeon/EPYC systems, keep tensors on the NUMA node closest to the cores using them. llama.cpp community has documented 20-40% speedup from proper NUMA binding on multi-socket machines.

### 4.4b Kernel Fusion Strategy (CUDA + Metal)

Our profiling revealed 28-32% of compute is "Other" — intermediate device memory round-trips between operations. This exists on **both** CUDA and Metal. Eliminating these is a major opportunity on all hardware.

**The problem — unfused operations:**
```
kernel 1: RMSNorm        → write to device memory
kernel 2: Q projection   → read from device, write to device
kernel 3: K projection   → read from device, write to device
kernel 4: attention       → read from device, write to device
kernel 5: output proj     → read from device, write to device
... each kernel launch has overhead + device memory round-trip
```

**The solution — fused operations:**
```
kernel 1: RMSNorm + QKV projection  → stays in registers / shared memory
         → attention                → stays in shared memory
         → output projection        → ONE write to device memory
```

#### Platform Comparison

| Concept | Metal 4 | CUDA |
|---|---|---|
| Keep intermediates in fast storage | **Cooperative tensors** (thread registers) | **Register tiling + shared memory** |
| Fuse multiple ops in one dispatch | **TensorOps chain** (single shader) | **Kernel fusion** (single kernel launch) |
| Hardware matrix acceleration | **Neural Accelerators** via execution_simdgroup | **Tensor Cores** via WMMA/MMA intrinsics |
| Tooling | TensorOps + MPP (clean API) | CUTLASS templates / Triton / hand-written CUDA |
| Maturity | New (Metal 4, 2025) | Established (since Volta, 2017) |

#### What ik_llama.cpp Already Fuses (Inherited)

| Fused Op | What It Eliminates | Status |
|---|---|---|
| `fused_up_gate` | FFN gate + up in one matmul | ✅ Already done |
| `fused_moe` (-fmoe) | MoE gate + up + down in one dispatch | ✅ Already done |
| `fused_mmad` | Matmul + add | ✅ Already done |
| Flash Attention (`-fa`) | Entire attention in one kernel | ✅ Already done |

#### Our Fusion Targets (What We Add)

| Fusion Target | Round-Trips Eliminated | Platform | Complexity | Impact |
|---|---|---|---|---|
| **RMSNorm + QKV projection** | 1 per layer | CUDA + Metal | Medium | ~3-5% speedup |
| **DeltaNet: proj + state update + output** | 3-4 per DeltaNet layer | CUDA + Metal | Hard (novel) | ~8-12% speedup |
| **FFN: norm + gate + SiLU + up + down** | 4 per layer | CUDA + Metal | Hard | ~10-15% speedup |
| **Attention: norm + QKV + flash_attn + out** | 2 per attn layer | CUDA + Metal | Medium | ~3-5% speedup |

**Estimated total impact per model:**

| Model | Layers | Fused Round-Trips Eliminated | Estimated Speedup |
|---|---|---|---|
| Qwen 3.5-9B | 32 (24 DeltaNet + 8 Attn) | ~100 per token | **15-25%** |
| Qwen 3.5-27B | 64 (48 DeltaNet + 16 Attn) | ~200 per token | **15-25%** |
| Qwen3-14B | 40 (all Attn) | ~120 per token | **10-20%** |

#### CUDA Implementation Path

```
Phase A: Use ik_llama.cpp's existing fusions (already done)
  • fused_up_gate, fused_moe, flash_attention — inherited

Phase B: Add RMSNorm + projection fusions (CUTLASS epilogues)
  • CUTLASS fused GEMM with custom epilogue: norm → matmul in one kernel
  • Applies to both QKV and FFN entry points
  • Template-based — works across quant types

Phase C: DeltaNet block fusion (novel — our unique contribution)
  • Fuse: qkv_mixed → delta_net_fused_raw → linear_attn_out
  • Keep recurrent state in shared memory across the 3 DeltaNet layers
  • No existing implementation anywhere — this is greenfield
  • Targets the 9.2% DeltaNet compute + eliminates intermediate traffic

Phase D: Full FFN block fusion
  • Fuse: norm → gate → SiLU → up → multiply → down
  • Requires careful shared memory management (FFN dims are large)
  • Biggest single-op impact (30-43% of compute made more efficient)
```

#### Metal Implementation Path

```
Same fusion targets, cleaner API:
  • Cooperative tensors for all intermediates
  • TensorOps chain within single compute dispatch
  • Shader ML for custom DeltaNet fusions
  • See Phase 2b Metal Backend section for full details
```

**The DeltaNet fusion is unique to us on both platforms.** Nobody else is optimizing for Qwen 3.5's hybrid architecture. Flash Attention was the last major fusion breakthrough in inference — DeltaNet block fusion could be the next one for hybrid models.

### 4.5 Phase 2: RAM Reduction Engine

#### 2a. Tiered KV Cache with CoT Eviction

```
┌───────────────────────────────────────────┐
│              KV Cache Tiers               │
├───────────────────────────────────────────┤
│ HOT    System prompt + recent context     │
│        Q8_KV • Always resident            │
├───────────────────────────────────────────┤
│ WARM   Conversation history               │
│        Q4 + Hadamard (-khad)              │
│        Resident, evictable under pressure │
├───────────────────────────────────────────┤
│ COLD   Thinking tokens (<think>...</think>)│
│        Evicted after </think> generated   │
│        Recovers 60-80% of reasoning cache │
└───────────────────────────────────────────┘
```

- Hadamard K-cache transforms (ik_llama's `-khad`) make Q4 K-cache viable
- Automatic tier assignment based on token role (system/user/thinking/answer)
- Eviction triggered on `</think>` token detection
- For Qwen 3.5: only 25% of layers have KV cache → already 75% smaller than pure transformer

#### 2b. Quantization Presets

Pre-built configs using ik_llama's `--custom-q` regex overrides:

**Qwen 3.5-27B targets:**
| Preset | Strategy | Model Size | + Cache | Total |
|---|---|---|---|---|
| quality | IQ4_K uniform | ~15 GB | ~2 GB | ~17 GB |
| balanced | attn=IQ4_K, ffn=IQ3_K | ~12 GB | ~1.5 GB | ~13.5 GB |
| lean | attn=IQ4_K, ffn=IQ3_K, Q4+khad cache | ~10 GB | ~1 GB | ~11 GB |
| ultra-lean | IQ2_K + Q4 cache | ~8 GB | ~1 GB | ~9 GB |

**Qwen 3.5-9B targets:**
| Preset | Strategy | Model Size | + Cache | Total |
|---|---|---|---|---|
| quality | IQ4_K uniform | ~5 GB | ~1 GB | ~6 GB |
| lean | attn=IQ4_K, ffn=IQ3_K, Q4+khad cache | ~3.5 GB | ~0.5 GB | ~4 GB |

#### 2c. Frequency-Aware Expert Paging (MoE)

Apply PowerInfer's hot/cold concept to MoE experts:

**Offline phase:**
- Profile expert activation frequencies on calibration corpus (~1M tokens)
- Rank experts by frequency across layers
- Generate placement policy: hot experts → GPU, warm → RAM, cold → SSD

**Online phase:**
- Hot experts (top 20-30%) always resident in VRAM
- Warm experts in system RAM, fetched on activation
- Cold experts paged from NVMe SSD with prefetching
- **Tensor grouping:** batch load `blk.N.ffn_up + ffn_gate + ffn_down` together per expert (fewer SSD seeks, better locality)
- **Anti-thrash cooldown:** experts must stay in their tier for N tokens before promotion/demotion (prevents marginal experts from ping-ponging between tiers)
- Combine with ik_llama's `-ser Kmin,t` to skip improbable experts entirely
- Combine with `-ot exps=CPU` for surgical placement

**For Qwen 3.5-MoE (397B-A17B):**
- 512 total experts, 10 routed + 1 shared per token
- Only ~11 experts active per layer per token (2.1% utilization)
- Hot 30% (~154 experts) covers majority of activations
- Estimated VRAM for hot experts: ~8-12 GB (fits RTX 3060)

### 4.6 Phase 2b: Metal Backend (Apple Silicon — M5/A19 Optimized)

#### The M5 Opportunity

Apple's M5 (Oct 2025) and M5 Pro/Max (Mar 2026) introduced **GPU Neural Accelerators** — dedicated matrix multiplication hardware inside every GPU core, functionally equivalent to NVIDIA tensor cores. Combined with Metal 4's first-class tensor support, this is the most significant Apple Silicon upgrade for local LLM inference.

**Hardware specs that matter for us:**

| Chip | GPU Cores | Neural Accels | FP16 Matmul | Memory | Bandwidth |
|---|---|---|---|---|---|
| M5 | 10 | 10 | ~14.8 TFLOPS | 32 GB | 153.6 GB/s |
| M5 Pro | 20 | 20 | ~29.6 TFLOPS | 64 GB | 307 GB/s |
| M5 Max (high) | 40 | 40 | ~70 TFLOPS | 128 GB | 614 GB/s |

**Apple's own LLM benchmarks (M5 vs M4):**

| Model | Prefill Speedup | Token Gen Speedup |
|---|---|---|
| Qwen 14B (4-bit) | 4.06x | 1.19x |
| Qwen 30B MoE (4-bit) | 3.52x | 1.25x |
| Qwen3 8B | 3.65x | 2.95x |

**Key insight: prefill is now cheap, decode is still expensive.** The Neural Accelerators deliver 3.5-4x prefill speedup (compute-bound), but token generation is still bandwidth-limited (~15 GB/s per core vs ~93 GB/s needed for full Neural Accelerator utilization). This validates LeanInfer's focus — every decode-time optimization we build (quantization, KV compression, expert paging, speculative decoding) directly addresses the M5's bottleneck.

#### Metal 4 TensorOps (Primary Compute Path)

Metal 4 (WWDC 2025) makes tensors first-class citizens in the API and Metal Shading Language. This is our primary compute interface — **not** hand-written Metal shaders.

```
┌─────────────────────────────────────────────────────────────┐
│                  Metal 4 Compute Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TensorOps API                                              │
│  • Direct programming interface to Neural Accelerators      │
│  • First-class multi-dimensional tensor types               │
│  • Hardware-accelerated matmul dispatched to Neural Accels  │
│  • Supported types: FP16, INT8 (native HW), BF16 (SW)      │
│                                                             │
│  Metal Performance Primitives (MPP)                         │
│  • High-perf matrix multiplication kernels                  │
│  • Convolution primitives                                   │
│  • Integrated into Shader ML for fused compute              │
│  • Optimal tiling for 32x32+ matrices                       │
│                                                             │
│  Shader ML                                                  │
│  • Embed ML inference directly within shaders               │
│  • No sync overhead between device memory and compute       │
│  • Fuse multiple ops into single dispatch                   │
│                                                             │
│  LeanInfer Integration Points:                              │
│  • GEMM: MPP matmul → Neural Accelerators                  │
│  • Attention: TensorOps for Q*K^T and attn*V               │
│  • DeltaNet: Fused gated delta rule via Shader ML           │
│  • Expert FFN: Per-expert matmul via MPP                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why TensorOps over hand-written kernels:**
- Neural Accelerators process matrices in hardware-optimal 32x32 tiles — TensorOps handles tiling automatically
- Matrix transpose is free in hardware — no need to pre-transpose weights
- Apple will optimize TensorOps across silicon generations; hand-written kernels won't get those gains
- MLX already has preliminary Neural Accelerator support via this path

**Backwards compatibility:** TensorOps API is the same across M1 through M5 (and A14+). On M5/A19 hardware, TensorOps dispatches to the dedicated Neural Accelerators (3.5-4x prefill gain). On M1-M4 hardware, the same API runs on standard GPU shader cores — still faster than CPU, just without hardware matrix acceleration. **One code path covers the entire ~100M+ M-series installed base.** No chip-specific branching, no separate kernels. This is a significant advantage over CUDA, where different GPU compute capabilities require different kernel configurations.

#### Cooperative Tensors (Eliminating Device Memory Round-Trips)

Metal 4 introduces three tensor storage types. The critical one for us is `cooperative_tensor`:

| Tensor Type | Storage Location | Use Case |
|---|---|---|
| `tensor_handle` | Device memory (unified RAM) | Model weights, KV cache — the large, persistent data |
| `tensor_inline` | Constructed on GPU | Build input tensors inside shaders on the fly |
| `cooperative_tensor` | **Thread registers + threadgroup SRAM** | **Intermediate results — never hits device memory** |

**How cooperative tensors save bandwidth:**

Traditional pipeline (without cooperative tensors):
```
weights → matmul → [write to device mem] → [read from device mem] → activation
→ [write to device mem] → [read from device mem] → next matmul → ...
```

With cooperative tensors:
```
weights → matmul → [stays in registers] → activation → [stays in registers]
→ next matmul → [stays in registers] → write final output to device mem
```

**Intermediate results never touch device memory.** The data is distributed across participating threads' register files (implementation-defined partitioning). This is analogous to NVIDIA's warp-level register tiling / WMMA fragments.

**Impact on LeanInfer decode performance:**

For Qwen 3.5-27B, each token processes 64 layers. Without cooperative tensors, each layer generates ~10+ intermediate device memory round-trips (RMSNorm output, Q/K/V projections, attention scores, FFN gate, FFN up, FFN down). With cooperative tensors fusing operations within a layer:

| Metric | Without Cooperative Tensors | With Cooperative Tensors |
|---|---|---|
| Device memory round-trips per layer | ~10+ | ~2 (load weights, write output) |
| Per-token device memory traffic (64 layers) | ~640+ accesses | ~128 accesses |
| Bandwidth pressure during decode | Very high | **~5x reduction in intermediate traffic** |

This directly attacks the decode bottleneck — the Neural Accelerators can compute faster than memory can feed them, so eliminating unnecessary memory traffic lets them stay fed longer.

**Fused operations enabled by cooperative tensors:**

```
Fused Attention Block (single dispatch, no device mem sync):
  RMSNorm → Q/K/V projection → attention scores → softmax → attn*V
  All intermediates in cooperative_tensor registers

Fused FFN Block (single dispatch):
  RMSNorm → gate projection → SiLU activation → up projection → multiply → down projection
  All intermediates in cooperative_tensor registers

Fused DeltaNet Block (single dispatch):
  Conv1D carry state → gated delta rule → state update → output projection
  All intermediates in cooperative_tensor registers
```

**Programming model:** Cooperative tensors are **explicit** — we design our Metal compute kernels to use them. They are not automatic. We choose the execution scope (`execution_simdgroup` for Neural Accelerator dispatch), declare intermediates as `cooperative_tensor`, and chain TensorOps within a single shader dispatch. The compiler handles the register allocation and thread distribution.

**Execution scopes for TensorOps:**

| Scope | Threads | Neural Accelerator? | Our Use |
|---|---|---|---|
| `execution_thread` | 1 | No | Fragment shaders, divergent paths |
| `execution_simdgroup` | 32 | **Yes (M5)** | **Primary: all GEMM, attention, FFN** |
| `execution_threadgroup` | All in group | No (perf drop) | Avoid for now |

**Optimal tile size:** 32×32 or larger for Neural Accelerator dispatch. K dimension must be multiple of 32. Optimal pipelining at **128×64×64** tiles (from llama.cpp Metal 4 integration benchmarks: 2.4x speedup on Llama 8B Q4_0 prefill).

**llama.cpp Metal 4 real-world results (M5):**
- Llama 8B Q4_0 pp512: **609 t/s** (vs 257 t/s baseline = 2.4x)
- Qwen3 0.6B FP16 pp512: **4,936 t/s** (vs 3,073 t/s = 1.6x)
- GPT-OSS 20B pp512: **846 t/s** (vs 415 t/s = ~2x)

#### Quantization Strategy for Neural Accelerators

The Neural Accelerators have native hardware support for specific types:

| Type | Hardware Accelerated? | Our Use |
|---|---|---|
| FP16 (FP16/FP32 accum) | **Yes — native** | Attention layers (quality-critical) |
| INT8 (INT32 accum) | **Yes — native** | Hot expert weights, KV cache |
| BF16 | Software via framework | Alternative to FP16 where supported |
| INT4 | **No — software dequant** | FFN/cold experts (size over speed) |

**Implication for our quant presets:**
- **Attention layers → Q8_0 or FP16** — gets full Neural Accelerator hardware path
- **FFN layers → Q4_K** — larger, less sensitive to precision, INT4 dequant overhead acceptable because FFN is the bigger bandwidth consumer
- **KV cache → Q8_KV** — hardware-accelerated INT8 reads during attention
- This is a different optimal strategy than CUDA (where everything benefits from INT4 equally)

#### ARM SME/SME2 on CPU (Fallback Path)

M5 CPU cores support **Scalable Matrix Extensions** (ARMv9.2a):
- `FMOPA` / `UMOPA` / `BFMOPA` — outer-product accumulate instructions
- Hardware-accelerated matrix ops on CPU, no GPU needed
- Use for: cold expert computation, small matmuls below GPU dispatch threshold, preprocessing
- ik_llama.cpp's ARM NEON path should be extended to exploit SME when available

#### Implementation Plan

```
Step 1: Metal 4 backend skeleton
  • MTLDevice, MTLCommandQueue, MTLComputePipelineState setup
  • Unified memory allocator using MTLBuffer shared heaps (zero-copy)
  • Backend registration in ggml (alongside CPU, CUDA)

Step 2: Core compute via TensorOps + MPP
  • GEMM dispatch through Metal Performance Primitives → Neural Accelerators
  • Attention kernels via TensorOps (Q*K^T, softmax, attn*V)
  • Fused RMSNorm + matmul where MPP supports it

Step 3: DeltaNet + hybrid memory on unified pool
  • Recurrent state + HOT KV tier share same MTLBuffer heap — no copies
  • DeltaNet gated delta rule as fused Shader ML kernel
  • 3× DeltaNet → 1× Attention block pattern exploited for prefetch

Step 4: Tiered memory on unified architecture
  • HOT: GPU-preferred MTLBuffer (attention, router, hot experts)
  • WARM: Shared MTLBuffer (compressed experts, older KV)
  • COLD: Disk-backed with async page-in to MTLBuffer
  • Anti-thrash cooldown on tier transitions
  • 32 MB SLC-aware: tensors < 32 MB that are reused stay in SLC for free

Step 5: Tile size auto-tuning (see below)

Step 6: Benchmarks + quant presets
  • M5-specific quant presets (Q8 attn + Q4 FFN, optimized for Neural Accelerator HW types)
  • Prefill benchmark (should show 3-4x vs CPU-only on M5)
  • Decode benchmark (should show bandwidth-limited gains)
  • macOS CI targets for Apple Silicon
```

#### TensorOps Tile Size Auto-Tuning

The Neural Accelerator's minimum tile is 32×32, but optimal performance depends on balancing register pressure, memory staging, and pipeline occupancy. The optimal tile varies by **matrix shape class** (not per-layer — all layers of the same type share a shape).

**Hardware constraints (fixed per chip generation):**

| Constraint | Value | Impact |
|---|---|---|
| Minimum tile | 32×32 | Hardware floor — K dimension must be multiple of 32 |
| SIMD group width | 32 threads | Cooperative tensor distributes across exactly 32 threads |
| Register file per thread | Limited (chip-specific) | Larger tiles = more register pressure, potential spills |
| Threadgroup SRAM | ~32-64 KB (chip-specific) | Limits how much data can be staged from device memory |
| K-split threshold | K ≥ 4096 | K-dimension splitting beneficial above this (from llama.cpp benchmarks) |

**Model-specific matrix shape classes (what we tune for):**

Each model architecture has ~5 distinct matrix shape classes. Once tuned, the tile config applies to every layer of that type:

```
┌──────────────────────────────────────────────────────────────────┐
│          Qwen 3.5-27B Matrix Shape Classes                       │
├──────────────────────┬───────────────────────────────────────────┤
│ Shape Class          │ Dimensions (decode, batch=1)              │
├──────────────────────┼───────────────────────────────────────────┤
│ Attention QKV proj   │ [1, 5120] × [5120, 5120+1280]            │
│ Attention scores     │ [1, n_ctx] × [n_ctx, 128]  (per head)    │
│ Attention output     │ [1, 5120] × [5120, 5120]                 │
│ FFN gate + up        │ [1, 5120] × [5120, 27648] (fused)        │
│ FFN down             │ [1, 27648] × [27648, 5120]               │
│ DeltaNet state       │ [128, 128] state matrix update           │
│ Output head          │ [1, 5120] × [5120, vocab]                │
├──────────────────────┴───────────────────────────────────────────┤
│ Prefill (batch=N): same shapes but M dimension = N              │
│ → larger M benefits from larger M-tiles                         │
└──────────────────────────────────────────────────────────────────┘

Qwen 3.5-9B: same classes, different dims (hidden=3584, ffn=18944)
DeepSeek-R1-14B: same classes, different dims (hidden=5120, ffn=13824)
MoE models: add per-expert FFN shape class (smaller, batched)
```

**Tile sweep strategy:**

```
Phase 1: Baseline (use llama.cpp's proven configs)
  • Default: M=128, N=64, K=64 (best general-purpose from llama.cpp Metal 4 PR)
  • K-split enabled for K ≥ 4096

Phase 2: Per-shape-class sweep (automated, ~50 runs per model)
  • Tile candidates for M: [32, 64, 128, 256]
  • Tile candidates for N: [32, 64, 128]
  • Tile candidates for K: [32, 64, 128, 256]
  • For each shape class: benchmark all valid combinations
  • Metric: tokens/sec for decode (batch=1) and prefill (batch=512)
  • Constraint: skip configs that exceed register budget (detected by compiler)

Phase 3: Store results in model presets
  • configs/presets/qwen35-27b-quality.conf gains tile config section:
    [metal_tiles]
    attn_qkv  = 128x64x64
    attn_score = 32x128x32
    attn_out  = 128x64x64
    ffn_gate  = 128x64x128
    ffn_down  = 64x64x128
    deltanet  = 32x32x128
    output    = 64x64x64

Phase 4: Runtime selection
  • On startup: load tile config from preset, compile Metal pipelines
    with constexpr tile dimensions baked into shader constants
  • If no preset: fall back to Phase 1 defaults
  • Optional: --metal-tile-sweep flag runs Phase 2 on first launch,
    caches results for future runs
```

**Why this works (it's NOT pure trial-and-error):**
- Only ~5 shape classes per model architecture, not 64 layers
- Each shape class has ~40-60 valid tile combinations to test
- One sweep takes minutes (short matmul benchmarks), results are permanent
- Ship presets for our target models (Qwen 3.5, DeepSeek-R1) — users never sweep
- This is exactly how cuBLAS works: pre-benchmarked tile lookup tables per matrix shape

**Decode vs Prefill tile strategy:**

| Phase | Matrix Shape | Optimal Tile Strategy |
|---|---|---|
| **Decode** (batch=1) | Tall-skinny: [1, K] × [K, N] | Small M-tile (32-64), larger K-tile for bandwidth |
| **Prefill** (batch=N) | Square-ish: [N, K] × [K, N] | Large M-tile (128-256), balanced K-tile for compute |

The decode path is bandwidth-limited (loading weights dominates), so tile strategy focuses on minimizing memory stalls. The prefill path is compute-limited (Neural Accelerators are fully utilized), so tile strategy maximizes arithmetic intensity. Cooperative tensors matter most during decode — keeping intermediates in registers saves the bandwidth that weights need.

**Model fit on M5 family:**

| Model | Q4 Size | FP16 Size | Fits on M5 (32GB) | Fits on M5 Pro (64GB) | Fits on M5 Max (128GB) |
|---|---|---|---|---|---|
| Qwen 3.5-9B | ~5 GB | ~18 GB | Q4 + FP16 both fit | Comfortable | Comfortable |
| Qwen 3.5-27B | ~15 GB | ~54 GB | Q4 fits | Q4 + room; FP16 fits | FP16 + full KV |
| DeepSeek-R1-14B | ~8 GB | ~28 GB | Q4 fits | Both fit | Both fit |
| Qwen 3.5-MoE-397B | ~55 GB | ~794 GB | Expert paging | Hot experts fit | ~40% resident |

**Bottom line:** M5 Max with 128 GB unified memory at 614 GB/s is the most powerful consumer device for local LLM inference. LeanInfer + Metal 4 TensorOps makes it sing — Neural Accelerators handle the compute, our bandwidth optimizations handle the decode bottleneck.

### 4.7 Phase 3: Speed & Intelligence

#### 3a. Reasoning-Aware Speculative Decoding

Thinking tokens are predictable (repetitive reasoning patterns):
- During `<think>` generation: aggressive n-gram speculation (high acceptance)
- During answer generation: conservative speculation
- Adaptive draft length based on rolling acceptance rate
- Use ik_llama's self-speculative / ngram implementation as base

#### 3b. Fused DeltaNet + Attention Pipeline

For Qwen 3.5's repeating `3× DeltaNet → 1× Attention` pattern:
- Pre-allocate buffers for the full 4-layer block
- Overlap DeltaNet state updates with attention KV writes
- CUDA: pipeline the 3 DeltaNet layers (independent of attention KV)
- CPU: batch the 3 DeltaNet updates before the attention layer

#### 3c. Default Runtime Repacking

ik_llama's `-rtr` gives 1.6x speedup with one-time startup cost. Make it default for reasoning workloads where sessions are long.

#### 3d. Predictive Expert Prefetch (MoE)

"Branch prediction for LLMs" — predict which experts will activate BEFORE the router runs:

```
┌──────────────────────────────────────────────────┐
│           Expert Prefetch Pipeline                │
│                                                  │
│  Layer N router fires                            │
│       ↓                                          │
│  Prefetch predictor uses:                        │
│  • Expert co-activation matrix (from Phase 0b)   │
│  • Last N layers' routing decisions              │
│  • Token hidden state (lightweight MLP or LUT)   │
│       ↓                                          │
│  Top-K candidate experts for Layer N+1           │
│       ↓                                          │
│  Async prefetch: SSD → RAM → VRAM               │
│       ↓                                          │
│  Layer N+1 router fires → expert already loaded  │
└──────────────────────────────────────────────────┘
```

- PowerInfer achieves >95% prediction accuracy for neuron activation
- MoE expert routing is less predictable but still strongly correlated layer-to-layer
- Start simple: lookup table from Phase 0b co-activation data (no neural predictor needed initially)
- Target: >90% prefetch hit rate → eliminates most page-fault stalls
- Fallback: if prediction misses, fall back to synchronous load (same as current behavior, no worse)
- Feeds directly from Phase 0b expert usage profiling data

#### 3e. Dynamic Operator-level CPU/GPU Routing

Route individual operations to optimal hardware at runtime:

```
if matrix_size > threshold:
    run_on_gpu()      # Large GEMMs, batched attention
else:
    run_on_cpu()      # Small matmuls, scalar ops, expert gating
```

- Threshold determined empirically from Phase 0a profiling data (not hardcoded)
- On Metal unified memory: threshold drops dramatically (no transfer cost)
- On CUDA: must account for PCIe round-trip (~5-10us) in threshold calculation
- Combine with ik_llama's tensor overrides (`-ot`) for static placement hints

### 4.8 Phase 4: ANE/CoreML Offload (Deprioritized)

> **Largely superseded by Metal 4 TensorOps.** On M5, the GPU Neural Accelerators deliver 3.5-4x prefill speedup via Metal 4 — the same compute class that ANE targeted but with full programmability and no CoreML op constraints. The 16-core Neural Engine (38 TOPS) still exists but is less relevant when TensorOps provides direct GPU-side matrix acceleration.

**Remaining ANE use cases (if ever pursued):**
- Battery-life optimization on MacBook (ANE is more power-efficient than GPU for sustained inference)
- Background inference while GPU is occupied by other workloads
- Small model inference (< 3B) where ANE alone may suffice

**Why Metal 4 TensorOps wins over CoreML/ANE for our workloads:**
- No op constraints — DeltaNet's gated delta rule, 1D causal convolution all expressible
- Same hardware acceleration path (Neural Accelerators inside GPU cores)
- Full control over memory layout, tiling, fusion
- No model export/conversion step — compute inline with our inference loop
- Apple optimizes TensorOps across silicon generations automatically

**Decision:** Skip Phase 4 unless battery-life profiling on MacBook shows a compelling case. Metal 4 TensorOps (Phase 2b) covers 99% of the compute acceleration need.

### 4.9 North Star: Streaming Neural Network

> **"Never load the full model. Only stream what's needed."**

```
Predict → Prefetch → Compute → Evict → Repeat
```

The model becomes a data stream, not a static object. This is the endgame vision that all phases build toward incrementally.

**Why the math works** — For Qwen 3.5-MoE (397B-A17B):
- Total model Q4: ~55 GB
- Active parameters per token: ~17B = ~9 GB Q4
- Active as % of total: **4.3%** — 95.7% is idle at any moment
- Perfect streaming working set: ~9-12 GB for a 397B model

**The progression from current state to streaming:**

```
Today (static):     Load entire model → run all layers → done
Phase 2 (tiered):   Load model, but hot/warm/cold placement
Phase 3 (prefetch): Predict what's needed → load just that → compute → evict
North Star:         Stream experts like video frames — never hold the full model
```

**The three constraints that determine how close we get:**

| Constraint | Challenge | Our Mitigation |
|---|---|---|
| Prediction accuracy | Must be >90% to avoid stalls | Phase 0b profiling → co-activation LUT → lightweight MLP predictor |
| Prefetch latency | NVMe: 0.2-0.6ms per expert load | Pipeline: overlap I/O with compute (PowerInfer-2 proved this) |
| Cold start | New topic = unknown expert needs | Keep top 20-30% most globally frequent experts always resident |

**What this is NOT:**
- NOT runtime subgraph memoization (violates attention mechanics — transformer hidden states are context-dependent, you can't cache and replay them)
- NOT sparse forward pass for SiLU models (near-zero is not zero — only works for ReLU models like SmallThinker)

**Decision point:** Evaluate feasibility of full streaming after Phase 3d (predictive prefetch) ships and we have real prediction accuracy numbers from Phase 0b profiling.

### 4.10 Idea Feasibility Matrix

Consolidated assessment of all optimization ideas explored during research:

| Idea | Feasible? | Wall Risk | Phase | Notes |
|---|---|---|---|---|
| **Instrumentation / profiling** | Yes (trivial) | None | **Phase 0** | Foundation — must ship first |
| **Tiered quantization** | Yes (trivial) | None | **Phase 2** | ik_llama `--custom-q` already exists |
| **Weight prepacking** | Yes (proven) | None | **Phase 2-3** | Store in kernel-native format; ik_llama `-rtr` is a start |
| **Tensor grouping** | Yes (trivial) | None | **Phase 2** | Batch load `blk.N.*` together, fewer seeks |
| **Anti-thrash cooldown** | Yes (trivial) | None | **Phase 2** | Minimum dwell time before tier transitions |
| **Static hot/cold expert split** | Yes (proven) | None | **Phase 2** | PowerInfer proved this at neuron level |
| **KV cache tiers + CoT eviction** | Yes | Low | **Phase 2** | Cache invalidation on `</think>` sentinel |
| **Hot/cold expert paging to NVMe** | Yes | Low-Medium | **Phase 2-3** | PowerInfer-2 does this on phones |
| **Predictive expert prefetch** | Yes (hard) | Medium | **Phase 3** | "Branch prediction for LLMs" — high reward |
| **Dynamic CPU/GPU operator routing** | Yes | Low | **Phase 3** | Trivial on Metal (unified memory) |
| **Fused DeltaNet pipeline** | Yes | Low | **Phase 3** | Exploits Qwen 3.5's 3:1 block pattern |
| **NUMA-aware placement** | Yes | Low | **Phase 2** | 20-40% gain on multi-socket; no gain on consumer |
| **Multi-resolution weights** | Yes (niche) | Low | Opportunistic | Speculative decoding variant |
| **Sparse forward pass (non-ReLU)** | ReLU only | **Medium** | Opportunistic | SiLU models: near-zero != zero |
| **Runtime subgraph memoization** | **No** | **HIGH** | **SKIP** | Violates attention context-dependence |
| **Full streaming neural network** | Incrementally | Medium | **North Star** | Each phase ships standalone value |

> **Killed ideas:** Runtime subgraph memoization (Idea 7 from research) is architecturally infeasible for transformers. Hidden states are context-dependent — the same input produces different representations depending on all prior tokens. Only viable for fixed system prompt caching, which llama.cpp already handles.

### 4.11 Project Structure

```
LeanInfer/
├── README.md
├── docs/
│   ├── ASSESSMENT.md            # This document
│   ├── MODEL_MATRIX.md          # Supported models + recommended configs
│   └── BENCHMARKS.md            # Reasoning-specific benchmarks
├── upstream/                     # ik_llama.cpp as git subtree
├── patches/
│   ├── 00-profiler-hooks.patch   # Phase 0a — instrumentation
│   ├── 01-hybrid-memory.patch    # Phase 1a
│   ├── 02-thinking-control.patch # Phase 1b
│   ├── 03-recurrent-quant.patch  # Phase 1c
│   ├── 04-kv-tiers.patch        # Phase 2a
│   ├── 05-expert-paging.patch   # Phase 2c
│   ├── 06-metal-backend.patch   # Phase 2b
│   └── 07-expert-prefetch.patch # Phase 3d
├── instrument/                   # Phase 0 — Measure Everything
│   ├── profiler.h                # Lightweight C hooks (#ifdef LEANINFER_PROFILE)
│   ├── profiler.cpp              # Timer + counter implementation
│   ├── trace_writer.cpp          # Chrome tracing JSON output
│   ├── expert_tracker.cpp        # MoE expert activation logger
│   └── analyze.py                # Post-run analysis: hotspots, expert frequency, bottleneck ID
├── configs/
│   ├── presets/
│   │   ├── qwen35-27b-quality.conf
│   │   ├── qwen35-27b-balanced.conf
│   │   ├── qwen35-27b-lean.conf
│   │   ├── qwen35-9b-lean.conf
│   │   ├── deepseek-r1-14b.conf
│   │   └── deepseek-r1-moe.conf
│   └── sampling/
│       ├── thinking.conf
│       ├── no-think.conf
│       └── coding.conf
├── profiles/                     # Expert frequency profiles (Phase 0b + 2c)
│   ├── profiler.py               # Offline expert profiling on calibration corpus
│   └── coactivation.py           # Expert co-activation matrix builder (Phase 3d input)
├── scripts/
│   ├── setup.sh                  # Clone upstream + apply patches
│   ├── quantize.sh               # One-command quant with presets
│   ├── benchmark.sh              # Reasoning-specific benchmarks (Phase 0c)
│   ├── tile_sweep.py             # Metal tile auto-tuning (Phase 2b Step 5)
│   └── rebase.sh                 # Rebase patches on new upstream
└── tests/
    ├── test_hybrid_memory.py
    ├── test_thinking_control.py
    ├── test_cot_eviction.py
    └── test_profiler.py           # Verify instrumentation correctness
```

### 4.12 Rebase Strategy

ik_llama.cpp is a hard fork with one primary developer (multiple commits daily):
- Track upstream via git subtree
- Patches applied on top, rebased periodically via `scripts/rebase.sh`
- CI tests patches against latest upstream on each rebase
- Cherry-pick specific upstream commits when relevant (e.g., new model support)

---

## 5. Target Models (Priority Order)

| Model | Type | Why |
|---|---|---|
| **Qwen 3.5-27B** | Hybrid DeltaNet + Attention | Best reasoning at size, hybrid arch is our differentiator |
| **Qwen 3.5-9B** | Hybrid DeltaNet + Attention | Fits 16 GB easily with our optimizations |
| **DeepSeek-R1-14B** | Dense distill | Excellent reasoning, well-supported in ik_llama |
| **Qwen 3.5-MoE-35B-A3B** | Hybrid + MoE | Expert paging showcase — 35B model at 3B active cost |
| **DeepSeek-R1 (671B)** | MoE | Stretch goal — expert paging makes it feasible on consumer HW |
| **SmallThinker-21B-A3B** | MoE + ReLU | PowerInfer-style sparsity fully applicable |

---

## 6. Competitive Positioning

| Feature | llama.cpp | ik_llama.cpp | Ollama/LM Studio | **LeanInfer** |
|---|---|---|---|---|
| Qwen 3.5 multi-turn | Broken | Buggy | Broken (uses llama.cpp) | **Fixed** |
| Thinking control | Broken | Partial | Broken | **Full control** |
| Inference profiling | Basic | sweep-bench | None | **Per-layer, per-expert, chrome://tracing** |
| CoT cache eviction | No | No | No | **Yes** |
| Reasoning quant presets | No | No | Auto (suboptimal) | **Optimized** |
| Expert frequency paging | No | No | No | **Yes** |
| Predictive expert prefetch | No | No | No | **Yes** |
| CPU inference speed | Baseline | 2-3x faster | ~Baseline | **2-3x (inherited)** |
| Quant quality | Good | Best | Good | **Best (inherited)** |
| Metal 4 TensorOps | No | No | No (via llama.cpp Metal) | **Yes — Neural Accelerator targeting** |
| M5 Neural Accel quant presets | No | No | No | **Q8 attn + Q4 FFN (HW-aware)** |
| Backend breadth | 18 | 2 | Via llama.cpp | **3: CPU + CUDA + Metal 4** |
| Model breadth | Universal | Wide | Universal | **Focused (reasoning)** |

**Our niche:** The best runtime for reasoning models on consumer hardware. Not trying to be everything for everyone.

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| ikawrakow abandons project | High | We have full source; can maintain fork independently |
| Qwen 3.5 architecture not adopted by others | Medium | DeltaNet/hybrid is a trend (Falcon-H1, Jamba also hybrid) |
| Mainline llama.cpp fixes Qwen 3.5 | Low (positive) | Cherry-pick their fixes; our reasoning optimizations remain unique |
| ik_llama.cpp diverges too far from mainline | Medium | Rebase strategy; patches are modular |
| ReLU models don't gain traction | Low | Our core optimizations don't require ReLU; expert paging works on any MoE |
| Metal backend maintenance burden | Medium | Metal is one backend (not 18); unified memory simplifies vs CUDA's split model |
| CoreML can't express DeltaNet ops | Low | Phase 4 is optional; Metal alone delivers ~90% of Apple Silicon benefit |

---

## 8. Research Sources

- **PowerInfer:** https://github.com/Tiiny-AI/PowerInfer (SJTU-IPADS)
- **ik_llama.cpp:** https://github.com/ikawrakow/ik_llama.cpp
- **llama.cpp:** https://github.com/ggml-org/llama.cpp
- **Qwen 3.5:** https://github.com/QwenLM/Qwen3.5
- **llama.cpp issue #20225:** Full prompt reprocessing for hybrid models
- **llama.cpp issue #20182:** enable_thinking broken
- **llama.cpp PR #20232:** Parallel slots fix for recurrent models
- **ik_llama.cpp issue #1471:** VRAM leak with SSM tensors
- **ik_llama.cpp issue #1487:** Endless tool calls on aarch64
- **ik_llama.cpp PR #1288:** Qwen3.5-MoE support
- **ik_llama.cpp PR #1326:** Qwen3.5 dense support
