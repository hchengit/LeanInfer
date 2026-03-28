# LeanInfer

**Purpose-built inference runtime for 8-27B reasoning models, including Qwen 3.5 hybrid models, on consumer hardware.**

A focused fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), incorporating key ideas from [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) and targeted optimizations for chain-of-thought reasoning workloads.

## Why This Exists

Modern reasoning models (Qwen 3.5, DeepSeek-R1, etc.) push consumer hardware to its limits:
- 27B models need 20+ GB RAM with standard tooling
- Reasoning chains generate thousands of throwaway thinking tokens that bloat KV cache
- Hybrid architectures (Gated DeltaNet + Attention) break existing memory management
- No inference engine optimizes specifically for the reasoning use case

LeanInfer targets **one thing well**: running 8-27B reasoning models on 16-32 GB machines at maximum efficiency.

## Target Hardware

- x86-64 CPUs (AVX2+) + NVIDIA GPUs (CUDA)
- Mac M1-M5: TensorOp and Cooperative Tensor 
- 16-32 GB system RAM
- Consumer GPUs (RTX 3060-4090, 8-24 GB VRAM)

## Key Features (Planned)

### Phase 1: Qwen 3.5 Compatibility
- Hybrid memory manager (recurrent state + KV cache)
- Server-level thinking control (`--no-think`)
- Recurrent state quantization (Q8)

### Phase 2: RAM Reduction
- Tiered KV cache with chain-of-thought eviction
- Reasoning-optimized quantization presets (IQK quants)
- Frequency-aware expert paging for MoE models

### Phase 3: Speed
- Reasoning-aware speculative decoding
- Fused DeltaNet + Attention pipeline
- Runtime repacking by default

## Architecture

Built on ik_llama.cpp for:
- 2-3x faster CPU inference vs mainline llama.cpp
- Superior IQK quantization (2.7x lower error at same bit-width)
- FlashMLA-3 for DeepSeek models
- Fused MoE operations
- Hadamard K-cache transforms

## Status

**Pre-development** — Architecture planning phase.

## License

MIT (inheriting from ik_llama.cpp / llama.cpp)
