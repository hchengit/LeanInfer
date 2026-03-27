#!/usr/bin/env python3
"""
LeanInfer Phase 2c: Expert Co-Activation Matrix Builder
========================================================
Builds a co-activation matrix for OLMoE experts using router weight similarity.
Co-activated experts tend to appear together in the same top-K selection.

Two experts tend to co-activate when their router weight vectors are similar
(they "attract" similar hidden states). The co-activation matrix
C[i,j] = cosine_similarity(W[:,i], W[:,j]) approximates joint selection probability.

Usage:
  # From profiler JSON (weight-based):
  python3 coactivation.py --profile olmoe_expert_profile.json

  # From runtime log (requires --log from profiler.py):
  python3 coactivation.py --log expert_activations.log \
                          --n-layers 16 --n-experts 64

Output: olmoe_coactivation.json  (per-layer co-activation matrices)

How this helps Phase 2c expert paging:
  - If expert A is hot, prefetch experts that strongly co-activate with A
  - Co-activation groups can be kept together in the same memory tier
  - Reduces cache misses when loading expert weights during inference
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
import struct


# ─── GGUF Router Weight Loader ───────────────────────────────────────────────
# (shared utilities duplicated here to keep tools standalone)

GGUF_TYPE_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}

def _read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def _skip_kv(f, vtype):
    if vtype == 8:
        n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        alen  = struct.unpack('<Q', f.read(8))[0]
        sz = GGUF_TYPE_SIZES.get(etype)
        if sz:
            f.read(alen * sz)
        else:
            for _ in range(alen):
                n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif vtype in GGUF_TYPE_SIZES:
        f.read(GGUF_TYPE_SIZES[vtype])

def load_router_weights(model_path):
    """Load all ffn_gate_inp tensors (router weights) from a GGUF file."""
    kv_data = {}
    tensors = {}

    with open(model_path, 'rb') as f:
        f.read(4)  # magic
        f.read(4)  # version
        n_tens = struct.unpack('<Q', f.read(8))[0]
        n_kv   = struct.unpack('<Q', f.read(8))[0]

        for _ in range(n_kv):
            key   = _read_str(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 8:
                val = _read_str(f)
                kv_data[key] = val
            else:
                _skip_kv(f, vtype)

        for _ in range(n_tens):
            name  = _read_str(f)
            ndim  = struct.unpack('<I', f.read(4))[0]
            dims  = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
            dtype = struct.unpack('<I', f.read(4))[0]
            off   = struct.unpack('<Q', f.read(8))[0]
            if 'ffn_gate_inp' in name:
                tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': off}

        arch  = kv_data.get('general.architecture', 'olmoe')
        align = 32
        pos = f.tell()
        data_start = ((pos + align - 1) // align) * align

    # Load the actual tensor data
    router_weights = {}
    with open(model_path, 'rb') as f:
        for name, info in tensors.items():
            f.seek(data_start + info['offset'])
            dims = info['dims']
            n_elem = 1
            for d in dims: n_elem *= d

            dtype_id = info['dtype']
            if dtype_id == 0:  # F32
                raw = f.read(n_elem * 4)
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif dtype_id == 1:  # F16
                raw = f.read(n_elem * 2)
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dtype_id == 19:  # BF16
                raw = f.read(n_elem * 2)
                arr_bf16 = np.frombuffer(raw, dtype=np.uint16)
                arr_f32  = np.zeros(n_elem, dtype=np.uint32)
                arr_f32[:] = arr_bf16.astype(np.uint32) << 16
                arr = arr_f32.view(np.float32)
            else:
                # Quantized — can't easily dequantize; skip
                continue

            router_weights[name] = arr.reshape(dims[::-1])  # [n_experts, n_embd]

    return router_weights, arch


# ─── Co-activation from Router Weights ───────────────────────────────────────

def build_weight_coactivation(router_w):
    """
    Build co-activation matrix from router weight cosine similarity.
    router_w: [n_experts, n_embd]
    Returns: [n_experts, n_experts] symmetric matrix in [0, 1]
    """
    # Normalize each expert vector to unit length
    norms = np.linalg.norm(router_w, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    W_norm = router_w / norms

    # Cosine similarity matrix: C[i,j] = dot(W[i], W[j])
    C = W_norm @ W_norm.T  # [n_experts, n_experts]

    # Shift from [-1, 1] to [0, 1]
    C = (C + 1.0) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


# ─── Co-activation from Runtime Log ──────────────────────────────────────────

def build_runtime_coactivation(log_path, n_layers, n_experts):
    """
    Build co-activation matrices from runtime expert activation log.
    For each token, mark all pairs of selected experts as co-activated.
    C[i,j] = fraction of tokens where both expert i and j were selected.
    """
    counts   = [np.zeros((n_experts, n_experts), dtype=np.int32)  for _ in range(n_layers)]
    totals   = [0] * n_layers

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            layer_str, experts_str = line.split(':', 1)
            layer = int(layer_str)
            if 0 <= layer < n_layers:
                experts = [int(e) for e in experts_str.split(',') if e.strip()]
                totals[layer] += 1
                for i in range(len(experts)):
                    for j in range(i, len(experts)):
                        counts[layer][experts[i], experts[j]] += 1
                        if i != j:
                            counts[layer][experts[j], experts[i]] += 1

    result = {}
    for layer in range(n_layers):
        if totals[layer] == 0:
            continue
        C = counts[layer].astype(np.float32) / totals[layer]
        result[str(layer)] = C.tolist()

    return result, totals


# ─── Expert Group Finder ──────────────────────────────────────────────────────

def find_expert_groups(C, threshold=0.75):
    """
    Greedily group experts with high mutual co-activation scores.
    Returns list of expert groups (each group is a list of expert IDs).
    """
    n = C.shape[0]
    assigned = [False] * n
    groups = []

    # Sort experts by sum of co-activation scores (most "connected" first)
    connectivity = C.sum(axis=1)
    order = np.argsort(-connectivity)

    for seed in order:
        if assigned[seed]:
            continue
        group = [seed]
        assigned[seed] = True
        # Find unassigned experts with high co-activation to all current group members
        for candidate in order:
            if assigned[candidate]:
                continue
            if all(C[candidate, g] >= threshold for g in group):
                group.append(candidate)
                assigned[candidate] = True
        if group:
            groups.append(sorted(int(x) for x in group))

    return groups


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='OLMoE Expert Co-Activation Matrix Builder')
    ap.add_argument('--model',     default=None, help='GGUF model file (for weight-based analysis)')
    ap.add_argument('--profile',   default=None, help='Profiler JSON output (from profiler.py)')
    ap.add_argument('--log',       default=None, help='Runtime expert activation log')
    ap.add_argument('--n-layers',  type=int, default=16)
    ap.add_argument('--n-experts', type=int, default=64)
    ap.add_argument('--threshold', type=float, default=0.75,
                    help='Co-activation similarity threshold for grouping')
    ap.add_argument('--out', default='olmoe_coactivation.json')
    args = ap.parse_args()

    output = {
        'n_layers':  args.n_layers,
        'n_experts': args.n_experts,
    }

    # ── Weight-based co-activation ──
    if args.model:
        print(f"[Weight-based Co-activation]")
        print(f"  Loading router weights from: {args.model}")
        router_weights, arch = load_router_weights(args.model)
        print(f"  Loaded {len(router_weights)} router tensors")

        weight_coact = {}
        group_map = {}

        for layer in range(args.n_layers):
            name = f'blk.{layer}.ffn_gate_inp.weight'
            if name not in router_weights:
                print(f"  Layer {layer}: no router weights (quantized or missing)")
                continue

            W = router_weights[name]  # [n_experts, n_embd]
            C = build_weight_coactivation(W)

            # Find expert groups
            groups = find_expert_groups(C, threshold=args.threshold)
            group_map[str(layer)] = groups

            # Top co-activating pairs for each of the top-5 experts
            norms = np.linalg.norm(W, axis=-1)
            top_experts = np.argsort(-norms)[:5]

            print(f"  Layer {layer:2d}: {len(groups)} groups | "
                  f"Top expert {top_experts[0]} co-activates with: "
                  f"{np.argsort(-C[top_experts[0]])[:5].tolist()}")

            weight_coact[str(layer)] = {
                'matrix_shape': list(C.shape),
                'top_pairs': [],
            }
            # Store top-10 co-activating pairs (for reporting)
            pairs = []
            for i in range(args.n_experts):
                for j in range(i+1, args.n_experts):
                    pairs.append((float(C[i, j]), i, j))
            pairs.sort(reverse=True)
            weight_coact[str(layer)]['top_pairs'] = [
                {'score': round(s, 4), 'expert_a': a, 'expert_b': b}
                for s, a, b in pairs[:20]
            ]

        output['weight_coactivation'] = weight_coact
        output['expert_groups'] = group_map

        print(f"\n[Expert Group Summary (threshold={args.threshold})]")
        for layer_id, groups in sorted(group_map.items(), key=lambda x: int(x[0])):
            large = [g for g in groups if len(g) >= 3]
            print(f"  L{int(layer_id):2d}: {len(groups)} groups, "
                  f"{len(large)} with 3+ experts — "
                  f"largest: {max(groups, key=len) if groups else []}")

    # ── Runtime co-activation ──
    if args.log:
        print(f"\n[Runtime Co-activation from: {args.log}]")
        runtime_coact, totals = build_runtime_coactivation(
            args.log, args.n_layers, args.n_experts)
        output['runtime_coactivation'] = runtime_coact
        output['runtime_totals'] = totals

    # ── Save ──
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nCo-activation data saved: {args.out}")

    # ── Paging strategy hint ──
    if 'expert_groups' in output:
        print("\n[Paging Strategy Hints]")
        print("  Keep co-activation groups together in the same memory tier.")
        print("  When loading a hot expert, prefetch its top co-activators.")
        total_experts = args.n_layers * args.n_experts
        hot_count = sum(
            len([g for g in groups if len(g) >= 2])
            for groups in group_map.values()
        )
        print(f"  Multi-expert groups: {hot_count} groups across {args.n_layers} layers")
        print(f"  = {100*hot_count/(args.n_layers):.1f}% of layers have co-activation groups")


if __name__ == '__main__':
    main()
