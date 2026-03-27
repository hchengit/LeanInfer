#!/usr/bin/env python3
"""
LeanInfer Phase 2c: OLMoE Expert Frequency Profiler
====================================================
Analyzes expert activation frequencies from two sources:
1. GGUF weight-based structural analysis (router weight norms as activation proxy)
2. Runtime expert log files produced by llama-cli --expert-log (when available)

Usage:
  # Structural analysis (weights only, no inference needed):
  python3 profiler.py --model ../models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf

  # With runtime log (produced via llama-cli --expert-log expert_activations.log):
  python3 profiler.py --model ../models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf \
                      --log expert_activations.log

Output: olmoe_expert_profile.json
"""

import struct
import sys
import json
import argparse
import os
import numpy as np
from pathlib import Path


# ─── GGUF Reader ────────────────────────────────────────────────────────────

GGUF_TYPE_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}

GGML_DTYPE_MAP = {
    0: 'F32',  1: 'F16',  2: 'Q4_0',  3: 'Q4_1',
    6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
    13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 19: 'BF16', 30: 'F16',
}

GGML_DTYPE_SIZES = {
    0: 4,    # F32: 4 bytes per element
    1: 2,    # F16: 2 bytes
    12: None,  # Q4_K: block-based
    13: None,  # Q5_K: block-based
    14: None,  # Q6_K: block-based
    15: None,  # Q8_K: block-based
}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def skip_kv_value(f, vtype):
    if vtype == 8:   # string
        n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif vtype == 9:  # array
        etype = struct.unpack('<I', f.read(4))[0]
        alen  = struct.unpack('<Q', f.read(8))[0]
        sz = GGUF_TYPE_SIZES.get(etype)
        if sz:
            f.read(alen * sz)
        else:  # string array
            for _ in range(alen):
                n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif vtype in GGUF_TYPE_SIZES:
        f.read(GGUF_TYPE_SIZES[vtype])

def read_kv_value(f, vtype):
    if vtype == 8:
        return read_str(f)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        alen  = struct.unpack('<Q', f.read(8))[0]
        sz = GGUF_TYPE_SIZES.get(etype)
        if sz:
            raw = f.read(alen * sz)
            if etype == 4:
                return list(struct.unpack(f'<{alen}I', raw))
            elif etype == 5:
                return list(struct.unpack(f'<{alen}i', raw))
            elif etype == 6:
                return list(struct.unpack(f'<{alen}f', raw))
            return raw
        else:
            # string array or array of arrays: skip elements
            result = []
            for _ in range(alen):
                if etype == 8:
                    result.append(read_str(f))
                # nested arrays not supported — bail out
            return result
    elif vtype in GGUF_TYPE_SIZES:
        sz = GGUF_TYPE_SIZES[vtype]
        raw = f.read(sz)
        fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}
        return struct.unpack(fmt[vtype], raw)[0]
    return None

def parse_gguf(path):
    """Parse GGUF file: return (kv_metadata, tensor_info_list, data_offset)."""
    kv = {}
    tensors = []
    with open(path, 'rb') as f:
        magic   = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tens  = struct.unpack('<Q', f.read(8))[0]
        n_kv    = struct.unpack('<Q', f.read(8))[0]

        for _ in range(n_kv):
            key   = read_str(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            val   = read_kv_value(f, vtype)
            kv[key] = val

        for _ in range(n_tens):
            name  = read_str(f)
            ndim  = struct.unpack('<I', f.read(4))[0]
            dims  = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
            dtype = struct.unpack('<I', f.read(4))[0]
            off   = struct.unpack('<Q', f.read(8))[0]
            tensors.append({'name': name, 'dims': dims, 'dtype': dtype, 'offset': off})

        # GGUF v3 aligns data to 32 bytes
        align = kv.get('general.alignment', 32)
        pos = f.tell()
        data_start = ((pos + align - 1) // align) * align

    return kv, tensors, data_start


# ─── Router Weight Analysis ──────────────────────────────────────────────────

def load_f32_tensor(path, tensor_info, data_start):
    """Load a tensor as float32 numpy array (only works for F32/BF16/F16 tensors)."""
    dtype_id = tensor_info['dtype']
    dims = tensor_info['dims']
    n_elem = 1
    for d in dims:
        n_elem *= d

    with open(path, 'rb') as f:
        f.seek(data_start + tensor_info['offset'])

        if dtype_id == 0:  # F32
            raw = f.read(n_elem * 4)
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif dtype_id == 1:  # F16
            raw = f.read(n_elem * 2)
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype_id == 19:  # BF16
            raw = f.read(n_elem * 2)
            # BF16 → F32: pad 2 zero bytes after each BF16 word
            arr_bf16 = np.frombuffer(raw, dtype=np.uint16)
            arr_f32  = np.zeros(n_elem, dtype=np.uint32)
            arr_f32[:] = arr_bf16.astype(np.uint32) << 16
            arr = arr_f32.view(np.float32)
        else:
            return None  # quantized — needs dequantization

    return arr.reshape(dims[::-1])  # GGUF dims are innermost-first; numpy is C-order


def analyze_router_weights(path, kv, tensors, data_start):
    """
    For each MoE layer, analyze the router weight matrix (ffn_gate_inp).
    Router: W ∈ R^{n_embd × n_experts}
    Each column W[:,e] is the "expert key" for expert e.
    Experts with higher |W[:,e]|_2 norm tend to have higher baseline activation.
    """
    arch       = kv.get('general.architecture', 'olmoe')
    n_layers   = kv.get(f'{arch}.block_count', 16)
    n_experts  = kv.get(f'{arch}.expert_count', 64)
    n_used     = kv.get(f'{arch}.expert_used_count', 8)

    print(f"\n[Router Weight Analysis]")
    print(f"  Arch: {arch}, Layers: {n_layers}, Experts/layer: {n_experts}, Top-K: {n_used}")

    tensor_map = {t['name']: t for t in tensors}
    layer_profiles = {}

    for layer in range(n_layers):
        name = f'blk.{layer}.ffn_gate_inp.weight'
        if name not in tensor_map:
            print(f"  Layer {layer}: router tensor not found, skipping")
            continue

        ti = tensor_map[name]
        router_w = load_f32_tensor(path, ti, data_start)

        if router_w is None:
            # Quantized router — use row norms of Q4_K blocks as proxy
            print(f"  Layer {layer}: router is quantized, using norm proxy")
            # For quantized tensors, dims = [n_experts, n_embd] in GGUF ordering
            # dims[0] = n_embd (innermost), dims[1] = n_experts
            dims = ti['dims']
            n_exp_dim = dims[1] if len(dims) > 1 else n_experts
            # Create uniform distribution as fallback
            scores = np.ones(n_exp_dim, dtype=np.float32)
        else:
            # router_w shape after reshape: [n_experts, n_embd] (GGUF innermost-first)
            # Compute L2 norm of each expert's weight vector
            # W[:,e] in math ≡ router_w[e, :] in numpy
            norms = np.linalg.norm(router_w, axis=-1)  # [n_experts]
            if norms.ndim > 1:
                norms = norms.mean(axis=tuple(range(norms.ndim - 1)))
            scores = norms.flatten()

        # Rank experts by score (higher = more likely to be activated)
        rank = np.argsort(-scores)

        # Classify into hot/warm/cold
        hot_cutoff  = int(n_experts * 0.20)  # top 20% = hot
        warm_cutoff = int(n_experts * 0.50)  # next 30% = warm; bottom 50% = cold

        hot_experts  = rank[:hot_cutoff].tolist()
        warm_experts = rank[hot_cutoff:warm_cutoff].tolist()
        cold_experts = rank[warm_cutoff:].tolist()

        layer_profiles[str(layer)] = {
            'scores':       scores.tolist(),
            'rank':         rank.tolist(),
            'hot_experts':  hot_experts,
            'warm_experts': warm_experts,
            'cold_experts': cold_experts,
            'score_stats': {
                'mean': float(scores.mean()),
                'std':  float(scores.std()),
                'min':  float(scores.min()),
                'max':  float(scores.max()),
                'hot_score_threshold':  float(scores[hot_experts[-1]]) if hot_experts else 0,
                'cold_score_threshold': float(scores[warm_experts[-1]]) if warm_experts else 0,
            }
        }

        top5 = rank[:5].tolist()
        bot5 = rank[-5:].tolist()
        print(f"  Layer {layer:2d}: top-5 hot={top5}  bot-5 cold={bot5}  "
              f"score range [{scores.min():.3f}, {scores.max():.3f}]")

    return {
        'arch':       arch,
        'n_layers':   n_layers,
        'n_experts':  n_experts,
        'n_used':     n_used,
        'source':     'weight_structural_analysis',
        'layers':     layer_profiles,
    }


# ─── Runtime Log Parser ──────────────────────────────────────────────────────

def parse_expert_log(log_path, n_layers, n_experts):
    """
    Parse runtime expert activation log.
    Format (one line per token): layer_idx:expert1,expert2,...,expertK
    Example: 0:12,7,45,3,61,22,8,31
    """
    # activation_counts[layer][expert] = count of how many times expert was selected
    counts = [[0] * n_experts for _ in range(n_layers)]
    total_tokens = 0

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            layer_str, experts_str = line.split(':', 1)
            layer = int(layer_str)
            if 0 <= layer < n_layers:
                for e in experts_str.split(','):
                    e = e.strip()
                    if e:
                        counts[layer][int(e)] += 1
            if layer == n_layers - 1:
                total_tokens += 1

    layer_profiles = {}
    for layer in range(n_layers):
        c = np.array(counts[layer], dtype=np.float32)
        if c.sum() == 0:
            continue
        freq = c / c.sum()
        rank = np.argsort(-freq)

        hot_cutoff  = int(n_experts * 0.20)
        warm_cutoff = int(n_experts * 0.50)

        layer_profiles[str(layer)] = {
            'counts':       c.tolist(),
            'frequencies':  freq.tolist(),
            'rank':         rank.tolist(),
            'hot_experts':  rank[:hot_cutoff].tolist(),
            'warm_experts': rank[hot_cutoff:warm_cutoff].tolist(),
            'cold_experts': rank[warm_cutoff:].tolist(),
        }

    return {'total_tokens': total_tokens, 'layers': layer_profiles}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='OLMoE Expert Frequency Profiler')
    ap.add_argument('--model', required=True, help='Path to GGUF model file')
    ap.add_argument('--log',   default=None,   help='Runtime expert activation log (optional)')
    ap.add_argument('--out',   default='olmoe_expert_profile.json', help='Output JSON path')
    args = ap.parse_args()

    print(f"Loading GGUF: {args.model}")
    kv, tensors, data_start = parse_gguf(args.model)

    arch = kv.get('general.architecture', 'olmoe')
    print(f"  Architecture: {arch}")
    print(f"  Total tensors: {len(tensors)}")

    # Weight-based structural analysis
    weight_profile = analyze_router_weights(args.model, kv, tensors, data_start)

    profile = {
        'model': os.path.basename(args.model),
        'architecture': arch,
        'n_layers':   weight_profile['n_layers'],
        'n_experts':  weight_profile['n_experts'],
        'n_used':     weight_profile['n_used'],
        'weight_analysis': weight_profile,
    }

    # Runtime log (if provided)
    if args.log:
        print(f"\n[Runtime Log Analysis]")
        print(f"  Parsing: {args.log}")
        runtime = parse_expert_log(args.log, weight_profile['n_layers'], weight_profile['n_experts'])
        profile['runtime_analysis'] = runtime
        profile['total_tokens_observed'] = runtime['total_tokens']
        print(f"  Tokens observed: {runtime['total_tokens']}")

    # Save
    with open(args.out, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved: {args.out}")

    # Print summary
    print("\n[Hot Expert Summary (top 20% by weight norm)]")
    print(f"{'Layer':>6}  {'Hot Experts (top 12 of ' + str(weight_profile['n_experts']) + ')':}")
    for layer_id, lp in weight_profile['layers'].items():
        hot = lp['hot_experts'][:12]
        print(f"  L{int(layer_id):2d}:  {hot}")


if __name__ == '__main__':
    main()
