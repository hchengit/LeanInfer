#!/usr/bin/env python3
"""
LeanInfer Phase 2c: Expert Placement Policy Generator
======================================================
Combines weight-based (structural) and runtime (activation frequency) expert
profiles to produce a placement decision for each expert in each layer.

Tiers
-----
  hot  – experts fired very frequently; keep in fastest memory (GPU VRAM / DRAM)
  warm – moderately active; can stay in DRAM or slow VRAM
  cold – rarely fired; can be paged to NVMe/CPU memory

Usage
-----
  python3 policy.py [--weight-profile olmoe_expert_profile.json]
                    [--runtime-profile olmoe_runtime_profile.json]
                    [--coact-profile olmoe_coactivation.json]
                    [--memory-budget-hot  <fraction, e.g. 0.20>]
                    [--memory-budget-warm <fraction, e.g. 0.30>]
                    [--output policy.json]
                    [--report]

Output JSON
-----------
  {
    "n_layers": 16,
    "n_experts": 64,
    "layers": {
      "0": {
        "hot":  [list of expert ids],
        "warm": [list of expert ids],
        "cold": [list of expert ids]
      },
      ...
    }
  }

The output can be consumed by a future paging pass in ik_llama.cpp.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def rank_experts_runtime(runtime_profile):
    """
    Return per-layer list of (expert_id, activation_count) sorted descending.
    """
    ranked = {}
    for il_str, info in runtime_profile['layers'].items():
        il = int(il_str)
        # Flatten the already-classified lists back to counts
        # The runtime profile stores hot/warm/cold lists, not raw counts.
        # Reconstitute rank from the ordering (position = rank).
        ordered = info['hot'] + info['warm'] + info['cold']
        ranked[il] = ordered   # highest-rank first
    return ranked


def rank_experts_weight(weight_profile):
    """
    Return per-layer list of expert_ids sorted by structural activation proxy
    (L2 norm of router weights, descending).
    """
    ranked = {}
    for layer in weight_profile.get('layers', []):
        il = layer['layer']
        experts = sorted(layer['experts'],
                         key=lambda e: e.get('l2_norm', 0.0), reverse=True)
        ranked[il] = [e['expert_id'] for e in experts]
    return ranked


def blend_rankings(weight_rank, runtime_rank, n_experts,
                   weight_alpha=0.30, runtime_alpha=0.70):
    """
    Blend weight-based and runtime-based rankings into a single score per expert.

    Score = runtime_alpha * (runtime_rank_score) + weight_alpha * (weight_rank_score)
    where rank_score[i] = (n_experts - rank_position) / n_experts  (higher is hotter)

    Returns dict: layer_index → list of expert_ids sorted by blended score desc.
    """
    all_layers = set(weight_rank) | set(runtime_rank)
    result = {}
    for il in all_layers:
        scores = {}
        w_list = weight_rank.get(il, [])
        r_list = runtime_rank.get(il, [])

        for rank, eid in enumerate(w_list):
            scores.setdefault(eid, 0.0)
            scores[eid] += weight_alpha * (n_experts - rank) / n_experts

        for rank, eid in enumerate(r_list):
            scores.setdefault(eid, 0.0)
            scores[eid] += runtime_alpha * (n_experts - rank) / n_experts

        # fill in experts not seen in either list
        for eid in range(n_experts):
            scores.setdefault(eid, 0.0)

        result[il] = sorted(scores, key=lambda e: scores[e], reverse=True)
    return result


def classify_by_budget(ordered_experts, budget_hot, budget_warm, n_experts):
    """
    Given experts sorted by score (best first), cut into hot / warm / cold
    according to fractional budgets.

    budget_hot  = fraction of n_experts to label 'hot'
    budget_warm = additional fraction to label 'warm'
    """
    n_hot  = max(1, round(n_experts * budget_hot))
    n_warm = max(1, round(n_experts * budget_warm))
    n_warm = min(n_warm, n_experts - n_hot)

    hot  = ordered_experts[:n_hot]
    warm = ordered_experts[n_hot:n_hot + n_warm]
    cold = ordered_experts[n_hot + n_warm:]
    return hot, warm, cold


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------

def print_report(policy, n_experts):
    n_layers = len(policy['layers'])
    print(f"\n{'='*60}")
    print(f"  OLMoE Expert Placement Policy Report")
    print(f"  {n_layers} layers × {n_experts} experts each")
    print(f"{'='*60}")

    for il_str, info in sorted(policy['layers'].items(), key=lambda x: int(x[0])):
        n_hot  = len(info['hot'])
        n_warm = len(info['warm'])
        n_cold = len(info['cold'])
        print(f"  Layer {int(il_str):2d}: "
              f"hot={n_hot:2d} warm={n_warm:2d} cold={n_cold:2d}  "
              f"hot experts: {info['hot'][:5]}{'...' if n_hot > 5 else ''}")

    total_experts = n_layers * n_experts
    total_hot  = sum(len(v['hot'])  for v in policy['layers'].values())
    total_warm = sum(len(v['warm']) for v in policy['layers'].values())
    total_cold = sum(len(v['cold']) for v in policy['layers'].values())
    print(f"\n  Total:  hot={total_hot} ({100*total_hot//total_experts}%)  "
          f"warm={total_warm} ({100*total_warm//total_experts}%)  "
          f"cold={total_cold} ({100*total_cold//total_experts}%)")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='LeanInfer expert placement policy generator')
    ap.add_argument('--weight-profile',  default=os.path.join(HERE, 'olmoe_expert_profile.json'))
    ap.add_argument('--runtime-profile', default=os.path.join(HERE, 'olmoe_runtime_profile.json'))
    ap.add_argument('--coact-profile',   default=os.path.join(HERE, 'olmoe_coactivation.json'))
    ap.add_argument('--memory-budget-hot',  type=float, default=0.20,
                    help='Fraction of experts per layer to keep in hot tier (default 0.20)')
    ap.add_argument('--memory-budget-warm', type=float, default=0.30,
                    help='Additional fraction for warm tier (default 0.30)')
    ap.add_argument('--weight-alpha', type=float, default=0.30,
                    help='Weight of structural (weight-based) signal (default 0.30)')
    ap.add_argument('--runtime-alpha', type=float, default=0.70,
                    help='Weight of runtime activation signal (default 0.70)')
    ap.add_argument('--output', default=os.path.join(HERE, 'policy.json'))
    ap.add_argument('--report', action='store_true', help='Print human-readable report')
    args = ap.parse_args()

    # Load profiles
    weight_profile  = None
    runtime_profile = None

    if os.path.exists(args.weight_profile):
        weight_profile = load_json(args.weight_profile)
        print(f'Loaded weight profile: {args.weight_profile}')
    else:
        print(f'[warn] weight profile not found: {args.weight_profile}', file=sys.stderr)

    if os.path.exists(args.runtime_profile):
        runtime_profile = load_json(args.runtime_profile)
        print(f'Loaded runtime profile: {args.runtime_profile}')
    else:
        print(f'[warn] runtime profile not found: {args.runtime_profile}', file=sys.stderr)

    if weight_profile is None and runtime_profile is None:
        print('ERROR: no profiles found.', file=sys.stderr)
        sys.exit(1)

    # Determine dimensions
    n_experts = (weight_profile or runtime_profile).get('n_experts', 64)
    n_layers  = (weight_profile or runtime_profile).get('n_layers',  16)

    # Build rankings
    w_rank = rank_experts_weight(weight_profile)  if weight_profile  else {}
    r_rank = rank_experts_runtime(runtime_profile) if runtime_profile else {}

    # If only one source, set its alpha to 1.0
    if not w_rank:
        args.weight_alpha, args.runtime_alpha = 0.0, 1.0
    elif not r_rank:
        args.weight_alpha, args.runtime_alpha = 1.0, 0.0

    # Blend
    blended = blend_rankings(w_rank, r_rank, n_experts,
                              weight_alpha=args.weight_alpha,
                              runtime_alpha=args.runtime_alpha)

    # Classify
    policy_layers = {}
    for il in range(n_layers):
        ordered = blended.get(il, list(range(n_experts)))
        hot, warm, cold = classify_by_budget(ordered,
                                              args.memory_budget_hot,
                                              args.memory_budget_warm,
                                              n_experts)
        policy_layers[str(il)] = {
            'hot':  [int(e) for e in hot],
            'warm': [int(e) for e in warm],
            'cold': [int(e) for e in cold],
        }

    policy = {
        'model':      'OLMoE-1B-7B',
        'n_layers':   n_layers,
        'n_experts':  n_experts,
        'budget_hot':  args.memory_budget_hot,
        'budget_warm': args.memory_budget_warm,
        'layers': policy_layers,
    }

    with open(args.output, 'w') as f:
        json.dump(policy, f, indent=2)
    print(f'Policy written to {args.output}')

    if args.report:
        print_report(policy, n_experts)


if __name__ == '__main__':
    main()
