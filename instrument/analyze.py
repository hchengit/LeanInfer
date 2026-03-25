#!/usr/bin/env python3
"""LeanInfer Phase 0a — Trace analyzer.

Reads chrome://tracing JSON and extracts per-token latency breakdown.

Usage:
    python3 analyze.py <trace.json>
"""

import json
import sys
from collections import defaultdict

def analyze(path):
    with open(path) as f:
        data = json.load(f)

    events = data["traceEvents"]

    # Match begin/end pairs by tracking open events per thread
    open_events = {}  # (tid, stack_depth) -> event
    durations = defaultdict(list)  # name -> [duration_us, ...]
    counters = defaultdict(list)   # name -> [value, ...]

    stack = defaultdict(int)  # tid -> current depth

    for ev in events:
        ph = ev["ph"]
        tid = ev["tid"]

        if ph == "B":
            depth = stack[tid]
            open_events[(tid, depth)] = ev
            stack[tid] += 1

        elif ph == "E":
            stack[tid] -= 1
            depth = stack[tid]
            key = (tid, depth)
            if key in open_events:
                begin = open_events.pop(key)
                dur = ev["ts"] - begin["ts"]
                durations[begin["name"]].append(dur)

        elif ph == "C":
            name = ev["name"]
            val = ev["args"].get(name, 0)
            counters[name].append(val)

    # Print report
    print("=" * 65)
    print(f"LeanInfer Trace Analysis: {path}")
    print(f"Total events: {len(events)}")
    print("=" * 65)

    print("\n--- Duration Events (microseconds) ---\n")
    print(f"{'Event':<20} {'Count':>6} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Total':>12}")
    print("-" * 80)

    for name in sorted(durations.keys()):
        vals = sorted(durations[name])
        n = len(vals)
        mean = sum(vals) / n
        p50 = vals[n // 2]
        p95 = vals[int(n * 0.95)]
        p99 = vals[int(n * 0.99)]
        total = sum(vals)
        print(f"{name:<20} {n:>6} {mean:>10.1f} {p50:>10} {p95:>10} {p99:>10} {total:>12}")

    print("\n--- Counters ---\n")
    for name in sorted(counters.keys()):
        vals = counters[name]
        print(f"{name}: min={min(vals)}, max={max(vals)}, samples={len(vals)}")

    # Token generation analysis
    decode_durs = durations.get("llama_decode", [])
    graph_durs = durations.get("graph_compute", [])

    if decode_durs:
        print("\n--- Token Generation Analysis ---\n")
        # First call is likely prompt eval (larger batch)
        if len(decode_durs) > 1:
            prompt_dur = decode_durs[0]
            gen_durs = decode_durs[1:]
            gen_mean = sum(gen_durs) / len(gen_durs)
            gen_p50 = sorted(gen_durs)[len(gen_durs) // 2]

            print(f"Prompt eval (first decode): {prompt_dur:,} us ({prompt_dur/1000:.1f} ms)")
            print(f"Token generation (n={len(gen_durs)}):")
            print(f"  Mean: {gen_mean:,.0f} us ({gen_mean/1000:.1f} ms) = {1_000_000/gen_mean:.1f} tok/s")
            print(f"  P50:  {gen_p50:,} us ({gen_p50/1000:.1f} ms)")

    if graph_durs:
        graph_total = sum(graph_durs)
        total_time = max(ev["ts"] for ev in events) - min(ev["ts"] for ev in events)
        print(f"\nGraph compute: {graph_total/1000:.1f} ms total ({100*graph_total/max(total_time,1):.1f}% of wall time)")

    print("\n" + "=" * 65)
    print("Open this trace in chrome://tracing or ui.perfetto.dev for visualization")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
