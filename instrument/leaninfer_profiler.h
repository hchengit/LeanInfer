#ifndef LEANINFER_PROFILER_H
#define LEANINFER_PROFILER_H

// LeanInfer Phase 0a Profiler
// Lightweight instrumentation for ik_llama.cpp inference
// Outputs chrome://tracing compatible JSON
//
// Usage:
//   Build with -DLEANINFER_PROFILE to enable
//   Run with --li-profile <output.json> to collect traces
//   Open chrome://tracing or ui.perfetto.dev to visualize

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Core profiler lifecycle
// ============================================================

// Initialize profiler. Call once at startup.
// max_events: pre-allocated event ring buffer size (0 = default 1M)
void li_profiler_init(int max_events);

// Finalize and write trace to file. Call once at shutdown.
void li_profiler_finish(const char * output_path);

// Enable/disable collection at runtime (default: enabled after init)
void li_profiler_enable(void);
void li_profiler_disable(void);
bool li_profiler_is_enabled(void);

// ============================================================
// Event recording — duration events (B/E pairs)
// ============================================================

// Begin a named duration event.
// category: grouping tag (e.g., "layer", "expert", "kv", "graph")
// name: event name (e.g., "layer_12_attn", "expert_42_load")
// Returns event ID for matching end call.
int64_t li_event_begin(const char * category, const char * name);

// End a duration event started with li_event_begin.
void li_event_end(int64_t event_id);

// ============================================================
// Event recording — instant events (markers)
// ============================================================

// Record an instantaneous event (cache hit/miss, eviction, etc.)
void li_event_instant(const char * category, const char * name);

// ============================================================
// Counter events (tracked values over time)
// ============================================================

// Record a counter value at current timestamp.
// name: counter name (e.g., "kv_cache_bytes", "experts_loaded")
// value: current value
void li_counter(const char * name, int64_t value);

// ============================================================
// Convenience macros — zero overhead when profiling disabled
// ============================================================

#ifdef LEANINFER_PROFILE

#define LI_PROFILE_INIT(n)           li_profiler_init(n)
#define LI_PROFILE_FINISH(path)      li_profiler_finish(path)

// Scoped timing — use in C code (manual begin/end)
#define LI_PROFILE_BEGIN(cat, name)  li_event_begin(cat, name)
#define LI_PROFILE_END(id)           li_event_end(id)
#define LI_PROFILE_INSTANT(cat, name) li_event_instant(cat, name)
#define LI_PROFILE_COUNTER(name, val) li_counter(name, val)

#else

#define LI_PROFILE_INIT(n)           ((void)0)
#define LI_PROFILE_FINISH(path)      ((void)0)
#define LI_PROFILE_BEGIN(cat, name)  ((int64_t)0)
#define LI_PROFILE_END(id)           ((void)0)
#define LI_PROFILE_INSTANT(cat, name) ((void)0)
#define LI_PROFILE_COUNTER(name, val) ((void)0)

#endif // LEANINFER_PROFILE

// ============================================================
// C++ RAII scope guard (zero overhead when disabled)
// ============================================================

#ifdef __cplusplus
} // extern "C"

#ifdef LEANINFER_PROFILE

struct LiProfileScope {
    int64_t id;
    LiProfileScope(const char * cat, const char * name)
        : id(li_event_begin(cat, name)) {}
    ~LiProfileScope() { li_event_end(id); }
    LiProfileScope(const LiProfileScope &) = delete;
    LiProfileScope & operator=(const LiProfileScope &) = delete;
};

#define LI_PROFILE_SCOPE(cat, name) \
    LiProfileScope _li_scope_##__LINE__(cat, name)

#else

#define LI_PROFILE_SCOPE(cat, name) ((void)0)

#endif // LEANINFER_PROFILE
#endif // __cplusplus

#endif // LEANINFER_PROFILER_H
