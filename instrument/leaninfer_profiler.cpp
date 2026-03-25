// LeanInfer Phase 0a Profiler Implementation
// Chrome Trace Event Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

#ifdef LEANINFER_PROFILE

#include "leaninfer_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include <atomic>
#include <vector>
#include <mutex>
#include <unordered_map>

// ============================================================
// Timing
// ============================================================

static int64_t li_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

// ============================================================
// Event storage
// ============================================================

// Chrome trace event types
enum li_event_type {
    LI_EVENT_BEGIN     = 'B',  // Duration begin
    LI_EVENT_END       = 'E',  // Duration end
    LI_EVENT_INSTANT   = 'i',  // Instant event
    LI_EVENT_COUNTER   = 'C',  // Counter
};

struct li_event {
    int64_t     ts_us;         // Timestamp in microseconds
    const char* category;      // Category string (static, not owned)
    const char* name;          // Event name (static, not owned)
    int64_t     value;         // Counter value or event ID
    uint32_t    tid;           // Thread ID
    uint8_t     type;          // li_event_type
    uint8_t     _pad[7];
};

struct li_profiler_state {
    std::vector<li_event>  events;
    std::mutex             lock;
    std::atomic<bool>      enabled{false};
    std::atomic<int64_t>   next_id{1};
    int64_t                start_time_us;
    int                    max_events;
    bool                   initialized;
};

static li_profiler_state g_profiler = {};

// ============================================================
// Thread ID helper
// ============================================================

static uint32_t li_get_tid(void) {
    // Compact thread ID — Perfetto needs small integers
    pthread_t t = pthread_self();
    uint64_t h = (uint64_t)t;
    return (uint32_t)((h ^ (h >> 32)) & 0xFFFF);
}

// ============================================================
// Public API
// ============================================================

void li_profiler_init(int max_events) {
    if (g_profiler.initialized) return;

    g_profiler.max_events = (max_events > 0) ? max_events : 1000000;
    g_profiler.events.reserve(g_profiler.max_events);
    g_profiler.start_time_us = li_time_us();
    g_profiler.initialized = true;
    g_profiler.enabled.store(true, std::memory_order_release);

    fprintf(stderr, "[LeanInfer Profiler] initialized, max events: %d\n",
            g_profiler.max_events);
}

void li_profiler_enable(void) {
    g_profiler.enabled.store(true, std::memory_order_release);
}

void li_profiler_disable(void) {
    g_profiler.enabled.store(false, std::memory_order_release);
}

bool li_profiler_is_enabled(void) {
    return g_profiler.enabled.load(std::memory_order_acquire);
}

static void li_record_event(uint8_t type, const char * category,
                            const char * name, int64_t value) {
    if (!g_profiler.enabled.load(std::memory_order_acquire)) return;

    li_event ev;
    ev.ts_us    = li_time_us() - g_profiler.start_time_us;
    ev.category = category;
    ev.name     = name;
    ev.value    = value;
    ev.tid      = li_get_tid();
    ev.type     = type;

    std::lock_guard<std::mutex> guard(g_profiler.lock);
    if ((int)g_profiler.events.size() < g_profiler.max_events) {
        g_profiler.events.push_back(ev);
    }
}

// Store begin event info for matching end events
struct li_open_event {
    const char * category;
    const char * name;
};

static std::mutex g_open_lock;
static std::unordered_map<int64_t, li_open_event> g_open_events;

int64_t li_event_begin(const char * category, const char * name) {
    int64_t id = g_profiler.next_id.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> guard(g_open_lock);
        g_open_events[id] = {category, name};
    }
    li_record_event(LI_EVENT_BEGIN, category, name, id);
    return id;
}

void li_event_end(int64_t event_id) {
    const char * cat = "";
    const char * name = "";
    {
        std::lock_guard<std::mutex> guard(g_open_lock);
        auto it = g_open_events.find(event_id);
        if (it != g_open_events.end()) {
            cat = it->second.category;
            name = it->second.name;
            g_open_events.erase(it);
        }
    }
    li_record_event(LI_EVENT_END, cat, name, event_id);
}

void li_event_instant(const char * category, const char * name) {
    li_record_event(LI_EVENT_INSTANT, category, name, 0);
}

void li_counter(const char * name, int64_t value) {
    li_record_event(LI_EVENT_COUNTER, "counter", name, value);
}

// ============================================================
// Chrome Trace JSON writer
// ============================================================

// Escape a string for JSON output
static void li_write_json_string(FILE * f, const char * s) {
    fputc('"', f);
    if (s) {
        for (const char * p = s; *p; p++) {
            switch (*p) {
                case '"':  fputs("\\\"", f); break;
                case '\\': fputs("\\\\", f); break;
                case '\n': fputs("\\n", f);  break;
                case '\t': fputs("\\t", f);  break;
                default:   fputc(*p, f);     break;
            }
        }
    }
    fputc('"', f);
}

void li_profiler_finish(const char * output_path) {
    if (!g_profiler.initialized) return;

    g_profiler.enabled.store(false, std::memory_order_release);

    FILE * f = fopen(output_path, "w");
    if (!f) {
        fprintf(stderr, "[LeanInfer Profiler] ERROR: cannot open %s for writing\n",
                output_path);
        return;
    }

    fprintf(f, "{\"traceEvents\":[\n");

    const auto & events = g_profiler.events;
    bool first = true;

    for (size_t i = 0; i < events.size(); i++) {
        const li_event & ev = events[i];

        if (!first) fprintf(f, ",\n");
        first = false;

        fprintf(f, "{\"ph\":\"%c\",\"ts\":%lld,\"pid\":1,\"tid\":%u",
                (char)ev.type,
                (long long)ev.ts_us,
                ev.tid);

        if (ev.type == LI_EVENT_COUNTER) {
            fprintf(f, ",\"name\":");
            li_write_json_string(f, ev.name);
            fprintf(f, ",\"args\":{");
            li_write_json_string(f, ev.name);
            fprintf(f, ":%lld}", (long long)ev.value);
        } else {
            fprintf(f, ",\"name\":");
            li_write_json_string(f, ev.name);
            fprintf(f, ",\"cat\":");
            li_write_json_string(f, ev.category);
            if (ev.type == LI_EVENT_INSTANT) {
                fprintf(f, ",\"s\":\"t\"");  // thread-scoped instant
            }
        }

        fprintf(f, "}");
    }

    fprintf(f, "\n],\n");
    fprintf(f, "\"displayTimeUnit\":\"ms\",\n");
    fprintf(f, "\"metadata\":{\"leaninfer_version\":\"0.1.0\",\"phase\":\"0a\"}\n");
    fprintf(f, "}\n");

    fclose(f);

    fprintf(stderr, "[LeanInfer Profiler] wrote %zu events to %s\n",
            events.size(), output_path);

    // Cleanup
    g_profiler.events.clear();
    g_profiler.events.shrink_to_fit();
    g_profiler.initialized = false;
}

#endif // LEANINFER_PROFILE
