#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace simd_bench {

// Performance counter backend types
enum class CounterBackend {
    NONE,
    PERF_EVENT,   // Linux perf_event
    PAPI,         // PAPI library
    LIKWID        // LIKWID library
};

// Counter value storage
struct CounterValues {
    std::unordered_map<CounterEvent, uint64_t> values;

    uint64_t get(CounterEvent event) const {
        auto it = values.find(event);
        return it != values.end() ? it->second : 0;
    }

    void set(CounterEvent event, uint64_t value) {
        values[event] = value;
    }
};

// Abstract performance counter interface
class IPerformanceCounters {
public:
    virtual ~IPerformanceCounters() = default;

    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    virtual bool add_event(CounterEvent event) = 0;
    virtual void clear_events() = 0;

    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual bool reset() = 0;

    virtual CounterValues read() = 0;

    virtual CounterBackend get_backend() const = 0;
    virtual std::string get_backend_name() const = 0;

    // Check if a specific event is supported
    virtual bool is_event_supported(CounterEvent event) const = 0;
};

// Factory for creating performance counters
class PerformanceCounterFactory {
public:
    static std::unique_ptr<IPerformanceCounters> create(CounterBackend backend = CounterBackend::NONE);
    static std::unique_ptr<IPerformanceCounters> create_best_available();
    static std::vector<CounterBackend> get_available_backends();
    static bool is_backend_available(CounterBackend backend);
};

// Perf event implementation (Linux)
#ifdef __linux__
class PerfEventCounters : public IPerformanceCounters {
public:
    PerfEventCounters();
    ~PerfEventCounters() override;

    bool initialize() override;
    void shutdown() override;

    bool add_event(CounterEvent event) override;
    void clear_events() override;

    bool start() override;
    bool stop() override;
    bool reset() override;

    CounterValues read() override;

    CounterBackend get_backend() const override { return CounterBackend::PERF_EVENT; }
    std::string get_backend_name() const override { return "perf_event"; }
    bool is_event_supported(CounterEvent event) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

// PAPI implementation
#ifdef SIMD_BENCH_HAS_PAPI
class PAPICounters : public IPerformanceCounters {
public:
    PAPICounters();
    ~PAPICounters() override;

    bool initialize() override;
    void shutdown() override;

    bool add_event(CounterEvent event) override;
    void clear_events() override;

    bool start() override;
    bool stop() override;
    bool reset() override;

    CounterValues read() override;

    CounterBackend get_backend() const override { return CounterBackend::PAPI; }
    std::string get_backend_name() const override { return "PAPI"; }
    bool is_event_supported(CounterEvent event) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

// LIKWID implementation
#ifdef SIMD_BENCH_HAS_LIKWID
class LIKWIDCounters : public IPerformanceCounters {
public:
    LIKWIDCounters();
    ~LIKWIDCounters() override;

    bool initialize() override;
    void shutdown() override;

    bool add_event(CounterEvent event) override;
    void clear_events() override;

    bool start() override;
    bool stop() override;
    bool reset() override;

    CounterValues read() override;

    CounterBackend get_backend() const override { return CounterBackend::LIKWID; }
    std::string get_backend_name() const override { return "LIKWID"; }
    bool is_event_supported(CounterEvent event) const override;

    // LIKWID-specific: set performance group
    bool set_performance_group(const std::string& group);
    std::vector<std::string> get_available_groups() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

// Null implementation (no hardware counters)
class NullCounters : public IPerformanceCounters {
public:
    bool initialize() override { return true; }
    void shutdown() override {}
    bool add_event(CounterEvent) override { return true; }
    void clear_events() override {}
    bool start() override { return true; }
    bool stop() override { return true; }
    bool reset() override { return true; }
    CounterValues read() override { return CounterValues{}; }
    CounterBackend get_backend() const override { return CounterBackend::NONE; }
    std::string get_backend_name() const override { return "none"; }
    bool is_event_supported(CounterEvent) const override { return false; }
};

// Scoped counter measurement
class ScopedCounters {
public:
    ScopedCounters(IPerformanceCounters& counters, CounterValues& result);
    ~ScopedCounters();

    ScopedCounters(const ScopedCounters&) = delete;
    ScopedCounters& operator=(const ScopedCounters&) = delete;

private:
    IPerformanceCounters& counters_;
    CounterValues& result_;
};

// Convert CounterEvent to string name
std::string counter_event_to_string(CounterEvent event);

// Parse string to CounterEvent
CounterEvent string_to_counter_event(const std::string& name);

// Get standard event sets for common measurements
std::vector<CounterEvent> get_flops_events();
std::vector<CounterEvent> get_cache_events();
std::vector<CounterEvent> get_memory_events();
std::vector<CounterEvent> get_tma_events();

}  // namespace simd_bench
