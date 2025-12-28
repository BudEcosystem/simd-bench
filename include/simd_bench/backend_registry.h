#pragma once

#include "types.h"
#include "performance_counters.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <optional>

namespace simd_bench {

// Backend capability flags
enum class BackendCapability : uint32_t {
    NONE                = 0,
    CORE_COUNTERS       = 1 << 0,   // Basic CPU counters (cycles, instructions)
    CACHE_COUNTERS      = 1 << 1,   // Cache hit/miss counters
    SIMD_COUNTERS       = 1 << 2,   // SIMD/FP operation counters
    TMA_COUNTERS        = 1 << 3,   // Top-down microarchitecture counters
    ENERGY_COUNTERS     = 1 << 4,   // RAPL energy counters
    UNCORE_COUNTERS     = 1 << 5,   // Uncore/IMC counters
    MULTIPLEXING        = 1 << 6,   // Supports counter multiplexing
    PER_THREAD          = 1 << 7,   // Per-thread measurement
    SYSTEM_WIDE         = 1 << 8,   // System-wide measurement
    SAMPLING            = 1 << 9,   // Interrupt-based sampling
    OFFCORE             = 1 << 10,  // Off-core response counters
    FIXED_COUNTERS      = 1 << 11,  // Fixed function counters
    PROGRAMMABLE        = 1 << 12   // Programmable counters
};

inline BackendCapability operator|(BackendCapability a, BackendCapability b) {
    return static_cast<BackendCapability>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline BackendCapability operator&(BackendCapability a, BackendCapability b) {
    return static_cast<BackendCapability>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b)
    );
}

inline bool has_capability(BackendCapability caps, BackendCapability cap) {
    return (static_cast<uint32_t>(caps) & static_cast<uint32_t>(cap)) != 0;
}

// Backend information structure
struct BackendInfo {
    std::string name;
    std::string version;
    std::string description;
    int priority = 0;                    // Higher = preferred
    BackendCapability capabilities = BackendCapability::NONE;
    std::vector<CounterEvent> supported_events;
    bool requires_root = false;
    bool requires_kernel_module = false;
    std::string platform;               // "linux", "windows", "macos", "any"
};

// Abstract counter backend interface (extended from IPerformanceCounters)
class ICounterBackend {
public:
    virtual ~ICounterBackend() = default;

    // Identity
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual BackendInfo info() const = 0;

    // Probing and initialization
    virtual bool probe() = 0;               // Check if backend is available
    virtual int priority() const = 0;       // Higher = preferred
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    // Event management
    virtual std::vector<CounterEvent> supported_events() const = 0;
    virtual bool is_event_supported(CounterEvent event) const = 0;
    virtual bool add_event(CounterEvent event) = 0;
    virtual bool remove_event(CounterEvent event) = 0;
    virtual void clear_events() = 0;
    virtual std::vector<CounterEvent> get_active_events() const = 0;

    // Measurement
    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual bool reset() = 0;
    virtual CounterValues read() = 0;

    // Capability checking
    virtual BackendCapability capabilities() const = 0;
    virtual bool has_capability(BackendCapability cap) const {
        return simd_bench::has_capability(capabilities(), cap);
    }

    // Configuration
    virtual bool configure(const std::string& key, const std::string& value) {
        (void)key; (void)value;
        return false;
    }

    // Diagnostics
    virtual std::string last_error() const { return ""; }
    virtual std::vector<std::string> diagnostics() const { return {}; }
};

// Backend factory function type
using BackendFactory = std::function<std::unique_ptr<ICounterBackend>()>;

// Backend registration info
struct BackendRegistration {
    std::string name;
    BackendFactory factory;
    int priority;
    std::string platform;
    std::function<bool()> probe_func;
};

// Backend Registry - singleton for managing counter backends
class BackendRegistry {
public:
    // Singleton access
    static BackendRegistry& instance();

    // Registration
    void register_backend(
        const std::string& name,
        BackendFactory factory,
        int priority = 0,
        const std::string& platform = "any",
        std::function<bool()> probe_func = nullptr
    );

    template<typename T>
    void register_backend(
        const std::string& name,
        int priority = 0,
        const std::string& platform = "any"
    ) {
        register_backend(
            name,
            []() { return std::make_unique<T>(); },
            priority,
            platform,
            []() {
                auto backend = std::make_unique<T>();
                return backend->probe();
            }
        );
    }

    void unregister_backend(const std::string& name);
    void clear();

    // Query
    std::vector<std::string> list_registered() const;
    std::vector<std::string> list_available() const;
    bool is_registered(const std::string& name) const;
    bool is_available(const std::string& name) const;

    // Get backend info without creating instance
    std::optional<BackendRegistration> get_registration(const std::string& name) const;

    // Factory methods
    std::unique_ptr<ICounterBackend> create(const std::string& name);
    std::unique_ptr<ICounterBackend> create_best_available();
    std::unique_ptr<ICounterBackend> create_with_capability(BackendCapability required);

    // Probing
    void probe_all();
    std::vector<std::string> get_probe_results() const;

    // Configuration
    void set_default_backend(const std::string& name);
    std::string get_default_backend() const;

private:
    BackendRegistry();
    ~BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    void register_builtin_backends();
    void probe_all_internal();  // Internal probe - assumes mutex already held

    mutable std::mutex mutex_;
    std::unordered_map<std::string, BackendRegistration> backends_;
    std::unordered_map<std::string, bool> probe_results_;
    std::string default_backend_;
    bool probed_ = false;
};

// Macro for static backend registration
#define SIMD_BENCH_REGISTER_BACKEND(BackendClass, name, priority) \
    namespace { \
        struct BackendRegistrar_##BackendClass { \
            BackendRegistrar_##BackendClass() { \
                simd_bench::BackendRegistry::instance().register_backend<BackendClass>( \
                    name, priority); \
            } \
        } backend_registrar_##BackendClass; \
    }

// ============================================================================
// Built-in Backend Implementations
// ============================================================================

// Null backend (always available, returns empty values)
class NullBackend : public ICounterBackend {
public:
    std::string name() const override { return "null"; }
    std::string version() const override { return "1.0"; }

    BackendInfo info() const override {
        BackendInfo bi;
        bi.name = "null";
        bi.version = "1.0";
        bi.description = "Null backend - returns empty values";
        bi.priority = -1000;
        bi.capabilities = BackendCapability::NONE;
        bi.requires_root = false;
        bi.platform = "any";
        return bi;
    }

    bool probe() override { return true; }
    int priority() const override { return -1000; }

    bool initialize() override { return true; }
    void shutdown() override {}

    std::vector<CounterEvent> supported_events() const override { return {}; }
    bool is_event_supported(CounterEvent) const override { return false; }
    bool add_event(CounterEvent) override { return true; }
    bool remove_event(CounterEvent) override { return true; }
    void clear_events() override {}
    std::vector<CounterEvent> get_active_events() const override { return {}; }

    bool start() override { return true; }
    bool stop() override { return true; }
    bool reset() override { return true; }
    CounterValues read() override { return CounterValues{}; }

    BackendCapability capabilities() const override { return BackendCapability::NONE; }
};

// Linux perf_event backend
#ifdef __linux__
class PerfEventBackend : public ICounterBackend {
public:
    PerfEventBackend();
    ~PerfEventBackend() override;

    std::string name() const override { return "perf_event"; }
    std::string version() const override;

    BackendInfo info() const override;

    bool probe() override;
    int priority() const override { return 100; }

    bool initialize() override;
    void shutdown() override;

    std::vector<CounterEvent> supported_events() const override;
    bool is_event_supported(CounterEvent event) const override;
    bool add_event(CounterEvent event) override;
    bool remove_event(CounterEvent event) override;
    void clear_events() override;
    std::vector<CounterEvent> get_active_events() const override;

    bool start() override;
    bool stop() override;
    bool reset() override;
    CounterValues read() override;

    BackendCapability capabilities() const override;

    std::string last_error() const override { return last_error_; }

    // perf_event specific
    bool set_pid(pid_t pid);
    bool set_cpu(int cpu);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;
};
#endif

// PAPI backend
#ifdef SIMD_BENCH_HAS_PAPI
class PAPIBackend : public ICounterBackend {
public:
    PAPIBackend();
    ~PAPIBackend() override;

    std::string name() const override { return "papi"; }
    std::string version() const override;

    BackendInfo info() const override;

    bool probe() override;
    int priority() const override { return 80; }

    bool initialize() override;
    void shutdown() override;

    std::vector<CounterEvent> supported_events() const override;
    bool is_event_supported(CounterEvent event) const override;
    bool add_event(CounterEvent event) override;
    bool remove_event(CounterEvent event) override;
    void clear_events() override;
    std::vector<CounterEvent> get_active_events() const override;

    bool start() override;
    bool stop() override;
    bool reset() override;
    CounterValues read() override;

    BackendCapability capabilities() const override;

    std::string last_error() const override { return last_error_; }

    // PAPI-specific
    int get_event_set() const;
    std::vector<std::string> list_native_events() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;
};
#endif

// LIKWID backend
#ifdef SIMD_BENCH_HAS_LIKWID
class LIKWIDBackend : public ICounterBackend {
public:
    LIKWIDBackend();
    ~LIKWIDBackend() override;

    std::string name() const override { return "likwid"; }
    std::string version() const override;

    BackendInfo info() const override;

    bool probe() override;
    int priority() const override { return 90; }

    bool initialize() override;
    void shutdown() override;

    std::vector<CounterEvent> supported_events() const override;
    bool is_event_supported(CounterEvent event) const override;
    bool add_event(CounterEvent event) override;
    bool remove_event(CounterEvent event) override;
    void clear_events() override;
    std::vector<CounterEvent> get_active_events() const override;

    bool start() override;
    bool stop() override;
    bool reset() override;
    CounterValues read() override;

    BackendCapability capabilities() const override;

    std::string last_error() const override { return last_error_; }

    // LIKWID-specific
    bool set_group(const std::string& group_name);
    std::vector<std::string> available_groups() const;
    std::string current_group() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;
};
#endif

// Simulated backend (for testing and demo purposes)
class SimulatedBackend : public ICounterBackend {
public:
    SimulatedBackend();
    ~SimulatedBackend() override = default;

    std::string name() const override { return "simulated"; }
    std::string version() const override { return "1.0"; }

    BackendInfo info() const override;

    bool probe() override { return true; }
    int priority() const override { return -500; }

    bool initialize() override { return true; }
    void shutdown() override {}

    std::vector<CounterEvent> supported_events() const override;
    bool is_event_supported(CounterEvent event) const override;
    bool add_event(CounterEvent event) override;
    bool remove_event(CounterEvent event) override;
    void clear_events() override;
    std::vector<CounterEvent> get_active_events() const override;

    bool start() override;
    bool stop() override;
    bool reset() override;
    CounterValues read() override;

    BackendCapability capabilities() const override;

    // Simulation control
    void set_simulation_values(const CounterValues& values);
    void set_cycles_per_second(uint64_t cps);
    void set_instructions_per_cycle(double ipc);

private:
    std::vector<CounterEvent> active_events_;
    CounterValues simulated_values_;
    uint64_t cycles_per_second_ = 3000000000ULL;  // 3 GHz
    double ipc_ = 2.5;
    std::chrono::steady_clock::time_point start_time_;
    bool running_ = false;
};

// ============================================================================
// Backend Adapter - Wraps IPerformanceCounters as ICounterBackend
// ============================================================================

class PerformanceCounterAdapter : public ICounterBackend {
public:
    explicit PerformanceCounterAdapter(std::unique_ptr<IPerformanceCounters> counter);

    std::string name() const override;
    std::string version() const override { return "1.0"; }
    BackendInfo info() const override;

    bool probe() override;
    int priority() const override;

    bool initialize() override;
    void shutdown() override;

    std::vector<CounterEvent> supported_events() const override;
    bool is_event_supported(CounterEvent event) const override;
    bool add_event(CounterEvent event) override;
    bool remove_event(CounterEvent event) override { (void)event; return false; }
    void clear_events() override;
    std::vector<CounterEvent> get_active_events() const override;

    bool start() override;
    bool stop() override;
    bool reset() override;
    CounterValues read() override;

    BackendCapability capabilities() const override;

private:
    std::unique_ptr<IPerformanceCounters> counter_;
    std::vector<CounterEvent> active_events_;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Get human-readable capability description
std::string capability_to_string(BackendCapability cap);
std::vector<std::string> capabilities_to_strings(BackendCapability caps);

// Create backend from IPerformanceCounters
std::unique_ptr<ICounterBackend> adapt_performance_counter(
    std::unique_ptr<IPerformanceCounters> counter
);

// Check if any backend with required capabilities is available
bool has_backend_with_capability(BackendCapability required);

// Get list of backends supporting specific event
std::vector<std::string> backends_supporting_event(CounterEvent event);

}  // namespace simd_bench
