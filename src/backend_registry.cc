#include "simd_bench/backend_registry.h"
#include <algorithm>
#include <sstream>
#include <cstring>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef SIMD_BENCH_HAS_PAPI
#include <papi.h>
#endif

#ifdef SIMD_BENCH_HAS_LIKWID
#include <likwid.h>
#endif

namespace simd_bench {

// ============================================================================
// BackendRegistry Implementation
// ============================================================================

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry instance;
    return instance;
}

BackendRegistry::BackendRegistry() {
    register_builtin_backends();
}

void BackendRegistry::register_builtin_backends() {
    // Always register null backend
    register_backend(
        "null",
        []() { return std::make_unique<NullBackend>(); },
        -1000,
        "any",
        []() { return true; }
    );

    // Register simulated backend
    register_backend(
        "simulated",
        []() { return std::make_unique<SimulatedBackend>(); },
        -500,
        "any",
        []() { return true; }
    );

#ifdef __linux__
    // Register perf_event backend
    register_backend(
        "perf_event",
        []() { return std::make_unique<PerfEventBackend>(); },
        100,
        "linux",
        []() {
            // Check if perf_event_open syscall is available
            struct perf_event_attr pe = {};
            pe.type = PERF_TYPE_HARDWARE;
            pe.size = sizeof(pe);
            pe.config = PERF_COUNT_HW_CPU_CYCLES;
            pe.disabled = 1;
            pe.exclude_kernel = 1;
            pe.exclude_hv = 1;

            int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            if (fd >= 0) {
                close(fd);
                return true;
            }
            return false;
        }
    );
#endif

#ifdef SIMD_BENCH_HAS_PAPI
    register_backend(
        "papi",
        []() { return std::make_unique<PAPIBackend>(); },
        80,
        "any",
        []() {
            int retval = PAPI_library_init(PAPI_VER_CURRENT);
            if (retval == PAPI_VER_CURRENT) {
                PAPI_shutdown();
                return true;
            }
            return false;
        }
    );
#endif

#ifdef SIMD_BENCH_HAS_LIKWID
    register_backend(
        "likwid",
        []() { return std::make_unique<LIKWIDBackend>(); },
        90,
        "linux",
        []() {
            // Try to initialize LIKWID
            int err = perfmon_init(1, nullptr);
            if (err == 0) {
                perfmon_finalize();
                return true;
            }
            return false;
        }
    );
#endif
}

void BackendRegistry::register_backend(
    const std::string& name,
    BackendFactory factory,
    int priority,
    const std::string& platform,
    std::function<bool()> probe_func
) {
    std::lock_guard<std::mutex> lock(mutex_);

    BackendRegistration reg;
    reg.name = name;
    reg.factory = std::move(factory);
    reg.priority = priority;
    reg.platform = platform;
    reg.probe_func = std::move(probe_func);

    backends_[name] = std::move(reg);
    probed_ = false;  // Need to re-probe
}

void BackendRegistry::unregister_backend(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    backends_.erase(name);
    probe_results_.erase(name);
}

void BackendRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    backends_.clear();
    probe_results_.clear();
    default_backend_.clear();
    probed_ = false;

    // Re-register built-in backends
    register_builtin_backends();
}

std::vector<std::string> BackendRegistry::list_registered() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(backends_.size());

    for (const auto& [name, reg] : backends_) {
        names.push_back(name);
    }

    // Sort by priority (highest first)
    std::sort(names.begin(), names.end(), [this](const std::string& a, const std::string& b) {
        return backends_.at(a).priority > backends_.at(b).priority;
    });

    return names;
}

std::vector<std::string> BackendRegistry::list_available() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!probed_) {
        const_cast<BackendRegistry*>(this)->probe_all_internal();
    }

    std::vector<std::string> available;
    for (const auto& [name, result] : probe_results_) {
        if (result) {
            available.push_back(name);
        }
    }

    // Sort by priority
    std::sort(available.begin(), available.end(), [this](const std::string& a, const std::string& b) {
        auto it_a = backends_.find(a);
        auto it_b = backends_.find(b);
        int prio_a = (it_a != backends_.end()) ? it_a->second.priority : 0;
        int prio_b = (it_b != backends_.end()) ? it_b->second.priority : 0;
        return prio_a > prio_b;
    });

    return available;
}

bool BackendRegistry::is_registered(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backends_.find(name) != backends_.end();
}

bool BackendRegistry::is_available(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!probed_) {
        const_cast<BackendRegistry*>(this)->probe_all_internal();
    }

    auto it = probe_results_.find(name);
    return it != probe_results_.end() && it->second;
}

std::optional<BackendRegistration> BackendRegistry::get_registration(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = backends_.find(name);
    if (it != backends_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::unique_ptr<ICounterBackend> BackendRegistry::create(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = backends_.find(name);
    if (it == backends_.end()) {
        return nullptr;
    }

    return it->second.factory();
}

std::unique_ptr<ICounterBackend> BackendRegistry::create_best_available() {
    auto available = list_available();

    if (available.empty()) {
        return std::make_unique<NullBackend>();
    }

    // Use default if set and available
    if (!default_backend_.empty()) {
        auto it = std::find(available.begin(), available.end(), default_backend_);
        if (it != available.end()) {
            return create(default_backend_);
        }
    }

    // Return highest priority available backend
    return create(available[0]);
}

std::unique_ptr<ICounterBackend> BackendRegistry::create_with_capability(BackendCapability required) {
    auto available = list_available();

    for (const auto& name : available) {
        auto backend = create(name);
        if (backend && backend->has_capability(required)) {
            return backend;
        }
    }

    return nullptr;
}

// Internal probe implementation (assumes mutex already held)
void BackendRegistry::probe_all_internal() {
    probe_results_.clear();

    for (const auto& [name, reg] : backends_) {
        bool available = false;

        // Check platform compatibility
#ifdef __linux__
        bool platform_ok = (reg.platform == "any" || reg.platform == "linux");
#elif defined(_WIN32)
        bool platform_ok = (reg.platform == "any" || reg.platform == "windows");
#elif defined(__APPLE__)
        bool platform_ok = (reg.platform == "any" || reg.platform == "macos");
#else
        bool platform_ok = (reg.platform == "any");
#endif

        if (platform_ok && reg.probe_func) {
            try {
                available = reg.probe_func();
            } catch (...) {
                available = false;
            }
        }

        probe_results_[name] = available;
    }

    probed_ = true;
}

void BackendRegistry::probe_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    probe_all_internal();
}

std::vector<std::string> BackendRegistry::get_probe_results() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> results;
    for (const auto& [name, available] : probe_results_) {
        std::ostringstream oss;
        oss << name << ": " << (available ? "available" : "not available");
        results.push_back(oss.str());
    }

    return results;
}

void BackendRegistry::set_default_backend(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_backend_ = name;
}

std::string BackendRegistry::get_default_backend() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return default_backend_;
}

// ============================================================================
// PerfEventBackend Implementation
// ============================================================================

#ifdef __linux__

struct PerfEventBackend::Impl {
    std::vector<int> fds;
    std::vector<CounterEvent> events;
    pid_t pid = 0;
    int cpu = -1;
    bool initialized = false;
    bool running = false;

    struct EventConfig {
        CounterEvent event;
        uint32_t type;
        uint64_t config;
    };

    std::vector<EventConfig> event_configs;

    static EventConfig get_event_config(CounterEvent event) {
        EventConfig cfg;
        cfg.event = event;

        switch (event) {
            case CounterEvent::CYCLES:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_CPU_CYCLES;
                break;

            case CounterEvent::INSTRUCTIONS:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_INSTRUCTIONS;
                break;

            case CounterEvent::CACHE_REFERENCES:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_CACHE_REFERENCES;
                break;

            case CounterEvent::CACHE_MISSES:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_CACHE_MISSES;
                break;

            case CounterEvent::BRANCH_INSTRUCTIONS:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
                break;

            case CounterEvent::BRANCH_MISSES:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_BRANCH_MISSES;
                break;

            case CounterEvent::L1D_READ_ACCESS:
                cfg.type = PERF_TYPE_HW_CACHE;
                cfg.config = (PERF_COUNT_HW_CACHE_L1D) |
                            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                            (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;

            case CounterEvent::L1D_READ_MISS:
                cfg.type = PERF_TYPE_HW_CACHE;
                cfg.config = (PERF_COUNT_HW_CACHE_L1D) |
                            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;

            case CounterEvent::L1D_WRITE_ACCESS:
                cfg.type = PERF_TYPE_HW_CACHE;
                cfg.config = (PERF_COUNT_HW_CACHE_L1D) |
                            (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
                            (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                break;

            case CounterEvent::L1D_WRITE_MISS:
                cfg.type = PERF_TYPE_HW_CACHE;
                cfg.config = (PERF_COUNT_HW_CACHE_L1D) |
                            (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
                            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                break;

            default:
                cfg.type = PERF_TYPE_HARDWARE;
                cfg.config = PERF_COUNT_HW_CPU_CYCLES;
                break;
        }

        return cfg;
    }
};

PerfEventBackend::PerfEventBackend() : impl_(std::make_unique<Impl>()) {}

PerfEventBackend::~PerfEventBackend() {
    shutdown();
}

std::string PerfEventBackend::version() const {
    return "1.0";
}

BackendInfo PerfEventBackend::info() const {
    BackendInfo bi;
    bi.name = "perf_event";
    bi.version = version();
    bi.description = "Linux perf_event subsystem";
    bi.priority = 100;
    bi.capabilities = capabilities();
    bi.requires_root = false;  // Depends on paranoid level
    bi.requires_kernel_module = false;
    bi.platform = "linux";
    return bi;
}

bool PerfEventBackend::probe() {
    struct perf_event_attr pe = {};
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd >= 0) {
        close(fd);
        return true;
    }
    return false;
}

bool PerfEventBackend::initialize() {
    if (impl_->initialized) return true;
    impl_->initialized = true;
    return true;
}

void PerfEventBackend::shutdown() {
    for (int fd : impl_->fds) {
        if (fd >= 0) {
            close(fd);
        }
    }
    impl_->fds.clear();
    impl_->events.clear();
    impl_->event_configs.clear();
    impl_->initialized = false;
    impl_->running = false;
}

std::vector<CounterEvent> PerfEventBackend::supported_events() const {
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::CACHE_REFERENCES,
        CounterEvent::CACHE_MISSES,
        CounterEvent::BRANCH_INSTRUCTIONS,
        CounterEvent::BRANCH_MISSES,
        CounterEvent::L1D_READ_ACCESS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L1D_WRITE_ACCESS,
        CounterEvent::L1D_WRITE_MISS
    };
}

bool PerfEventBackend::is_event_supported(CounterEvent event) const {
    auto supported = supported_events();
    return std::find(supported.begin(), supported.end(), event) != supported.end();
}

bool PerfEventBackend::add_event(CounterEvent event) {
    if (!is_event_supported(event)) {
        last_error_ = "Event not supported";
        return false;
    }

    auto cfg = Impl::get_event_config(event);

    struct perf_event_attr pe = {};
    pe.type = cfg.type;
    pe.size = sizeof(pe);
    pe.config = cfg.config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

    int leader_fd = impl_->fds.empty() ? -1 : impl_->fds[0];

    int fd = syscall(__NR_perf_event_open, &pe, impl_->pid, impl_->cpu, leader_fd, 0);
    if (fd < 0) {
        last_error_ = "Failed to open perf event: " + std::string(strerror(errno));
        return false;
    }

    impl_->fds.push_back(fd);
    impl_->events.push_back(event);
    impl_->event_configs.push_back(cfg);

    return true;
}

bool PerfEventBackend::remove_event(CounterEvent event) {
    for (size_t i = 0; i < impl_->events.size(); ++i) {
        if (impl_->events[i] == event) {
            close(impl_->fds[i]);
            impl_->fds.erase(impl_->fds.begin() + i);
            impl_->events.erase(impl_->events.begin() + i);
            impl_->event_configs.erase(impl_->event_configs.begin() + i);
            return true;
        }
    }
    return false;
}

void PerfEventBackend::clear_events() {
    for (int fd : impl_->fds) {
        if (fd >= 0) {
            close(fd);
        }
    }
    impl_->fds.clear();
    impl_->events.clear();
    impl_->event_configs.clear();
}

std::vector<CounterEvent> PerfEventBackend::get_active_events() const {
    return impl_->events;
}

bool PerfEventBackend::start() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    impl_->running = true;
    return true;
}

bool PerfEventBackend::stop() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    }
    impl_->running = false;
    return true;
}

bool PerfEventBackend::reset() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    }
    return true;
}

CounterValues PerfEventBackend::read() {
    CounterValues values;

    struct read_format {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    };

    for (size_t i = 0; i < impl_->fds.size(); ++i) {
        read_format data;
        ssize_t n = ::read(impl_->fds[i], &data, sizeof(data));

        if (n == sizeof(data)) {
            uint64_t scaled_value = data.value;
            if (data.time_running > 0 && data.time_enabled > data.time_running) {
                // Apply multiplexing scaling
                scaled_value = static_cast<uint64_t>(
                    static_cast<double>(data.value) *
                    static_cast<double>(data.time_enabled) /
                    static_cast<double>(data.time_running)
                );
            }
            values.set(impl_->events[i], scaled_value);
        }
    }

    return values;
}

BackendCapability PerfEventBackend::capabilities() const {
    return BackendCapability::CORE_COUNTERS |
           BackendCapability::CACHE_COUNTERS |
           BackendCapability::MULTIPLEXING |
           BackendCapability::PER_THREAD |
           BackendCapability::FIXED_COUNTERS |
           BackendCapability::PROGRAMMABLE;
}

bool PerfEventBackend::set_pid(pid_t pid) {
    if (impl_->running) return false;
    impl_->pid = pid;
    return true;
}

bool PerfEventBackend::set_cpu(int cpu) {
    if (impl_->running) return false;
    impl_->cpu = cpu;
    return true;
}

#endif  // __linux__

// ============================================================================
// PAPIBackend Implementation
// ============================================================================

#ifdef SIMD_BENCH_HAS_PAPI

struct PAPIBackend::Impl {
    int event_set = PAPI_NULL;
    std::vector<CounterEvent> events;
    std::vector<int> papi_events;
    bool initialized = false;
    bool running = false;

    static int to_papi_event(CounterEvent event) {
        switch (event) {
            case CounterEvent::CYCLES: return PAPI_TOT_CYC;
            case CounterEvent::INSTRUCTIONS: return PAPI_TOT_INS;
            case CounterEvent::CACHE_REFERENCES: return PAPI_L1_TCA;
            case CounterEvent::CACHE_MISSES: return PAPI_L1_TCM;
            case CounterEvent::BRANCH_INSTRUCTIONS: return PAPI_BR_INS;
            case CounterEvent::BRANCH_MISSES: return PAPI_BR_MSP;
            case CounterEvent::L1D_READ_ACCESS: return PAPI_L1_DCA;
            case CounterEvent::L1D_READ_MISS: return PAPI_L1_DCM;
            case CounterEvent::L2_READ_ACCESS: return PAPI_L2_DCA;
            case CounterEvent::L2_READ_MISS: return PAPI_L2_DCM;
            case CounterEvent::L3_READ_ACCESS: return PAPI_L3_TCA;
            case CounterEvent::L3_READ_MISS: return PAPI_L3_TCM;
            default: return PAPI_NULL;
        }
    }
};

PAPIBackend::PAPIBackend() : impl_(std::make_unique<Impl>()) {}

PAPIBackend::~PAPIBackend() {
    shutdown();
}

std::string PAPIBackend::version() const {
    return std::to_string(PAPI_VERSION_MAJOR(PAPI_VER_CURRENT)) + "." +
           std::to_string(PAPI_VERSION_MINOR(PAPI_VER_CURRENT));
}

BackendInfo PAPIBackend::info() const {
    BackendInfo bi;
    bi.name = "papi";
    bi.version = version();
    bi.description = "Performance Application Programming Interface (PAPI)";
    bi.priority = 80;
    bi.capabilities = capabilities();
    bi.requires_root = false;
    bi.requires_kernel_module = false;
    bi.platform = "any";
    return bi;
}

bool PAPIBackend::probe() {
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval == PAPI_VER_CURRENT) {
        PAPI_shutdown();
        return true;
    }
    return false;
}

bool PAPIBackend::initialize() {
    if (impl_->initialized) return true;

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        last_error_ = "PAPI library init failed";
        return false;
    }

    impl_->event_set = PAPI_NULL;
    retval = PAPI_create_eventset(&impl_->event_set);
    if (retval != PAPI_OK) {
        last_error_ = "Failed to create PAPI event set";
        PAPI_shutdown();
        return false;
    }

    impl_->initialized = true;
    return true;
}

void PAPIBackend::shutdown() {
    if (!impl_->initialized) return;

    if (impl_->running) {
        PAPI_stop(impl_->event_set, nullptr);
    }

    if (impl_->event_set != PAPI_NULL) {
        PAPI_cleanup_eventset(impl_->event_set);
        PAPI_destroy_eventset(&impl_->event_set);
    }

    PAPI_shutdown();
    impl_->initialized = false;
    impl_->running = false;
    impl_->events.clear();
    impl_->papi_events.clear();
}

std::vector<CounterEvent> PAPIBackend::supported_events() const {
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::CACHE_REFERENCES,
        CounterEvent::CACHE_MISSES,
        CounterEvent::BRANCH_INSTRUCTIONS,
        CounterEvent::BRANCH_MISSES,
        CounterEvent::L1D_READ_ACCESS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L2_READ_ACCESS,
        CounterEvent::L2_READ_MISS,
        CounterEvent::L3_READ_ACCESS,
        CounterEvent::L3_READ_MISS
    };
}

bool PAPIBackend::is_event_supported(CounterEvent event) const {
    int papi_event = Impl::to_papi_event(event);
    if (papi_event == PAPI_NULL) return false;

    return PAPI_query_event(papi_event) == PAPI_OK;
}

bool PAPIBackend::add_event(CounterEvent event) {
    int papi_event = Impl::to_papi_event(event);
    if (papi_event == PAPI_NULL) {
        last_error_ = "Unknown event";
        return false;
    }

    int retval = PAPI_add_event(impl_->event_set, papi_event);
    if (retval != PAPI_OK) {
        last_error_ = PAPI_strerror(retval);
        return false;
    }

    impl_->events.push_back(event);
    impl_->papi_events.push_back(papi_event);
    return true;
}

bool PAPIBackend::remove_event(CounterEvent event) {
    for (size_t i = 0; i < impl_->events.size(); ++i) {
        if (impl_->events[i] == event) {
            PAPI_remove_event(impl_->event_set, impl_->papi_events[i]);
            impl_->events.erase(impl_->events.begin() + i);
            impl_->papi_events.erase(impl_->papi_events.begin() + i);
            return true;
        }
    }
    return false;
}

void PAPIBackend::clear_events() {
    for (int papi_event : impl_->papi_events) {
        PAPI_remove_event(impl_->event_set, papi_event);
    }
    impl_->events.clear();
    impl_->papi_events.clear();
}

std::vector<CounterEvent> PAPIBackend::get_active_events() const {
    return impl_->events;
}

bool PAPIBackend::start() {
    int retval = PAPI_start(impl_->event_set);
    if (retval != PAPI_OK) {
        last_error_ = PAPI_strerror(retval);
        return false;
    }
    impl_->running = true;
    return true;
}

bool PAPIBackend::stop() {
    std::vector<long long> values(impl_->events.size());
    int retval = PAPI_stop(impl_->event_set, values.data());
    if (retval != PAPI_OK) {
        last_error_ = PAPI_strerror(retval);
        return false;
    }
    impl_->running = false;
    return true;
}

bool PAPIBackend::reset() {
    int retval = PAPI_reset(impl_->event_set);
    return retval == PAPI_OK;
}

CounterValues PAPIBackend::read() {
    CounterValues result;

    std::vector<long long> values(impl_->events.size());
    int retval = PAPI_read(impl_->event_set, values.data());

    if (retval == PAPI_OK) {
        for (size_t i = 0; i < impl_->events.size(); ++i) {
            result.set(impl_->events[i], static_cast<uint64_t>(values[i]));
        }
    }

    return result;
}

BackendCapability PAPIBackend::capabilities() const {
    return BackendCapability::CORE_COUNTERS |
           BackendCapability::CACHE_COUNTERS |
           BackendCapability::MULTIPLEXING |
           BackendCapability::PER_THREAD;
}

int PAPIBackend::get_event_set() const {
    return impl_->event_set;
}

std::vector<std::string> PAPIBackend::list_native_events() const {
    std::vector<std::string> events;

    int code = PAPI_NATIVE_MASK;
    int retval = PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, 0);

    while (retval == PAPI_OK) {
        PAPI_event_info_t info;
        if (PAPI_get_event_info(code, &info) == PAPI_OK) {
            events.push_back(info.symbol);
        }
        retval = PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, 0);
    }

    return events;
}

#endif  // SIMD_BENCH_HAS_PAPI

// ============================================================================
// LIKWIDBackend Implementation
// ============================================================================

#ifdef SIMD_BENCH_HAS_LIKWID

struct LIKWIDBackend::Impl {
    std::vector<CounterEvent> events;
    std::string current_group;
    int group_id = -1;
    bool initialized = false;
    bool running = false;
};

LIKWIDBackend::LIKWIDBackend() : impl_(std::make_unique<Impl>()) {}

LIKWIDBackend::~LIKWIDBackend() {
    shutdown();
}

std::string LIKWIDBackend::version() const {
    return LIKWID_VERSION;
}

BackendInfo LIKWIDBackend::info() const {
    BackendInfo bi;
    bi.name = "likwid";
    bi.version = version();
    bi.description = "LIKWID performance monitoring library";
    bi.priority = 90;
    bi.capabilities = capabilities();
    bi.requires_root = false;  // With access daemon
    bi.requires_kernel_module = true;  // likwid-accessD
    bi.platform = "linux";
    return bi;
}

bool LIKWIDBackend::probe() {
    int err = perfmon_init(1, nullptr);
    if (err == 0) {
        perfmon_finalize();
        return true;
    }
    return false;
}

bool LIKWIDBackend::initialize() {
    if (impl_->initialized) return true;

    topology_init();
    int err = perfmon_init(1, nullptr);
    if (err != 0) {
        last_error_ = "LIKWID perfmon init failed";
        return false;
    }

    impl_->initialized = true;
    return true;
}

void LIKWIDBackend::shutdown() {
    if (!impl_->initialized) return;

    if (impl_->running) {
        perfmon_stopCounters();
    }

    perfmon_finalize();
    topology_finalize();

    impl_->initialized = false;
    impl_->running = false;
    impl_->events.clear();
    impl_->group_id = -1;
}

std::vector<CounterEvent> LIKWIDBackend::supported_events() const {
    // LIKWID uses performance groups, not individual events
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::CACHE_REFERENCES,
        CounterEvent::CACHE_MISSES,
        CounterEvent::L1D_READ_ACCESS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L2_READ_ACCESS,
        CounterEvent::L2_READ_MISS,
        CounterEvent::L3_READ_ACCESS,
        CounterEvent::L3_READ_MISS
    };
}

bool LIKWIDBackend::is_event_supported(CounterEvent event) const {
    auto supported = supported_events();
    return std::find(supported.begin(), supported.end(), event) != supported.end();
}

bool LIKWIDBackend::add_event(CounterEvent event) {
    impl_->events.push_back(event);
    return true;
}

bool LIKWIDBackend::remove_event(CounterEvent event) {
    auto it = std::find(impl_->events.begin(), impl_->events.end(), event);
    if (it != impl_->events.end()) {
        impl_->events.erase(it);
        return true;
    }
    return false;
}

void LIKWIDBackend::clear_events() {
    impl_->events.clear();
}

std::vector<CounterEvent> LIKWIDBackend::get_active_events() const {
    return impl_->events;
}

bool LIKWIDBackend::start() {
    // Set default group if not set
    if (impl_->group_id < 0) {
        if (!set_group("FLOPS_DP")) {
            if (!set_group("MEM")) {
                last_error_ = "No suitable performance group found";
                return false;
            }
        }
    }

    int err = perfmon_startCounters();
    if (err != 0) {
        last_error_ = "Failed to start LIKWID counters";
        return false;
    }

    impl_->running = true;
    return true;
}

bool LIKWIDBackend::stop() {
    int err = perfmon_stopCounters();
    if (err != 0) {
        last_error_ = "Failed to stop LIKWID counters";
        return false;
    }

    impl_->running = false;
    return true;
}

bool LIKWIDBackend::reset() {
    // LIKWID doesn't have a reset - need to stop and start
    if (impl_->running) {
        perfmon_stopCounters();
        perfmon_startCounters();
    }
    return true;
}

CounterValues LIKWIDBackend::read() {
    CounterValues result;

    if (impl_->group_id < 0) return result;

    perfmon_readCounters();

    // Read specific metrics based on group
    // This is a simplified implementation - real one would need to map
    // LIKWID metric names to our CounterEvent enum
    for (int i = 0; i < perfmon_getNumberOfMetrics(impl_->group_id); ++i) {
        double value [[maybe_unused]] = perfmon_getLastMetric(impl_->group_id, i, 0);
        const char* name = perfmon_getMetricName(impl_->group_id, i);

        if (strstr(name, "Runtime") != nullptr) {
            // Runtime is in seconds, convert to cycles approximately
            // This is a rough approximation
        } else if (strstr(name, "CPI") != nullptr) {
            // CPI metric
        }
        // ... map other metrics
    }

    return result;
}

BackendCapability LIKWIDBackend::capabilities() const {
    return BackendCapability::CORE_COUNTERS |
           BackendCapability::CACHE_COUNTERS |
           BackendCapability::SIMD_COUNTERS |
           BackendCapability::ENERGY_COUNTERS |
           BackendCapability::UNCORE_COUNTERS |
           BackendCapability::PER_THREAD;
}

bool LIKWIDBackend::set_group(const std::string& group_name) {
    impl_->group_id = perfmon_addEventSet(group_name.c_str());
    if (impl_->group_id < 0) {
        last_error_ = "Failed to add performance group: " + group_name;
        return false;
    }

    int err = perfmon_setupCounters(impl_->group_id);
    if (err != 0) {
        last_error_ = "Failed to setup counters for group: " + group_name;
        return false;
    }

    impl_->current_group = group_name;
    return true;
}

std::vector<std::string> LIKWIDBackend::available_groups() const {
    std::vector<std::string> groups;

    // Common LIKWID groups
    const char* common_groups[] = {
        "FLOPS_DP", "FLOPS_SP", "FLOPS_AVX",
        "MEM", "L2", "L3",
        "BRANCH", "ICACHE",
        "ENERGY", "UOPS"
    };

    for (const char* group : common_groups) {
        if (perfmon_addEventSet(group) >= 0) {
            groups.push_back(group);
        }
    }

    return groups;
}

std::string LIKWIDBackend::current_group() const {
    return impl_->current_group;
}

#endif  // SIMD_BENCH_HAS_LIKWID

// ============================================================================
// SimulatedBackend Implementation
// ============================================================================

SimulatedBackend::SimulatedBackend() {
    // Initialize with reasonable default simulated values
    simulated_values_.set(CounterEvent::CYCLES, 0);
    simulated_values_.set(CounterEvent::INSTRUCTIONS, 0);
}

BackendInfo SimulatedBackend::info() const {
    BackendInfo bi;
    bi.name = "simulated";
    bi.version = "1.0";
    bi.description = "Simulated counters for testing";
    bi.priority = -500;
    bi.capabilities = capabilities();
    bi.requires_root = false;
    bi.platform = "any";
    return bi;
}

std::vector<CounterEvent> SimulatedBackend::supported_events() const {
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::CACHE_REFERENCES,
        CounterEvent::CACHE_MISSES,
        CounterEvent::BRANCH_INSTRUCTIONS,
        CounterEvent::BRANCH_MISSES,
        CounterEvent::L1D_READ_ACCESS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L2_READ_ACCESS,
        CounterEvent::L2_READ_MISS,
        CounterEvent::L3_READ_ACCESS,
        CounterEvent::L3_READ_MISS
    };
}

bool SimulatedBackend::is_event_supported(CounterEvent event) const {
    auto supported = supported_events();
    return std::find(supported.begin(), supported.end(), event) != supported.end();
}

bool SimulatedBackend::add_event(CounterEvent event) {
    if (!is_event_supported(event)) return false;
    active_events_.push_back(event);
    return true;
}

bool SimulatedBackend::remove_event(CounterEvent event) {
    auto it = std::find(active_events_.begin(), active_events_.end(), event);
    if (it != active_events_.end()) {
        active_events_.erase(it);
        return true;
    }
    return false;
}

void SimulatedBackend::clear_events() {
    active_events_.clear();
}

std::vector<CounterEvent> SimulatedBackend::get_active_events() const {
    return active_events_;
}

bool SimulatedBackend::start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
    return true;
}

bool SimulatedBackend::stop() {
    running_ = false;
    return true;
}

bool SimulatedBackend::reset() {
    start_time_ = std::chrono::steady_clock::now();
    return true;
}

CounterValues SimulatedBackend::read() {
    CounterValues result;

    auto now = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(now - start_time_).count();

    // Simulate based on elapsed time
    uint64_t simulated_cycles = static_cast<uint64_t>(elapsed_seconds * cycles_per_second_);
    uint64_t simulated_instructions = static_cast<uint64_t>(simulated_cycles * ipc_);

    for (CounterEvent event : active_events_) {
        switch (event) {
            case CounterEvent::CYCLES:
                result.set(event, simulated_cycles);
                break;
            case CounterEvent::INSTRUCTIONS:
                result.set(event, simulated_instructions);
                break;
            case CounterEvent::CACHE_REFERENCES:
                result.set(event, simulated_instructions / 10);  // ~10% are memory ops
                break;
            case CounterEvent::CACHE_MISSES:
                result.set(event, simulated_instructions / 100); // ~1% miss rate
                break;
            case CounterEvent::BRANCH_INSTRUCTIONS:
                result.set(event, simulated_instructions / 5);   // ~20% branches
                break;
            case CounterEvent::BRANCH_MISSES:
                result.set(event, simulated_instructions / 100); // ~1% misprediction
                break;
            default:
                result.set(event, simulated_values_.get(event));
                break;
        }
    }

    return result;
}

BackendCapability SimulatedBackend::capabilities() const {
    return BackendCapability::CORE_COUNTERS |
           BackendCapability::CACHE_COUNTERS |
           BackendCapability::PER_THREAD;
}

void SimulatedBackend::set_simulation_values(const CounterValues& values) {
    simulated_values_ = values;
}

void SimulatedBackend::set_cycles_per_second(uint64_t cps) {
    cycles_per_second_ = cps;
}

void SimulatedBackend::set_instructions_per_cycle(double ipc) {
    ipc_ = ipc;
}

// ============================================================================
// PerformanceCounterAdapter Implementation
// ============================================================================

PerformanceCounterAdapter::PerformanceCounterAdapter(
    std::unique_ptr<IPerformanceCounters> counter
) : counter_(std::move(counter)) {}

std::string PerformanceCounterAdapter::name() const {
    return counter_ ? counter_->get_backend_name() : "null";
}

BackendInfo PerformanceCounterAdapter::info() const {
    BackendInfo bi;
    bi.name = name();
    bi.version = "1.0";
    bi.description = "Adapted IPerformanceCounters backend";
    bi.priority = priority();
    bi.capabilities = capabilities();
    return bi;
}

bool PerformanceCounterAdapter::probe() {
    return counter_ != nullptr;
}

int PerformanceCounterAdapter::priority() const {
    if (!counter_) return -1000;

    switch (counter_->get_backend()) {
        case CounterBackend::PERF_EVENT: return 100;
        case CounterBackend::LIKWID: return 90;
        case CounterBackend::PAPI: return 80;
        default: return 0;
    }
}

bool PerformanceCounterAdapter::initialize() {
    return counter_ && counter_->initialize();
}

void PerformanceCounterAdapter::shutdown() {
    if (counter_) counter_->shutdown();
}

std::vector<CounterEvent> PerformanceCounterAdapter::supported_events() const {
    // Return common events
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::CACHE_REFERENCES,
        CounterEvent::CACHE_MISSES
    };
}

bool PerformanceCounterAdapter::is_event_supported(CounterEvent event) const {
    return counter_ && counter_->is_event_supported(event);
}

bool PerformanceCounterAdapter::add_event(CounterEvent event) {
    if (!counter_) return false;
    if (counter_->add_event(event)) {
        active_events_.push_back(event);
        return true;
    }
    return false;
}

void PerformanceCounterAdapter::clear_events() {
    if (counter_) counter_->clear_events();
    active_events_.clear();
}

std::vector<CounterEvent> PerformanceCounterAdapter::get_active_events() const {
    return active_events_;
}

bool PerformanceCounterAdapter::start() {
    return counter_ && counter_->start();
}

bool PerformanceCounterAdapter::stop() {
    return counter_ && counter_->stop();
}

bool PerformanceCounterAdapter::reset() {
    return counter_ && counter_->reset();
}

CounterValues PerformanceCounterAdapter::read() {
    return counter_ ? counter_->read() : CounterValues{};
}

BackendCapability PerformanceCounterAdapter::capabilities() const {
    if (!counter_) return BackendCapability::NONE;

    BackendCapability caps = BackendCapability::CORE_COUNTERS;

    // Infer capabilities from backend type
    switch (counter_->get_backend()) {
        case CounterBackend::PERF_EVENT:
            caps = caps | BackendCapability::CACHE_COUNTERS |
                   BackendCapability::MULTIPLEXING |
                   BackendCapability::PER_THREAD;
            break;
        case CounterBackend::LIKWID:
            caps = caps | BackendCapability::CACHE_COUNTERS |
                   BackendCapability::SIMD_COUNTERS |
                   BackendCapability::ENERGY_COUNTERS |
                   BackendCapability::UNCORE_COUNTERS;
            break;
        case CounterBackend::PAPI:
            caps = caps | BackendCapability::CACHE_COUNTERS |
                   BackendCapability::MULTIPLEXING;
            break;
        default:
            break;
    }

    return caps;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string capability_to_string(BackendCapability cap) {
    switch (cap) {
        case BackendCapability::NONE: return "none";
        case BackendCapability::CORE_COUNTERS: return "core_counters";
        case BackendCapability::CACHE_COUNTERS: return "cache_counters";
        case BackendCapability::SIMD_COUNTERS: return "simd_counters";
        case BackendCapability::TMA_COUNTERS: return "tma_counters";
        case BackendCapability::ENERGY_COUNTERS: return "energy_counters";
        case BackendCapability::UNCORE_COUNTERS: return "uncore_counters";
        case BackendCapability::MULTIPLEXING: return "multiplexing";
        case BackendCapability::PER_THREAD: return "per_thread";
        case BackendCapability::SYSTEM_WIDE: return "system_wide";
        case BackendCapability::SAMPLING: return "sampling";
        case BackendCapability::OFFCORE: return "offcore";
        case BackendCapability::FIXED_COUNTERS: return "fixed_counters";
        case BackendCapability::PROGRAMMABLE: return "programmable";
        default: return "unknown";
    }
}

std::vector<std::string> capabilities_to_strings(BackendCapability caps) {
    std::vector<std::string> result;

    for (int i = 0; i < 32; ++i) {
        BackendCapability cap = static_cast<BackendCapability>(1 << i);
        if (has_capability(caps, cap)) {
            result.push_back(capability_to_string(cap));
        }
    }

    return result;
}

std::unique_ptr<ICounterBackend> adapt_performance_counter(
    std::unique_ptr<IPerformanceCounters> counter
) {
    return std::make_unique<PerformanceCounterAdapter>(std::move(counter));
}

bool has_backend_with_capability(BackendCapability required) {
    auto& registry = BackendRegistry::instance();
    auto available = registry.list_available();

    for (const auto& name : available) {
        auto backend = registry.create(name);
        if (backend && backend->has_capability(required)) {
            return true;
        }
    }

    return false;
}

std::vector<std::string> backends_supporting_event(CounterEvent event) {
    std::vector<std::string> result;

    auto& registry = BackendRegistry::instance();
    auto available = registry.list_available();

    for (const auto& name : available) {
        auto backend = registry.create(name);
        if (backend && backend->is_event_supported(event)) {
            result.push_back(name);
        }
    }

    return result;
}

}  // namespace simd_bench
