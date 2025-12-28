#include "simd_bench/performance_counters.h"
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <string>

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#ifdef SIMD_BENCH_HAS_PAPI
#include <papi.h>
#endif

#ifdef SIMD_BENCH_HAS_LIKWID
extern "C" {
#include <likwid.h>
}
#endif

namespace simd_bench {

// Event name mappings (comprehensive for all architectures)
static const std::unordered_map<CounterEvent, std::string> event_names = {
    // Basic events
    {CounterEvent::CYCLES, "CYCLES"},
    {CounterEvent::INSTRUCTIONS, "INSTRUCTIONS"},
    {CounterEvent::CACHE_REFERENCES, "CACHE_REFERENCES"},
    {CounterEvent::CACHE_MISSES, "CACHE_MISSES"},
    {CounterEvent::BRANCH_INSTRUCTIONS, "BRANCH_INSTRUCTIONS"},
    {CounterEvent::BRANCH_MISSES, "BRANCH_MISSES"},

    // Intel FP events
    {CounterEvent::FP_ARITH_SCALAR_SINGLE, "FP_ARITH_SCALAR_SINGLE"},
    {CounterEvent::FP_ARITH_SCALAR_DOUBLE, "FP_ARITH_SCALAR_DOUBLE"},
    {CounterEvent::FP_ARITH_128B_PACKED_SINGLE, "FP_ARITH_128B_PACKED_SINGLE"},
    {CounterEvent::FP_ARITH_128B_PACKED_DOUBLE, "FP_ARITH_128B_PACKED_DOUBLE"},
    {CounterEvent::FP_ARITH_256B_PACKED_SINGLE, "FP_ARITH_256B_PACKED_SINGLE"},
    {CounterEvent::FP_ARITH_256B_PACKED_DOUBLE, "FP_ARITH_256B_PACKED_DOUBLE"},
    {CounterEvent::FP_ARITH_512B_PACKED_SINGLE, "FP_ARITH_512B_PACKED_SINGLE"},
    {CounterEvent::FP_ARITH_512B_PACKED_DOUBLE, "FP_ARITH_512B_PACKED_DOUBLE"},

    // AMD FP events
    {CounterEvent::AMD_FP_RET_SSE_AVX_OPS, "AMD_FP_RET_SSE_AVX_OPS"},
    {CounterEvent::AMD_FP_RET_X87_OPS, "AMD_FP_RET_X87_OPS"},
    {CounterEvent::AMD_FP_DISP_PIPE_ASSIGN, "AMD_FP_DISP_PIPE_ASSIGN"},
    {CounterEvent::AMD_FP_SCH_NO_LOW_RES, "AMD_FP_SCH_NO_LOW_RES"},

    // ARM SIMD events
    {CounterEvent::ARM_ASE_SPEC, "ARM_ASE_SPEC"},
    {CounterEvent::ARM_SVE_INST_SPEC, "ARM_SVE_INST_SPEC"},
    {CounterEvent::ARM_SVE_PRED_PARTIAL, "ARM_SVE_PRED_PARTIAL"},
    {CounterEvent::ARM_SVE_PRED_FULL, "ARM_SVE_PRED_FULL"},
    {CounterEvent::ARM_SVE_PRED_EMPTY, "ARM_SVE_PRED_EMPTY"},
    {CounterEvent::ARM_FP_SPEC, "ARM_FP_SPEC"},
    {CounterEvent::ARM_VFP_SPEC, "ARM_VFP_SPEC"},

    // RISC-V Vector events
    {CounterEvent::RVV_VECTOR_INST, "RVV_VECTOR_INST"},
    {CounterEvent::RVV_VECTOR_ELEMENTS, "RVV_VECTOR_ELEMENTS"},
    {CounterEvent::RVV_VL_ACTIVE, "RVV_VL_ACTIVE"},

    // IBM Power VSX events
    {CounterEvent::POWER_VSX_EXEC, "POWER_VSX_EXEC"},
    {CounterEvent::POWER_VSX_SINGLE, "POWER_VSX_SINGLE"},
    {CounterEvent::POWER_VSX_DOUBLE, "POWER_VSX_DOUBLE"},

    // Cache events
    {CounterEvent::L1D_READ_ACCESS, "L1D_READ_ACCESS"},
    {CounterEvent::L1D_READ_MISS, "L1D_READ_MISS"},
    {CounterEvent::L1D_WRITE_ACCESS, "L1D_WRITE_ACCESS"},
    {CounterEvent::L1D_WRITE_MISS, "L1D_WRITE_MISS"},
    {CounterEvent::L2_READ_ACCESS, "L2_READ_ACCESS"},
    {CounterEvent::L2_READ_MISS, "L2_READ_MISS"},
    {CounterEvent::L3_READ_ACCESS, "L3_READ_ACCESS"},
    {CounterEvent::L3_READ_MISS, "L3_READ_MISS"},

    // Cache line split events
    {CounterEvent::LD_BLOCKS_STORE_FORWARD, "LD_BLOCKS_STORE_FORWARD"},
    {CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS, "MEM_INST_RETIRED_SPLIT_LOADS"},
    {CounterEvent::MEM_INST_RETIRED_SPLIT_STORES, "MEM_INST_RETIRED_SPLIT_STORES"},
    {CounterEvent::MISALIGN_MEM_REF_LOADS, "MISALIGN_MEM_REF_LOADS"},
    {CounterEvent::MISALIGN_MEM_REF_STORES, "MISALIGN_MEM_REF_STORES"},

    // Memory events
    {CounterEvent::MEM_LOAD_RETIRED, "MEM_LOAD_RETIRED"},
    {CounterEvent::MEM_STORE_RETIRED, "MEM_STORE_RETIRED"},
    {CounterEvent::OFFCORE_REQUESTS_DEMAND_DATA_RD, "OFFCORE_REQUESTS_DEMAND_DATA_RD"},
    {CounterEvent::OFFCORE_REQUESTS_DEMAND_RFO, "OFFCORE_REQUESTS_DEMAND_RFO"},

    // TMA events
    {CounterEvent::UOPS_RETIRED_SLOTS, "UOPS_RETIRED_SLOTS"},
    {CounterEvent::UOPS_ISSUED_ANY, "UOPS_ISSUED_ANY"},
    {CounterEvent::INT_MISC_RECOVERY_CYCLES, "INT_MISC_RECOVERY_CYCLES"},
    {CounterEvent::CYCLE_ACTIVITY_STALLS_MEM, "CYCLE_ACTIVITY_STALLS_MEM"},
    {CounterEvent::CYCLE_ACTIVITY_STALLS_L1D, "CYCLE_ACTIVITY_STALLS_L1D"},
    {CounterEvent::CYCLE_ACTIVITY_STALLS_L2, "CYCLE_ACTIVITY_STALLS_L2"},
    {CounterEvent::CYCLE_ACTIVITY_STALLS_L3, "CYCLE_ACTIVITY_STALLS_L3"},

    // DSB events
    {CounterEvent::IDQ_DSB_UOPS, "IDQ_DSB_UOPS"},
    {CounterEvent::IDQ_MITE_UOPS, "IDQ_MITE_UOPS"},
    {CounterEvent::IDQ_MS_UOPS, "IDQ_MS_UOPS"},

    // Port utilization events
    {CounterEvent::UOPS_DISPATCHED_PORT_0, "UOPS_DISPATCHED_PORT_0"},
    {CounterEvent::UOPS_DISPATCHED_PORT_1, "UOPS_DISPATCHED_PORT_1"},
    {CounterEvent::UOPS_DISPATCHED_PORT_5, "UOPS_DISPATCHED_PORT_5"},
    {CounterEvent::UOPS_DISPATCHED_PORT_6, "UOPS_DISPATCHED_PORT_6"},

    // AVX-512 frequency events
    {CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE, "CORE_POWER_LVL0_TURBO_LICENSE"},
    {CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE, "CORE_POWER_LVL1_TURBO_LICENSE"},
    {CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE, "CORE_POWER_LVL2_TURBO_LICENSE"},

    // IMC events
    {CounterEvent::IMC_READS, "IMC_READS"},
    {CounterEvent::IMC_WRITES, "IMC_WRITES"},
    {CounterEvent::IMC_CAS_COUNT_RD, "IMC_CAS_COUNT_RD"},
    {CounterEvent::IMC_CAS_COUNT_WR, "IMC_CAS_COUNT_WR"},

    // Energy events
    {CounterEvent::RAPL_ENERGY_PKG, "RAPL_ENERGY_PKG"},
    {CounterEvent::RAPL_ENERGY_CORES, "RAPL_ENERGY_CORES"},
    {CounterEvent::RAPL_ENERGY_RAM, "RAPL_ENERGY_RAM"},
    {CounterEvent::RAPL_ENERGY_GPU, "RAPL_ENERGY_GPU"},
    {CounterEvent::AMD_RAPL_PKG, "AMD_RAPL_PKG"},
    {CounterEvent::AMD_RAPL_CORES, "AMD_RAPL_CORES"},

    // Register spill indicators
    {CounterEvent::MEM_INST_RETIRED_STLB_MISS_LOADS, "MEM_INST_RETIRED_STLB_MISS_LOADS"},
    {CounterEvent::MEM_LOAD_L1_HIT_RETIRED, "MEM_LOAD_L1_HIT_RETIRED"},

    // Horizontal operation indicators
    {CounterEvent::FP_ASSIST_ANY, "FP_ASSIST_ANY"},
    {CounterEvent::OTHER_ASSISTS_ANY, "OTHER_ASSISTS_ANY"},
};

// CPU vendor detection from CPUID or /proc/cpuinfo
static CPUVendor detect_cpu_vendor() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0) return CPUVendor::INTEL;
    if (strcmp(vendor, "AuthenticAMD") == 0) return CPUVendor::AMD;
    return CPUVendor::UNKNOWN;
#elif defined(__aarch64__)
    // Check for Apple Silicon via /proc/cpuinfo or system_profiler
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("Apple") != std::string::npos) {
                return CPUVendor::APPLE_SILICON;
            }
        }
    }
    return CPUVendor::ARM;
#elif defined(__riscv)
    return CPUVendor::RISCV;
#elif defined(__powerpc__) || defined(__powerpc64__)
    return CPUVendor::IBM_POWER;
#else
    return CPUVendor::UNKNOWN;
#endif
}

// Raw event code structure for perf_event
struct RawEventCode {
    uint64_t config;
    uint64_t config1;  // For offcore/uncore
    bool valid;
};

// Intel raw event codes (Skylake/ICL/SPR)
static RawEventCode get_intel_raw_event(CounterEvent event) {
    RawEventCode code = {0, 0, false};

    switch (event) {
        // FP_ARITH_INST_RETIRED.* events
        case CounterEvent::FP_ARITH_SCALAR_SINGLE:
            code = {0x01C7, 0, true};  // umask=0x01, event=0xC7
            break;
        case CounterEvent::FP_ARITH_SCALAR_DOUBLE:
            code = {0x02C7, 0, true};  // umask=0x02
            break;
        case CounterEvent::FP_ARITH_128B_PACKED_SINGLE:
            code = {0x04C7, 0, true};  // umask=0x04
            break;
        case CounterEvent::FP_ARITH_128B_PACKED_DOUBLE:
            code = {0x08C7, 0, true};  // umask=0x08
            break;
        case CounterEvent::FP_ARITH_256B_PACKED_SINGLE:
            code = {0x10C7, 0, true};  // umask=0x10
            break;
        case CounterEvent::FP_ARITH_256B_PACKED_DOUBLE:
            code = {0x20C7, 0, true};  // umask=0x20
            break;
        case CounterEvent::FP_ARITH_512B_PACKED_SINGLE:
            code = {0x40C7, 0, true};  // umask=0x40
            break;
        case CounterEvent::FP_ARITH_512B_PACKED_DOUBLE:
            code = {0x80C7, 0, true};  // umask=0x80
            break;

        // Cache line split events
        case CounterEvent::LD_BLOCKS_STORE_FORWARD:
            code = {0x0203, 0, true};  // LD_BLOCKS.STORE_FORWARD
            break;
        case CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS:
            code = {0x41D0, 0, true};  // MEM_INST_RETIRED.SPLIT_LOADS
            break;
        case CounterEvent::MEM_INST_RETIRED_SPLIT_STORES:
            code = {0x42D0, 0, true};  // MEM_INST_RETIRED.SPLIT_STORES
            break;
        case CounterEvent::MISALIGN_MEM_REF_LOADS:
            code = {0x0105, 0, true};  // MISALIGN_MEM_REF.LOADS
            break;
        case CounterEvent::MISALIGN_MEM_REF_STORES:
            code = {0x0205, 0, true};  // MISALIGN_MEM_REF.STORES
            break;

        // DSB events
        case CounterEvent::IDQ_DSB_UOPS:
            code = {0x0879, 0, true};  // IDQ.DSB_UOPS
            break;
        case CounterEvent::IDQ_MITE_UOPS:
            code = {0x0479, 0, true};  // IDQ.MITE_UOPS
            break;
        case CounterEvent::IDQ_MS_UOPS:
            code = {0x3079, 0, true};  // IDQ.MS_UOPS
            break;

        // Port utilization
        case CounterEvent::UOPS_DISPATCHED_PORT_0:
            code = {0x01A1, 0, true};  // UOPS_DISPATCHED_PORT.PORT_0
            break;
        case CounterEvent::UOPS_DISPATCHED_PORT_1:
            code = {0x02A1, 0, true};  // UOPS_DISPATCHED_PORT.PORT_1
            break;
        case CounterEvent::UOPS_DISPATCHED_PORT_5:
            code = {0x20A1, 0, true};  // UOPS_DISPATCHED_PORT.PORT_5
            break;
        case CounterEvent::UOPS_DISPATCHED_PORT_6:
            code = {0x40A1, 0, true};  // UOPS_DISPATCHED_PORT.PORT_6
            break;

        // AVX-512 frequency license levels
        case CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE:
            code = {0x0728, 0, true};  // CORE_POWER.LVL0_TURBO_LICENSE
            break;
        case CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE:
            code = {0x1828, 0, true};  // CORE_POWER.LVL1_TURBO_LICENSE
            break;
        case CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE:
            code = {0x2028, 0, true};  // CORE_POWER.LVL2_TURBO_LICENSE
            break;

        // TMA events
        case CounterEvent::UOPS_RETIRED_SLOTS:
            code = {0x02C2, 0, true};  // UOPS_RETIRED.SLOTS
            break;
        case CounterEvent::UOPS_ISSUED_ANY:
            code = {0x010E, 0, true};  // UOPS_ISSUED.ANY
            break;
        case CounterEvent::INT_MISC_RECOVERY_CYCLES:
            code = {0x010D, 0, true};  // INT_MISC.RECOVERY_CYCLES
            break;
        case CounterEvent::CYCLE_ACTIVITY_STALLS_MEM:
            code = {0x14A3, 0, true};  // CYCLE_ACTIVITY.STALLS_MEM_ANY
            break;
        case CounterEvent::CYCLE_ACTIVITY_STALLS_L1D:
            code = {0x0CA3, 0, true};  // CYCLE_ACTIVITY.STALLS_L1D_MISS
            break;
        case CounterEvent::CYCLE_ACTIVITY_STALLS_L2:
            code = {0x05A3, 0, true};  // CYCLE_ACTIVITY.STALLS_L2_MISS
            break;
        case CounterEvent::CYCLE_ACTIVITY_STALLS_L3:
            code = {0x06A3, 0, true};  // CYCLE_ACTIVITY.STALLS_L3_MISS
            break;

        // Memory events
        case CounterEvent::MEM_LOAD_RETIRED:
            code = {0x81D0, 0, true};  // MEM_INST_RETIRED.ALL_LOADS
            break;
        case CounterEvent::MEM_STORE_RETIRED:
            code = {0x82D0, 0, true};  // MEM_INST_RETIRED.ALL_STORES
            break;
        case CounterEvent::MEM_LOAD_L1_HIT_RETIRED:
            code = {0x01D1, 0, true};  // MEM_LOAD_RETIRED.L1_HIT
            break;

        // FP assist events
        case CounterEvent::FP_ASSIST_ANY:
            code = {0x1ECA, 0, true};  // FP_ASSIST.ANY
            break;
        case CounterEvent::OTHER_ASSISTS_ANY:
            code = {0x3FC1, 0, true};  // OTHER_ASSISTS.ANY
            break;

        default:
            break;
    }
    return code;
}

// AMD Zen raw event codes
static RawEventCode get_amd_raw_event(CounterEvent event) {
    RawEventCode code = {0, 0, false};

    switch (event) {
        // AMD FpRetSseAvxOps - Combined SSE/AVX operations
        // Event 0x03, different umasks for different widths
        case CounterEvent::AMD_FP_RET_SSE_AVX_OPS:
            code = {0xFF03, 0, true};  // All SSE/AVX ops
            break;
        case CounterEvent::FP_ARITH_SCALAR_SINGLE:
            code = {0x0103, 0, true};  // SSE scalar single
            break;
        case CounterEvent::FP_ARITH_SCALAR_DOUBLE:
            code = {0x0203, 0, true};  // SSE scalar double
            break;
        case CounterEvent::FP_ARITH_128B_PACKED_SINGLE:
            code = {0x0403, 0, true};  // 128-bit packed single
            break;
        case CounterEvent::FP_ARITH_128B_PACKED_DOUBLE:
            code = {0x0803, 0, true};  // 128-bit packed double
            break;
        case CounterEvent::FP_ARITH_256B_PACKED_SINGLE:
            code = {0x1003, 0, true};  // 256-bit packed single
            break;
        case CounterEvent::FP_ARITH_256B_PACKED_DOUBLE:
            code = {0x2003, 0, true};  // 256-bit packed double
            break;
        case CounterEvent::FP_ARITH_512B_PACKED_SINGLE:
            code = {0x4003, 0, true};  // 512-bit (dual-pump) single
            break;
        case CounterEvent::FP_ARITH_512B_PACKED_DOUBLE:
            code = {0x8003, 0, true};  // 512-bit (dual-pump) double
            break;

        // AMD x87 FP ops
        case CounterEvent::AMD_FP_RET_X87_OPS:
            code = {0x0002, 0, true};  // x87 FP operations
            break;

        // AMD FP pipe dispatch (for dual-pump detection)
        case CounterEvent::AMD_FP_DISP_PIPE_ASSIGN:
            code = {0x0100, 0, true};  // FP scheduler pipe assignments
            break;

        // Cache events for AMD
        case CounterEvent::L1D_READ_MISS:
            code = {0x0164, 0, true};  // L1 DC miss
            break;
        case CounterEvent::L2_READ_MISS:
            code = {0x0364, 0, true};  // L2 cache miss
            break;
        case CounterEvent::L3_READ_MISS:
            code = {0x0664, 0, true};  // L3 cache miss
            break;

        default:
            break;
    }
    return code;
}

// ARM PMU event codes (Cortex-A/Neoverse)
static RawEventCode get_arm_raw_event(CounterEvent event) {
    RawEventCode code = {0, 0, false};

    switch (event) {
        // ARM ASIMD events
        case CounterEvent::ARM_ASE_SPEC:
            code = {0x74, 0, true};  // ASE_SPEC: ASIMD speculation
            break;
        case CounterEvent::ARM_SVE_INST_SPEC:
            code = {0x8006, 0, true};  // SVE_INST_SPEC
            break;
        case CounterEvent::ARM_FP_SPEC:
            code = {0x73, 0, true};  // FP_SPEC: FP speculation
            break;
        case CounterEvent::ARM_VFP_SPEC:
            code = {0x75, 0, true};  // VFP_SPEC
            break;

        // SVE predicate events
        case CounterEvent::ARM_SVE_PRED_PARTIAL:
            code = {0x8007, 0, true};  // SVE_PRED_PARTIAL_MATCH
            break;
        case CounterEvent::ARM_SVE_PRED_FULL:
            code = {0x8008, 0, true};  // SVE_PRED_FULL_MATCH
            break;
        case CounterEvent::ARM_SVE_PRED_EMPTY:
            code = {0x8009, 0, true};  // SVE_PRED_EMPTY_MATCH
            break;

        // ARM cache events
        case CounterEvent::L1D_READ_ACCESS:
            code = {0x04, 0, true};  // L1D_CACHE_RD
            break;
        case CounterEvent::L1D_READ_MISS:
            code = {0x03, 0, true};  // L1D_CACHE_REFILL
            break;
        case CounterEvent::L2_READ_ACCESS:
            code = {0x16, 0, true};  // L2D_CACHE_RD
            break;
        case CounterEvent::L2_READ_MISS:
            code = {0x17, 0, true};  // L2D_CACHE_REFILL
            break;

        default:
            break;
    }
    return code;
}

std::string counter_event_to_string(CounterEvent event) {
    auto it = event_names.find(event);
    if (it != event_names.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

CounterEvent string_to_counter_event(const std::string& name) {
    for (const auto& [event, event_name] : event_names) {
        if (event_name == name) {
            return event;
        }
    }
    return CounterEvent::CYCLES;  // Default
}

std::vector<CounterEvent> get_flops_events() {
    return {
        CounterEvent::FP_ARITH_SCALAR_SINGLE,
        CounterEvent::FP_ARITH_128B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_256B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_512B_PACKED_SINGLE,
    };
}

std::vector<CounterEvent> get_cache_events() {
    return {
        CounterEvent::L1D_READ_ACCESS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L2_READ_ACCESS,
        CounterEvent::L2_READ_MISS,
        CounterEvent::L3_READ_ACCESS,
        CounterEvent::L3_READ_MISS,
    };
}

std::vector<CounterEvent> get_memory_events() {
    return {
        CounterEvent::MEM_LOAD_RETIRED,
        CounterEvent::MEM_STORE_RETIRED,
    };
}

std::vector<CounterEvent> get_tma_events() {
    return {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::UOPS_RETIRED_SLOTS,
        CounterEvent::UOPS_ISSUED_ANY,
        CounterEvent::INT_MISC_RECOVERY_CYCLES,
        CounterEvent::CYCLE_ACTIVITY_STALLS_MEM,
        CounterEvent::CYCLE_ACTIVITY_STALLS_L1D,
        CounterEvent::CYCLE_ACTIVITY_STALLS_L2,
        CounterEvent::CYCLE_ACTIVITY_STALLS_L3,
    };
}

// Factory implementation
std::unique_ptr<IPerformanceCounters> PerformanceCounterFactory::create(CounterBackend backend) {
    switch (backend) {
#ifdef __linux__
        case CounterBackend::PERF_EVENT:
            return std::make_unique<PerfEventCounters>();
#endif
#ifdef SIMD_BENCH_HAS_PAPI
        case CounterBackend::PAPI:
            return std::make_unique<PAPICounters>();
#endif
#ifdef SIMD_BENCH_HAS_LIKWID
        case CounterBackend::LIKWID:
            return std::make_unique<LIKWIDCounters>();
#endif
        case CounterBackend::NONE:
        default:
            return std::make_unique<NullCounters>();
    }
}

std::unique_ptr<IPerformanceCounters> PerformanceCounterFactory::create_best_available() {
    auto backends = get_available_backends();

#ifdef SIMD_BENCH_HAS_LIKWID
    if (std::find(backends.begin(), backends.end(), CounterBackend::LIKWID) != backends.end()) {
        auto counters = create(CounterBackend::LIKWID);
        if (counters->initialize()) {
            return counters;
        }
    }
#endif

#ifdef SIMD_BENCH_HAS_PAPI
    if (std::find(backends.begin(), backends.end(), CounterBackend::PAPI) != backends.end()) {
        auto counters = create(CounterBackend::PAPI);
        if (counters->initialize()) {
            return counters;
        }
    }
#endif

#ifdef __linux__
    auto counters = create(CounterBackend::PERF_EVENT);
    if (counters->initialize()) {
        return counters;
    }
#endif

    return std::make_unique<NullCounters>();
}

std::vector<CounterBackend> PerformanceCounterFactory::get_available_backends() {
    std::vector<CounterBackend> backends;
    backends.push_back(CounterBackend::NONE);

#ifdef __linux__
    backends.push_back(CounterBackend::PERF_EVENT);
#endif

#ifdef SIMD_BENCH_HAS_PAPI
    backends.push_back(CounterBackend::PAPI);
#endif

#ifdef SIMD_BENCH_HAS_LIKWID
    backends.push_back(CounterBackend::LIKWID);
#endif

    return backends;
}

bool PerformanceCounterFactory::is_backend_available(CounterBackend backend) {
    auto backends = get_available_backends();
    return std::find(backends.begin(), backends.end(), backend) != backends.end();
}

// ScopedCounters implementation
ScopedCounters::ScopedCounters(IPerformanceCounters& counters, CounterValues& result)
    : counters_(counters), result_(result) {
    counters_.start();
}

ScopedCounters::~ScopedCounters() {
    counters_.stop();
    result_ = counters_.read();
}

// PerfEventCounters implementation
#ifdef __linux__
struct PerfEventCounters::Impl {
    std::vector<int> fds;
    std::vector<CounterEvent> events;
    bool running = false;
};

PerfEventCounters::PerfEventCounters() : impl_(std::make_unique<Impl>()) {}
PerfEventCounters::~PerfEventCounters() { shutdown(); }

bool PerfEventCounters::initialize() {
    return true;
}

void PerfEventCounters::shutdown() {
    for (int fd : impl_->fds) {
        if (fd >= 0) close(fd);
    }
    impl_->fds.clear();
    impl_->events.clear();
}

static long perf_event_open(struct perf_event_attr* hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

bool PerfEventCounters::add_event(CounterEvent event) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    bool use_raw = false;
    RawEventCode raw_code = {0, 0, false};

    switch (event) {
        case CounterEvent::CYCLES:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CPU_CYCLES;
            break;
        case CounterEvent::INSTRUCTIONS:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_INSTRUCTIONS;
            break;
        case CounterEvent::CACHE_REFERENCES:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
            break;
        case CounterEvent::CACHE_MISSES:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CACHE_MISSES;
            break;
        case CounterEvent::BRANCH_INSTRUCTIONS:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
            break;
        case CounterEvent::BRANCH_MISSES:
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_BRANCH_MISSES;
            break;
        default:
            // Use architecture-specific raw event codes
            use_raw = true;
            break;
    }

    if (use_raw) {
        // Detect CPU vendor and get appropriate raw event code
        CPUVendor vendor = detect_cpu_vendor();

        switch (vendor) {
            case CPUVendor::INTEL:
                raw_code = get_intel_raw_event(event);
                break;
            case CPUVendor::AMD:
                raw_code = get_amd_raw_event(event);
                break;
            case CPUVendor::ARM:
            case CPUVendor::APPLE_SILICON:
                raw_code = get_arm_raw_event(event);
                break;
            default:
                // Try Intel codes as fallback for x86
#if defined(__x86_64__) || defined(_M_X64)
                raw_code = get_intel_raw_event(event);
#endif
                break;
        }

        if (!raw_code.valid) {
            return false;  // Event not supported on this architecture
        }

        pe.type = PERF_TYPE_RAW;
        pe.config = raw_code.config;
        if (raw_code.config1 != 0) {
            pe.config1 = raw_code.config1;
        }
    }

    int fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd < 0) {
        return false;
    }

    impl_->fds.push_back(fd);
    impl_->events.push_back(event);
    return true;
}

void PerfEventCounters::clear_events() {
    shutdown();
}

bool PerfEventCounters::start() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    impl_->running = true;
    return true;
}

bool PerfEventCounters::stop() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    }
    impl_->running = false;
    return true;
}

bool PerfEventCounters::reset() {
    for (int fd : impl_->fds) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    }
    return true;
}

CounterValues PerfEventCounters::read() {
    CounterValues values;
    for (size_t i = 0; i < impl_->fds.size(); ++i) {
        uint64_t count;
        if (::read(impl_->fds[i], &count, sizeof(count)) == sizeof(count)) {
            values.set(impl_->events[i], count);
        }
    }
    return values;
}

bool PerfEventCounters::is_event_supported(CounterEvent event) const {
    // Basic hardware events are always supported
    switch (event) {
        case CounterEvent::CYCLES:
        case CounterEvent::INSTRUCTIONS:
        case CounterEvent::CACHE_REFERENCES:
        case CounterEvent::CACHE_MISSES:
        case CounterEvent::BRANCH_INSTRUCTIONS:
        case CounterEvent::BRANCH_MISSES:
            return true;
        default:
            break;
    }

    // Check architecture-specific events
    CPUVendor vendor = detect_cpu_vendor();
    RawEventCode code = {0, 0, false};

    switch (vendor) {
        case CPUVendor::INTEL:
            code = get_intel_raw_event(event);
            break;
        case CPUVendor::AMD:
            code = get_amd_raw_event(event);
            break;
        case CPUVendor::ARM:
        case CPUVendor::APPLE_SILICON:
            code = get_arm_raw_event(event);
            break;
        default:
#if defined(__x86_64__) || defined(_M_X64)
            code = get_intel_raw_event(event);
#endif
            break;
    }

    return code.valid;
}
#endif

// PAPI implementation
#ifdef SIMD_BENCH_HAS_PAPI
struct PAPICounters::Impl {
    int event_set = PAPI_NULL;
    std::vector<int> events;
    std::vector<CounterEvent> event_types;
    bool initialized = false;
};

PAPICounters::PAPICounters() : impl_(std::make_unique<Impl>()) {}
PAPICounters::~PAPICounters() { shutdown(); }

bool PAPICounters::initialize() {
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        return false;
    }

    if (PAPI_create_eventset(&impl_->event_set) != PAPI_OK) {
        return false;
    }

    impl_->initialized = true;
    return true;
}

void PAPICounters::shutdown() {
    if (impl_->initialized) {
        PAPI_cleanup_eventset(impl_->event_set);
        PAPI_destroy_eventset(&impl_->event_set);
        impl_->initialized = false;
    }
}

bool PAPICounters::add_event(CounterEvent event) {
    int papi_event = PAPI_TOT_CYC;

    switch (event) {
        case CounterEvent::CYCLES:
            papi_event = PAPI_TOT_CYC;
            break;
        case CounterEvent::INSTRUCTIONS:
            papi_event = PAPI_TOT_INS;
            break;
        case CounterEvent::L1D_READ_MISS:
            papi_event = PAPI_L1_DCM;
            break;
        case CounterEvent::L2_READ_MISS:
            papi_event = PAPI_L2_DCM;
            break;
        case CounterEvent::BRANCH_MISSES:
            papi_event = PAPI_BR_MSP;
            break;
        default:
            return false;
    }

    if (PAPI_add_event(impl_->event_set, papi_event) != PAPI_OK) {
        return false;
    }

    impl_->events.push_back(papi_event);
    impl_->event_types.push_back(event);
    return true;
}

void PAPICounters::clear_events() {
    PAPI_cleanup_eventset(impl_->event_set);
    impl_->events.clear();
    impl_->event_types.clear();
}

bool PAPICounters::start() {
    return PAPI_start(impl_->event_set) == PAPI_OK;
}

bool PAPICounters::stop() {
    std::vector<long long> values(impl_->events.size());
    return PAPI_stop(impl_->event_set, values.data()) == PAPI_OK;
}

bool PAPICounters::reset() {
    return PAPI_reset(impl_->event_set) == PAPI_OK;
}

CounterValues PAPICounters::read() {
    CounterValues result;
    std::vector<long long> values(impl_->events.size());

    if (PAPI_read(impl_->event_set, values.data()) == PAPI_OK) {
        for (size_t i = 0; i < impl_->event_types.size(); ++i) {
            result.set(impl_->event_types[i], static_cast<uint64_t>(values[i]));
        }
    }

    return result;
}

bool PAPICounters::is_event_supported(CounterEvent event) const {
    switch (event) {
        case CounterEvent::CYCLES:
        case CounterEvent::INSTRUCTIONS:
        case CounterEvent::L1D_READ_MISS:
        case CounterEvent::L2_READ_MISS:
        case CounterEvent::BRANCH_MISSES:
            return true;
        default:
            return false;
    }
}
#endif

// LIKWID implementation
#ifdef SIMD_BENCH_HAS_LIKWID
struct LIKWIDCounters::Impl {
    int gid = -1;
    std::string current_group;
    bool initialized = false;
};

LIKWIDCounters::LIKWIDCounters() : impl_(std::make_unique<Impl>()) {}
LIKWIDCounters::~LIKWIDCounters() { shutdown(); }

bool LIKWIDCounters::initialize() {
    // likwid_markerInit() returns void, so we just call it and assume success
    // It internally sets up the marker API
    likwid_markerInit();
    impl_->initialized = true;
    return true;
}

void LIKWIDCounters::shutdown() {
    if (impl_->initialized) {
        likwid_markerClose();
        impl_->initialized = false;
    }
}

bool LIKWIDCounters::add_event(CounterEvent) {
    // LIKWID uses performance groups, not individual events
    return true;
}

void LIKWIDCounters::clear_events() {
    impl_->current_group.clear();
}

bool LIKWIDCounters::start() {
    likwid_markerStartRegion("benchmark");
    return true;
}

bool LIKWIDCounters::stop() {
    likwid_markerStopRegion("benchmark");
    return true;
}

bool LIKWIDCounters::reset() {
    return true;
}

CounterValues LIKWIDCounters::read() {
    // LIKWID results are read via likwid-perfctr wrapper
    return CounterValues{};
}

bool LIKWIDCounters::is_event_supported(CounterEvent) const {
    return true;  // LIKWID supports most events through groups
}

bool LIKWIDCounters::set_performance_group(const std::string& group) {
    impl_->current_group = group;
    return true;
}

std::vector<std::string> LIKWIDCounters::get_available_groups() const {
    return {"FLOPS_SP", "FLOPS_DP", "FLOPS_AVX", "L2CACHE", "L3CACHE", "MEM", "ENERGY"};
}
#endif

}  // namespace simd_bench
