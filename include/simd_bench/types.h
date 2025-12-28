#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <chrono>

namespace simd_bench {

// Forward declarations
class Kernel;
class BenchmarkResult;
class HardwareInfo;

// Basic types
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::duration<double>;
using Nanoseconds = std::chrono::nanoseconds;

// Performance metrics
struct PerformanceMetrics {
    double gflops = 0.0;           // Giga floating-point ops per second
    double gops = 0.0;             // Giga integer ops per second
    double throughput = 0.0;       // Elements per second
    double latency_ns = 0.0;       // Average latency in nanoseconds
    double ipc = 0.0;              // Instructions per cycle
    double cpi = 0.0;              // Cycles per instruction
    uint64_t cycles = 0;           // Total cycles
    uint64_t instructions = 0;     // Total instructions
    double elapsed_seconds = 0.0;  // Wall-clock time
};

// SIMD-specific metrics
struct SIMDMetrics {
    double vectorization_ratio = 0.0;     // packed_ops / (packed_ops + scalar_ops)
    double vector_capacity_usage = 0.0;   // actual_width / max_width
    double simd_efficiency = 0.0;         // theoretical vs actual speedup
    double fma_utilization = 0.0;         // FMA vs separate MUL+ADD
    double lane_utilization = 0.0;        // active lanes / total lanes

    uint64_t scalar_ops = 0;
    uint64_t packed_128_ops = 0;
    uint64_t packed_256_ops = 0;
    uint64_t packed_512_ops = 0;
    uint64_t fma_ops = 0;
    uint64_t mul_ops = 0;
    uint64_t add_ops = 0;
};

// Memory metrics
struct MemoryMetrics {
    double memory_bandwidth_gbps = 0.0;   // GB/s
    double arithmetic_intensity = 0.0;    // FLOP/byte
    double l1_hit_rate = 0.0;             // L1 cache efficiency
    double l2_hit_rate = 0.0;             // L2 cache efficiency
    double l3_hit_rate = 0.0;             // L3 cache efficiency
    double llc_miss_rate = 0.0;           // Last-level cache miss rate

    uint64_t l1_hits = 0;
    uint64_t l1_misses = 0;
    uint64_t l2_hits = 0;
    uint64_t l2_misses = 0;
    uint64_t l3_hits = 0;
    uint64_t l3_misses = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
};

// Top-down microarchitecture analysis metrics
struct TMAMetrics {
    // Level 1
    double retiring = 0.0;           // Useful work (0-1)
    double bad_speculation = 0.0;    // Wasted due to misprediction
    double frontend_bound = 0.0;     // Stalled on instruction delivery
    double backend_bound = 0.0;      // Stalled on execution

    // Level 2 - Retiring breakdown
    double retiring_base = 0.0;
    double retiring_vectorized = 0.0;

    // Level 2 - Backend breakdown
    double core_bound = 0.0;
    double memory_bound = 0.0;

    // Level 3 - Memory bound breakdown
    double l1_bound = 0.0;
    double l2_bound = 0.0;
    double l3_bound = 0.0;
    double dram_bound = 0.0;
};

// Energy metrics
struct EnergyMetrics {
    double package_power_watts = 0.0;     // Total CPU power
    double core_power_watts = 0.0;        // Core-only power
    double dram_power_watts = 0.0;        // Memory power
    double energy_joules = 0.0;           // Total energy consumed
    double energy_per_op_nj = 0.0;        // nJ per FLOP
    double energy_delay_product = 0.0;    // J * s
};

// Correctness metrics
struct CorrectnessMetrics {
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double max_ulp_error = 0.0;
    double mean_absolute_error = 0.0;
    double mean_relative_error = 0.0;
    int nan_count = 0;
    int inf_count = 0;
    bool passed = true;
    std::string failure_reason;
};

// Combined metrics for a kernel run
struct KernelMetrics {
    PerformanceMetrics performance;
    SIMDMetrics simd;
    MemoryMetrics memory;
    TMAMetrics tma;
    EnergyMetrics energy;
    CorrectnessMetrics correctness;

    // Metadata
    std::string kernel_name;
    std::string variant_name;
    size_t problem_size = 0;
    size_t iterations = 0;
};

// Roofline model point
struct RooflinePoint {
    double arithmetic_intensity;  // FLOP/byte
    double achieved_gflops;
    std::string bound;           // "compute", "l1", "l2", "l3", "memory"
    double efficiency;           // achieved / theoretical max
};

// Hardware performance counter event types
enum class CounterEvent {
    // Basic events
    CYCLES,
    INSTRUCTIONS,
    CACHE_REFERENCES,
    CACHE_MISSES,
    BRANCH_INSTRUCTIONS,
    BRANCH_MISSES,

    // Intel Floating-point events (FP_ARITH_INST_RETIRED.*)
    FP_ARITH_SCALAR_SINGLE,
    FP_ARITH_SCALAR_DOUBLE,
    FP_ARITH_128B_PACKED_SINGLE,
    FP_ARITH_128B_PACKED_DOUBLE,
    FP_ARITH_256B_PACKED_SINGLE,
    FP_ARITH_256B_PACKED_DOUBLE,
    FP_ARITH_512B_PACKED_SINGLE,
    FP_ARITH_512B_PACKED_DOUBLE,

    // AMD Floating-point events (Zen architecture)
    AMD_FP_RET_SSE_AVX_OPS,           // Combined SSE/AVX operations
    AMD_FP_RET_X87_OPS,               // x87 FP operations
    AMD_FP_DISP_PIPE_ASSIGN,          // FP pipe assignment (dual-pump detection)
    AMD_FP_SCH_NO_LOW_RES,            // Scheduler: no low resource

    // ARM Floating-point/SIMD events
    ARM_ASE_SPEC,                     // ASIMD speculation count
    ARM_SVE_INST_SPEC,                // SVE instruction speculation
    ARM_SVE_PRED_PARTIAL,             // SVE partial predicate operations
    ARM_SVE_PRED_FULL,                // SVE full predicate operations
    ARM_SVE_PRED_EMPTY,               // SVE empty predicate operations
    ARM_FP_SPEC,                      // FP operations speculated
    ARM_VFP_SPEC,                     // VFP operations speculated

    // RISC-V Vector events
    RVV_VECTOR_INST,                  // Vector instructions executed
    RVV_VECTOR_ELEMENTS,              // Vector elements processed
    RVV_VL_ACTIVE,                    // Vector length utilization

    // IBM Power VSX events
    POWER_VSX_EXEC,                   // VSX instructions executed
    POWER_VSX_SINGLE,                 // VSX single-precision
    POWER_VSX_DOUBLE,                 // VSX double-precision

    // Cache events
    L1D_READ_ACCESS,
    L1D_READ_MISS,
    L1D_WRITE_ACCESS,
    L1D_WRITE_MISS,
    L2_READ_ACCESS,
    L2_READ_MISS,
    L3_READ_ACCESS,
    L3_READ_MISS,

    // Cache line split events (Intel)
    LD_BLOCKS_STORE_FORWARD,          // Loads blocked by store forwarding
    MEM_INST_RETIRED_SPLIT_LOADS,     // Split load operations
    MEM_INST_RETIRED_SPLIT_STORES,    // Split store operations
    MISALIGN_MEM_REF_LOADS,           // Misaligned load references
    MISALIGN_MEM_REF_STORES,          // Misaligned store references

    // Memory events
    MEM_LOAD_RETIRED,
    MEM_STORE_RETIRED,
    OFFCORE_REQUESTS_DEMAND_DATA_RD,  // Off-core data reads
    OFFCORE_REQUESTS_DEMAND_RFO,      // Read-for-ownership requests

    // TMA events (Intel Top-Down)
    UOPS_RETIRED_SLOTS,
    UOPS_ISSUED_ANY,
    INT_MISC_RECOVERY_CYCLES,
    CYCLE_ACTIVITY_STALLS_MEM,
    CYCLE_ACTIVITY_STALLS_L1D,
    CYCLE_ACTIVITY_STALLS_L2,
    CYCLE_ACTIVITY_STALLS_L3,

    // DSB (Decoded Stream Buffer) events
    IDQ_DSB_UOPS,                     // µops from DSB
    IDQ_MITE_UOPS,                    // µops from MITE
    IDQ_MS_UOPS,                      // µops from MS

    // Port utilization events (Intel)
    UOPS_DISPATCHED_PORT_0,           // Port 0 (ALU/FMA)
    UOPS_DISPATCHED_PORT_1,           // Port 1 (ALU/FMA)
    UOPS_DISPATCHED_PORT_5,           // Port 5 (shuffle/permute)
    UOPS_DISPATCHED_PORT_6,           // Port 6 (branch)

    // AVX-512 frequency events (Intel)
    CORE_POWER_LVL0_TURBO_LICENSE,    // Light AVX-512 (L0)
    CORE_POWER_LVL1_TURBO_LICENSE,    // Medium AVX-512 (L1)
    CORE_POWER_LVL2_TURBO_LICENSE,    // Heavy AVX-512 (L2)

    // IMC (Uncore) events for memory bandwidth
    IMC_READS,                        // Memory controller reads
    IMC_WRITES,                       // Memory controller writes
    IMC_CAS_COUNT_RD,                 // CAS read count
    IMC_CAS_COUNT_WR,                 // CAS write count

    // Energy events
    RAPL_ENERGY_PKG,
    RAPL_ENERGY_CORES,
    RAPL_ENERGY_RAM,
    RAPL_ENERGY_GPU,                  // Integrated GPU energy (if present)

    // AMD RAPL energy
    AMD_RAPL_PKG,                     // AMD package energy
    AMD_RAPL_CORES,                   // AMD core energy

    // Register spill indicators
    MEM_INST_RETIRED_STLB_MISS_LOADS, // STLB miss loads (can indicate spilling)
    MEM_LOAD_L1_HIT_RETIRED,          // L1 hits (for spill detection pattern)

    // Horizontal operation indicators
    FP_ASSIST_ANY,                    // FP assists (often from horizontal ops)
    OTHER_ASSISTS_ANY                 // Other microcode assists
};

// CPU vendor detection
enum class CPUVendor {
    UNKNOWN,
    INTEL,
    AMD,
    ARM,
    APPLE_SILICON,
    RISCV,
    IBM_POWER
};

// MPKI (Misses Per Kilo-Instructions) thresholds
struct MPKIThresholds {
    static constexpr double L1_ACCEPTABLE = 5.0;   // L1 MPKI < 5 is acceptable
    static constexpr double L2_ACCEPTABLE = 2.0;   // L2 MPKI < 2 is acceptable
    static constexpr double L3_ACCEPTABLE = 1.0;   // L3 MPKI < 1 is acceptable
    static constexpr double L1_CRITICAL = 20.0;    // L1 MPKI > 20 is critical
    static constexpr double L2_CRITICAL = 10.0;    // L2 MPKI > 10 is critical
    static constexpr double L3_CRITICAL = 5.0;     // L3 MPKI > 5 is critical
};

// Quality thresholds for derived metrics
struct QualityThresholds {
    // Performance health indicators
    static constexpr double IPC_HEALTHY = 2.0;              // IPC > 2.0 is healthy
    static constexpr double IPC_GOOD = 3.0;                 // IPC > 3.0 is good
    static constexpr double VECTORIZATION_ACCEPTABLE = 0.9; // > 90% vectorized
    static constexpr double VECTORIZATION_GOOD = 0.95;      // > 95% vectorized
    static constexpr double VECTOR_WIDTH_GOOD = 0.8;        // > 80% width used
    static constexpr double VECTOR_WIDTH_OPTIMAL = 0.9;     // > 90% width used

    // TMA thresholds
    static constexpr double RETIRING_HEALTHY = 0.4;         // > 40% retiring
    static constexpr double RETIRING_GOOD = 0.6;            // > 60% retiring
    static constexpr double BACKEND_BOUND_ACCEPTABLE = 0.3; // < 30% backend bound
    static constexpr double FRONTEND_BOUND_GOOD = 0.1;      // < 10% frontend bound
    static constexpr double BAD_SPECULATION_GOOD = 0.05;    // < 5% bad speculation

    // Cache thresholds
    static constexpr double DSB_COVERAGE_ACCEPTABLE = 0.7;  // > 70% DSB coverage
    static constexpr double DSB_COVERAGE_GOOD = 0.85;       // > 85% DSB coverage
    static constexpr double CACHE_HIT_RATE_GOOD = 0.95;     // > 95% cache hit rate

    // Port balance
    static constexpr double PORT_SATURATION_WARNING = 0.8;  // > 80% port usage
    static constexpr double PORT_SATURATION_CRITICAL = 0.95;// > 95% port usage
};

// Extended SIMD metrics with quality ratings
struct ExtendedSIMDMetrics {
    // Basic metrics
    double vectorization_ratio = 0.0;
    double vector_width_utilization = 0.0;
    double fma_utilization = 0.0;
    double lane_utilization = 0.0;

    // Architecture-specific
    double avx512_license_level = 0.0;    // 0=L0, 1=L1, 2=L2 (Intel)
    double amd_dual_pump_ratio = 0.0;     // 512-bit dual-pump ratio (AMD)
    double sve_predicate_efficiency = 0.0; // SVE predicate utilization (ARM)

    // Quality ratings (0.0 = poor, 1.0 = excellent)
    double quality_score = 0.0;
    std::string quality_rating;           // "Poor", "Acceptable", "Good", "Excellent"
    std::vector<std::string> issues;      // Detected issues
    std::vector<std::string> suggestions; // Optimization suggestions
};

// MPKI metrics with evaluation
struct MPKIMetrics {
    double l1_mpki = 0.0;
    double l2_mpki = 0.0;
    double l3_mpki = 0.0;

    // Evaluation results
    bool l1_acceptable = true;
    bool l2_acceptable = true;
    bool l3_acceptable = true;
    bool has_cache_issues = false;
    std::string evaluation;
};

// DSB (Decoded Stream Buffer) metrics
struct DSBMetrics {
    double dsb_coverage = 0.0;      // Fraction from DSB vs MITE
    double mite_coverage = 0.0;     // Fraction from MITE (legacy decode)
    double ms_coverage = 0.0;       // Fraction from microcode sequencer
    uint64_t dsb_uops = 0;
    uint64_t mite_uops = 0;
    uint64_t ms_uops = 0;
    bool is_dsb_efficient = true;
    std::string recommendation;
};

// Port saturation metrics
struct PortMetrics {
    double port0_utilization = 0.0;  // FMA/ALU
    double port1_utilization = 0.0;  // FMA/ALU
    double port5_utilization = 0.0;  // Shuffle/permute (critical for SIMD)
    double port6_utilization = 0.0;  // Branch

    bool port5_saturated = false;    // Shuffle port bottleneck
    bool fma_ports_balanced = true;  // FMA port balance
    std::string bottleneck;
};

// Cache line split metrics
struct CacheLineSplitMetrics {
    uint64_t split_loads = 0;
    uint64_t split_stores = 0;
    uint64_t total_loads = 0;
    uint64_t total_stores = 0;
    double split_load_ratio = 0.0;
    double split_store_ratio = 0.0;
    bool has_alignment_issues = false;
    std::string recommendation;
};

// AVX-512 frequency transition metrics
struct AVX512FrequencyMetrics {
    uint64_t l0_cycles = 0;   // Light (minimal downclocking)
    uint64_t l1_cycles = 0;   // Medium downclocking
    uint64_t l2_cycles = 0;   // Heavy downclocking
    double avg_license_level = 0.0;
    double frequency_penalty_estimate = 0.0;  // Estimated % slowdown
    bool has_frequency_penalty = false;
    std::string recommendation;
};

// IMC (Memory Controller) bandwidth metrics
struct IMCBandwidthMetrics {
    double read_bandwidth_gbps = 0.0;
    double write_bandwidth_gbps = 0.0;
    double total_bandwidth_gbps = 0.0;
    double bandwidth_utilization = 0.0;  // vs theoretical max
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
};

// Kernel function signature
using KernelFunction = std::function<void(void* data, size_t size, size_t iterations)>;
using SetupFunction = std::function<void*(size_t size)>;
using TeardownFunction = std::function<void(void* data)>;
using VerifyFunction = std::function<bool(const void* result, const void* reference, size_t size)>;

// Kernel variant definition
struct KernelVariant {
    std::string name;
    KernelFunction func;
    std::string isa;  // "scalar", "sse", "avx2", "avx512", "neon", "sve"
    bool is_reference = false;
};

// Kernel configuration
struct KernelConfig {
    std::string name;
    std::string description;
    std::string category;

    double arithmetic_intensity = 0.0;  // FLOP/byte (theoretical)
    size_t flops_per_element = 0;
    size_t bytes_per_element = 0;

    std::vector<KernelVariant> variants;
    std::vector<size_t> sizes;
    size_t default_iterations = 1000;

    SetupFunction setup;
    TeardownFunction teardown;
    VerifyFunction verify;
};

// Benchmark configuration
struct BenchmarkConfig {
    size_t warmup_iterations = 5;
    size_t benchmark_iterations = 100;
    bool enable_hardware_counters = true;
    bool enable_energy_profiling = true;
    bool enable_correctness_check = true;
    bool enable_roofline = true;
    bool enable_tma = true;

    std::vector<CounterEvent> counter_events;
    std::vector<size_t> sizes_override;

    double regression_threshold = 0.05;  // 5%
    std::string output_format = "json";  // "json", "html", "markdown"
    std::string output_path;
};

// Result for a single kernel variant at a specific size
struct VariantResult {
    std::string variant_name;
    size_t problem_size;
    KernelMetrics metrics;
    RooflinePoint roofline;
    std::vector<std::string> recommendations;
};

// Complete benchmark result
struct BenchmarkResult {
    std::string kernel_name;
    std::vector<VariantResult> results;
    std::string best_variant;
    double speedup_vs_scalar = 0.0;

    // Aggregated stats
    double avg_vectorization_ratio = 0.0;
    double avg_efficiency = 0.0;
};

// Report types
enum class ReportFormat {
    JSON,
    HTML,
    MARKDOWN,
    CSV
};

}  // namespace simd_bench
