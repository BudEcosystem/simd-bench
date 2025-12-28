#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>
#include <cstdint>

namespace simd_bench {

// Prefetch effectiveness metrics
struct PrefetchMetrics {
    // Software prefetch counts
    uint64_t sw_prefetch_issued = 0;      // Total SW prefetches issued
    uint64_t sw_prefetch_l1 = 0;          // Prefetch to L1
    uint64_t sw_prefetch_l2 = 0;          // Prefetch to L2
    uint64_t sw_prefetch_nta = 0;         // Non-temporal prefetch

    // Hardware prefetch effectiveness
    uint64_t hw_prefetch_useful = 0;      // HW prefetch that was used
    uint64_t hw_prefetch_useless = 0;     // HW prefetch evicted before use

    // Derived metrics
    double prefetch_coverage = 0.0;       // Fraction of DRAM accesses covered by prefetch
    double prefetch_accuracy = 0.0;       // Useful / (useful + useless)
    double prefetch_timeliness = 0.0;     // How early prefetch arrives (0-1)

    // Recommendations
    int recommended_prefetch_distance = 0;  // Cache lines ahead to prefetch
    std::string recommendation;
    std::vector<std::string> suggestions;
};

// Prefetch analyzer for measuring prefetch effectiveness
class PrefetchAnalyzer {
public:
    // Analyze prefetch behavior from counter values
    static PrefetchMetrics analyze(
        uint64_t l2_lines_in_all,          // L2_LINES_IN.ALL
        uint64_t l2_lines_out_useless_hwpf,// L2_LINES_OUT.USELESS_HWPF
        uint64_t sw_prefetch_access,       // SW_PREFETCH_ACCESS.*
        uint64_t dram_requests,            // Total DRAM requests
        uint64_t total_instructions        // For normalization
    );

    // Calculate optimal prefetch distance for a given access pattern
    static int calculate_prefetch_distance(
        double memory_latency_ns,          // DRAM latency
        double loop_iteration_ns,          // Time per iteration
        size_t stride_bytes                // Access stride
    );

    // Calculate optimal prefetch distance with cache hierarchy
    static int calculate_prefetch_distance_detailed(
        double l3_latency_ns,
        double dram_latency_ns,
        double loop_iteration_ns,
        size_t stride_bytes,
        size_t l2_size_kb,
        size_t cache_line_bytes = 64
    );

    // Generate prefetch code suggestion
    static std::string generate_prefetch_code(
        int prefetch_distance,
        size_t stride_bytes,
        const std::string& pointer_name = "ptr",
        bool use_nta = false
    );

    // Analyze if hardware prefetch is sufficient
    struct HardwarePrefetchAnalysis {
        bool hw_prefetch_sufficient = false;
        double hw_prefetch_efficiency = 0.0;
        bool needs_sw_prefetch = false;
        std::string reason;
        std::vector<std::string> recommendations;
    };

    static HardwarePrefetchAnalysis analyze_hw_prefetch(
        double l1_miss_rate,
        double l2_miss_rate,
        double l3_miss_rate,
        size_t stride_bytes,
        size_t working_set_bytes
    );
};

// Prefetch pattern detector
struct PrefetchPattern {
    enum class Type {
        SEQUENTIAL,       // Unit stride, HW prefetch effective
        STRIDED,          // Regular stride, may need SW prefetch
        INDIRECT,         // Pointer chasing, prefetch limited
        RANDOM,           // No pattern, prefetch useless
        STREAMING         // Large sequential, use NTA
    };

    Type type = Type::SEQUENTIAL;
    size_t stride = 0;                    // Detected stride in bytes
    double regularity = 0.0;              // How regular the pattern is (0-1)
    bool hw_prefetch_effective = true;    // Will HW prefetch work?
    int optimal_sw_prefetch_distance = 0; // If SW prefetch needed
    std::string description;
};

// Detect prefetch pattern from access behavior
PrefetchPattern detect_prefetch_pattern(
    uint64_t sequential_reads,       // Sequential reads detected
    uint64_t strided_reads,          // Strided reads detected
    uint64_t random_reads,           // Random reads detected
    size_t detected_stride,          // Average stride if strided
    double memory_latency_ns,
    double iteration_time_ns
);

// Prefetch tuning recommendations
struct PrefetchTuningResult {
    bool needs_prefetch = false;
    int l1_prefetch_distance = 0;
    int l2_prefetch_distance = 0;
    bool use_nta = false;
    double expected_improvement = 0.0;
    std::string code_suggestion;
    std::vector<std::string> explanation;
};

// Generate prefetch tuning recommendations
PrefetchTuningResult recommend_prefetch_tuning(
    const PrefetchMetrics& metrics,
    const PrefetchPattern& pattern,
    const HardwareInfo& hw,
    size_t working_set_bytes,
    size_t stride_bytes
);

// Helper: Calculate memory latency from miss costs
inline double estimate_memory_latency_ns(
    uint64_t cycles,
    uint64_t l3_misses,
    double frequency_ghz
) {
    if (l3_misses == 0 || frequency_ghz <= 0) return 100.0;  // Default ~100ns
    double cycles_per_miss = static_cast<double>(cycles) / l3_misses;
    return cycles_per_miss / frequency_ghz;  // Convert to ns
}

// Helper: Common DRAM latencies
struct DRAMLatency {
    static constexpr double DDR4_2400_NS = 80.0;
    static constexpr double DDR4_3200_NS = 70.0;
    static constexpr double DDR5_4800_NS = 65.0;
    static constexpr double DDR5_6400_NS = 55.0;

    // Typical loaded latency (higher due to queuing)
    static constexpr double LOADED_MULTIPLIER = 1.5;
};

}  // namespace simd_bench
