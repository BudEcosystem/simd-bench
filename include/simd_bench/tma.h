#pragma once

#include "types.h"
#include "performance_counters.h"
#include <string>
#include <vector>

namespace simd_bench {

// Top-Down Microarchitecture Analysis (TMA)
// Based on Intel's Top-down Microarchitecture Analysis Method

// TMA level for analysis depth
enum class TMALevel {
    LEVEL1,  // Basic: Retiring, Bad Speculation, Frontend, Backend
    LEVEL2,  // Detailed breakdown of each category
    LEVEL3   // Most detailed (memory bound breakdown)
};

// TMA category classification
enum class TMACategory {
    RETIRING,
    BAD_SPECULATION,
    FRONTEND_BOUND,
    BACKEND_BOUND,
    // Level 2
    RETIRING_BASE,
    RETIRING_VECTORIZED,
    BRANCH_MISPREDICTS,
    MACHINE_CLEARS,
    FETCH_LATENCY,
    FETCH_BANDWIDTH,
    CORE_BOUND,
    MEMORY_BOUND,
    // Level 3
    DIVIDER,
    PORTS_UTILIZATION,
    L1_BOUND,
    L2_BOUND,
    L3_BOUND,
    DRAM_BOUND
};

// TMA result for a single category
struct TMACategoryResult {
    TMACategory category;
    std::string name;
    double ratio;           // 0.0 to 1.0
    std::string description;
    std::vector<std::string> recommendations;
};

// Complete TMA analysis result
struct TMAResult {
    TMAMetrics metrics;
    std::vector<TMACategoryResult> categories;

    // Summary
    std::string primary_bottleneck;
    std::string secondary_bottleneck;
    double bottleneck_ratio;

    // Recommendations
    std::vector<std::string> recommendations;

    // Is the code well-optimized?
    bool is_well_optimized() const {
        return metrics.retiring > 0.7 && metrics.backend_bound < 0.2;
    }
};

// TMA Analyzer
class TMAAnalyzer {
public:
    TMAAnalyzer();
    explicit TMAAnalyzer(IPerformanceCounters* counters);

    // Set the performance counter backend
    void set_counters(IPerformanceCounters* counters);

    // Configure analysis level
    void set_level(TMALevel level) { level_ = level; }
    TMALevel get_level() const { return level_; }

    // Run analysis on counter values
    TMAResult analyze(const CounterValues& values) const;

    // Run analysis by measuring a function
    TMAResult measure_and_analyze(
        const std::function<void()>& func,
        size_t iterations = 1
    );

    // Get required events for the current analysis level
    std::vector<CounterEvent> get_required_events() const;

    // Check if TMA is supported on current hardware
    static bool is_supported();

private:
    IPerformanceCounters* counters_ = nullptr;
    TMALevel level_ = TMALevel::LEVEL1;

    // Calculate TMA metrics from raw counter values
    TMAMetrics calculate_metrics(const CounterValues& values) const;

    // Classify bottleneck
    TMACategoryResult classify_category(TMACategory cat, const TMAMetrics& metrics) const;

    // Generate recommendations based on bottleneck
    std::vector<std::string> generate_recommendations(const TMAMetrics& metrics) const;
};

// TMA visualization (text-based bar chart)
std::string format_tma_bar_chart(const TMAResult& result, int width = 60);

// TMA to string for reporting
std::string tma_category_to_string(TMACategory category);

}  // namespace simd_bench
