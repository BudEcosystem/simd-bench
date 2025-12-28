#pragma once

#include "types.h"
#include "performance_counters.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace simd_bench {

// Forward declarations
class IPerformanceCounters;

// Comprehensive metrics analyzer for SIMD kernel profiling
class MetricsAnalyzer {
public:
    MetricsAnalyzer();
    explicit MetricsAnalyzer(IPerformanceCounters* counters);

    void set_counters(IPerformanceCounters* counters);
    void set_cpu_vendor(CPUVendor vendor);

    // MPKI (Misses Per Kilo-Instructions) calculation
    MPKIMetrics calculate_mpki(const CounterValues& values) const;

    // DSB (Decoded Stream Buffer) coverage
    DSBMetrics calculate_dsb_coverage(const CounterValues& values) const;

    // Port utilization metrics
    PortMetrics calculate_port_utilization(const CounterValues& values) const;

    // Cache line split metrics
    CacheLineSplitMetrics calculate_cache_line_splits(const CounterValues& values) const;

    // AVX-512 frequency transition metrics
    AVX512FrequencyMetrics calculate_avx512_frequency(const CounterValues& values) const;

    // Extended SIMD metrics with quality scoring
    ExtendedSIMDMetrics calculate_extended_simd(const CounterValues& values,
                                                 double elapsed_seconds,
                                                 size_t total_flops) const;

    // Calculate vectorization ratio from counter values
    double calculate_vectorization_ratio(const CounterValues& values) const;

    // Calculate vector width utilization
    double calculate_vector_width_utilization(const CounterValues& values) const;

    // Calculate IPC
    double calculate_ipc(const CounterValues& values) const;

    // Evaluate quality and generate recommendations
    std::vector<std::string> evaluate_quality(const CounterValues& values) const;

    // Measure and analyze a function
    struct AnalysisResult {
        MPKIMetrics mpki;
        DSBMetrics dsb;
        PortMetrics ports;
        CacheLineSplitMetrics cache_splits;
        AVX512FrequencyMetrics avx512_freq;
        ExtendedSIMDMetrics extended_simd;
        double ipc = 0.0;
        double vectorization_ratio = 0.0;
        std::vector<std::string> recommendations;
        double overall_quality_score = 0.0;
    };

    AnalysisResult measure_and_analyze(
        const std::function<void()>& func,
        size_t iterations,
        size_t flops_per_iteration
    );

    // Get required events for comprehensive analysis
    std::vector<CounterEvent> get_required_events() const;

    // Get architecture-specific events
    std::vector<CounterEvent> get_intel_events() const;
    std::vector<CounterEvent> get_amd_events() const;
    std::vector<CounterEvent> get_arm_events() const;

private:
    IPerformanceCounters* counters_ = nullptr;
    CPUVendor cpu_vendor_ = CPUVendor::UNKNOWN;

    // Helper to detect CPU vendor
    CPUVendor detect_vendor() const;

    // Quality scoring helpers
    double score_ipc(double ipc) const;
    double score_vectorization(double ratio) const;
    double score_cache_efficiency(const MPKIMetrics& mpki) const;
    double score_dsb_efficiency(const DSBMetrics& dsb) const;
    double score_port_balance(const PortMetrics& ports) const;
};

// IMC (Integrated Memory Controller) bandwidth analyzer
class IMCBandwidthAnalyzer {
public:
    IMCBandwidthAnalyzer();

    // Initialize with memory controller device
    bool initialize();
    void shutdown();

    // Start/stop measurement
    bool start();
    bool stop();

    // Get bandwidth metrics
    IMCBandwidthMetrics get_metrics() const;

    // Check if IMC uncore counters are available
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper functions for metric evaluation
inline bool is_ipc_healthy(double ipc) {
    return ipc >= QualityThresholds::IPC_HEALTHY;
}

inline bool is_vectorization_acceptable(double ratio) {
    return ratio >= QualityThresholds::VECTORIZATION_ACCEPTABLE;
}

inline bool is_retiring_healthy(double retiring) {
    return retiring >= QualityThresholds::RETIRING_HEALTHY;
}

inline bool is_backend_bound_acceptable(double backend_bound) {
    return backend_bound <= QualityThresholds::BACKEND_BOUND_ACCEPTABLE;
}

inline bool is_dsb_coverage_acceptable(double coverage) {
    return coverage >= QualityThresholds::DSB_COVERAGE_ACCEPTABLE;
}

inline bool is_mpki_acceptable(double l1, double l2, double l3) {
    return l1 < MPKIThresholds::L1_ACCEPTABLE &&
           l2 < MPKIThresholds::L2_ACCEPTABLE &&
           l3 < MPKIThresholds::L3_ACCEPTABLE;
}

// Utility: Calculate FLOPS from counter values
uint64_t calculate_total_flops(const CounterValues& values, CPUVendor vendor);

// Utility: Get human-readable quality rating
std::string get_quality_rating(double score);

}  // namespace simd_bench
