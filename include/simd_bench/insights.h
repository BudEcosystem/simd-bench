#pragma once

#include "simd_bench/types.h"
#include "simd_bench/hardware.h"
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace simd_bench {

// ============================================================================
// Insight Categories and Severity Levels
// ============================================================================

enum class InsightCategory {
    MEMORY_BOUND,           // Memory bandwidth limitations
    COMPUTE_BOUND,          // Compute throughput limitations
    VECTORIZATION,          // SIMD vectorization issues
    CACHE_EFFICIENCY,       // Cache utilization problems
    DATA_ALIGNMENT,         // Memory alignment issues
    LOOP_OPTIMIZATION,      // Loop structure improvements
    BRANCH_PREDICTION,      // Branch-related performance
    INSTRUCTION_MIX,        // Instruction pipeline issues
    REGISTER_PRESSURE,      // Register spilling issues
    MEMORY_ACCESS_PATTERN,  // Strided/scattered access
    ARITHMETIC_INTENSITY,   // AI improvement opportunities
    PARALLELISM,           // Threading/SIMD parallelism
    LATENCY_HIDING,        // Prefetching/pipelining
    GENERAL                // General recommendations
};

enum class InsightSeverity {
    CRITICAL,   // Major performance issue (>50% potential improvement)
    HIGH,       // Significant issue (20-50% improvement)
    MEDIUM,     // Moderate issue (5-20% improvement)
    LOW,        // Minor issue (<5% improvement)
    INFO        // Informational only
};

enum class InsightConfidence {
    HIGH,       // Strong evidence from multiple metrics
    MEDIUM,     // Moderate evidence
    LOW         // Weak evidence, needs further investigation
};

// ============================================================================
// Insight Data Structures
// ============================================================================

struct OptimizationInsight {
    InsightCategory category;
    InsightSeverity severity;
    InsightConfidence confidence;

    std::string title;
    std::string description;
    std::string evidence;           // What data supports this insight
    std::string recommendation;     // Specific action to take
    std::string code_example;       // Optional code snippet

    double potential_speedup;       // Estimated improvement factor
    std::vector<std::string> references;  // Links to documentation

    // Metrics that triggered this insight
    std::map<std::string, double> triggering_metrics;
};

struct KernelAnalysis {
    std::string kernel_name;
    std::string variant_name;
    size_t problem_size;

    // Performance classification
    std::string primary_bottleneck;     // "memory", "compute", "latency", etc.
    double efficiency_vs_peak;
    double efficiency_vs_roofline;

    // Detailed insights
    std::vector<OptimizationInsight> insights;

    // Summary metrics
    double achieved_gflops;
    double theoretical_max_gflops;
    double arithmetic_intensity;
    double memory_bandwidth_utilization;
    double vectorization_ratio;
    double ipc;
    double cache_miss_rate;

    // Recommended next steps (prioritized)
    std::vector<std::string> next_steps;
};

struct InsightsReport {
    std::string timestamp;
    HardwareInfo hardware;

    std::vector<KernelAnalysis> kernel_analyses;

    // Cross-kernel insights
    std::vector<OptimizationInsight> global_insights;

    // Summary statistics
    int total_insights;
    int critical_count;
    int high_count;
    int medium_count;
    int low_count;
};

// ============================================================================
// Threshold Configuration
// ============================================================================

struct InsightThresholds {
    // Roofline thresholds
    double ridge_point_margin = 0.1;        // Within 10% of ridge point
    double memory_bound_ai_threshold = 1.0; // AI < 1.0 = memory bound
    double compute_bound_ai_threshold = 10.0;

    // Efficiency thresholds
    double excellent_efficiency = 0.7;      // >70% = excellent
    double good_efficiency = 0.5;           // 50-70% = good
    double poor_efficiency = 0.25;          // <25% = poor

    // Speedup thresholds
    double expected_simd_speedup_avx2 = 4.0;    // 8 floats/vector, ~50% efficiency
    double expected_simd_speedup_avx512 = 8.0;  // 16 floats/vector, ~50% efficiency
    double min_worthwhile_speedup = 1.5;

    // Cache thresholds
    double high_l1_miss_rate = 0.05;        // >5% L1 miss rate
    double high_l2_miss_rate = 0.10;        // >10% L2 miss rate
    double high_llc_miss_rate = 0.20;       // >20% LLC miss rate

    // IPC thresholds (for modern x86)
    double excellent_ipc = 3.0;
    double good_ipc = 2.0;
    double poor_ipc = 1.0;

    // Memory bandwidth utilization
    double high_bw_utilization = 0.7;
    double medium_bw_utilization = 0.4;

    // Vectorization thresholds
    double good_vectorization_ratio = 0.8;
    double poor_vectorization_ratio = 0.3;

    // Size thresholds for cache fitting
    double l1_fit_factor = 0.5;     // Use 50% of L1 for working set
    double l2_fit_factor = 0.75;    // Use 75% of L2
    double l3_fit_factor = 0.9;     // Use 90% of L3
};

// ============================================================================
// Insights Engine
// ============================================================================

class InsightsEngine {
public:
    InsightsEngine();
    explicit InsightsEngine(const HardwareInfo& hw);

    // Configure thresholds
    void set_thresholds(const InsightThresholds& thresholds);
    InsightThresholds& thresholds() { return thresholds_; }

    // Analyze a single variant result
    KernelAnalysis analyze_variant(
        const VariantResult& result,
        const KernelConfig* config = nullptr
    ) const;

    // Analyze a complete benchmark result (all variants)
    std::vector<KernelAnalysis> analyze_benchmark(
        const BenchmarkResult& result,
        const KernelConfig* config = nullptr
    ) const;

    // Analyze multiple benchmarks
    InsightsReport analyze_all(
        const std::vector<BenchmarkResult>& results,
        const std::map<std::string, const KernelConfig*>& configs = {}
    ) const;

    // Generate insights from raw metrics
    std::vector<OptimizationInsight> generate_insights(
        double gflops,
        double arithmetic_intensity,
        double memory_bandwidth_gbps,
        double ipc,
        double cache_miss_rate,
        double vectorization_ratio,
        size_t problem_size,
        bool is_scalar_baseline = false
    ) const;

    // Specific analysis functions
    std::vector<OptimizationInsight> analyze_roofline_position(
        double ai, double achieved_gflops
    ) const;

    std::vector<OptimizationInsight> analyze_cache_behavior(
        size_t working_set_bytes,
        double l1_miss_rate,
        double l2_miss_rate,
        double l3_miss_rate
    ) const;

    std::vector<OptimizationInsight> analyze_vectorization(
        double vectorization_ratio,
        double speedup_vs_scalar,
        int vector_width
    ) const;

    std::vector<OptimizationInsight> analyze_memory_access(
        double bandwidth_utilization,
        double arithmetic_intensity,
        size_t stride = 1
    ) const;

    std::vector<OptimizationInsight> analyze_ipc(
        double ipc,
        double branch_miss_rate = 0
    ) const;

    // Format output
    std::string format_insights_text(const KernelAnalysis& analysis) const;
    std::string format_insights_markdown(const KernelAnalysis& analysis) const;
    std::string format_insights_json(const KernelAnalysis& analysis) const;
    std::string format_report_markdown(const InsightsReport& report) const;
    std::string format_report_json(const InsightsReport& report) const;

    // Utility functions
    static std::string category_to_string(InsightCategory cat);
    static std::string severity_to_string(InsightSeverity sev);
    static std::string confidence_to_string(InsightConfidence conf);

private:
    HardwareInfo hw_;
    InsightThresholds thresholds_;

    // Internal analysis helpers
    std::string classify_bottleneck(double ai, double efficiency) const;
    double calculate_theoretical_max(double ai) const;
    double estimate_potential_speedup(const OptimizationInsight& insight) const;
    std::vector<std::string> generate_next_steps(
        const std::vector<OptimizationInsight>& insights
    ) const;

    // Cross-kernel analysis
    std::vector<OptimizationInsight> analyze_cross_kernel_patterns(
        const std::vector<KernelAnalysis>& analyses
    ) const;
};

// ============================================================================
// Rule-Based Insight Generator
// ============================================================================

// Rule function type: takes metrics, returns optional insight
using InsightRule = std::function<std::optional<OptimizationInsight>(
    const KernelMetrics& metrics,
    const HardwareInfo& hw,
    const InsightThresholds& thresholds,
    const KernelConfig* config
)>;

class InsightRuleEngine {
public:
    InsightRuleEngine();

    // Register custom rules
    void add_rule(const std::string& name, InsightRule rule);
    void remove_rule(const std::string& name);

    // Run all rules
    std::vector<OptimizationInsight> evaluate(
        const KernelMetrics& metrics,
        const HardwareInfo& hw,
        const InsightThresholds& thresholds,
        const KernelConfig* config = nullptr
    ) const;

    // Get built-in rules
    static std::vector<std::pair<std::string, InsightRule>> get_default_rules();

private:
    std::map<std::string, InsightRule> rules_;
};

// ============================================================================
// Predefined Insight Templates
// ============================================================================

namespace insight_templates {

// Memory-bound insights
OptimizationInsight memory_bandwidth_limited(double utilization, double achieved_bw);
OptimizationInsight low_arithmetic_intensity(double ai, double ridge_point);
OptimizationInsight strided_access_pattern(int stride);
OptimizationInsight cache_thrashing(size_t working_set, size_t cache_size);

// Compute-bound insights
OptimizationInsight low_vectorization(double ratio, double expected);
OptimizationInsight suboptimal_simd_speedup(double speedup, double expected);
OptimizationInsight high_scalar_operations(uint64_t scalar_ops, uint64_t vector_ops);

// Cache insights
OptimizationInsight high_l1_miss_rate(double rate);
OptimizationInsight high_l2_miss_rate(double rate);
OptimizationInsight high_llc_miss_rate(double rate);
OptimizationInsight working_set_exceeds_cache(size_t ws, const std::string& cache_level);

// Vectorization insights
OptimizationInsight recommend_loop_unrolling(int current_factor, int recommended);
OptimizationInsight recommend_data_alignment(int current_alignment, int required);
OptimizationInsight dependency_prevents_vectorization();
OptimizationInsight recommend_simd_intrinsics();

// Loop optimization insights
OptimizationInsight recommend_loop_tiling(size_t tile_size);
OptimizationInsight recommend_loop_interchange();
OptimizationInsight recommend_loop_fusion();
OptimizationInsight recommend_software_prefetch(int prefetch_distance);

// Instruction-level insights
OptimizationInsight low_ipc(double ipc, double expected);
OptimizationInsight high_branch_misprediction(double rate);
OptimizationInsight recommend_branchless_code();
OptimizationInsight recommend_fma_usage();

// Data layout insights
OptimizationInsight recommend_aos_to_soa();
OptimizationInsight recommend_struct_padding();
OptimizationInsight recommend_data_packing();

// General insights
OptimizationInsight excellent_performance();
OptimizationInsight near_optimal_performance(double efficiency);

}  // namespace insight_templates

// ============================================================================
// Code Pattern Analyzer (for source code analysis)
// ============================================================================

struct CodePattern {
    std::string pattern_name;
    std::string description;
    bool is_antipattern;
    std::vector<std::string> file_locations;  // file:line
    OptimizationInsight related_insight;
};

class CodePatternAnalyzer {
public:
    // Analyze source code for optimization patterns
    std::vector<CodePattern> analyze_source(const std::string& source_code) const;

    // Check for specific patterns
    bool has_loop_carried_dependency(const std::string& loop_code) const;
    bool has_non_unit_stride(const std::string& loop_code) const;
    bool has_function_call_in_loop(const std::string& loop_code) const;
    bool has_conditional_in_loop(const std::string& loop_code) const;
    bool uses_restrict_qualifier(const std::string& code) const;
    bool has_alignment_attributes(const std::string& code) const;

    // Estimate vectorization potential
    double estimate_vectorization_potential(const std::string& loop_code) const;

private:
    // Pattern matching helpers
    std::vector<std::string> find_loops(const std::string& code) const;
    std::vector<std::string> find_array_accesses(const std::string& code) const;
};

}  // namespace simd_bench
