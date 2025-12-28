#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>
#include <functional>
#include <cstdint>

namespace simd_bench {

// Thread count measurement result
struct ThreadMeasurement {
    int thread_count;
    double elapsed_seconds;
    double throughput;           // Operations per second
    double gflops;               // If applicable
    double speedup;              // Relative to single thread
    double efficiency;           // speedup / thread_count
};

// Scaling analysis result
struct ScalingResult {
    std::vector<ThreadMeasurement> measurements;
    std::vector<int> thread_counts;
    std::vector<double> speedups;

    // Derived metrics
    double parallel_efficiency;        // Average efficiency across thread counts
    int optimal_thread_count;          // Best performance
    int diminishing_returns_point;     // Where efficiency drops below threshold
    std::string scaling_category;      // "linear", "sublinear", "saturated", "negative"
    std::string bottleneck;            // "memory_bandwidth", "false_sharing", "contention", etc.

    // Amdahl's law analysis
    double serial_fraction;            // Estimated serial portion (Amdahl's law)
    double max_theoretical_speedup;    // Based on serial fraction

    // Recommendations
    std::vector<std::string> recommendations;
};

// Scaling analyzer for multi-core performance analysis
class ScalingAnalyzer {
public:
    using BenchmarkFunc = std::function<double(int thread_count)>;  // Returns elapsed time

    ScalingAnalyzer();
    explicit ScalingAnalyzer(const HardwareInfo& hw);

    // Configure thread counts to test
    void set_thread_counts(const std::vector<int>& counts);
    void set_default_thread_counts();  // 1, 2, 4, 8, ... up to hardware threads

    // Set iterations for stable timing
    void set_iterations(size_t iterations);

    // Set expected FLOPS per operation (for GFLOPS calculation)
    void set_flops_per_operation(uint64_t flops);

    // Run scaling analysis
    ScalingResult analyze(BenchmarkFunc benchmark, size_t problem_size);

    // Run with kernel config
    ScalingResult analyze(const KernelConfig& kernel, size_t problem_size);

    // Detect specific scaling issues
    struct ScalingIssue {
        std::string type;          // "false_sharing", "lock_contention", etc.
        double severity;           // 0-1
        std::string description;
        std::string fix_suggestion;
    };

    std::vector<ScalingIssue> detect_scaling_issues(const ScalingResult& result);

    // Predict performance at different thread counts
    double predict_speedup(int thread_count, double serial_fraction) const;

    // Get hardware info
    const HardwareInfo& hardware() const { return hw_; }

private:
    HardwareInfo hw_;
    std::vector<int> thread_counts_;
    size_t iterations_ = 10;
    uint64_t flops_per_op_ = 0;

    // Estimate serial fraction using Amdahl's law
    double estimate_serial_fraction(const std::vector<ThreadMeasurement>& measurements);

    // Classify scaling behavior
    std::string classify_scaling(const std::vector<ThreadMeasurement>& measurements);

    // Detect bottleneck type
    std::string detect_bottleneck(const std::vector<ThreadMeasurement>& measurements);
};

// Scaling model types
enum class ScalingModelType {
    AMDAHL,        // Amdahl's law: S = 1 / (s + (1-s)/p)
    GUSTAFSON,     // Gustafson's law: S = s + p(1-s)
    ROOFLINE,      // Memory bandwidth limited
    CUSTOM         // Fitted model
};

// Scaling model for prediction
class ScalingModel {
public:
    ScalingModel(ScalingModelType type = ScalingModelType::AMDAHL);

    // Fit model to measurements
    void fit(const std::vector<ThreadMeasurement>& measurements);

    // Predict speedup for thread count
    double predict(int thread_count) const;

    // Get model parameters
    double get_serial_fraction() const { return serial_fraction_; }
    double get_memory_factor() const { return memory_factor_; }

    // Calculate R² fit quality
    double get_r_squared() const { return r_squared_; }

    // Get model description
    std::string get_description() const;

private:
    ScalingModelType type_;
    double serial_fraction_ = 0.0;
    double memory_factor_ = 0.0;
    double r_squared_ = 0.0;

    // Fit Amdahl's law parameters
    void fit_amdahl(const std::vector<ThreadMeasurement>& measurements);

    // Fit memory-limited model
    void fit_roofline(const std::vector<ThreadMeasurement>& measurements);
};

// NUMA analysis
struct NUMAAnalysis {
    int numa_nodes;
    std::vector<double> node_bandwidths;     // GB/s per node
    double remote_access_penalty;             // Ratio of remote vs local latency
    bool numa_aware_beneficial;               // Would NUMA-aware allocation help?
    std::string recommendation;
};

NUMAAnalysis analyze_numa_effects(
    const HardwareInfo& hw,
    const ScalingResult& scaling
);

// False sharing detection
struct FalseSharingAnalysis {
    bool detected = false;
    double severity = 0.0;                    // 0-1
    std::string evidence;
    std::vector<std::string> recommendations;
};

FalseSharingAnalysis detect_false_sharing(
    const std::vector<ThreadMeasurement>& measurements,
    size_t working_set_bytes,
    size_t element_size
);

// Thread contention analysis
struct ContentionAnalysis {
    bool has_contention = false;
    double contention_factor = 0.0;           // Overhead ratio
    std::string contention_type;              // "lock", "atomic", "memory_order"
    std::vector<std::string> recommendations;
};

ContentionAnalysis analyze_contention(
    const std::vector<ThreadMeasurement>& measurements
);

// Scaling thresholds
struct ScalingThresholds {
    // Efficiency thresholds
    static constexpr double EXCELLENT_EFFICIENCY = 0.9;   // > 90%
    static constexpr double GOOD_EFFICIENCY = 0.7;        // > 70%
    static constexpr double ACCEPTABLE_EFFICIENCY = 0.5;  // > 50%

    // Speedup thresholds relative to linear
    static constexpr double LINEAR_THRESHOLD = 0.85;      // > 85% of linear
    static constexpr double SUBLINEAR_THRESHOLD = 0.5;    // > 50% of linear
    static constexpr double SATURATION_THRESHOLD = 1.1;   // < 10% improvement

    // Serial fraction thresholds
    static constexpr double HIGHLY_PARALLEL = 0.05;       // < 5% serial
    static constexpr double PARALLEL = 0.20;              // < 20% serial
    static constexpr double PARTIALLY_PARALLEL = 0.50;    // < 50% serial
};

}  // namespace simd_bench
