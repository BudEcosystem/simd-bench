#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>

namespace simd_bench {

// Single benchmark measurement for comparison
struct BenchmarkMeasurement {
    std::string kernel_name;
    std::string variant_name;
    size_t problem_size;

    double gflops;
    double elapsed_seconds;
    double efficiency;
    double vectorization_ratio;

    // Optional detailed metrics
    std::optional<double> cache_miss_rate;
    std::optional<double> ipc;

    // Metadata
    std::string timestamp;
    std::string git_commit;
    std::string cpu_model;
};

// Comparison result for a single benchmark
struct RegressionResult {
    std::string kernel_name;
    std::string variant_name;
    size_t problem_size;

    double baseline_gflops;
    double current_gflops;
    double change_percent;     // Positive = faster, negative = slower

    bool is_regression;        // True if significantly slower
    bool is_improvement;       // True if significantly faster
    bool is_unchanged;         // Within noise threshold

    std::string status;        // "regression", "improvement", "unchanged"
};

// Complete regression report
struct RegressionReport {
    std::vector<RegressionResult> results;

    // Summary statistics
    std::vector<std::string> regressions;    // Kernel names with >5% slowdown
    std::vector<std::string> improvements;   // Kernel names with >5% speedup
    std::vector<std::string> unchanged;      // Within threshold

    std::map<std::string, double> all_changes;  // Kernel -> percent change

    // Overall status
    bool has_critical_regressions = false;
    int regression_count = 0;
    int improvement_count = 0;
    int total_benchmarks = 0;

    double worst_regression_percent = 0.0;
    double best_improvement_percent = 0.0;
    std::string worst_regression_kernel;
    std::string best_improvement_kernel;
};

// Regression tracker for CI integration
class RegressionTracker {
public:
    RegressionTracker();

    // Set baseline from file
    bool set_baseline(const std::string& baseline_file);

    // Set baseline from git commit
    bool set_baseline(const std::string& git_commit,
                      const std::string& results_dir);

    // Set baseline from measurements
    void set_baseline(const std::vector<BenchmarkMeasurement>& measurements);

    // Compare current results against baseline
    RegressionReport compare(const std::vector<BenchmarkResult>& current);
    RegressionReport compare(const std::vector<BenchmarkMeasurement>& current);

    // Set regression threshold (default 5%)
    void set_threshold(double percent) { threshold_percent_ = percent; }

    // Set noise threshold for "unchanged" classification (default 2%)
    void set_noise_threshold(double percent) { noise_percent_ = percent; }

    // CI integration
    int exit_code() const;  // 0 = pass, 1 = regressions found

    // Generate reports
    std::string generate_markdown_report(const RegressionReport& report) const;
    std::string generate_github_comment(const RegressionReport& report) const;
    std::string generate_json_report(const RegressionReport& report) const;

    // Save/load baseline
    bool save_baseline(const std::string& filepath) const;
    bool load_baseline(const std::string& filepath);

private:
    std::vector<BenchmarkMeasurement> baseline_;
    double threshold_percent_ = 5.0;
    double noise_percent_ = 2.0;
    bool has_baseline_ = false;

    // Find matching baseline measurement
    std::optional<BenchmarkMeasurement> find_baseline(
        const std::string& kernel,
        const std::string& variant,
        size_t size
    ) const;

    // Convert BenchmarkResult to measurement
    std::vector<BenchmarkMeasurement> to_measurements(
        const std::vector<BenchmarkResult>& results
    ) const;
};

// Baseline file format
struct BaselineFile {
    std::string version = "1.0";
    std::string created_at;
    std::string git_commit;
    std::string git_branch;
    std::string cpu_model;
    std::string os_version;
    std::vector<BenchmarkMeasurement> measurements;
};

// Baseline file I/O
bool write_baseline_file(const BaselineFile& baseline, const std::string& filepath);
std::optional<BaselineFile> read_baseline_file(const std::string& filepath);

// Trend analysis for multiple runs
struct TrendAnalysis {
    std::string kernel_name;
    std::vector<double> gflops_history;
    std::vector<std::string> timestamps;

    double mean_gflops;
    double stddev_gflops;
    double coefficient_of_variation;

    bool is_stable;            // CV < 5%
    bool is_trending_up;       // Improving over time
    bool is_trending_down;     // Degrading over time

    std::string trend_description;
};

std::vector<TrendAnalysis> analyze_trends(
    const std::vector<BaselineFile>& history,
    size_t min_samples = 3
);

// Performance gates for CI
struct PerformanceGate {
    std::string name;
    double threshold_gflops;
    std::string comparison;  // ">=", "<=", ">", "<"
    bool required = true;    // Fail CI if not met
};

struct GateResult {
    std::string gate_name;
    bool passed;
    double achieved_gflops;
    double threshold_gflops;
    std::string message;
};

std::vector<GateResult> check_performance_gates(
    const std::vector<BenchmarkResult>& results,
    const std::vector<PerformanceGate>& gates
);

}  // namespace simd_bench
