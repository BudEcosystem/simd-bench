#include "simd_bench/scaling.h"
#include "simd_bench/timing.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <thread>

namespace simd_bench {

// ============================================================================
// ScalingAnalyzer implementation
// ============================================================================

ScalingAnalyzer::ScalingAnalyzer() {
    hw_ = HardwareInfo::detect();
    set_default_thread_counts();
}

ScalingAnalyzer::ScalingAnalyzer(const HardwareInfo& hw) : hw_(hw) {
    set_default_thread_counts();
}

void ScalingAnalyzer::set_thread_counts(const std::vector<int>& counts) {
    thread_counts_ = counts;
    std::sort(thread_counts_.begin(), thread_counts_.end());
}

void ScalingAnalyzer::set_default_thread_counts() {
    thread_counts_.clear();
    int max_threads = hw_.logical_cores > 0 ? hw_.logical_cores :
                       static_cast<int>(std::thread::hardware_concurrency());

    // 1, 2, 4, 8, ... up to max
    for (int t = 1; t <= max_threads; t *= 2) {
        thread_counts_.push_back(t);
    }

    // Add max if not already included
    if (thread_counts_.back() != max_threads) {
        thread_counts_.push_back(max_threads);
    }
}

void ScalingAnalyzer::set_iterations(size_t iterations) {
    iterations_ = iterations;
}

void ScalingAnalyzer::set_flops_per_operation(uint64_t flops) {
    flops_per_op_ = flops;
}

ScalingResult ScalingAnalyzer::analyze(BenchmarkFunc benchmark, size_t problem_size) {
    ScalingResult result;
    result.thread_counts = thread_counts_;

    double single_thread_time = 0.0;

    for (int threads : thread_counts_) {
        ThreadMeasurement measurement;
        measurement.thread_count = threads;

        // Run benchmark multiple times for stability
        double total_time = 0.0;
        for (size_t iter = 0; iter < iterations_; ++iter) {
            total_time += benchmark(threads);
        }

        measurement.elapsed_seconds = total_time / iterations_;
        measurement.throughput = problem_size / measurement.elapsed_seconds;

        if (flops_per_op_ > 0) {
            measurement.gflops = FlopsCalculator::to_gflops(
                flops_per_op_ * problem_size, measurement.elapsed_seconds);
        }

        // Calculate speedup relative to single thread
        if (threads == 1) {
            single_thread_time = measurement.elapsed_seconds;
            measurement.speedup = 1.0;
            measurement.efficiency = 1.0;
        } else {
            measurement.speedup = single_thread_time / measurement.elapsed_seconds;
            measurement.efficiency = measurement.speedup / threads;
        }

        result.measurements.push_back(measurement);
        result.speedups.push_back(measurement.speedup);
    }

    // Calculate derived metrics
    result.serial_fraction = estimate_serial_fraction(result.measurements);
    result.max_theoretical_speedup = 1.0 / result.serial_fraction;

    // Find optimal thread count
    auto max_it = std::max_element(result.speedups.begin(), result.speedups.end());
    size_t max_idx = std::distance(result.speedups.begin(), max_it);
    result.optimal_thread_count = result.thread_counts[max_idx];

    // Find diminishing returns point (efficiency < 50%)
    result.diminishing_returns_point = result.thread_counts.back();
    for (const auto& m : result.measurements) {
        if (m.efficiency < ScalingThresholds::ACCEPTABLE_EFFICIENCY) {
            result.diminishing_returns_point = m.thread_count;
            break;
        }
    }

    // Calculate average efficiency (excluding single-threaded measurement)
    double total_efficiency = 0.0;
    size_t multi_thread_count = 0;
    for (const auto& m : result.measurements) {
        if (m.thread_count > 1) {  // Exclude single-threaded
            total_efficiency += m.efficiency;
            ++multi_thread_count;
        }
    }
    result.parallel_efficiency = (multi_thread_count > 0) ?
        (total_efficiency / multi_thread_count) : 1.0;

    // Classify scaling behavior
    result.scaling_category = classify_scaling(result.measurements);

    // Detect bottleneck
    result.bottleneck = detect_bottleneck(result.measurements);

    // Generate recommendations
    if (result.scaling_category == "saturated") {
        result.recommendations.push_back(
            "Scaling saturates early - likely memory bandwidth limited");
        result.recommendations.push_back(
            "Consider improving cache utilization or reducing memory traffic");
    } else if (result.scaling_category == "sublinear") {
        result.recommendations.push_back(
            "Sublinear scaling - some overhead present");
        if (result.serial_fraction > 0.1) {
            result.recommendations.push_back(
                "Serial fraction is " +
                std::to_string(static_cast<int>(result.serial_fraction * 100)) +
                "% - look for parallelization opportunities");
        }
    } else if (result.scaling_category == "negative") {
        result.recommendations.push_back(
            "Negative scaling detected - parallelization overhead exceeds benefit");
        result.recommendations.push_back(
            "Check for false sharing, lock contention, or memory thrashing");
    }

    if (result.bottleneck == "memory_bandwidth") {
        result.recommendations.push_back(
            "Memory bandwidth limited - use fewer threads or improve cache locality");
    } else if (result.bottleneck == "contention") {
        result.recommendations.push_back(
            "Thread contention detected - reduce synchronization or use lock-free structures");
    }

    return result;
}

ScalingResult ScalingAnalyzer::analyze(const KernelConfig& kernel, size_t problem_size) {
    // Wrap kernel into benchmark function
    // This is a simplified version - real impl would set up threading
    auto benchmark = [&kernel, problem_size](int thread_count) -> double {
        // Setup
        void* data = kernel.setup ? kernel.setup(problem_size) : nullptr;

        Timer timer;
        timer.start();

        // Execute kernel (thread handling would be in the kernel itself)
        if (!kernel.variants.empty()) {
            kernel.variants[0].func(data, problem_size, 1);
        }

        timer.stop();

        // Teardown
        if (kernel.teardown && data) {
            kernel.teardown(data);
        }

        return timer.elapsed_seconds();
    };

    set_flops_per_operation(kernel.flops_per_element);
    return analyze(benchmark, problem_size);
}

double ScalingAnalyzer::estimate_serial_fraction(
    const std::vector<ThreadMeasurement>& measurements
) {
    if (measurements.size() < 2) return 0.0;

    // Use Amdahl's law: S = 1 / (s + (1-s)/p)
    // Solving for s: s = (1/S - 1/p) / (1 - 1/p)

    // Use the highest thread count measurement for estimation
    const auto& last = measurements.back();
    if (last.thread_count <= 1) return 0.0;

    double S = last.speedup;
    double p = last.thread_count;

    // s = (p - S) / (S * (p - 1))
    double s = (p - S) / (S * (p - 1));

    // Clamp to valid range
    return std::max(0.0, std::min(1.0, s));
}

std::string ScalingAnalyzer::classify_scaling(
    const std::vector<ThreadMeasurement>& measurements
) {
    if (measurements.size() < 2) return "unknown";

    // Check for negative scaling
    for (size_t i = 1; i < measurements.size(); ++i) {
        if (measurements[i].speedup < measurements[i-1].speedup * 0.95) {
            return "negative";
        }
    }

    // Check last measurement efficiency
    const auto& last = measurements.back();

    if (last.efficiency >= ScalingThresholds::LINEAR_THRESHOLD) {
        return "linear";
    } else if (last.efficiency >= ScalingThresholds::SUBLINEAR_THRESHOLD) {
        return "sublinear";
    } else {
        // Check if speedup plateaus
        if (measurements.size() >= 3) {
            double prev_speedup = measurements[measurements.size() - 2].speedup;
            double improvement = last.speedup / prev_speedup;
            if (improvement < ScalingThresholds::SATURATION_THRESHOLD) {
                return "saturated";
            }
        }
        return "sublinear";
    }
}

std::string ScalingAnalyzer::detect_bottleneck(
    const std::vector<ThreadMeasurement>& measurements
) {
    if (measurements.size() < 2) return "unknown";

    const auto& last = measurements.back();

    // Memory bandwidth saturation: efficiency drops sharply at higher thread counts
    bool sharp_efficiency_drop = false;
    for (size_t i = 2; i < measurements.size(); ++i) {
        double efficiency_ratio = measurements[i].efficiency / measurements[i-1].efficiency;
        if (efficiency_ratio < 0.7) {
            sharp_efficiency_drop = true;
            break;
        }
    }

    if (sharp_efficiency_drop && last.thread_count >= 4) {
        return "memory_bandwidth";
    }

    // Contention: negative scaling or very low efficiency
    if (last.efficiency < 0.3) {
        return "contention";
    }

    // Serial bottleneck: high serial fraction
    double serial_fraction = estimate_serial_fraction(measurements);
    if (serial_fraction > 0.3) {
        return "serial_portion";
    }

    // Load imbalance: moderate efficiency loss
    if (last.efficiency < 0.6) {
        return "load_imbalance";
    }

    return "none";
}

double ScalingAnalyzer::predict_speedup(int thread_count, double serial_fraction) const {
    // Amdahl's law
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / thread_count);
}

std::vector<ScalingAnalyzer::ScalingIssue> ScalingAnalyzer::detect_scaling_issues(
    const ScalingResult& result
) {
    std::vector<ScalingIssue> issues;

    // Check for false sharing signature
    auto false_sharing = detect_false_sharing(result.measurements, 0, sizeof(float));
    if (false_sharing.detected) {
        ScalingIssue issue;
        issue.type = "false_sharing";
        issue.severity = false_sharing.severity;
        issue.description = false_sharing.evidence;
        issue.fix_suggestion = "Pad data structures to cache line boundaries (64 bytes)";
        issues.push_back(issue);
    }

    // Check for contention
    auto contention = analyze_contention(result.measurements);
    if (contention.has_contention) {
        ScalingIssue issue;
        issue.type = "contention";
        issue.severity = contention.contention_factor;
        issue.description = "Lock or atomic contention detected";
        issue.fix_suggestion = contention.recommendations.empty() ? "" :
                               contention.recommendations[0];
        issues.push_back(issue);
    }

    // Check for NUMA effects
    if (hw_.numa_nodes > 1) {
        auto numa = analyze_numa_effects(hw_, result);
        if (numa.numa_aware_beneficial) {
            ScalingIssue issue;
            issue.type = "numa";
            issue.severity = 0.5;
            issue.description = "NUMA-aware allocation may improve scaling";
            issue.fix_suggestion = numa.recommendation;
            issues.push_back(issue);
        }
    }

    return issues;
}

// ============================================================================
// ScalingModel implementation
// ============================================================================

ScalingModel::ScalingModel(ScalingModelType type) : type_(type) {}

void ScalingModel::fit(const std::vector<ThreadMeasurement>& measurements) {
    switch (type_) {
        case ScalingModelType::AMDAHL:
            fit_amdahl(measurements);
            break;
        case ScalingModelType::ROOFLINE:
            fit_roofline(measurements);
            break;
        default:
            fit_amdahl(measurements);
    }
}

void ScalingModel::fit_amdahl(const std::vector<ThreadMeasurement>& measurements) {
    if (measurements.size() < 2) {
        serial_fraction_ = 0.0;
        return;
    }

    // Least squares fit for serial fraction
    // S = 1 / (s + (1-s)/p)
    // Linearize: 1/S = s + (1-s)/p = s(1 - 1/p) + 1/p

    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    int n = 0;

    for (const auto& m : measurements) {
        if (m.thread_count <= 0 || m.speedup <= 0) continue;

        double x = 1.0 - 1.0 / m.thread_count;  // (1 - 1/p)
        double y = 1.0 / m.speedup - 1.0 / m.thread_count;  // 1/S - 1/p

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        ++n;
    }

    if (n < 2 || sum_xx * n - sum_x * sum_x == 0) {
        serial_fraction_ = 0.0;
        return;
    }

    // Linear regression: y = s * x
    serial_fraction_ = (sum_xy * n - sum_x * sum_y) / (sum_xx * n - sum_x * sum_x);
    serial_fraction_ = std::max(0.0, std::min(1.0, serial_fraction_));

    // Calculate R²
    double ss_res = 0.0, ss_tot = 0.0;
    double mean_y = sum_y / n;

    for (const auto& m : measurements) {
        if (m.thread_count <= 0 || m.speedup <= 0) continue;

        double predicted = predict(m.thread_count);
        double actual = m.speedup;
        double y = 1.0 / m.speedup - 1.0 / m.thread_count;

        ss_res += (actual - predicted) * (actual - predicted);
        ss_tot += (y - mean_y) * (y - mean_y);
    }

    r_squared_ = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0.0;
}

void ScalingModel::fit_roofline(const std::vector<ThreadMeasurement>& measurements) {
    // Memory-limited model: S = min(p, bandwidth_limit)
    // Find where scaling saturates

    if (measurements.empty()) return;

    double max_speedup = 0.0;
    for (const auto& m : measurements) {
        max_speedup = std::max(max_speedup, m.speedup);
    }

    memory_factor_ = max_speedup;

    // Simple serial fraction estimate
    const auto& last = measurements.back();
    if (last.thread_count > 1) {
        serial_fraction_ = (last.thread_count - last.speedup) /
                           (last.speedup * (last.thread_count - 1));
        serial_fraction_ = std::max(0.0, std::min(1.0, serial_fraction_));
    }
}

double ScalingModel::predict(int thread_count) const {
    switch (type_) {
        case ScalingModelType::AMDAHL:
            return 1.0 / (serial_fraction_ + (1.0 - serial_fraction_) / thread_count);

        case ScalingModelType::GUSTAFSON:
            return serial_fraction_ + thread_count * (1.0 - serial_fraction_);

        case ScalingModelType::ROOFLINE:
            // Amdahl limited by memory bandwidth
            {
                double amdahl = 1.0 / (serial_fraction_ +
                                        (1.0 - serial_fraction_) / thread_count);
                return std::min(amdahl, memory_factor_);
            }

        default:
            return static_cast<double>(thread_count);
    }
}

std::string ScalingModel::get_description() const {
    std::ostringstream oss;

    switch (type_) {
        case ScalingModelType::AMDAHL:
            oss << "Amdahl's Law: serial_fraction = "
                << std::to_string(static_cast<int>(serial_fraction_ * 100)) << "%"
                << ", max_speedup = " << (1.0 / serial_fraction_);
            break;

        case ScalingModelType::GUSTAFSON:
            oss << "Gustafson's Law: serial_fraction = "
                << std::to_string(static_cast<int>(serial_fraction_ * 100)) << "%";
            break;

        case ScalingModelType::ROOFLINE:
            oss << "Memory-limited: max_speedup = " << memory_factor_;
            break;

        default:
            oss << "Custom model";
    }

    oss << ", R² = " << r_squared_;
    return oss.str();
}

// ============================================================================
// Analysis functions
// ============================================================================

NUMAAnalysis analyze_numa_effects(
    const HardwareInfo& hw,
    const ScalingResult& scaling
) {
    NUMAAnalysis analysis;
    analysis.numa_nodes = hw.numa_nodes;

    if (hw.numa_nodes <= 1) {
        analysis.numa_aware_beneficial = false;
        analysis.recommendation = "Single NUMA node - NUMA optimization not applicable";
        return analysis;
    }

    // Check if scaling degrades when crossing NUMA boundaries
    // Typically happens when thread count exceeds cores per node

    int cores_per_node = hw.physical_cores / hw.numa_nodes;
    bool crosses_numa = false;

    for (const auto& m : scaling.measurements) {
        if (m.thread_count > cores_per_node) {
            crosses_numa = true;
            // Check efficiency drop
            for (size_t i = 0; i < scaling.measurements.size(); ++i) {
                if (scaling.measurements[i].thread_count == cores_per_node) {
                    double prev_eff = scaling.measurements[i].efficiency;
                    if (m.efficiency < prev_eff * 0.8) {
                        analysis.numa_aware_beneficial = true;
                        analysis.remote_access_penalty = prev_eff / m.efficiency;
                    }
                    break;
                }
            }
            break;
        }
    }

    if (analysis.numa_aware_beneficial) {
        analysis.recommendation =
            "Use NUMA-aware memory allocation (numactl or libnuma) to "
            "keep data local to threads";
    } else if (crosses_numa) {
        analysis.recommendation =
            "Scaling across NUMA nodes appears acceptable";
    }

    return analysis;
}

FalseSharingAnalysis detect_false_sharing(
    const std::vector<ThreadMeasurement>& measurements,
    size_t /* working_set_bytes */,
    size_t /* element_size */
) {
    FalseSharingAnalysis analysis;

    if (measurements.size() < 2) return analysis;

    // False sharing signature: efficiency drops dramatically with more threads
    // especially at small thread counts (2-4)

    for (size_t i = 1; i < measurements.size(); ++i) {
        if (measurements[i].thread_count <= 4) {
            double expected_efficiency = 0.9;  // Should be near-linear at low counts
            if (measurements[i].efficiency < expected_efficiency * 0.6) {
                analysis.detected = true;
                analysis.severity = 1.0 - (measurements[i].efficiency / expected_efficiency);
                analysis.evidence =
                    "Efficiency drops to " +
                    std::to_string(static_cast<int>(measurements[i].efficiency * 100)) +
                    "% at " + std::to_string(measurements[i].thread_count) +
                    " threads - characteristic of false sharing";
                break;
            }
        }
    }

    if (analysis.detected) {
        analysis.recommendations.push_back(
            "Align shared data structures to cache line boundaries");
        analysis.recommendations.push_back(
            "Use padding between per-thread data: struct alignas(64) ThreadData { ... };");
        analysis.recommendations.push_back(
            "Consider thread-local storage for intermediate results");
    }

    return analysis;
}

ContentionAnalysis analyze_contention(
    const std::vector<ThreadMeasurement>& measurements
) {
    ContentionAnalysis analysis;

    if (measurements.size() < 3) return analysis;

    // Contention signature: non-monotonic speedup or speedup decrease

    for (size_t i = 2; i < measurements.size(); ++i) {
        if (measurements[i].speedup < measurements[i-1].speedup) {
            analysis.has_contention = true;
            analysis.contention_factor =
                (measurements[i-1].speedup - measurements[i].speedup) /
                measurements[i-1].speedup;

            analysis.contention_type = "lock";  // Most common
            analysis.recommendations.push_back(
                "Reduce lock granularity or use lock-free algorithms");
            analysis.recommendations.push_back(
                "Consider replacing locks with atomics where possible");
            break;
        }
    }

    // Check for linear efficiency decline (atomic contention)
    if (!analysis.has_contention && measurements.back().efficiency < 0.4) {
        double efficiency_slope = (measurements.back().efficiency - measurements[1].efficiency) /
                                   (measurements.back().thread_count - measurements[1].thread_count);

        if (efficiency_slope < -0.05) {  // > 5% drop per thread added
            analysis.has_contention = true;
            analysis.contention_factor = -efficiency_slope * 10;
            analysis.contention_type = "atomic";
            analysis.recommendations.push_back(
                "Atomic operations causing memory bus contention");
            analysis.recommendations.push_back(
                "Batch atomic updates or use thread-local accumulators");
        }
    }

    return analysis;
}

}  // namespace simd_bench
