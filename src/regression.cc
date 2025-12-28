#include "simd_bench/regression.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>

namespace simd_bench {

// ============================================================================
// RegressionTracker implementation
// ============================================================================

RegressionTracker::RegressionTracker() {}

bool RegressionTracker::set_baseline(const std::string& baseline_file) {
    return load_baseline(baseline_file);
}

bool RegressionTracker::set_baseline(const std::string& git_commit,
                                       const std::string& results_dir) {
    // Look for baseline file matching commit
    std::string filepath = results_dir + "/" + git_commit + "_baseline.json";
    return load_baseline(filepath);
}

void RegressionTracker::set_baseline(
    const std::vector<BenchmarkMeasurement>& measurements
) {
    baseline_ = measurements;
    has_baseline_ = !measurements.empty();
}

std::vector<BenchmarkMeasurement> RegressionTracker::to_measurements(
    const std::vector<BenchmarkResult>& results
) const {
    std::vector<BenchmarkMeasurement> measurements;

    for (const auto& result : results) {
        for (const auto& variant_result : result.results) {
            BenchmarkMeasurement m;
            m.kernel_name = result.kernel_name;
            m.variant_name = variant_result.variant_name;
            m.problem_size = variant_result.problem_size;
            m.gflops = variant_result.metrics.performance.gflops;
            m.elapsed_seconds = variant_result.metrics.performance.elapsed_seconds;
            m.efficiency = variant_result.roofline.efficiency;
            m.vectorization_ratio = variant_result.metrics.simd.vectorization_ratio;

            if (variant_result.metrics.memory.llc_miss_rate > 0) {
                m.cache_miss_rate = variant_result.metrics.memory.llc_miss_rate;
            }
            if (variant_result.metrics.performance.ipc > 0) {
                m.ipc = variant_result.metrics.performance.ipc;
            }

            measurements.push_back(m);
        }
    }

    return measurements;
}

std::optional<BenchmarkMeasurement> RegressionTracker::find_baseline(
    const std::string& kernel,
    const std::string& variant,
    size_t size
) const {
    for (const auto& m : baseline_) {
        if (m.kernel_name == kernel &&
            m.variant_name == variant &&
            m.problem_size == size) {
            return m;
        }
    }
    return std::nullopt;
}

RegressionReport RegressionTracker::compare(
    const std::vector<BenchmarkResult>& current
) {
    return compare(to_measurements(current));
}

RegressionReport RegressionTracker::compare(
    const std::vector<BenchmarkMeasurement>& current
) {
    RegressionReport report;
    report.total_benchmarks = current.size();

    for (const auto& curr : current) {
        auto baseline = find_baseline(
            curr.kernel_name, curr.variant_name, curr.problem_size);

        RegressionResult result;
        result.kernel_name = curr.kernel_name;
        result.variant_name = curr.variant_name;
        result.problem_size = curr.problem_size;
        result.current_gflops = curr.gflops;

        if (baseline) {
            result.baseline_gflops = baseline->gflops;

            // Calculate percent change (positive = improvement)
            if (baseline->gflops > 0) {
                result.change_percent =
                    ((curr.gflops - baseline->gflops) / baseline->gflops) * 100.0;
            } else {
                result.change_percent = 0.0;
            }

            // Classify change
            if (result.change_percent < -threshold_percent_) {
                result.is_regression = true;
                result.status = "regression";
                report.regression_count++;

                std::string key = curr.kernel_name + "/" + curr.variant_name;
                report.regressions.push_back(key);

                if (result.change_percent < report.worst_regression_percent) {
                    report.worst_regression_percent = result.change_percent;
                    report.worst_regression_kernel = key;
                }
            } else if (result.change_percent > threshold_percent_) {
                result.is_improvement = true;
                result.status = "improvement";
                report.improvement_count++;

                std::string key = curr.kernel_name + "/" + curr.variant_name;
                report.improvements.push_back(key);

                if (result.change_percent > report.best_improvement_percent) {
                    report.best_improvement_percent = result.change_percent;
                    report.best_improvement_kernel = key;
                }
            } else if (std::abs(result.change_percent) <= noise_percent_) {
                result.is_unchanged = true;
                result.status = "unchanged";
                report.unchanged.push_back(curr.kernel_name + "/" + curr.variant_name);
            } else {
                result.status = "within_threshold";
            }

            report.all_changes[curr.kernel_name + "/" + curr.variant_name] =
                result.change_percent;
        } else {
            // No baseline for comparison
            result.baseline_gflops = 0.0;
            result.change_percent = 0.0;
            result.status = "no_baseline";
        }

        report.results.push_back(result);
    }

    // Determine if critical regressions exist
    report.has_critical_regressions = (report.regression_count > 0);

    return report;
}

int RegressionTracker::exit_code() const {
    // 0 = pass, 1 = regressions found
    // This would be called after compare()
    return 0;  // Placeholder - real impl checks last report
}

std::string RegressionTracker::generate_markdown_report(
    const RegressionReport& report
) const {
    std::ostringstream md;

    md << "# Benchmark Regression Report\n\n";

    // Summary
    md << "## Summary\n\n";
    md << "| Metric | Value |\n";
    md << "|--------|-------|\n";
    md << "| Total Benchmarks | " << report.total_benchmarks << " |\n";
    md << "| Regressions | " << report.regression_count << " |\n";
    md << "| Improvements | " << report.improvement_count << " |\n";
    md << "| Unchanged | " << report.unchanged.size() << " |\n\n";

    // Status
    if (report.has_critical_regressions) {
        md << "**Status: FAILED** - Performance regressions detected\n\n";
    } else {
        md << "**Status: PASSED** - No significant regressions\n\n";
    }

    // Regressions
    if (!report.regressions.empty()) {
        md << "## Regressions (>" << threshold_percent_ << "% slower)\n\n";
        md << "| Benchmark | Change |\n";
        md << "|-----------|--------|\n";
        for (const auto& r : report.regressions) {
            auto it = report.all_changes.find(r);
            if (it != report.all_changes.end()) {
                md << "| " << r << " | " << std::fixed << std::setprecision(1)
                   << it->second << "% |\n";
            }
        }
        md << "\n";
    }

    // Improvements
    if (!report.improvements.empty()) {
        md << "## Improvements (>" << threshold_percent_ << "% faster)\n\n";
        md << "| Benchmark | Change |\n";
        md << "|-----------|--------|\n";
        for (const auto& i : report.improvements) {
            auto it = report.all_changes.find(i);
            if (it != report.all_changes.end()) {
                md << "| " << i << " | +" << std::fixed << std::setprecision(1)
                   << it->second << "% |\n";
            }
        }
        md << "\n";
    }

    // Detailed results
    md << "## All Results\n\n";
    md << "| Benchmark | Variant | Size | Baseline | Current | Change |\n";
    md << "|-----------|---------|------|----------|---------|--------|\n";

    for (const auto& r : report.results) {
        md << "| " << r.kernel_name
           << " | " << r.variant_name
           << " | " << r.problem_size
           << " | " << std::fixed << std::setprecision(2) << r.baseline_gflops
           << " | " << std::fixed << std::setprecision(2) << r.current_gflops
           << " | ";

        if (r.change_percent >= 0) md << "+";
        md << std::fixed << std::setprecision(1) << r.change_percent << "% |\n";
    }

    return md.str();
}

std::string RegressionTracker::generate_github_comment(
    const RegressionReport& report
) const {
    std::ostringstream comment;

    // Header with status
    if (report.has_critical_regressions) {
        comment << "## :x: Performance Regression Detected\n\n";
    } else if (report.improvement_count > 0) {
        comment << "## :rocket: Performance Improvements\n\n";
    } else {
        comment << "## :white_check_mark: No Performance Regressions\n\n";
    }

    // Quick summary
    comment << "**" << report.regression_count << "** regressions, "
            << "**" << report.improvement_count << "** improvements, "
            << "**" << report.unchanged.size() << "** unchanged\n\n";

    // Highlight worst regression
    if (report.has_critical_regressions) {
        comment << ":warning: Worst regression: **" << report.worst_regression_kernel
                << "** (" << std::fixed << std::setprecision(1)
                << report.worst_regression_percent << "%)\n\n";
    }

    // Highlight best improvement
    if (report.improvement_count > 0) {
        comment << ":star: Best improvement: **" << report.best_improvement_kernel
                << "** (+" << std::fixed << std::setprecision(1)
                << report.best_improvement_percent << "%)\n\n";
    }

    // Collapsible details
    comment << "<details>\n<summary>Full Results</summary>\n\n";

    comment << "| Benchmark | Change |\n";
    comment << "|-----------|--------|\n";

    for (const auto& [name, change] : report.all_changes) {
        std::string icon;
        if (change < -threshold_percent_) icon = ":small_red_triangle_down:";
        else if (change > threshold_percent_) icon = ":small_blue_diamond:";
        else icon = ":white_small_square:";

        comment << "| " << icon << " " << name << " | ";
        if (change >= 0) comment << "+";
        comment << std::fixed << std::setprecision(1) << change << "% |\n";
    }

    comment << "\n</details>\n";

    return comment.str();
}

std::string RegressionTracker::generate_json_report(
    const RegressionReport& report
) const {
    std::ostringstream json;

    json << "{\n";
    json << "  \"summary\": {\n";
    json << "    \"total_benchmarks\": " << report.total_benchmarks << ",\n";
    json << "    \"regressions\": " << report.regression_count << ",\n";
    json << "    \"improvements\": " << report.improvement_count << ",\n";
    json << "    \"unchanged\": " << report.unchanged.size() << ",\n";
    json << "    \"has_critical_regressions\": "
         << (report.has_critical_regressions ? "true" : "false") << "\n";
    json << "  },\n";

    json << "  \"results\": [\n";
    for (size_t i = 0; i < report.results.size(); ++i) {
        const auto& r = report.results[i];
        json << "    {\n";
        json << "      \"kernel\": \"" << r.kernel_name << "\",\n";
        json << "      \"variant\": \"" << r.variant_name << "\",\n";
        json << "      \"size\": " << r.problem_size << ",\n";
        json << "      \"baseline_gflops\": " << r.baseline_gflops << ",\n";
        json << "      \"current_gflops\": " << r.current_gflops << ",\n";
        json << "      \"change_percent\": " << r.change_percent << ",\n";
        json << "      \"status\": \"" << r.status << "\"\n";
        json << "    }";
        if (i < report.results.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    return json.str();
}

bool RegressionTracker::save_baseline(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time);
    timestamp.pop_back();  // Remove newline

    file << "{\n";
    file << "  \"version\": \"1.0\",\n";
    file << "  \"created_at\": \"" << timestamp << "\",\n";
    file << "  \"measurements\": [\n";

    for (size_t i = 0; i < baseline_.size(); ++i) {
        const auto& m = baseline_[i];
        file << "    {\n";
        file << "      \"kernel_name\": \"" << m.kernel_name << "\",\n";
        file << "      \"variant_name\": \"" << m.variant_name << "\",\n";
        file << "      \"problem_size\": " << m.problem_size << ",\n";
        file << "      \"gflops\": " << m.gflops << ",\n";
        file << "      \"elapsed_seconds\": " << m.elapsed_seconds << ",\n";
        file << "      \"efficiency\": " << m.efficiency << ",\n";
        file << "      \"vectorization_ratio\": " << m.vectorization_ratio << "\n";
        file << "    }";
        if (i < baseline_.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    return true;
}

bool RegressionTracker::load_baseline(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    baseline_.clear();

    // Simple JSON parsing
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Parse measurements array (simplified parser)
    size_t pos = content.find("\"measurements\"");
    if (pos == std::string::npos) return false;

    // Find each measurement object
    size_t start = pos;
    while ((start = content.find("{", start)) != std::string::npos) {
        size_t end = content.find("}", start);
        if (end == std::string::npos) break;

        std::string obj = content.substr(start, end - start + 1);
        start = end + 1;

        // Skip if this is a top-level object
        if (obj.find("\"kernel_name\"") == std::string::npos) continue;

        BenchmarkMeasurement m;

        // Extract fields (simplified)
        auto extract = [&obj](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":";
            size_t p = obj.find(search);
            if (p == std::string::npos) return "";
            p += search.length();
            while (p < obj.size() && (obj[p] == ' ' || obj[p] == '"')) p++;
            size_t e = p;
            bool in_string = (obj[p-1] == '"');
            if (in_string) {
                e = obj.find("\"", p);
            } else {
                while (e < obj.size() && obj[e] != ',' && obj[e] != '}') e++;
            }
            return obj.substr(p, e - p);
        };

        m.kernel_name = extract("kernel_name");
        m.variant_name = extract("variant_name");

        std::string size_str = extract("problem_size");
        if (!size_str.empty()) m.problem_size = std::stoull(size_str);

        std::string gflops_str = extract("gflops");
        if (!gflops_str.empty()) m.gflops = std::stod(gflops_str);

        std::string elapsed_str = extract("elapsed_seconds");
        if (!elapsed_str.empty()) m.elapsed_seconds = std::stod(elapsed_str);

        std::string eff_str = extract("efficiency");
        if (!eff_str.empty()) m.efficiency = std::stod(eff_str);

        std::string vec_str = extract("vectorization_ratio");
        if (!vec_str.empty()) m.vectorization_ratio = std::stod(vec_str);

        if (!m.kernel_name.empty()) {
            baseline_.push_back(m);
        }
    }

    has_baseline_ = !baseline_.empty();
    return has_baseline_;
}

// ============================================================================
// Baseline file I/O
// ============================================================================

bool write_baseline_file(const BaselineFile& baseline, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "{\n";
    file << "  \"version\": \"" << baseline.version << "\",\n";
    file << "  \"created_at\": \"" << baseline.created_at << "\",\n";
    file << "  \"git_commit\": \"" << baseline.git_commit << "\",\n";
    file << "  \"git_branch\": \"" << baseline.git_branch << "\",\n";
    file << "  \"cpu_model\": \"" << baseline.cpu_model << "\",\n";
    file << "  \"os_version\": \"" << baseline.os_version << "\",\n";
    file << "  \"measurements\": [\n";

    for (size_t i = 0; i < baseline.measurements.size(); ++i) {
        const auto& m = baseline.measurements[i];
        file << "    {\n";
        file << "      \"kernel_name\": \"" << m.kernel_name << "\",\n";
        file << "      \"variant_name\": \"" << m.variant_name << "\",\n";
        file << "      \"problem_size\": " << m.problem_size << ",\n";
        file << "      \"gflops\": " << m.gflops << ",\n";
        file << "      \"elapsed_seconds\": " << m.elapsed_seconds << ",\n";
        file << "      \"efficiency\": " << m.efficiency << ",\n";
        file << "      \"vectorization_ratio\": " << m.vectorization_ratio << "\n";
        file << "    }";
        if (i < baseline.measurements.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    return true;
}

std::optional<BaselineFile> read_baseline_file(const std::string& filepath) {
    RegressionTracker tracker;
    if (!tracker.load_baseline(filepath)) {
        return std::nullopt;
    }

    // For now, return a basic baseline file
    // Full implementation would parse all fields
    BaselineFile baseline;
    baseline.version = "1.0";
    // measurements would be extracted from tracker

    return baseline;
}

// ============================================================================
// Trend analysis
// ============================================================================

std::vector<TrendAnalysis> analyze_trends(
    const std::vector<BaselineFile>& history,
    size_t min_samples
) {
    std::vector<TrendAnalysis> trends;

    if (history.size() < min_samples) {
        return trends;
    }

    // Build map of kernel -> measurements over time
    std::map<std::string, std::vector<std::pair<std::string, double>>> kernel_history;

    for (const auto& baseline : history) {
        for (const auto& m : baseline.measurements) {
            std::string key = m.kernel_name + "/" + m.variant_name;
            kernel_history[key].emplace_back(baseline.created_at, m.gflops);
        }
    }

    // Analyze each kernel
    for (const auto& [kernel, samples] : kernel_history) {
        if (samples.size() < min_samples) continue;

        TrendAnalysis trend;
        trend.kernel_name = kernel;

        for (const auto& [ts, gflops] : samples) {
            trend.timestamps.push_back(ts);
            trend.gflops_history.push_back(gflops);
        }

        // Calculate statistics
        double sum = 0.0;
        for (double g : trend.gflops_history) sum += g;
        trend.mean_gflops = sum / trend.gflops_history.size();

        double sq_diff_sum = 0.0;
        for (double g : trend.gflops_history) {
            sq_diff_sum += (g - trend.mean_gflops) * (g - trend.mean_gflops);
        }
        trend.stddev_gflops = std::sqrt(sq_diff_sum / trend.gflops_history.size());

        trend.coefficient_of_variation =
            (trend.mean_gflops > 0) ? trend.stddev_gflops / trend.mean_gflops : 0.0;

        trend.is_stable = (trend.coefficient_of_variation < 0.05);

        // Simple trend detection (linear regression slope)
        if (samples.size() >= 3) {
            double first_avg = (samples[0].second + samples[1].second) / 2.0;
            double last_avg = (samples[samples.size()-1].second +
                               samples[samples.size()-2].second) / 2.0;

            double change = (last_avg - first_avg) / first_avg;
            if (change > 0.05) {
                trend.is_trending_up = true;
                trend.trend_description = "Improving over time (+" +
                    std::to_string(static_cast<int>(change * 100)) + "%)";
            } else if (change < -0.05) {
                trend.is_trending_down = true;
                trend.trend_description = "Degrading over time (" +
                    std::to_string(static_cast<int>(change * 100)) + "%)";
            } else {
                trend.trend_description = "Stable";
            }
        }

        trends.push_back(trend);
    }

    return trends;
}

// ============================================================================
// Performance gates
// ============================================================================

std::vector<GateResult> check_performance_gates(
    const std::vector<BenchmarkResult>& results,
    const std::vector<PerformanceGate>& gates
) {
    std::vector<GateResult> gate_results;

    for (const auto& gate : gates) {
        GateResult result;
        result.gate_name = gate.name;
        result.threshold_gflops = gate.threshold_gflops;
        result.achieved_gflops = 0.0;

        // Find matching benchmark
        bool found = false;
        for (const auto& bench : results) {
            if (bench.kernel_name == gate.name ||
                bench.kernel_name.find(gate.name) != std::string::npos) {

                // Get best variant performance
                for (const auto& v : bench.results) {
                    result.achieved_gflops = std::max(
                        result.achieved_gflops,
                        v.metrics.performance.gflops);
                }
                found = true;
                break;
            }
        }

        if (!found) {
            result.passed = !gate.required;
            result.message = "Benchmark not found";
            gate_results.push_back(result);
            continue;
        }

        // Check gate condition
        if (gate.comparison == ">=") {
            result.passed = (result.achieved_gflops >= gate.threshold_gflops);
        } else if (gate.comparison == ">") {
            result.passed = (result.achieved_gflops > gate.threshold_gflops);
        } else if (gate.comparison == "<=") {
            result.passed = (result.achieved_gflops <= gate.threshold_gflops);
        } else if (gate.comparison == "<") {
            result.passed = (result.achieved_gflops < gate.threshold_gflops);
        } else {
            result.passed = (result.achieved_gflops >= gate.threshold_gflops);
        }

        if (result.passed) {
            result.message = "PASSED: " + std::to_string(result.achieved_gflops) +
                             " GFLOPS " + gate.comparison + " " +
                             std::to_string(gate.threshold_gflops) + " GFLOPS";
        } else {
            result.message = "FAILED: " + std::to_string(result.achieved_gflops) +
                             " GFLOPS does not meet threshold of " +
                             std::to_string(gate.threshold_gflops) + " GFLOPS";
        }

        gate_results.push_back(result);
    }

    return gate_results;
}

}  // namespace simd_bench
