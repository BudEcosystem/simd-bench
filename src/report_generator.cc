#include "simd_bench/report_generator.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>

namespace simd_bench {

// Factory implementation
std::unique_ptr<IReportGenerator> ReportGeneratorFactory::create(ReportFormat format) {
    switch (format) {
        case ReportFormat::JSON:
            return std::make_unique<JSONReportGenerator>();
        case ReportFormat::HTML:
            return std::make_unique<HTMLReportGenerator>();
        case ReportFormat::MARKDOWN:
            return std::make_unique<MarkdownReportGenerator>();
        case ReportFormat::CSV:
            return std::make_unique<CSVReportGenerator>();
        default:
            return std::make_unique<JSONReportGenerator>();
    }
}

std::unique_ptr<IReportGenerator> ReportGeneratorFactory::create(const std::string& format_name) {
    std::string lower = format_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "json") return create(ReportFormat::JSON);
    if (lower == "html") return create(ReportFormat::HTML);
    if (lower == "markdown" || lower == "md") return create(ReportFormat::MARKDOWN);
    if (lower == "csv") return create(ReportFormat::CSV);

    return create(ReportFormat::JSON);
}

// JSON Report Generator
JSONReportGenerator::JSONReportGenerator() {
    json_["metadata"] = {
        {"tool", "simd-bench"},
        {"version", "1.0.0"},
        {"timestamp", ""}
    };
}

void JSONReportGenerator::set_config(const ReportConfig& config) {
    config_ = config;
    json_["metadata"]["title"] = config.title;
}

void JSONReportGenerator::add_hardware_info(const HardwareInfo& hw) {
    json_["hardware"] = {
        {"cpu", hw.cpu_brand},
        {"vendor", hw.cpu_vendor},
        {"architecture", hw.architecture},
        {"physical_cores", hw.physical_cores},
        {"logical_cores", hw.logical_cores},
        {"base_frequency_ghz", hw.base_frequency_ghz},
        {"max_frequency_ghz", hw.max_frequency_ghz},
        {"measured_frequency_ghz", hw.measured_frequency_ghz},
        {"simd_extensions", hw.get_simd_string()},
        {"max_vector_bits", hw.max_vector_bits},
        {"l1_cache_kb", hw.cache.l1d_size_kb},
        {"l2_cache_kb", hw.cache.l2_size_kb},
        {"l3_cache_kb", hw.cache.l3_size_kb},
        {"memory_bandwidth_gbps", hw.measured_memory_bw_gbps},
        {"theoretical_peak_sp_gflops", hw.theoretical_peak_sp_gflops},
        {"theoretical_peak_dp_gflops", hw.theoretical_peak_dp_gflops}
    };
}

void JSONReportGenerator::add_benchmark_result(const BenchmarkResult& result) {
    nlohmann::json kernel_json;
    kernel_json["name"] = result.kernel_name;
    kernel_json["best_variant"] = result.best_variant;
    kernel_json["speedup_vs_scalar"] = result.speedup_vs_scalar;
    kernel_json["avg_vectorization_ratio"] = result.avg_vectorization_ratio;

    nlohmann::json variants_json = nlohmann::json::array();
    for (const auto& vr : result.results) {
        nlohmann::json vr_json;
        vr_json["variant"] = vr.variant_name;
        vr_json["size"] = vr.problem_size;
        vr_json["gflops"] = vr.metrics.performance.gflops;
        vr_json["elapsed_seconds"] = vr.metrics.performance.elapsed_seconds;
        vr_json["arithmetic_intensity"] = vr.roofline.arithmetic_intensity;
        vr_json["efficiency"] = vr.roofline.efficiency;
        vr_json["bound"] = vr.roofline.bound;
        variants_json.push_back(vr_json);
    }

    kernel_json["results"] = variants_json;
    json_["kernels"][result.kernel_name] = kernel_json;
}

void JSONReportGenerator::add_roofline_model(const RooflineModel& model,
                                              const std::vector<RooflinePoint>& points) {
    nlohmann::json roofline_json;

    nlohmann::json ceilings_json = nlohmann::json::array();
    for (const auto& ceiling : model.get_ceilings()) {
        ceilings_json.push_back({
            {"name", ceiling.name},
            {"bandwidth_gbps", ceiling.bandwidth_gbps},
            {"is_compute", ceiling.is_compute_ceiling}
        });
    }
    roofline_json["ceilings"] = ceilings_json;

    nlohmann::json points_json = nlohmann::json::array();
    for (const auto& point : points) {
        points_json.push_back({
            {"arithmetic_intensity", point.arithmetic_intensity},
            {"gflops", point.achieved_gflops},
            {"bound", point.bound},
            {"efficiency", point.efficiency}
        });
    }
    roofline_json["points"] = points_json;

    json_["roofline"] = roofline_json;
}

void JSONReportGenerator::add_tma_result(const TMAResult& tma) {
    json_["tma"] = {
        {"retiring", tma.metrics.retiring},
        {"bad_speculation", tma.metrics.bad_speculation},
        {"frontend_bound", tma.metrics.frontend_bound},
        {"backend_bound", tma.metrics.backend_bound},
        {"memory_bound", tma.metrics.memory_bound},
        {"core_bound", tma.metrics.core_bound},
        {"primary_bottleneck", tma.primary_bottleneck},
        {"recommendations", tma.recommendations}
    };
}

void JSONReportGenerator::add_summary(const std::string& summary) {
    json_["summary"] = summary;
}

void JSONReportGenerator::add_recommendations(const std::vector<std::string>& recommendations) {
    json_["recommendations"] = recommendations;
}

std::string JSONReportGenerator::generate() {
    // Update timestamp
    auto now = std::time(nullptr);
    json_["metadata"]["timestamp"] = std::ctime(&now);

    if (config_.pretty_print) {
        return json_.dump(config_.indent);
    }
    return json_.dump();
}

void JSONReportGenerator::generate_to_file(const std::string& path) {
    std::ofstream file(path);
    file << generate();
}

// HTML Report Generator
HTMLReportGenerator::HTMLReportGenerator() {}

void HTMLReportGenerator::set_config(const ReportConfig& config) {
    config_ = config;
}

void HTMLReportGenerator::add_hardware_info(const HardwareInfo& hw) {
    std::ostringstream oss;
    oss << "<section id=\"hardware\">\n";
    oss << "  <h2>Hardware Configuration</h2>\n";
    oss << "  <table class=\"info-table\">\n";
    oss << "    <tr><td>CPU</td><td>" << hw.cpu_brand << "</td></tr>\n";
    oss << "    <tr><td>Architecture</td><td>" << hw.architecture << "</td></tr>\n";
    oss << "    <tr><td>Cores</td><td>" << hw.physical_cores << " physical / "
        << hw.logical_cores << " logical</td></tr>\n";
    oss << "    <tr><td>Frequency</td><td>" << std::fixed << std::setprecision(2)
        << hw.measured_frequency_ghz << " GHz</td></tr>\n";
    oss << "    <tr><td>SIMD</td><td>" << hw.get_simd_string() << "</td></tr>\n";
    oss << "    <tr><td>Vector Width</td><td>" << hw.max_vector_bits << " bits</td></tr>\n";
    oss << "    <tr><td>L1/L2/L3 Cache</td><td>" << hw.cache.l1d_size_kb << " / "
        << hw.cache.l2_size_kb << " / " << hw.cache.l3_size_kb << " KB</td></tr>\n";
    oss << "    <tr><td>Memory Bandwidth</td><td>" << std::fixed << std::setprecision(1)
        << hw.measured_memory_bw_gbps << " GB/s</td></tr>\n";
    oss << "    <tr><td>Peak GFLOPS (SP)</td><td>" << std::fixed << std::setprecision(1)
        << hw.theoretical_peak_sp_gflops << "</td></tr>\n";
    oss << "  </table>\n";
    oss << "</section>\n";
    hardware_section_ = oss.str();
}

void HTMLReportGenerator::add_benchmark_result(const BenchmarkResult& result) {
    std::ostringstream oss;
    oss << "<section class=\"kernel-result\">\n";
    oss << "  <h3>" << result.kernel_name << "</h3>\n";
    oss << "  <p>Best variant: <strong>" << result.best_variant
        << "</strong> (" << std::fixed << std::setprecision(1)
        << result.speedup_vs_scalar << "x speedup vs scalar)</p>\n";

    oss << "  <table class=\"results-table\">\n";
    oss << "    <thead><tr><th>Variant</th><th>Size</th><th>GFLOPS</th><th>Time (s)</th><th>Efficiency</th></tr></thead>\n";
    oss << "    <tbody>\n";

    for (const auto& vr : result.results) {
        oss << "      <tr><td>" << vr.variant_name << "</td>"
            << "<td>" << vr.problem_size << "</td>"
            << "<td>" << std::fixed << std::setprecision(2) << vr.metrics.performance.gflops << "</td>"
            << "<td>" << std::scientific << std::setprecision(3) << vr.metrics.performance.elapsed_seconds << "</td>"
            << "<td>" << std::fixed << std::setprecision(1) << (vr.roofline.efficiency * 100) << "%</td></tr>\n";
    }

    oss << "    </tbody>\n";
    oss << "  </table>\n";
    oss << "</section>\n";

    benchmark_sections_.push_back(oss.str());
}

void HTMLReportGenerator::add_roofline_model(const RooflineModel& model,
                                              const std::vector<RooflinePoint>& points) {
    std::ostringstream oss;
    oss << "<section id=\"roofline\">\n";
    oss << "  <h2>Roofline Analysis</h2>\n";
    oss << "  " << model.generate_svg(points) << "\n";
    oss << "</section>\n";
    roofline_section_ = oss.str();
}

void HTMLReportGenerator::add_tma_result(const TMAResult& tma) {
    std::ostringstream oss;
    oss << "<section id=\"tma\">\n";
    oss << "  <h2>Top-Down Microarchitecture Analysis</h2>\n";
    oss << "  <pre>" << format_tma_bar_chart(tma) << "</pre>\n";
    oss << "</section>\n";
    tma_section_ = oss.str();
}

void HTMLReportGenerator::add_summary(const std::string& summary) {
    summary_section_ = "<section id=\"summary\"><h2>Summary</h2><p>" + summary + "</p></section>\n";
}

void HTMLReportGenerator::add_recommendations(const std::vector<std::string>& recommendations) {
    std::ostringstream oss;
    oss << "<section id=\"recommendations\">\n";
    oss << "  <h2>Recommendations</h2>\n";
    oss << "  <ul>\n";
    for (const auto& rec : recommendations) {
        oss << "    <li>" << rec << "</li>\n";
    }
    oss << "  </ul>\n";
    oss << "</section>\n";
    recommendations_section_ = oss.str();
}

std::string HTMLReportGenerator::generate_css() const {
    return R"(
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
    h2 { color: #555; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #007bff; color: white; }
    tr:nth-child(even) { background-color: #f9f9f9; }
    tr:hover { background-color: #f5f5f5; }
    .info-table td:first-child { font-weight: bold; width: 200px; }
    pre { background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; }
    ul { line-height: 1.8; }
    )";
}

std::string HTMLReportGenerator::generate_header() const {
    std::ostringstream oss;
    oss << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
    oss << "  <meta charset=\"UTF-8\">\n";
    oss << "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
    oss << "  <title>" << config_.title << "</title>\n";
    oss << "  <style>" << generate_css() << "</style>\n";
    oss << "</head>\n<body>\n<div class=\"container\">\n";
    oss << "  <h1>" << config_.title << "</h1>\n";
    return oss.str();
}

std::string HTMLReportGenerator::generate_footer() const {
    auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << "  <footer><p>Generated: " << std::ctime(&now) << "</p></footer>\n";
    oss << "</div>\n</body>\n</html>\n";
    return oss.str();
}

std::string HTMLReportGenerator::generate() {
    std::ostringstream oss;
    oss << generate_header();
    oss << hardware_section_;
    oss << summary_section_;

    oss << "<section id=\"kernels\"><h2>Kernel Results</h2>\n";
    for (const auto& section : benchmark_sections_) {
        oss << section;
    }
    oss << "</section>\n";

    oss << roofline_section_;
    oss << tma_section_;
    oss << recommendations_section_;
    oss << generate_footer();

    return oss.str();
}

void HTMLReportGenerator::generate_to_file(const std::string& path) {
    std::ofstream file(path);
    file << generate();
}

// Markdown Report Generator
MarkdownReportGenerator::MarkdownReportGenerator() {}

void MarkdownReportGenerator::set_config(const ReportConfig& config) {
    config_ = config;
}

void MarkdownReportGenerator::add_hardware_info(const HardwareInfo& hw) {
    content_ << "# " << config_.title << "\n\n";
    content_ << "## Hardware Configuration\n\n";
    content_ << "| Property | Value |\n";
    content_ << "|----------|-------|\n";
    content_ << "| CPU | " << hw.cpu_brand << " |\n";
    content_ << "| Cores | " << hw.physical_cores << " physical / " << hw.logical_cores << " logical |\n";
    content_ << "| Frequency | " << std::fixed << std::setprecision(2) << hw.measured_frequency_ghz << " GHz |\n";
    content_ << "| SIMD | " << hw.get_simd_string() << " |\n";
    content_ << "| Peak GFLOPS | " << std::fixed << std::setprecision(1) << hw.theoretical_peak_sp_gflops << " |\n";
    content_ << "\n";
}

void MarkdownReportGenerator::add_benchmark_result(const BenchmarkResult& result) {
    content_ << "## " << result.kernel_name << "\n\n";
    content_ << "Best variant: **" << result.best_variant << "** ("
             << std::fixed << std::setprecision(1) << result.speedup_vs_scalar << "x speedup)\n\n";

    content_ << "| Variant | Size | GFLOPS | Efficiency |\n";
    content_ << "|---------|------|--------|------------|\n";

    for (const auto& vr : result.results) {
        content_ << "| " << vr.variant_name
                 << " | " << vr.problem_size
                 << " | " << std::fixed << std::setprecision(2) << vr.metrics.performance.gflops
                 << " | " << std::fixed << std::setprecision(1) << (vr.roofline.efficiency * 100) << "% |\n";
    }
    content_ << "\n";
}

void MarkdownReportGenerator::add_roofline_model(const RooflineModel&,
                                                  const std::vector<RooflinePoint>&) {
    content_ << "## Roofline Analysis\n\n";
    content_ << "*See generated SVG file for roofline plot*\n\n";
}

void MarkdownReportGenerator::add_tma_result(const TMAResult& tma) {
    content_ << "## Top-Down Analysis\n\n";
    content_ << "```\n" << format_tma_bar_chart(tma) << "```\n\n";
}

void MarkdownReportGenerator::add_summary(const std::string& summary) {
    content_ << "## Summary\n\n" << summary << "\n\n";
}

void MarkdownReportGenerator::add_recommendations(const std::vector<std::string>& recommendations) {
    content_ << "## Recommendations\n\n";
    for (const auto& rec : recommendations) {
        content_ << "- " << rec << "\n";
    }
    content_ << "\n";
}

std::string MarkdownReportGenerator::generate() {
    return content_.str();
}

void MarkdownReportGenerator::generate_to_file(const std::string& path) {
    std::ofstream file(path);
    file << generate();
}

// CSV Report Generator
CSVReportGenerator::CSVReportGenerator() {
    rows_.push_back({"kernel", "variant", "size", "gflops", "elapsed_s", "ai", "efficiency", "bound"});
}

void CSVReportGenerator::set_config(const ReportConfig& config) {
    config_ = config;
}

void CSVReportGenerator::add_hardware_info(const HardwareInfo&) {}

void CSVReportGenerator::add_benchmark_result(const BenchmarkResult& result) {
    for (const auto& vr : result.results) {
        rows_.push_back({
            result.kernel_name,
            vr.variant_name,
            std::to_string(vr.problem_size),
            std::to_string(vr.metrics.performance.gflops),
            std::to_string(vr.metrics.performance.elapsed_seconds),
            std::to_string(vr.roofline.arithmetic_intensity),
            std::to_string(vr.roofline.efficiency),
            vr.roofline.bound
        });
    }
}

void CSVReportGenerator::add_roofline_model(const RooflineModel&, const std::vector<RooflinePoint>&) {}
void CSVReportGenerator::add_tma_result(const TMAResult&) {}
void CSVReportGenerator::add_summary(const std::string&) {}
void CSVReportGenerator::add_recommendations(const std::vector<std::string>&) {}

std::string CSVReportGenerator::generate() {
    std::ostringstream oss;
    for (const auto& row : rows_) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) oss << ",";
            oss << row[i];
        }
        oss << "\n";
    }
    return oss.str();
}

void CSVReportGenerator::generate_to_file(const std::string& path) {
    std::ofstream file(path);
    file << generate();
}

// Format utilities
namespace format {

std::string format_gflops(double gflops, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << gflops << " GFLOPS";
    return oss.str();
}

std::string format_bandwidth(double gbps, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << gbps << " GB/s";
    return oss.str();
}

std::string format_percentage(double ratio, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << (ratio * 100) << "%";
    return oss.str();
}

std::string format_time(double seconds, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << seconds << " s";
    return oss.str();
}

std::string format_energy(double joules, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << joules << " J";
    return oss.str();
}

std::string format_power(double watts, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << watts << " W";
    return oss.str();
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && unit < 4) {
        size /= 1024;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

std::string format_count(uint64_t count) {
    std::string str = std::to_string(count);
    std::string result;

    int n = 0;
    for (auto it = str.rbegin(); it != str.rend(); ++it) {
        if (n > 0 && n % 3 == 0) {
            result = ',' + result;
        }
        result = *it + result;
        n++;
    }

    return result;
}

std::string ascii_bar(double value, double max_value, int width) {
    int filled = static_cast<int>((value / max_value) * width);
    filled = std::clamp(filled, 0, width);
    return std::string(filled, '#') + std::string(width - filled, '-');
}

std::string ascii_table(
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
) {
    // Calculate column widths
    std::vector<size_t> widths(headers.size(), 0);
    for (size_t i = 0; i < headers.size(); ++i) {
        widths[i] = headers[i].size();
    }
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size() && i < widths.size(); ++i) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }

    std::ostringstream oss;

    // Header
    oss << "|";
    for (size_t i = 0; i < headers.size(); ++i) {
        oss << " " << std::setw(widths[i]) << std::left << headers[i] << " |";
    }
    oss << "\n|";
    for (size_t i = 0; i < headers.size(); ++i) {
        oss << std::string(widths[i] + 2, '-') << "|";
    }
    oss << "\n";

    // Rows
    for (const auto& row : rows) {
        oss << "|";
        for (size_t i = 0; i < row.size(); ++i) {
            oss << " " << std::setw(widths[i]) << std::left << row[i] << " |";
        }
        oss << "\n";
    }

    return oss.str();
}

}  // namespace format

// Regression comparison
RegressionReport compare_results(
    const nlohmann::json& baseline,
    const nlohmann::json& current,
    double threshold
) {
    RegressionReport report;

    if (!baseline.contains("kernels") || !current.contains("kernels")) {
        return report;
    }

    for (auto& [name, kernel] : current["kernels"].items()) {
        if (!baseline["kernels"].contains(name)) {
            continue;
        }

        double baseline_gflops = baseline["kernels"][name].value("gflops", 0.0);
        double current_gflops = kernel.value("gflops", 0.0);

        if (baseline_gflops <= 0) continue;

        double change = (current_gflops - baseline_gflops) / baseline_gflops;

        RegressionReport::KernelComparison comp;
        comp.kernel_name = name;
        comp.baseline_gflops = baseline_gflops;
        comp.current_gflops = current_gflops;
        comp.change_percent = change * 100;
        comp.is_regression = (change < -threshold);

        if (change < -0.15) {
            comp.severity = "critical";
            report.critical_count++;
        } else if (change < -threshold) {
            comp.severity = "warning";
            report.warning_count++;
        } else {
            comp.severity = "none";
        }

        if (comp.is_regression) {
            report.has_regressions = true;
        }

        report.comparisons.push_back(comp);
    }

    return report;
}

std::string format_regression_report(const RegressionReport& report) {
    std::ostringstream oss;

    oss << "Regression Report\n";
    oss << "=================\n\n";

    if (report.has_regressions) {
        oss << "WARNING: Performance regression detected!\n";
        oss << "(" << report.critical_count << " critical, "
            << report.warning_count << " warnings)\n\n";
    } else {
        oss << "No regression detected - all benchmarks within tolerance\n\n";
    }

    oss << "| Kernel | Baseline | Current | Change |\n";
    oss << "|--------|----------|---------|--------|\n";

    for (const auto& comp : report.comparisons) {
        oss << "| " << comp.kernel_name
            << " | " << std::fixed << std::setprecision(2) << comp.baseline_gflops
            << " | " << comp.current_gflops
            << " | " << (comp.change_percent >= 0 ? "+" : "") << comp.change_percent << "%";

        if (comp.severity == "critical") {
            oss << " ❌";
        } else if (comp.severity == "warning") {
            oss << " ⚠️";
        }
        oss << " |\n";
    }

    return oss.str();
}

}  // namespace simd_bench
