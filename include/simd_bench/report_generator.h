#pragma once

#include "types.h"
#include "hardware.h"
#include "roofline.h"
#include "tma.h"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <nlohmann/json.hpp>

namespace simd_bench {

// Forward declarations
class RooflineModel;

// Report configuration
struct ReportConfig {
    std::string title = "SIMD-Bench Report";
    std::string author;
    bool include_hardware_info = true;
    bool include_roofline = true;
    bool include_tma = true;
    bool include_energy = true;
    bool include_recommendations = true;
    bool include_raw_data = false;

    // HTML-specific
    bool embed_charts = true;
    std::string css_theme = "default";

    // Markdown-specific
    bool github_flavored = true;

    // JSON-specific
    bool pretty_print = true;
    int indent = 2;
};

// Abstract report generator interface
class IReportGenerator {
public:
    virtual ~IReportGenerator() = default;

    virtual void set_config(const ReportConfig& config) = 0;

    virtual void add_hardware_info(const HardwareInfo& hw) = 0;
    virtual void add_benchmark_result(const BenchmarkResult& result) = 0;
    virtual void add_roofline_model(const RooflineModel& model,
                                    const std::vector<RooflinePoint>& points) = 0;
    virtual void add_tma_result(const TMAResult& tma) = 0;
    virtual void add_summary(const std::string& summary) = 0;
    virtual void add_recommendations(const std::vector<std::string>& recommendations) = 0;

    virtual std::string generate() = 0;
    virtual void generate_to_file(const std::string& path) = 0;

    virtual ReportFormat get_format() const = 0;
};

// Factory for report generators
class ReportGeneratorFactory {
public:
    static std::unique_ptr<IReportGenerator> create(ReportFormat format);
    static std::unique_ptr<IReportGenerator> create(const std::string& format_name);
};

// JSON report generator
class JSONReportGenerator : public IReportGenerator {
public:
    JSONReportGenerator();

    void set_config(const ReportConfig& config) override;

    void add_hardware_info(const HardwareInfo& hw) override;
    void add_benchmark_result(const BenchmarkResult& result) override;
    void add_roofline_model(const RooflineModel& model,
                           const std::vector<RooflinePoint>& points) override;
    void add_tma_result(const TMAResult& tma) override;
    void add_summary(const std::string& summary) override;
    void add_recommendations(const std::vector<std::string>& recommendations) override;

    std::string generate() override;
    void generate_to_file(const std::string& path) override;

    ReportFormat get_format() const override { return ReportFormat::JSON; }

    // Get raw JSON object
    const nlohmann::json& get_json() const { return json_; }

private:
    ReportConfig config_;
    nlohmann::json json_;
};

// HTML report generator
class HTMLReportGenerator : public IReportGenerator {
public:
    HTMLReportGenerator();

    void set_config(const ReportConfig& config) override;

    void add_hardware_info(const HardwareInfo& hw) override;
    void add_benchmark_result(const BenchmarkResult& result) override;
    void add_roofline_model(const RooflineModel& model,
                           const std::vector<RooflinePoint>& points) override;
    void add_tma_result(const TMAResult& tma) override;
    void add_summary(const std::string& summary) override;
    void add_recommendations(const std::vector<std::string>& recommendations) override;

    std::string generate() override;
    void generate_to_file(const std::string& path) override;

    ReportFormat get_format() const override { return ReportFormat::HTML; }

private:
    ReportConfig config_;
    std::string hardware_section_;
    std::vector<std::string> benchmark_sections_;
    std::string roofline_section_;
    std::string tma_section_;
    std::string summary_section_;
    std::string recommendations_section_;

    std::string generate_header() const;
    std::string generate_css() const;
    std::string generate_js() const;
    std::string generate_footer() const;
};

// Markdown report generator
class MarkdownReportGenerator : public IReportGenerator {
public:
    MarkdownReportGenerator();

    void set_config(const ReportConfig& config) override;

    void add_hardware_info(const HardwareInfo& hw) override;
    void add_benchmark_result(const BenchmarkResult& result) override;
    void add_roofline_model(const RooflineModel& model,
                           const std::vector<RooflinePoint>& points) override;
    void add_tma_result(const TMAResult& tma) override;
    void add_summary(const std::string& summary) override;
    void add_recommendations(const std::vector<std::string>& recommendations) override;

    std::string generate() override;
    void generate_to_file(const std::string& path) override;

    ReportFormat get_format() const override { return ReportFormat::MARKDOWN; }

private:
    ReportConfig config_;
    std::ostringstream content_;

    std::string format_table(
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& rows
    ) const;
};

// CSV report generator (for raw data export)
class CSVReportGenerator : public IReportGenerator {
public:
    CSVReportGenerator();

    void set_config(const ReportConfig& config) override;

    void add_hardware_info(const HardwareInfo& hw) override;
    void add_benchmark_result(const BenchmarkResult& result) override;
    void add_roofline_model(const RooflineModel& model,
                           const std::vector<RooflinePoint>& points) override;
    void add_tma_result(const TMAResult& tma) override;
    void add_summary(const std::string& summary) override;
    void add_recommendations(const std::vector<std::string>& recommendations) override;

    std::string generate() override;
    void generate_to_file(const std::string& path) override;

    ReportFormat get_format() const override { return ReportFormat::CSV; }

private:
    ReportConfig config_;
    std::vector<std::vector<std::string>> rows_;
};

// Utility functions for formatting
namespace format {

std::string format_gflops(double gflops, int precision = 2);
std::string format_bandwidth(double gbps, int precision = 2);
std::string format_percentage(double ratio, int precision = 1);
std::string format_time(double seconds, int precision = 3);
std::string format_energy(double joules, int precision = 3);
std::string format_power(double watts, int precision = 2);
std::string format_bytes(size_t bytes);
std::string format_count(uint64_t count);

// Generate ASCII bar chart
std::string ascii_bar(double value, double max_value, int width = 40);

// Generate ASCII table
std::string ascii_table(
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
);

}  // namespace format

// Comparison report for regression testing
struct RegressionReport {
    std::string baseline_version;
    std::string current_version;

    struct KernelComparison {
        std::string kernel_name;
        double baseline_gflops;
        double current_gflops;
        double change_percent;
        bool is_regression;
        std::string severity;  // "none", "warning", "critical"
    };

    std::vector<KernelComparison> comparisons;
    bool has_regressions = false;
    int warning_count = 0;
    int critical_count = 0;
};

RegressionReport compare_results(
    const nlohmann::json& baseline,
    const nlohmann::json& current,
    double threshold = 0.05  // 5% regression threshold
);

std::string format_regression_report(const RegressionReport& report);

}  // namespace simd_bench
