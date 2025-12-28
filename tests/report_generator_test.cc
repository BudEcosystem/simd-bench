#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/report_generator.h"
#include <fstream>
#include <filesystem>

namespace simd_bench {
namespace testing {

class ReportGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create sample data
        hw_.cpu_brand = "Intel Core i7-10700K";
        hw_.cpu_vendor = "GenuineIntel";
        hw_.physical_cores = 8;
        hw_.logical_cores = 16;
        hw_.base_frequency_ghz = 3.8;
        hw_.max_vector_bits = 256;
        hw_.theoretical_peak_sp_gflops = 120.0;

        result_.kernel_name = "dot_product";
        result_.best_variant = "avx2_8x";
        result_.speedup_vs_scalar = 4.5;
        result_.avg_vectorization_ratio = 0.98;

        VariantResult vr;
        vr.variant_name = "scalar";
        vr.problem_size = 1024;
        vr.metrics.performance.gflops = 1.5;
        vr.roofline.arithmetic_intensity = 0.25;
        vr.roofline.achieved_gflops = 1.5;
        result_.results.push_back(vr);

        vr.variant_name = "avx2_8x";
        vr.metrics.performance.gflops = 6.75;
        vr.roofline.achieved_gflops = 6.75;
        result_.results.push_back(vr);
    }

    void TearDown() override {
        // Clean up temp files
        if (std::filesystem::exists(temp_path_)) {
            std::filesystem::remove(temp_path_);
        }
    }

    HardwareInfo hw_;
    BenchmarkResult result_;
    std::string temp_path_ = "/tmp/simd_bench_test_report";
};

// Test factory
TEST_F(ReportGeneratorTest, FactoryCreateJSON) {
    auto gen = ReportGeneratorFactory::create(ReportFormat::JSON);
    ASSERT_NE(gen, nullptr);
    EXPECT_EQ(gen->get_format(), ReportFormat::JSON);
}

TEST_F(ReportGeneratorTest, FactoryCreateHTML) {
    auto gen = ReportGeneratorFactory::create(ReportFormat::HTML);
    ASSERT_NE(gen, nullptr);
    EXPECT_EQ(gen->get_format(), ReportFormat::HTML);
}

TEST_F(ReportGeneratorTest, FactoryCreateMarkdown) {
    auto gen = ReportGeneratorFactory::create(ReportFormat::MARKDOWN);
    ASSERT_NE(gen, nullptr);
    EXPECT_EQ(gen->get_format(), ReportFormat::MARKDOWN);
}

TEST_F(ReportGeneratorTest, FactoryCreateCSV) {
    auto gen = ReportGeneratorFactory::create(ReportFormat::CSV);
    ASSERT_NE(gen, nullptr);
    EXPECT_EQ(gen->get_format(), ReportFormat::CSV);
}

TEST_F(ReportGeneratorTest, FactoryCreateByString) {
    auto json = ReportGeneratorFactory::create("json");
    auto html = ReportGeneratorFactory::create("html");
    auto md = ReportGeneratorFactory::create("markdown");

    ASSERT_NE(json, nullptr);
    ASSERT_NE(html, nullptr);
    ASSERT_NE(md, nullptr);
}

// Test JSON generator
TEST_F(ReportGeneratorTest, JSONGeneratorBasic) {
    JSONReportGenerator gen;
    gen.add_hardware_info(hw_);

    std::string json = gen.generate();

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("Intel Core i7-10700K"), std::string::npos);
}

TEST_F(ReportGeneratorTest, JSONGeneratorWithBenchmarkResult) {
    JSONReportGenerator gen;
    gen.add_hardware_info(hw_);
    gen.add_benchmark_result(result_);

    std::string json = gen.generate();

    EXPECT_NE(json.find("dot_product"), std::string::npos);
    EXPECT_NE(json.find("avx2_8x"), std::string::npos);
}

TEST_F(ReportGeneratorTest, JSONGeneratorParseable) {
    JSONReportGenerator gen;
    gen.add_hardware_info(hw_);
    gen.add_benchmark_result(result_);

    std::string json = gen.generate();

    // Parse should not throw
    EXPECT_NO_THROW({
        nlohmann::json parsed = nlohmann::json::parse(json);
        EXPECT_TRUE(parsed.contains("hardware") || parsed.contains("metadata"));
    });
}

TEST_F(ReportGeneratorTest, JSONGeneratorToFile) {
    JSONReportGenerator gen;
    gen.add_hardware_info(hw_);

    std::string path = temp_path_ + ".json";
    gen.generate_to_file(path);

    EXPECT_TRUE(std::filesystem::exists(path));

    std::ifstream file(path);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());

    std::filesystem::remove(path);
}

TEST_F(ReportGeneratorTest, JSONGeneratorWithRoofline) {
    JSONReportGenerator gen;

    RooflineModel model;
    model.set_peak_gflops(100.0);
    model.add_ceiling("DRAM", 20.0);

    std::vector<RooflinePoint> points = {
        {0.5, 8.0, "DRAM", 0.8}
    };

    gen.add_roofline_model(model, points);
    std::string json = gen.generate();

    EXPECT_NE(json.find("roofline"), std::string::npos);
}

TEST_F(ReportGeneratorTest, JSONGeneratorWithTMA) {
    JSONReportGenerator gen;

    TMAResult tma;
    tma.metrics.retiring = 0.75;
    tma.metrics.backend_bound = 0.15;
    tma.primary_bottleneck = "Backend Bound";

    gen.add_tma_result(tma);
    std::string json = gen.generate();

    EXPECT_NE(json.find("tma"), std::string::npos);
}

// Test HTML generator
TEST_F(ReportGeneratorTest, HTMLGeneratorBasic) {
    HTMLReportGenerator gen;
    gen.add_hardware_info(hw_);

    std::string html = gen.generate();

    EXPECT_FALSE(html.empty());
    EXPECT_NE(html.find("<!DOCTYPE html>"), std::string::npos);
    EXPECT_NE(html.find("<html"), std::string::npos);  // May have attributes like lang="en"
    EXPECT_NE(html.find("</html>"), std::string::npos);
}

TEST_F(ReportGeneratorTest, HTMLGeneratorIncludesCSS) {
    HTMLReportGenerator gen;

    std::string html = gen.generate();

    EXPECT_NE(html.find("<style>"), std::string::npos);
}

TEST_F(ReportGeneratorTest, HTMLGeneratorWithBenchmarkResult) {
    HTMLReportGenerator gen;
    gen.add_hardware_info(hw_);
    gen.add_benchmark_result(result_);

    std::string html = gen.generate();

    EXPECT_NE(html.find("dot_product"), std::string::npos);
}

// Test Markdown generator
TEST_F(ReportGeneratorTest, MarkdownGeneratorBasic) {
    MarkdownReportGenerator gen;
    gen.add_hardware_info(hw_);

    std::string md = gen.generate();

    EXPECT_FALSE(md.empty());
    EXPECT_NE(md.find("# "), std::string::npos);  // Has headings
}

TEST_F(ReportGeneratorTest, MarkdownGeneratorWithTable) {
    MarkdownReportGenerator gen;
    gen.add_benchmark_result(result_);

    std::string md = gen.generate();

    EXPECT_NE(md.find("|"), std::string::npos);  // Has table
}

TEST_F(ReportGeneratorTest, MarkdownGeneratorWithRecommendations) {
    MarkdownReportGenerator gen;
    gen.add_recommendations({"Use FMA instructions", "Improve cache locality"});

    std::string md = gen.generate();

    EXPECT_NE(md.find("FMA"), std::string::npos);
    EXPECT_NE(md.find("cache"), std::string::npos);
}

// Test CSV generator
TEST_F(ReportGeneratorTest, CSVGeneratorBasic) {
    CSVReportGenerator gen;
    gen.add_benchmark_result(result_);

    std::string csv = gen.generate();

    EXPECT_FALSE(csv.empty());
    EXPECT_NE(csv.find(","), std::string::npos);
}

// Test format utilities
TEST_F(ReportGeneratorTest, FormatGflops) {
    EXPECT_EQ(format::format_gflops(123.456), "123.46 GFLOPS");
    EXPECT_EQ(format::format_gflops(1.5, 1), "1.5 GFLOPS");
}

TEST_F(ReportGeneratorTest, FormatBandwidth) {
    EXPECT_EQ(format::format_bandwidth(25.5), "25.50 GB/s");
}

TEST_F(ReportGeneratorTest, FormatPercentage) {
    EXPECT_EQ(format::format_percentage(0.85), "85.0%");
    EXPECT_EQ(format::format_percentage(0.999, 2), "99.90%");
}

TEST_F(ReportGeneratorTest, FormatTime) {
    EXPECT_EQ(format::format_time(0.001), "0.001 s");
    EXPECT_EQ(format::format_time(1.5, 1), "1.5 s");
}

TEST_F(ReportGeneratorTest, FormatBytes) {
    EXPECT_EQ(format::format_bytes(1024), "1.00 KB");
    EXPECT_EQ(format::format_bytes(1048576), "1.00 MB");
    EXPECT_EQ(format::format_bytes(1073741824), "1.00 GB");
}

TEST_F(ReportGeneratorTest, FormatCount) {
    EXPECT_EQ(format::format_count(1000), "1,000");
    EXPECT_EQ(format::format_count(1000000), "1,000,000");
}

TEST_F(ReportGeneratorTest, ASCIIBar) {
    std::string bar = format::ascii_bar(0.5, 1.0, 20);
    EXPECT_EQ(bar.length(), 20u);
}

TEST_F(ReportGeneratorTest, ASCIITable) {
    std::vector<std::string> headers = {"Name", "Value"};
    std::vector<std::vector<std::string>> rows = {
        {"A", "1"},
        {"B", "2"}
    };

    std::string table = format::ascii_table(headers, rows);
    EXPECT_NE(table.find("Name"), std::string::npos);
    EXPECT_NE(table.find("Value"), std::string::npos);
}

// Test regression report
TEST_F(ReportGeneratorTest, CompareResultsNoRegression) {
    nlohmann::json baseline = {
        {"kernels", {
            {"dot_product", {{"gflops", 10.0}}}
        }}
    };

    nlohmann::json current = {
        {"kernels", {
            {"dot_product", {{"gflops", 10.5}}}  // 5% improvement
        }}
    };

    RegressionReport report = compare_results(baseline, current, 0.05);

    EXPECT_FALSE(report.has_regressions);
    EXPECT_EQ(report.critical_count, 0);
}

TEST_F(ReportGeneratorTest, CompareResultsWithRegression) {
    nlohmann::json baseline = {
        {"kernels", {
            {"dot_product", {{"gflops", 10.0}}}
        }}
    };

    nlohmann::json current = {
        {"kernels", {
            {"dot_product", {{"gflops", 8.0}}}  // 20% regression
        }}
    };

    RegressionReport report = compare_results(baseline, current, 0.05);

    EXPECT_TRUE(report.has_regressions);
    EXPECT_GT(report.critical_count + report.warning_count, 0);
}

TEST_F(ReportGeneratorTest, FormatRegressionReport) {
    RegressionReport report;
    report.baseline_version = "v1.0.0";
    report.current_version = "v1.1.0";
    report.has_regressions = true;
    report.comparisons.push_back({
        "dot_product", 10.0, 8.0, -20.0, true, "critical"
    });

    std::string formatted = format_regression_report(report);

    EXPECT_NE(formatted.find("dot_product"), std::string::npos);
    EXPECT_NE(formatted.find("regression"), std::string::npos);
}

// Test report config
TEST_F(ReportGeneratorTest, ReportConfigApplied) {
    ReportConfig config;
    config.title = "Custom Title";
    config.include_roofline = false;

    JSONReportGenerator gen;
    gen.set_config(config);
    gen.add_hardware_info(hw_);

    std::string json = gen.generate();

    EXPECT_NE(json.find("Custom Title"), std::string::npos);
}

}  // namespace testing
}  // namespace simd_bench
