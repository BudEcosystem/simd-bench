// ============================================================================
// SIMD-Bench Insights Demo
// ============================================================================
// Demonstrates the InsightsEngine for automatic performance analysis
// and optimization recommendations based on benchmark results.
// ============================================================================

#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <iostream>
#include <fstream>

namespace hn = hwy::HWY_NAMESPACE;

using namespace simd_bench;

// Simple scalar implementation
float scalar_sum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// Optimized SIMD implementation
float simd_sum(const float* data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4*N <= n; i += 4*N) {
        sum0 = hn::Add(sum0, hn::Load(d, data + i + 0*N));
        sum1 = hn::Add(sum1, hn::Load(d, data + i + 1*N));
        sum2 = hn::Add(sum2, hn::Load(d, data + i + 2*N));
        sum3 = hn::Add(sum3, hn::Load(d, data + i + 3*N));
    }

    float result = hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "           SIMD-Bench Insights Engine Demo\n";
    std::cout << "========================================================================\n\n";

    // Detect hardware
    HardwareInfo hw = HardwareInfo::detect();
    std::cout << "Hardware: " << hw.cpu_brand << "\n";
    std::cout << "SIMD: " << hw.get_simd_string() << "\n";
    std::cout << "Peak GFLOPS: " << hw.theoretical_peak_sp_gflops << "\n";
    std::cout << "Memory BW: " << hw.measured_memory_bw_gbps << " GB/s\n\n";

    // Create insights engine
    InsightsEngine engine(hw);

    // Configure thresholds (can customize)
    InsightThresholds thresholds;
    thresholds.excellent_efficiency = 0.7;
    thresholds.good_efficiency = 0.5;
    engine.set_thresholds(thresholds);

    // Setup benchmark data
    struct TestData {
        hwy::AlignedFreeUniquePtr<float[]> data;
        float result;
        size_t size;
    };

    auto setup = [](size_t size) -> void* {
        auto* td = new TestData();
        td->size = size;
        td->data = hwy::AllocateAligned<float>(size);
        RandomInputGenerator gen(42);
        gen.generate_uniform(td->data.get(), size, -1.0f, 1.0f);
        return td;
    };

    auto teardown = [](void* ptr) {
        delete static_cast<TestData*>(ptr);
    };

    // Register kernel
    KernelBuilder("sum_reduction_demo")
        .description("Sum reduction benchmark for insights demo")
        .category("reduction")
        .arithmetic_intensity(0.25)  // 1 FLOP / 4 bytes
        .flops_per_element(1)
        .bytes_per_element(4)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* td = static_cast<TestData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                td->result = scalar_sum(td->data.get(), td->size);
            }
            do_not_optimize(td->result);
        }, "scalar", true)
        .add_variant("simd_4x", [](void* ptr, size_t, size_t iterations) {
            auto* td = static_cast<TestData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                td->result = simd_sum(td->data.get(), td->size);
            }
            do_not_optimize(td->result);
        }, "avx2", false)
        .sizes({1024, 16384, 262144, 4194304})
        .default_iterations(500)
        .setup(setup)
        .teardown(teardown)
        .register_kernel();

    // Run benchmark
    std::cout << "Running benchmark...\n\n";

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(500);

    BenchmarkResult result = runner.run("sum_reduction_demo");

    // Get kernel config for analysis
    const KernelConfig* config = KernelRegistry::instance().get_kernel("sum_reduction_demo");

    // Analyze results with insights engine
    std::cout << "========================================================================\n";
    std::cout << "                    INSIGHTS ANALYSIS\n";
    std::cout << "========================================================================\n\n";

    auto analyses = engine.analyze_benchmark(result, config);

    // Print insights for each variant/size combination
    for (const auto& analysis : analyses) {
        std::cout << engine.format_insights_text(analysis);
        std::cout << "\n" << std::string(72, '=') << "\n\n";
    }

    // Generate full report
    std::vector<BenchmarkResult> all_results = {result};
    std::map<std::string, const KernelConfig*> configs;
    configs["sum_reduction_demo"] = config;

    InsightsReport report = engine.analyze_all(all_results, configs);

    // Save markdown report
    std::string markdown = engine.format_report_markdown(report);
    std::ofstream md_out("/tmp/simd_bench_insights.md");
    md_out << markdown;
    md_out.close();
    std::cout << "Markdown report saved to: /tmp/simd_bench_insights.md\n";

    // Save JSON report
    std::string json = engine.format_report_json(report);
    std::ofstream json_out("/tmp/simd_bench_insights.json");
    json_out << json;
    json_out.close();
    std::cout << "JSON report saved to: /tmp/simd_bench_insights.json\n\n";

    // Summary
    std::cout << "========================================================================\n";
    std::cout << "                       INSIGHTS SUMMARY\n";
    std::cout << "========================================================================\n\n";

    std::cout << "Total insights generated: " << report.total_insights << "\n";
    std::cout << "  Critical: " << report.critical_count << "\n";
    std::cout << "  High: " << report.high_count << "\n";
    std::cout << "  Medium: " << report.medium_count << "\n";
    std::cout << "  Low: " << report.low_count << "\n\n";

    // Print global insights
    if (!report.global_insights.empty()) {
        std::cout << "Global Insights:\n";
        for (const auto& gi : report.global_insights) {
            std::cout << "  - " << gi.title << "\n";
        }
        std::cout << "\n";
    }

    // Cleanup
    KernelRegistry::instance().clear();

    return 0;
}
