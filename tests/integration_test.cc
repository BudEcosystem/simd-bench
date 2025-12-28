#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace simd_bench {
namespace testing {

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        KernelRegistry::instance().clear();
    }

    void TearDown() override {
        KernelRegistry::instance().clear();
    }
};

// Scalar dot product implementation
float scalar_dot_product(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// SIMD dot product using Highway
float simd_dot_product(const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4*N <= count; i += 4*N) {
        sum0 = hn::MulAdd(hn::Load(d, a + i + 0*N), hn::Load(d, b + i + 0*N), sum0);
        sum1 = hn::MulAdd(hn::Load(d, a + i + 1*N), hn::Load(d, b + i + 1*N), sum1);
        sum2 = hn::MulAdd(hn::Load(d, a + i + 2*N), hn::Load(d, b + i + 2*N), sum2);
        sum3 = hn::MulAdd(hn::Load(d, a + i + 3*N), hn::Load(d, b + i + 3*N), sum3);
    }

    auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// Test end-to-end workflow
TEST_F(IntegrationTest, EndToEndBenchmark) {
    // Setup
    struct DotData {
        hwy::AlignedFreeUniquePtr<float[]> a;
        hwy::AlignedFreeUniquePtr<float[]> b;
        float result;
        size_t size;
    };

    auto setup = [](size_t size) -> void* {
        auto* data = new DotData();
        data->size = size;
        data->a = hwy::AllocateAligned<float>(size);
        data->b = hwy::AllocateAligned<float>(size);
        for (size_t i = 0; i < size; ++i) {
            data->a[i] = static_cast<float>(i % 100) / 100.0f;
            data->b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
        }
        return data;
    };

    auto teardown = [](void* ptr) {
        delete static_cast<DotData*>(ptr);
    };

    auto scalar_kernel = [](void* ptr, size_t, size_t iterations) {
        auto* data = static_cast<DotData*>(ptr);
        for (size_t iter = 0; iter < iterations; ++iter) {
            data->result = scalar_dot_product(data->a.get(), data->b.get(), data->size);
        }
        do_not_optimize(data->result);
    };

    auto simd_kernel = [](void* ptr, size_t, size_t iterations) {
        auto* data = static_cast<DotData*>(ptr);
        for (size_t iter = 0; iter < iterations; ++iter) {
            data->result = simd_dot_product(data->a.get(), data->b.get(), data->size);
        }
        do_not_optimize(data->result);
    };

    // Register kernel
    KernelBuilder("dot_product_test")
        .description("Dot product benchmark")
        .category("BLAS")
        .arithmetic_intensity(0.25)
        .flops_per_element(2)
        .bytes_per_element(8)
        .add_variant("scalar", scalar_kernel, "scalar", true)
        .add_variant("simd_4x", simd_kernel, "avx2", false)
        .sizes({1024, 4096, 16384})
        .default_iterations(1000)
        .setup(setup)
        .teardown(teardown)
        .register_kernel();

    // Run benchmark
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);  // May require sudo
    runner.enable_energy_profiling(false);   // May require sudo
    runner.enable_correctness_check(false);  // Skip for speed
    runner.set_benchmark_iterations(100);

    BenchmarkResult result = runner.run("dot_product_test");

    // Verify results
    EXPECT_EQ(result.kernel_name, "dot_product_test");
    EXPECT_EQ(result.results.size(), 6u);  // 2 variants × 3 sizes

    // SIMD should be faster than scalar
    double scalar_gflops = 0, simd_gflops = 0;
    for (const auto& vr : result.results) {
        if (vr.variant_name == "scalar") {
            scalar_gflops = std::max(scalar_gflops, vr.metrics.performance.gflops);
        } else if (vr.variant_name == "simd_4x") {
            simd_gflops = std::max(simd_gflops, vr.metrics.performance.gflops);
        }
    }

    EXPECT_GT(simd_gflops, scalar_gflops);
    EXPECT_EQ(result.best_variant, "simd_4x");
}

// Test correctness verification integration
TEST_F(IntegrationTest, CorrectnessVerification) {
    // Create test data
    const size_t size = 1024;
    auto a = hwy::AllocateAligned<float>(size);
    auto b = hwy::AllocateAligned<float>(size);

    RandomInputGenerator gen(42);
    gen.generate_uniform(a.get(), size, -10.0f, 10.0f);
    gen.generate_uniform(b.get(), size, -10.0f, 10.0f);

    // Calculate with both implementations
    float scalar_result = scalar_dot_product(a.get(), b.get(), size);
    float simd_result = simd_dot_product(a.get(), b.get(), size);

    // Verify they match
    EXPECT_NEAR(scalar_result, simd_result, std::abs(scalar_result) * 1e-5);
}

// Test roofline analysis integration
TEST_F(IntegrationTest, RooflineAnalysis) {
    // Get hardware info
    HardwareInfo hw = HardwareInfo::detect();

    // Create roofline model
    RooflineModel model;
    model.configure_from_hardware(hw);

    // Analyze a typical dot product performance
    // AI = 2 FLOPS / 8 bytes = 0.25 FLOP/byte
    // Use a fraction of theoretical peak to ensure efficiency <= 1
    double theoretical_max = model.get_theoretical_max(0.25);
    double achieved_gflops = theoretical_max * 0.5;  // 50% of theoretical max
    RooflinePoint point = model.analyze(0.25, achieved_gflops);

    EXPECT_EQ(point.arithmetic_intensity, 0.25);
    EXPECT_DOUBLE_EQ(point.achieved_gflops, achieved_gflops);
    EXPECT_FALSE(point.bound.empty());
    EXPECT_GE(point.efficiency, 0.0);
    EXPECT_LE(point.efficiency, 1.01);  // Allow small tolerance for rounding
}

// Test report generation integration
TEST_F(IntegrationTest, ReportGeneration) {
    // Detect hardware
    HardwareInfo hw = HardwareInfo::detect();

    // Create sample results
    BenchmarkResult result;
    result.kernel_name = "test_kernel";
    result.best_variant = "simd";
    result.speedup_vs_scalar = 4.0;

    VariantResult vr;
    vr.variant_name = "scalar";
    vr.problem_size = 1024;
    vr.metrics.performance.gflops = 2.0;
    vr.metrics.performance.elapsed_seconds = 0.001;
    result.results.push_back(vr);

    vr.variant_name = "simd";
    vr.metrics.performance.gflops = 8.0;
    vr.metrics.performance.elapsed_seconds = 0.00025;
    result.results.push_back(vr);

    // Generate reports in all formats
    JSONReportGenerator json_gen;
    json_gen.add_hardware_info(hw);
    json_gen.add_benchmark_result(result);
    std::string json = json_gen.generate();
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("test_kernel"), std::string::npos);

    HTMLReportGenerator html_gen;
    html_gen.add_hardware_info(hw);
    html_gen.add_benchmark_result(result);
    std::string html = html_gen.generate();
    EXPECT_FALSE(html.empty());
    EXPECT_NE(html.find("<html"), std::string::npos);  // May have attributes like lang="en"

    MarkdownReportGenerator md_gen;
    md_gen.add_hardware_info(hw);
    md_gen.add_benchmark_result(result);
    std::string md = md_gen.generate();
    EXPECT_FALSE(md.empty());
    EXPECT_NE(md.find("#"), std::string::npos);
}

// Test timing accuracy
TEST_F(IntegrationTest, TimingAccuracy) {
    Timer timer;

    // Time a known workload
    const int iterations = 1000000;
    timer.start();
    volatile int dummy = 0;
    for (int i = 0; i < iterations; ++i) {
        dummy += i;
    }
    timer.stop();

    double elapsed = timer.elapsed_seconds();
    uint64_t cycles = timer.elapsed_cycles();

    EXPECT_GT(elapsed, 0.0);
    EXPECT_GT(cycles, 0u);

    // Cycles should correlate with time
    double freq_ghz = static_cast<double>(cycles) / (elapsed * 1e9);
    EXPECT_GT(freq_ghz, 0.5);  // At least 500 MHz
    EXPECT_LT(freq_ghz, 10.0); // Less than 10 GHz
}

// Test multiple kernel registration and execution
TEST_F(IntegrationTest, MultipleKernelExecution) {
    auto dummy = [](void*, size_t, size_t) {};
    auto setup = [](size_t) -> void* { return nullptr; };
    auto teardown = [](void*) {};

    // Register multiple kernels
    for (int i = 0; i < 5; ++i) {
        KernelBuilder("kernel_" + std::to_string(i))
            .category("test")
            .add_variant("scalar", dummy)
            .sizes({1024})
            .setup(setup)
            .teardown(teardown)
            .register_kernel();
    }

    EXPECT_EQ(KernelRegistry::instance().size(), 5u);

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    auto results = runner.run_all();
    EXPECT_EQ(results.size(), 5u);

    for (const auto& result : results) {
        EXPECT_TRUE(result.kernel_name.find("kernel_") != std::string::npos);
        EXPECT_FALSE(result.results.empty());
    }
}

// Test SIMD vectorization metrics
TEST_F(IntegrationTest, SIMDMetricsCollection) {
    // With hardware counters disabled, we can still verify structure
    SIMDMetrics metrics;
    metrics.scalar_ops = 1000;
    metrics.packed_256_ops = 8000;

    double vec_ratio = static_cast<double>(metrics.packed_256_ops) /
                       (metrics.scalar_ops + metrics.packed_256_ops);

    EXPECT_NEAR(vec_ratio, 0.889, 0.01);  // 8000 / 9000 ≈ 0.889
}

// Test memory bandwidth measurement
TEST_F(IntegrationTest, MemoryBandwidthMeasurement) {
    double bw = measure_memory_bandwidth_gbps(16);  // 16 MB

    EXPECT_GT(bw, 0.0);
    EXPECT_LT(bw, 500.0);  // Reasonable upper bound
}

// Test version string
TEST_F(IntegrationTest, VersionString) {
    std::string version = get_version_string();
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find("."), std::string::npos);
}

// Test empirical roofline
TEST_F(IntegrationTest, EmpiricalRooflineCreation) {
    EmpiricalRoofline empirical;
    empirical.measure_bandwidths();

    EXPECT_GT(empirical.get_l1_bandwidth_gbps(), 0.0);
    EXPECT_GT(empirical.get_dram_bandwidth_gbps(), 0.0);
    EXPECT_GT(empirical.get_peak_gflops(), 0.0);

    RooflineModel model = empirical.create_model();
    EXPECT_FALSE(model.get_ceilings().empty());
}

// Test full pipeline with SIMD kernel
TEST_F(IntegrationTest, FullPipelineWithSIMDKernel) {
    // This is a comprehensive test of the entire system

    struct TestData {
        hwy::AlignedFreeUniquePtr<float[]> input;
        hwy::AlignedFreeUniquePtr<float[]> output;
        size_t size;
    };

    auto setup = [](size_t size) -> void* {
        auto* data = new TestData();
        data->size = size;
        data->input = hwy::AllocateAligned<float>(size);
        data->output = hwy::AllocateAligned<float>(size);
        for (size_t i = 0; i < size; ++i) {
            data->input[i] = static_cast<float>(i);
        }
        return data;
    };

    auto teardown = [](void* ptr) {
        delete static_cast<TestData*>(ptr);
    };

    // Scalar: multiply by 2
    auto scalar_kernel = [](void* ptr, size_t, size_t iterations) {
        auto* data = static_cast<TestData*>(ptr);
        for (size_t iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < data->size; ++i) {
                data->output[i] = data->input[i] * 2.0f;
            }
        }
        do_not_optimize(data->output[0]);
    };

    // SIMD: multiply by 2
    auto simd_kernel = [](void* ptr, size_t, size_t iterations) {
        auto* data = static_cast<TestData*>(ptr);
        const hn::ScalableTag<float> d;
        const size_t N = hn::Lanes(d);
        const auto two = hn::Set(d, 2.0f);

        for (size_t iter = 0; iter < iterations; ++iter) {
            size_t i = 0;
            for (; i + N <= data->size; i += N) {
                auto v = hn::Load(d, data->input.get() + i);
                hn::Store(hn::Mul(v, two), d, data->output.get() + i);
            }
            for (; i < data->size; ++i) {
                data->output[i] = data->input[i] * 2.0f;
            }
        }
        do_not_optimize(data->output[0]);
    };

    KernelBuilder("scale_kernel")
        .description("Scale by 2")
        .category("element-wise")
        .flops_per_element(1)
        .bytes_per_element(8)
        .add_variant("scalar", scalar_kernel, "scalar", true)
        .add_variant("simd", simd_kernel, "avx2", false)
        .sizes({4096, 16384, 65536})
        .default_iterations(100)
        .setup(setup)
        .teardown(teardown)
        .register_kernel();

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(50);

    BenchmarkResult result = runner.run("scale_kernel");

    // Verify complete result structure
    EXPECT_EQ(result.kernel_name, "scale_kernel");
    EXPECT_EQ(result.results.size(), 6u);  // 2 variants × 3 sizes
    EXPECT_GE(result.speedup_vs_scalar, 1.0);  // SIMD should be at least as fast

    // Generate and verify report
    JSONReportGenerator gen;
    gen.add_hardware_info(runner.get_hardware_info());
    gen.add_benchmark_result(result);

    std::string json = gen.generate();
    EXPECT_FALSE(json.empty());

    // Parse and verify JSON structure
    nlohmann::json parsed = nlohmann::json::parse(json);
    EXPECT_TRUE(parsed.contains("kernels") || parsed.contains("results"));
}

}  // namespace testing
}  // namespace simd_bench
