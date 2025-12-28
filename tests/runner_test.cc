#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/runner.h"
#include <atomic>

namespace simd_bench {
namespace testing {

class RunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry
        KernelRegistry::instance().clear();

        // Register a simple test kernel
        auto scalar_add = [](void* data, size_t size, size_t iterations) {
            auto* vd = static_cast<VectorData*>(data);
            for (size_t iter = 0; iter < iterations; ++iter) {
                for (size_t i = 0; i < vd->size; ++i) {
                    vd->c[i] = vd->a[i] + vd->b[i];
                }
            }
        };

        auto setup = [](size_t size) -> void* {
            auto* data = new VectorData();
            data->size = size;
            data->a = new float[size];
            data->b = new float[size];
            data->c = new float[size];
            for (size_t i = 0; i < size; ++i) {
                data->a[i] = static_cast<float>(i);
                data->b[i] = static_cast<float>(i * 2);
            }
            return data;
        };

        auto teardown = [](void* ptr) {
            auto* data = static_cast<VectorData*>(ptr);
            delete[] data->a;
            delete[] data->b;
            delete[] data->c;
            delete data;
        };

        KernelBuilder("test_add")
            .description("Simple vector addition")
            .category("test")
            .arithmetic_intensity(0.125)
            .flops_per_element(1)
            .bytes_per_element(12)
            .add_variant("scalar", scalar_add, "scalar", true)
            .sizes({1024, 4096})
            .default_iterations(100)
            .setup(setup)
            .teardown(teardown)
            .register_kernel();
    }

    void TearDown() override {
        KernelRegistry::instance().clear();
    }
};

// Test runner construction
TEST_F(RunnerTest, ConstructionSucceeds) {
    EXPECT_NO_THROW({
        BenchmarkRunner runner;
    });
}

// Test hardware detection
TEST_F(RunnerTest, GetHardwareInfoReturnsValid) {
    BenchmarkRunner runner;
    const HardwareInfo& hw = runner.get_hardware_info();

    EXPECT_FALSE(hw.cpu_brand.empty());
    EXPECT_GT(hw.physical_cores, 0);
}

// Test configuration
TEST_F(RunnerTest, SetConfigurationOptions) {
    BenchmarkRunner runner;

    runner.set_warmup_iterations(10);
    runner.set_benchmark_iterations(50);
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);

    const BenchmarkConfig& config = runner.get_config();
    EXPECT_EQ(config.warmup_iterations, 10u);
    EXPECT_EQ(config.benchmark_iterations, 50u);
    EXPECT_FALSE(config.enable_hardware_counters);
    EXPECT_FALSE(config.enable_energy_profiling);
}

TEST_F(RunnerTest, SetSizesOverride) {
    BenchmarkRunner runner;
    runner.set_sizes({256, 512, 1024});

    // Should use overridden sizes instead of kernel defaults
}

// Test progress callback
TEST_F(RunnerTest, ProgressCallbackCalled) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(5);

    std::atomic<int> callback_count{0};
    runner.set_progress_callback([&](const std::string&, const std::string&,
                                     size_t, size_t, size_t, size_t) {
        callback_count++;
    });

    BenchmarkResult result = runner.run("test_add");

    EXPECT_GT(callback_count.load(), 0);
}

// Test running a single kernel
TEST_F(RunnerTest, RunKernelByName) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    BenchmarkResult result = runner.run("test_add");

    EXPECT_EQ(result.kernel_name, "test_add");
    EXPECT_FALSE(result.results.empty());
    EXPECT_FALSE(result.best_variant.empty());
}

TEST_F(RunnerTest, RunKernelByConfig) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    const KernelConfig* config = KernelRegistry::instance().get_kernel("test_add");
    ASSERT_NE(config, nullptr);

    BenchmarkResult result = runner.run(*config);

    EXPECT_EQ(result.kernel_name, "test_add");
}

// Test running all kernels
TEST_F(RunnerTest, RunAllKernels) {
    // Register another kernel
    auto dummy = [](void*, size_t, size_t) {};
    KernelBuilder("test_kernel2")
        .category("test")
        .add_variant("scalar", dummy)
        .sizes({1024})
        .register_kernel();

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(5);

    std::vector<BenchmarkResult> results = runner.run_all();

    EXPECT_EQ(results.size(), 2u);
}

// Test running by category
TEST_F(RunnerTest, RunByCategory) {
    auto dummy = [](void*, size_t, size_t) {};
    KernelBuilder("other_kernel")
        .category("other")
        .add_variant("scalar", dummy)
        .sizes({1024})
        .register_kernel();

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(5);

    std::vector<BenchmarkResult> test_results = runner.run_category("test");
    std::vector<BenchmarkResult> other_results = runner.run_category("other");

    EXPECT_EQ(test_results.size(), 1u);
    EXPECT_EQ(other_results.size(), 1u);
}

// Test variant result structure
TEST_F(RunnerTest, VariantResultHasMetrics) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    BenchmarkResult result = runner.run("test_add");

    ASSERT_FALSE(result.results.empty());
    const VariantResult& vr = result.results[0];

    EXPECT_FALSE(vr.variant_name.empty());
    EXPECT_GT(vr.problem_size, 0u);
    EXPECT_GT(vr.metrics.performance.elapsed_seconds, 0.0);
}

// Test roofline model
TEST_F(RunnerTest, GetRooflineModelValid) {
    BenchmarkRunner runner;
    const RooflineModel& model = runner.get_roofline_model();

    double max = model.get_theoretical_max(100.0);
    EXPECT_GT(max, 0.0);
}

// Test report generation
TEST_F(RunnerTest, GenerateJSONReport) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(5);

    BenchmarkResult result = runner.run("test_add");
    std::vector<BenchmarkResult> results = {result};

    std::string path = "/tmp/simd_bench_test_report.json";
    EXPECT_NO_THROW({
        runner.generate_report(results, ReportFormat::JSON, path);
    });

    EXPECT_TRUE(std::filesystem::exists(path));
    std::filesystem::remove(path);
}

// Test quick_benchmark helper
TEST_F(RunnerTest, QuickBenchmarkExecutes) {
    auto func = [](void* data, size_t size, size_t iterations) {
        float* arr = static_cast<float*>(data);
        for (size_t iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < size; ++i) {
                arr[i] += 1.0f;
            }
        }
    };

    auto setup = [](size_t size) -> void* {
        return new float[size]();
    };

    auto teardown = [](void* data) {
        delete[] static_cast<float*>(data);
    };

    BenchmarkResult result = quick_benchmark(
        "quick_test",
        func,
        setup,
        teardown,
        1024,
        1  // 1 FLOP per element
    );

    EXPECT_EQ(result.kernel_name, "quick_test");
    EXPECT_FALSE(result.results.empty());
}

// Test CLI option parsing
TEST_F(RunnerTest, ParseCLIArgs) {
    const char* argv[] = {
        "simd-bench",
        "--kernel", "dot_product",
        "--iterations", "100",
        "--output", "report.json"
    };
    int argc = 7;

    CLIOptions options = parse_cli_args(argc, const_cast<char**>(argv));

    EXPECT_THAT(options.kernel_names, ::testing::Contains("dot_product"));
    EXPECT_EQ(options.iterations, 100u);
    EXPECT_EQ(options.output_path, "report.json");
}

TEST_F(RunnerTest, ParseCLIArgsHelp) {
    const char* argv[] = {"simd-bench", "--help"};
    int argc = 2;

    CLIOptions options = parse_cli_args(argc, const_cast<char**>(argv));

    EXPECT_TRUE(options.help);
}

TEST_F(RunnerTest, ParseCLIArgsListKernels) {
    const char* argv[] = {"simd-bench", "--list-kernels"};
    int argc = 2;

    CLIOptions options = parse_cli_args(argc, const_cast<char**>(argv));

    EXPECT_TRUE(options.list_kernels);
}

TEST_F(RunnerTest, ParseCLIArgsQuiet) {
    const char* argv[] = {"simd-bench", "--quiet"};
    int argc = 2;

    CLIOptions options = parse_cli_args(argc, const_cast<char**>(argv));

    EXPECT_TRUE(options.quiet);
}

TEST_F(RunnerTest, ParseCLIArgsVerbose) {
    const char* argv[] = {"simd-bench", "--verbose"};
    int argc = 2;

    CLIOptions options = parse_cli_args(argc, const_cast<char**>(argv));

    EXPECT_TRUE(options.verbose);
}

// Test finding best variant
TEST_F(RunnerTest, BestVariantIsIdentified) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    BenchmarkResult result = runner.run("test_add");

    EXPECT_FALSE(result.best_variant.empty());
}

// Test speedup calculation
TEST_F(RunnerTest, SpeedupCalculation) {
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(10);

    BenchmarkResult result = runner.run("test_add");

    // With only one variant (scalar), speedup should be ~1.0
    EXPECT_GE(result.speedup_vs_scalar, 0.5);
    EXPECT_LE(result.speedup_vs_scalar, 2.0);
}

}  // namespace testing
}  // namespace simd_bench
