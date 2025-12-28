#pragma once

#include "types.h"
#include "hardware.h"
#include "kernel_registry.h"
#include "performance_counters.h"
#include "roofline.h"
#include "tma.h"
#include "energy.h"
#include "correctness.h"
#include "report_generator.h"
#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace simd_bench {

// Benchmark run progress callback
using ProgressCallback = std::function<void(
    const std::string& kernel_name,
    const std::string& variant_name,
    size_t current_size,
    size_t total_sizes,
    size_t current_iteration,
    size_t total_iterations
)>;

// Benchmark runner - orchestrates the entire benchmarking process
class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner();

    // Configuration
    void set_config(const BenchmarkConfig& config);
    const BenchmarkConfig& get_config() const { return config_; }

    void set_warmup_iterations(size_t iterations);
    void set_benchmark_iterations(size_t iterations);

    void enable_hardware_counters(bool enable);
    void enable_energy_profiling(bool enable);
    void enable_correctness_check(bool enable);
    void enable_roofline(bool enable);
    void enable_tma(bool enable);

    void set_counter_events(const std::vector<CounterEvent>& events);
    void set_sizes(const std::vector<size_t>& sizes);

    // Set progress callback
    void set_progress_callback(ProgressCallback callback);

    // Run benchmarks
    BenchmarkResult run(const KernelConfig& kernel);
    BenchmarkResult run(const std::string& kernel_name);
    std::vector<BenchmarkResult> run_all();
    std::vector<BenchmarkResult> run_category(const std::string& category);

    // Run a single variant at a single size
    VariantResult run_variant(
        const KernelConfig& kernel,
        const KernelVariant& variant,
        size_t size
    );

    // Generate report
    void generate_report(
        const std::vector<BenchmarkResult>& results,
        ReportFormat format,
        const std::string& output_path
    );

    // Get detected hardware info
    const HardwareInfo& get_hardware_info() const { return hardware_; }

    // Get roofline model
    const RooflineModel& get_roofline_model() const { return roofline_; }

private:
    BenchmarkConfig config_;
    HardwareInfo hardware_;
    RooflineModel roofline_;

    std::unique_ptr<IPerformanceCounters> counters_;
    std::unique_ptr<IEnergyMonitor> energy_monitor_;
    std::unique_ptr<CorrectnessVerifier> verifier_;
    std::unique_ptr<TMAAnalyzer> tma_analyzer_;

    ProgressCallback progress_callback_;

    void initialize();

    // Internal benchmarking methods
    PerformanceMetrics measure_performance(
        const KernelFunction& func,
        void* data,
        size_t size,
        size_t iterations
    );

    SIMDMetrics measure_simd_metrics(const CounterValues& counters);
    MemoryMetrics measure_memory_metrics(const CounterValues& counters, size_t bytes_processed);
    TMAMetrics measure_tma_metrics(const CounterValues& counters);
    EnergyMetrics measure_energy_metrics(uint64_t total_flops);

    CorrectnessMetrics verify_correctness(
        const KernelConfig& kernel,
        const KernelVariant& variant,
        const KernelVariant* reference,
        void* data,
        size_t size
    );

    RooflinePoint calculate_roofline_point(
        double arithmetic_intensity,
        double achieved_gflops
    );

    std::vector<std::string> generate_recommendations(
        const KernelMetrics& metrics,
        const RooflinePoint& roofline_point
    );

    // Find best variant
    static std::string find_best_variant(const std::vector<VariantResult>& results);
    static double calculate_speedup(const std::vector<VariantResult>& results);
};

// Quick benchmark helper functions
BenchmarkResult quick_benchmark(
    const std::string& name,
    KernelFunction func,
    SetupFunction setup,
    TeardownFunction teardown,
    size_t size,
    size_t flops_per_element
);

// Convenience macro for benchmarking a function
#define SIMD_BENCH_RUN(name, func, size, flops_per_elem) \
    simd_bench::quick_benchmark( \
        name, \
        [](void* data, size_t sz, size_t iters) { \
            for (size_t i = 0; i < iters; ++i) func; \
        }, \
        nullptr, nullptr, size, flops_per_elem \
    )

// Command-line interface
struct CLIOptions {
    std::vector<std::string> kernel_names;
    std::string category;
    std::vector<size_t> sizes;
    size_t iterations = 100;
    bool enable_counters = true;
    bool enable_energy = true;
    bool enable_correctness = true;
    ReportFormat output_format = ReportFormat::JSON;
    std::string output_path;
    std::string baseline_path;
    double regression_threshold = 0.05;
    bool verbose = false;
    bool quiet = false;
    bool list_kernels = false;
    bool list_groups = false;
    bool help = false;
};

CLIOptions parse_cli_args(int argc, char* argv[]);
void print_help();
int run_cli(int argc, char* argv[]);

}  // namespace simd_bench
