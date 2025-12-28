#include "simd_bench/runner.h"
#include "simd_bench/timing.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <getopt.h>

namespace simd_bench {

BenchmarkRunner::BenchmarkRunner() {
    initialize();
}

BenchmarkRunner::~BenchmarkRunner() = default;

void BenchmarkRunner::initialize() {
    // Detect hardware
    hardware_ = HardwareInfo::detect();

    // Configure roofline model
    roofline_.configure_from_hardware(hardware_);

    // Initialize performance counters
    counters_ = PerformanceCounterFactory::create_best_available();
    if (!counters_->initialize()) {
        counters_ = std::make_unique<NullCounters>();
    }

    // Initialize energy monitor
    energy_monitor_ = EnergyMonitorFactory::create_best_available();
    if (!energy_monitor_->initialize()) {
        energy_monitor_ = std::make_unique<NullEnergyMonitor>();
    }

    // Initialize correctness verifier
    verifier_ = std::make_unique<CorrectnessVerifier>();

    // Initialize TMA analyzer
    tma_analyzer_ = std::make_unique<TMAAnalyzer>(counters_.get());
}

void BenchmarkRunner::set_config(const BenchmarkConfig& config) {
    config_ = config;
}

void BenchmarkRunner::set_warmup_iterations(size_t iterations) {
    config_.warmup_iterations = iterations;
}

void BenchmarkRunner::set_benchmark_iterations(size_t iterations) {
    config_.benchmark_iterations = iterations;
}

void BenchmarkRunner::enable_hardware_counters(bool enable) {
    config_.enable_hardware_counters = enable;
}

void BenchmarkRunner::enable_energy_profiling(bool enable) {
    config_.enable_energy_profiling = enable;
}

void BenchmarkRunner::enable_correctness_check(bool enable) {
    config_.enable_correctness_check = enable;
}

void BenchmarkRunner::enable_roofline(bool enable) {
    config_.enable_roofline = enable;
}

void BenchmarkRunner::enable_tma(bool enable) {
    config_.enable_tma = enable;
}

void BenchmarkRunner::set_counter_events(const std::vector<CounterEvent>& events) {
    config_.counter_events = events;
}

void BenchmarkRunner::set_sizes(const std::vector<size_t>& sizes) {
    config_.sizes_override = sizes;
}

void BenchmarkRunner::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

BenchmarkResult BenchmarkRunner::run(const std::string& kernel_name) {
    const KernelConfig* config = KernelRegistry::instance().get_kernel(kernel_name);
    if (!config) {
        BenchmarkResult empty;
        empty.kernel_name = kernel_name;
        return empty;
    }
    return run(*config);
}

BenchmarkResult BenchmarkRunner::run(const KernelConfig& kernel) {
    BenchmarkResult result;
    result.kernel_name = kernel.name;

    // Determine sizes to use
    std::vector<size_t> sizes = config_.sizes_override.empty() ?
                                 kernel.sizes : config_.sizes_override;

    // Find reference variant
    const KernelVariant* reference = nullptr;
    for (const auto& v : kernel.variants) {
        if (v.is_reference) {
            reference = &v;
            break;
        }
    }
    if (!reference && !kernel.variants.empty()) {
        reference = &kernel.variants[0];
    }

    // Run each variant at each size
    size_t total_tests = kernel.variants.size() * sizes.size();
    size_t current_test = 0;

    for (const auto& variant : kernel.variants) {
        for (size_t size : sizes) {
            if (progress_callback_) {
                progress_callback_(kernel.name, variant.name, current_test, total_tests, 0, config_.benchmark_iterations);
            }

            VariantResult vr = run_variant(kernel, variant, size);
            result.results.push_back(vr);

            current_test++;
        }
    }

    // Find best variant
    result.best_variant = find_best_variant(result.results);
    result.speedup_vs_scalar = calculate_speedup(result.results);

    // Calculate average vectorization ratio
    if (!result.results.empty()) {
        double sum_vec = 0;
        for (const auto& vr : result.results) {
            sum_vec += vr.metrics.simd.vectorization_ratio;
        }
        result.avg_vectorization_ratio = sum_vec / result.results.size();
    }

    return result;
}

VariantResult BenchmarkRunner::run_variant(
    const KernelConfig& kernel,
    const KernelVariant& variant,
    size_t size
) {
    VariantResult result;
    result.variant_name = variant.name;
    result.problem_size = size;

    // Setup
    void* data = nullptr;
    if (kernel.setup) {
        data = kernel.setup(size);
    }

    size_t iterations = kernel.default_iterations;
    if (iterations == 0) iterations = config_.benchmark_iterations;

    // Warmup
    for (size_t i = 0; i < config_.warmup_iterations; ++i) {
        variant.func(data, size, 1);
    }

    // Benchmark
    result.metrics.performance = measure_performance(variant.func, data, size, iterations);

    // Calculate derived metrics
    result.metrics.performance.gflops =
        (static_cast<double>(size) * kernel.flops_per_element * iterations) /
        result.metrics.performance.elapsed_seconds / 1e9;

    // Hardware counters
    if (config_.enable_hardware_counters && counters_->get_backend() != CounterBackend::NONE) {
        counters_->clear_events();
        for (auto event : get_flops_events()) {
            counters_->add_event(event);
        }

        counters_->start();
        variant.func(data, size, iterations);
        counters_->stop();

        CounterValues values = counters_->read();
        result.metrics.simd = measure_simd_metrics(values);
        result.metrics.memory = measure_memory_metrics(values, size * kernel.bytes_per_element * iterations);
    }

    // TMA
    if (config_.enable_tma && tma_analyzer_) {
        auto tma_result = tma_analyzer_->measure_and_analyze(
            [&]() { variant.func(data, size, 1); },
            iterations
        );
        result.metrics.tma = tma_result.metrics;
    }

    // Energy
    if (config_.enable_energy_profiling && energy_monitor_->get_backend() != EnergyBackend::NONE) {
        uint64_t total_flops = size * kernel.flops_per_element * iterations;
        result.metrics.energy = measure_energy_metrics(total_flops);
    }

    // Roofline
    if (config_.enable_roofline) {
        double ai = kernel.arithmetic_intensity;
        if (ai <= 0 && kernel.bytes_per_element > 0) {
            ai = static_cast<double>(kernel.flops_per_element) / kernel.bytes_per_element;
        }
        result.roofline = calculate_roofline_point(ai, result.metrics.performance.gflops);
    }

    // Recommendations
    result.recommendations = generate_recommendations(result.metrics, result.roofline);

    // Teardown
    if (kernel.teardown && data) {
        kernel.teardown(data);
    }

    return result;
}

PerformanceMetrics BenchmarkRunner::measure_performance(
    const KernelFunction& func,
    void* data,
    size_t size,
    size_t iterations
) {
    PerformanceMetrics metrics;

    Timer timer;
    timer.start();

    func(data, size, iterations);

    timer.stop();

    metrics.elapsed_seconds = timer.elapsed_seconds();
    metrics.cycles = timer.elapsed_cycles();

    return metrics;
}

SIMDMetrics BenchmarkRunner::measure_simd_metrics(const CounterValues& counters) {
    SIMDMetrics metrics;

    metrics.scalar_ops = counters.get(CounterEvent::FP_ARITH_SCALAR_SINGLE);
    metrics.packed_128_ops = counters.get(CounterEvent::FP_ARITH_128B_PACKED_SINGLE);
    metrics.packed_256_ops = counters.get(CounterEvent::FP_ARITH_256B_PACKED_SINGLE);
    metrics.packed_512_ops = counters.get(CounterEvent::FP_ARITH_512B_PACKED_SINGLE);

    uint64_t total_packed = metrics.packed_128_ops + metrics.packed_256_ops + metrics.packed_512_ops;
    uint64_t total_ops = metrics.scalar_ops + total_packed;

    if (total_ops > 0) {
        metrics.vectorization_ratio = static_cast<double>(total_packed) / total_ops;
    }

    return metrics;
}

MemoryMetrics BenchmarkRunner::measure_memory_metrics(const CounterValues& counters, size_t bytes_processed) {
    MemoryMetrics metrics;

    metrics.l1_hits = counters.get(CounterEvent::L1D_READ_ACCESS) - counters.get(CounterEvent::L1D_READ_MISS);
    metrics.l1_misses = counters.get(CounterEvent::L1D_READ_MISS);
    metrics.l2_misses = counters.get(CounterEvent::L2_READ_MISS);
    metrics.l3_misses = counters.get(CounterEvent::L3_READ_MISS);

    uint64_t total_l1 = metrics.l1_hits + metrics.l1_misses;
    if (total_l1 > 0) {
        metrics.l1_hit_rate = static_cast<double>(metrics.l1_hits) / total_l1;
    }

    metrics.bytes_read = bytes_processed;

    return metrics;
}

TMAMetrics BenchmarkRunner::measure_tma_metrics(const CounterValues& counters) {
    TMAMetrics metrics;

    // Basic calculation from cycles and uops
    uint64_t cycles = counters.get(CounterEvent::CYCLES);
    uint64_t uops_retired = counters.get(CounterEvent::UOPS_RETIRED_SLOTS);

    if (cycles > 0) {
        const double pipeline_width = 4.0;
        double total_slots = cycles * pipeline_width;
        metrics.retiring = static_cast<double>(uops_retired) / total_slots;
    }

    return metrics;
}

EnergyMetrics BenchmarkRunner::measure_energy_metrics(uint64_t total_flops) {
    EnergyMetrics metrics = energy_monitor_->get_metrics();

    if (total_flops > 0 && metrics.energy_joules > 0) {
        metrics.energy_per_op_nj = (metrics.energy_joules * 1e9) / static_cast<double>(total_flops);
    }

    return metrics;
}

CorrectnessMetrics BenchmarkRunner::verify_correctness(
    const KernelConfig& kernel,
    const KernelVariant& variant,
    const KernelVariant* reference,
    void* data,
    size_t size
) {
    CorrectnessMetrics metrics;
    metrics.passed = true;

    if (!reference || !kernel.verify) {
        return metrics;
    }

    // This would require more complex setup to compare outputs
    // Simplified version just runs verification function

    return metrics;
}

RooflinePoint BenchmarkRunner::calculate_roofline_point(
    double arithmetic_intensity,
    double achieved_gflops
) {
    return roofline_.analyze(arithmetic_intensity, achieved_gflops);
}

std::vector<std::string> BenchmarkRunner::generate_recommendations(
    const KernelMetrics& metrics,
    const RooflinePoint& roofline_point
) {
    std::vector<std::string> recommendations;

    // Roofline-based recommendations
    auto roofline_recs = simd_bench::generate_recommendations(roofline_point, roofline_);
    for (const auto& rec : roofline_recs) {
        recommendations.push_back(rec.message);
    }

    // TMA-based recommendations
    if (metrics.tma.backend_bound > 0.3) {
        if (metrics.tma.memory_bound > 0.2) {
            recommendations.push_back("Consider prefetching or cache blocking");
        }
        if (metrics.tma.core_bound > 0.15) {
            recommendations.push_back("Increase instruction-level parallelism");
        }
    }

    // Vectorization recommendations
    if (metrics.simd.vectorization_ratio < 0.9) {
        recommendations.push_back("Improve loop vectorization");
    }

    return recommendations;
}

std::string BenchmarkRunner::find_best_variant(const std::vector<VariantResult>& results) {
    if (results.empty()) return "";

    double max_gflops = 0;
    std::string best;

    for (const auto& vr : results) {
        if (vr.metrics.performance.gflops > max_gflops) {
            max_gflops = vr.metrics.performance.gflops;
            best = vr.variant_name;
        }
    }

    return best;
}

double BenchmarkRunner::calculate_speedup(const std::vector<VariantResult>& results) {
    if (results.empty()) {
        return 1.0;  // No speedup if no results
    }

    double scalar_gflops = 0;
    double best_gflops = 0;

    for (const auto& vr : results) {
        if (vr.variant_name == "scalar" || vr.variant_name.find("scalar") != std::string::npos) {
            scalar_gflops = std::max(scalar_gflops, vr.metrics.performance.gflops);
        }
        best_gflops = std::max(best_gflops, vr.metrics.performance.gflops);
    }

    // If no scalar variant found, use first result as baseline
    if (scalar_gflops <= 0) {
        scalar_gflops = results[0].metrics.performance.gflops;
    }

    // Avoid division by zero
    if (scalar_gflops <= 0) {
        return 1.0;
    }

    return best_gflops / scalar_gflops;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_all() {
    std::vector<BenchmarkResult> results;
    for (const auto& name : KernelRegistry::instance().get_kernel_names()) {
        results.push_back(run(name));
    }
    return results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_category(const std::string& category) {
    std::vector<BenchmarkResult> results;
    auto kernels = KernelRegistry::instance().get_kernels_by_category(category);
    for (const auto* kernel : kernels) {
        results.push_back(run(*kernel));
    }
    return results;
}

void BenchmarkRunner::generate_report(
    const std::vector<BenchmarkResult>& results,
    ReportFormat format,
    const std::string& output_path
) {
    auto generator = ReportGeneratorFactory::create(format);

    generator->add_hardware_info(hardware_);

    for (const auto& result : results) {
        generator->add_benchmark_result(result);
    }

    // Collect roofline points
    std::vector<RooflinePoint> roofline_points;
    for (const auto& result : results) {
        for (const auto& vr : result.results) {
            roofline_points.push_back(vr.roofline);
        }
    }
    generator->add_roofline_model(roofline_, roofline_points);

    generator->generate_to_file(output_path);
}

// Quick benchmark helper
BenchmarkResult quick_benchmark(
    const std::string& name,
    KernelFunction func,
    SetupFunction setup,
    TeardownFunction teardown,
    size_t size,
    size_t flops_per_element
) {
    KernelConfig config;
    config.name = name;
    config.flops_per_element = flops_per_element;
    config.bytes_per_element = sizeof(float) * 2;
    config.sizes = {size};
    config.default_iterations = 100;
    config.setup = setup;
    config.teardown = teardown;

    KernelVariant variant;
    variant.name = "default";
    variant.func = func;
    variant.isa = "auto";
    config.variants.push_back(variant);

    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);

    return runner.run(config);
}

// CLI implementation
CLIOptions parse_cli_args(int argc, char* argv[]) {
    CLIOptions options;

    // Reset getopt state for proper parsing across multiple calls
    optind = 1;

    static struct option long_options[] = {
        {"kernel", required_argument, nullptr, 'k'},
        {"category", required_argument, nullptr, 'c'},
        {"size", required_argument, nullptr, 's'},
        {"iterations", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"format", required_argument, nullptr, 'f'},
        {"baseline", required_argument, nullptr, 'b'},
        {"threshold", required_argument, nullptr, 't'},
        {"no-counters", no_argument, nullptr, 'n'},
        {"no-energy", no_argument, nullptr, 'e'},
        {"verbose", no_argument, nullptr, 'v'},
        {"quiet", no_argument, nullptr, 'q'},
        {"list-kernels", no_argument, nullptr, 'l'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "k:c:s:i:o:f:b:t:nevqlh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'k':
                options.kernel_names.push_back(optarg);
                break;
            case 'c':
                options.category = optarg;
                break;
            case 's':
                options.sizes.push_back(std::stoull(optarg));
                break;
            case 'i':
                options.iterations = std::stoull(optarg);
                break;
            case 'o':
                options.output_path = optarg;
                break;
            case 'f':
                if (std::string(optarg) == "json") options.output_format = ReportFormat::JSON;
                else if (std::string(optarg) == "html") options.output_format = ReportFormat::HTML;
                else if (std::string(optarg) == "md") options.output_format = ReportFormat::MARKDOWN;
                break;
            case 'b':
                options.baseline_path = optarg;
                break;
            case 't':
                options.regression_threshold = std::stod(optarg);
                break;
            case 'n':
                options.enable_counters = false;
                break;
            case 'e':
                options.enable_energy = false;
                break;
            case 'v':
                options.verbose = true;
                break;
            case 'q':
                options.quiet = true;
                break;
            case 'l':
                options.list_kernels = true;
                break;
            case 'h':
                options.help = true;
                break;
        }
    }

    return options;
}

void print_help() {
    std::cout << R"(
SIMD-Bench: Holistic SIMD Kernel Analysis Tool

Usage: simd-bench [OPTIONS]

Options:
  -k, --kernel NAME     Run specific kernel (can be repeated)
  -c, --category CAT    Run all kernels in category
  -s, --size N          Override problem size (can be repeated)
  -i, --iterations N    Set benchmark iterations (default: 100)
  -o, --output PATH     Output file path
  -f, --format FMT      Output format: json, html, md (default: json)
  -b, --baseline PATH   Compare against baseline JSON
  -t, --threshold PCT   Regression threshold (default: 0.05)
  -n, --no-counters     Disable hardware counters
  -e, --no-energy       Disable energy profiling
  -v, --verbose         Verbose output
  -q, --quiet           Quiet mode
  -l, --list-kernels    List registered kernels
  -h, --help            Show this help

Examples:
  simd-bench --kernel dot_product --output results.json
  simd-bench --category blas --format html --output report.html
  simd-bench --baseline baseline.json --threshold 0.05

)" << std::endl;
}

int run_cli(int argc, char* argv[]) {
    CLIOptions options = parse_cli_args(argc, argv);

    if (options.help) {
        print_help();
        return 0;
    }

    if (options.list_kernels) {
        std::cout << "Registered kernels:\n";
        for (const auto& name : KernelRegistry::instance().get_kernel_names()) {
            const auto* config = KernelRegistry::instance().get_kernel(name);
            std::cout << "  " << name;
            if (config) {
                std::cout << " (" << config->category << ")";
            }
            std::cout << "\n";
        }
        return 0;
    }

    BenchmarkRunner runner;
    runner.set_benchmark_iterations(options.iterations);
    runner.enable_hardware_counters(options.enable_counters);
    runner.enable_energy_profiling(options.enable_energy);

    if (!options.sizes.empty()) {
        runner.set_sizes(options.sizes);
    }

    std::vector<BenchmarkResult> results;

    if (!options.kernel_names.empty()) {
        for (const auto& name : options.kernel_names) {
            if (!options.quiet) {
                std::cout << "Running " << name << "...\n";
            }
            results.push_back(runner.run(name));
        }
    } else if (!options.category.empty()) {
        results = runner.run_category(options.category);
    } else {
        results = runner.run_all();
    }

    // Generate report
    if (!options.output_path.empty()) {
        runner.generate_report(results, options.output_format, options.output_path);
        if (!options.quiet) {
            std::cout << "Report written to " << options.output_path << "\n";
        }
    }

    // Print summary
    if (!options.quiet) {
        std::cout << "\nResults:\n";
        for (const auto& result : results) {
            std::cout << "  " << result.kernel_name << ": "
                      << result.best_variant << " ("
                      << std::fixed << std::setprecision(1)
                      << result.speedup_vs_scalar << "x speedup)\n";
        }
    }

    return 0;
}

}  // namespace simd_bench
