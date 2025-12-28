#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

namespace hn = hwy::HWY_NAMESPACE;

using namespace simd_bench;

// ============================================================================
// Dot Product Implementations
// ============================================================================

float scalar_dot_product(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float simd_dot_product(const float* a, const float* b, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4*N <= n; i += 4*N) {
        sum0 = hn::MulAdd(hn::Load(d, a + i + 0*N), hn::Load(d, b + i + 0*N), sum0);
        sum1 = hn::MulAdd(hn::Load(d, a + i + 1*N), hn::Load(d, b + i + 1*N), sum1);
        sum2 = hn::MulAdd(hn::Load(d, a + i + 2*N), hn::Load(d, b + i + 2*N), sum2);
        sum3 = hn::MulAdd(hn::Load(d, a + i + 3*N), hn::Load(d, b + i + 3*N), sum3);
    }

    auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ============================================================================
// Vector Scale Implementations
// ============================================================================

void scalar_scale(float* out, const float* in, float alpha, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = in[i] * alpha;
    }
}

void simd_scale(float* out, const float* in, float alpha, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto va = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::Load(d, in + i);
        hn::Store(hn::Mul(v, va), d, out + i);
    }
    for (; i < n; ++i) {
        out[i] = in[i] * alpha;
    }
}

// ============================================================================
// Vector Add Implementations
// ============================================================================

void scalar_add(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void simd_add(float* out, const float* a, const float* b, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        hn::Store(hn::Add(va, vb), d, out + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

// ============================================================================
// Sum Reduction Implementations
// ============================================================================

float scalar_sum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

float simd_sum(const float* data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);

    size_t i = 0;
    for (; i + 2*N <= n; i += 2*N) {
        sum0 = hn::Add(sum0, hn::Load(d, data + i));
        sum1 = hn::Add(sum1, hn::Load(d, data + i + N));
    }

    float result = hn::ReduceSum(d, hn::Add(sum0, sum1));
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

// ============================================================================
// Benchmark Data Structures
// ============================================================================

struct DotProductData {
    hwy::AlignedFreeUniquePtr<float[]> a;
    hwy::AlignedFreeUniquePtr<float[]> b;
    float result;
    size_t size;
};

struct VectorOpData {
    hwy::AlignedFreeUniquePtr<float[]> input;
    hwy::AlignedFreeUniquePtr<float[]> input2;
    hwy::AlignedFreeUniquePtr<float[]> output;
    float scalar_result;
    size_t size;
};

// ============================================================================
// Register Kernels
// ============================================================================

void register_kernels() {
    // Dot Product kernel
    KernelBuilder("dot_product")
        .description("Vector dot product: sum(a[i] * b[i])")
        .category("BLAS-1")
        .arithmetic_intensity(0.25)  // 2 FLOPS / 8 bytes
        .flops_per_element(2)
        .bytes_per_element(8)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<DotProductData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = scalar_dot_product(data->a.get(), data->b.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "scalar", true)
        .add_variant("simd_4x_unroll", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<DotProductData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = simd_dot_product(data->a.get(), data->b.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "avx2", false)
        .sizes({1024, 4096, 16384, 65536, 262144, 1048576})
        .default_iterations(1000)
        .setup([](size_t size) -> void* {
            auto* data = new DotProductData();
            data->size = size;
            data->a = hwy::AllocateAligned<float>(size);
            data->b = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(42);
            gen.generate_uniform(data->a.get(), size, -1.0f, 1.0f);
            gen.generate_uniform(data->b.get(), size, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<DotProductData*>(ptr); })
        .register_kernel();

    // Vector Scale kernel
    KernelBuilder("vector_scale")
        .description("Scale vector: out[i] = alpha * in[i]")
        .category("BLAS-1")
        .arithmetic_intensity(0.125)  // 1 FLOP / 8 bytes
        .flops_per_element(1)
        .bytes_per_element(8)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                scalar_scale(data->output.get(), data->input.get(), 2.5f, data->size);
            }
            do_not_optimize(data->output[0]);
        }, "scalar", true)
        .add_variant("simd", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                simd_scale(data->output.get(), data->input.get(), 2.5f, data->size);
            }
            do_not_optimize(data->output[0]);
        }, "avx2", false)
        .sizes({1024, 4096, 16384, 65536, 262144, 1048576})
        .default_iterations(1000)
        .setup([](size_t size) -> void* {
            auto* data = new VectorOpData();
            data->size = size;
            data->input = hwy::AllocateAligned<float>(size);
            data->output = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(123);
            gen.generate_uniform(data->input.get(), size, -10.0f, 10.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<VectorOpData*>(ptr); })
        .register_kernel();

    // Vector Add kernel
    KernelBuilder("vector_add")
        .description("Vector addition: out[i] = a[i] + b[i]")
        .category("BLAS-1")
        .arithmetic_intensity(0.083)  // 1 FLOP / 12 bytes
        .flops_per_element(1)
        .bytes_per_element(12)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                scalar_add(data->output.get(), data->input.get(), data->input2.get(), data->size);
            }
            do_not_optimize(data->output[0]);
        }, "scalar", true)
        .add_variant("simd", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                simd_add(data->output.get(), data->input.get(), data->input2.get(), data->size);
            }
            do_not_optimize(data->output[0]);
        }, "avx2", false)
        .sizes({1024, 4096, 16384, 65536, 262144, 1048576})
        .default_iterations(1000)
        .setup([](size_t size) -> void* {
            auto* data = new VectorOpData();
            data->size = size;
            data->input = hwy::AllocateAligned<float>(size);
            data->input2 = hwy::AllocateAligned<float>(size);
            data->output = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(456);
            gen.generate_uniform(data->input.get(), size, -10.0f, 10.0f);
            gen.generate_uniform(data->input2.get(), size, -10.0f, 10.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<VectorOpData*>(ptr); })
        .register_kernel();

    // Sum Reduction kernel
    KernelBuilder("sum_reduction")
        .description("Sum reduction: result = sum(data[i])")
        .category("reduction")
        .arithmetic_intensity(0.25)  // 1 FLOP / 4 bytes
        .flops_per_element(1)
        .bytes_per_element(4)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->scalar_result = scalar_sum(data->input.get(), data->size);
            }
            do_not_optimize(data->scalar_result);
        }, "scalar", true)
        .add_variant("simd_2x_unroll", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<VectorOpData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->scalar_result = simd_sum(data->input.get(), data->size);
            }
            do_not_optimize(data->scalar_result);
        }, "avx2", false)
        .sizes({1024, 4096, 16384, 65536, 262144, 1048576})
        .default_iterations(1000)
        .setup([](size_t size) -> void* {
            auto* data = new VectorOpData();
            data->size = size;
            data->input = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(789);
            gen.generate_uniform(data->input.get(), size, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<VectorOpData*>(ptr); })
        .register_kernel();
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "        SIMD-Bench: Comprehensive BLAS-1 Benchmark Suite\n";
    std::cout << "========================================================================\n\n";

    // Detect hardware
    HardwareInfo hw = HardwareInfo::detect();
    std::cout << "Hardware Configuration:\n";
    std::cout << "  CPU: " << hw.cpu_brand << "\n";
    std::cout << "  Vendor: " << hw.cpu_vendor << "\n";
    std::cout << "  Cores: " << hw.physical_cores << " physical, " << hw.logical_cores << " logical\n";
    std::cout << "  Base Frequency: " << std::fixed << std::setprecision(2) << hw.base_frequency_ghz << " GHz\n";
    std::cout << "  Measured Frequency: " << hw.measured_frequency_ghz << " GHz\n";
    std::cout << "  L1d Cache: " << hw.cache.l1d_size_kb << " KB\n";
    std::cout << "  L2 Cache: " << hw.cache.l2_size_kb << " KB\n";
    std::cout << "  L3 Cache: " << hw.cache.l3_size_kb / 1024 << " MB\n";
    std::cout << "  Vector Width: " << hw.max_vector_bits << " bits\n";
    std::cout << "  SIMD Extensions: " << hw.get_simd_string() << "\n";
    std::cout << "  Peak SP GFLOPS: " << hw.theoretical_peak_sp_gflops << "\n";
    std::cout << "  Peak DP GFLOPS: " << hw.theoretical_peak_dp_gflops << "\n";
    std::cout << "  Memory Bandwidth: " << hw.measured_memory_bw_gbps << " GB/s\n";
    std::cout << "\n";

    // Register kernels
    register_kernels();
    std::cout << "Registered " << KernelRegistry::instance().size() << " kernels\n\n";

    // Configure runner
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);  // Disabled for cleaner output
    runner.enable_energy_profiling(false);   // Requires root
    runner.enable_correctness_check(false);  // Skip for speed
    runner.set_benchmark_iterations(500);    // Good precision
    runner.set_warmup_iterations(50);

    // Progress callback
    runner.set_progress_callback([](const std::string& kernel, const std::string& variant,
                                    size_t current_size, size_t total_sizes,
                                    size_t current_iter, size_t total_iter) {
        int progress = static_cast<int>((current_size * 100) / total_sizes);
        std::cout << "\r  [" << progress << "%] " << kernel << "::" << variant
                  << " iter " << current_iter << "/" << total_iter << std::flush;
    });

    std::cout << "Running benchmarks...\n";
    std::cout << "========================================================================\n";

    // Run all benchmarks
    auto results = runner.run_all();

    std::cout << "\n\n========================================================================\n";
    std::cout << "                         Results Summary\n";
    std::cout << "========================================================================\n\n";

    // Print summary table
    for (const auto& result : results) {
        std::cout << "Kernel: " << result.kernel_name << "\n";
        std::cout << "  Best variant: " << result.best_variant << "\n";
        std::cout << "  Speedup vs scalar: " << std::fixed << std::setprecision(2)
                  << result.speedup_vs_scalar << "x\n";

        std::cout << "\n  Size          Scalar (ns)        SIMD (ns)   Speedup     GFLOPS\n";
        std::cout << "  ----------------------------------------------------------------\n";

        // Group by size
        std::map<size_t, std::pair<double, double>> size_times;  // size -> (scalar_ns, simd_ns)
        std::map<size_t, double> size_gflops; // size -> gflops

        for (const auto& vr : result.results) {
            double ns = vr.metrics.performance.elapsed_seconds * 1e9;
            if (vr.variant_name.find("scalar") != std::string::npos) {
                size_times[vr.problem_size].first = ns;
            } else {
                size_times[vr.problem_size].second = ns;
                size_gflops[vr.problem_size] = vr.metrics.performance.gflops;
            }
        }

        for (const auto& [size, times] : size_times) {
            double speedup = times.second > 0 ? times.first / times.second : 0;
            double gflops = size_gflops[size];
            std::cout << "  " << std::setw(8) << size
                      << std::setw(18) << std::fixed << std::setprecision(1) << times.first
                      << std::setw(18) << times.second
                      << std::setw(10) << std::setprecision(2) << speedup << "x"
                      << std::setw(10) << std::setprecision(2) << gflops << "\n";
        }
        std::cout << "\n";
    }

    // Generate roofline analysis
    std::cout << "========================================================================\n";
    std::cout << "                       Roofline Analysis\n";
    std::cout << "========================================================================\n\n";

    RooflineModel roofline;
    roofline.configure_from_hardware(hw);

    double ridge_point = roofline.get_ridge_point();
    std::cout << "Peak GFLOPS (SP): " << hw.theoretical_peak_sp_gflops << "\n";
    std::cout << "Memory BW: " << hw.measured_memory_bw_gbps << " GB/s\n";
    std::cout << "Ridge Point: " << ridge_point << " FLOP/byte\n\n";

    for (const auto& result : results) {
        // Find best SIMD result at largest size
        double best_gflops = 0;
        double ai = 0.25;  // default

        // Get AI from kernel config
        auto* kernel = KernelRegistry::instance().get_kernel(result.kernel_name);
        if (kernel) {
            ai = kernel->arithmetic_intensity;
        }

        for (const auto& vr : result.results) {
            if (vr.variant_name.find("simd") != std::string::npos ||
                vr.variant_name.find("unroll") != std::string::npos) {
                if (vr.metrics.performance.gflops > best_gflops) {
                    best_gflops = vr.metrics.performance.gflops;
                }
            }
        }

        RooflinePoint point = roofline.analyze(ai, best_gflops);
        double theoretical_max = roofline.get_theoretical_max(ai);

        std::cout << result.kernel_name << ":\n";
        std::cout << "  Arithmetic Intensity: " << std::fixed << std::setprecision(3) << point.arithmetic_intensity << " FLOP/byte\n";
        std::cout << "  Achieved GFLOPS: " << std::setprecision(2) << point.achieved_gflops << "\n";
        std::cout << "  Theoretical Max: " << theoretical_max << " GFLOPS\n";
        std::cout << "  Efficiency: " << (point.efficiency * 100) << "%\n";
        std::cout << "  Bound: " << point.bound << "\n\n";
    }

    // Generate JSON report
    std::cout << "========================================================================\n";
    std::cout << "                     Generating JSON Report\n";
    std::cout << "========================================================================\n\n";

    JSONReportGenerator json_gen;
    json_gen.add_hardware_info(hw);

    for (const auto& result : results) {
        json_gen.add_benchmark_result(result);
    }

    std::string json = json_gen.generate();

    // Write to file
    std::ofstream out("/tmp/simd_bench_results.json");
    out << json;
    out.close();

    std::cout << "JSON report written to: /tmp/simd_bench_results.json\n\n";

    // Print the JSON
    std::cout << "========================================================================\n";
    std::cout << "                         JSON Output\n";
    std::cout << "========================================================================\n\n";
    std::cout << json << "\n";

    return 0;
}
