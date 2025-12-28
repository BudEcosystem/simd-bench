// Dot Product Example - Demonstrates SIMD-Bench usage for benchmarking dot product
#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <iostream>
#include <iomanip>
#include <random>

namespace hn = hwy::HWY_NAMESPACE;
using namespace simd_bench;

// Scalar dot product (baseline)
float scalar_dot_product(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// SIMD dot product using Highway
float simd_dot_product(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum = hn::Zero(d);
    size_t i = 0;

    // Main SIMD loop
    for (; i + N <= count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        sum = hn::MulAdd(va, vb, sum);
    }

    // Reduce vector to scalar
    float result = hn::ReduceSum(d, sum);

    // Handle remaining elements
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// SIMD dot product with unrolling
float simd_dot_product_unrolled(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;

    // 4x unrolled loop
    for (; i + 4 * N <= count; i += 4 * N) {
        auto va0 = hn::Load(d, a + i);
        auto vb0 = hn::Load(d, b + i);
        sum0 = hn::MulAdd(va0, vb0, sum0);

        auto va1 = hn::Load(d, a + i + N);
        auto vb1 = hn::Load(d, b + i + N);
        sum1 = hn::MulAdd(va1, vb1, sum1);

        auto va2 = hn::Load(d, a + i + 2 * N);
        auto vb2 = hn::Load(d, b + i + 2 * N);
        sum2 = hn::MulAdd(va2, vb2, sum2);

        auto va3 = hn::Load(d, a + i + 3 * N);
        auto vb3 = hn::Load(d, b + i + 3 * N);
        sum3 = hn::MulAdd(va3, vb3, sum3);
    }

    // Combine partial sums
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, sum2);

    // Handle remaining elements in vector
    for (; i + N <= count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        sum0 = hn::MulAdd(va, vb, sum0);
    }

    float result = hn::ReduceSum(d, sum0);

    // Handle remaining scalar elements
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

void print_separator() {
    std::cout << std::string(70, '=') << std::endl;
}

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "\n";
    print_separator();
    std::cout << "            SIMD-Bench: Dot Product Benchmark Example\n";
    print_separator();
    std::cout << "\n";

    // Detect and print hardware info
    HardwareInfo hw = HardwareInfo::detect();
    std::cout << "Hardware Info:\n";
    std::cout << "  CPU: " << hw.cpu_brand << "\n";
    std::cout << "  Cores: " << hw.physical_cores << " physical, " << hw.logical_cores << " logical\n";
    std::cout << "  L1 Cache: " << hw.cache.l1d_size_kb << " KB\n";
    std::cout << "  L2 Cache: " << hw.cache.l2_size_kb << " KB\n";
    std::cout << "  L3 Cache: " << hw.cache.l3_size_kb / 1024 << " MB\n";
    std::cout << "  SIMD: ";
    if (has_extension(hw.simd_extensions, SIMDExtension::AVX512F)) std::cout << "AVX-512 ";
    if (has_extension(hw.simd_extensions, SIMDExtension::AVX2)) std::cout << "AVX2 ";
    if (has_extension(hw.simd_extensions, SIMDExtension::AVX)) std::cout << "AVX ";
    if (has_extension(hw.simd_extensions, SIMDExtension::SSE4_2)) std::cout << "SSE4.2 ";
    std::cout << "\n\n";

    // Test sizes from L1 cache to main memory
    std::vector<size_t> sizes = {
        1024,           // 4 KB - fits in L1
        8 * 1024,       // 32 KB - L1 boundary
        64 * 1024,      // 256 KB - L2 boundary
        512 * 1024,     // 2 MB - L3 boundary
        4 * 1024 * 1024 // 16 MB - main memory
    };

    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t max_size = sizes.back();
    auto a = hwy::AllocateAligned<float>(max_size);
    auto b = hwy::AllocateAligned<float>(max_size);

    for (size_t i = 0; i < max_size; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    // Print results header
    std::cout << std::setw(12) << "Size"
              << std::setw(15) << "Scalar (ns)"
              << std::setw(15) << "SIMD (ns)"
              << std::setw(15) << "Unrolled (ns)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "GFlops"
              << "\n";
    std::cout << std::string(81, '-') << "\n";

    for (size_t size : sizes) {
        // Determine iteration count based on size
        size_t iterations = std::max(size_t(1), 10000000 / size);

        // Benchmark scalar version
        Timer scalar_timer;
        volatile float scalar_result = 0;
        scalar_timer.start();
        for (size_t i = 0; i < iterations; ++i) {
            scalar_result = scalar_dot_product(a.get(), b.get(), size);
        }
        scalar_timer.stop();
        double scalar_ns = scalar_timer.elapsed_nanoseconds() / iterations;

        // Benchmark SIMD version
        Timer simd_timer;
        volatile float simd_result = 0;
        simd_timer.start();
        for (size_t i = 0; i < iterations; ++i) {
            simd_result = simd_dot_product(a.get(), b.get(), size);
        }
        simd_timer.stop();
        double simd_ns = simd_timer.elapsed_nanoseconds() / iterations;

        // Benchmark unrolled SIMD version
        Timer unrolled_timer;
        volatile float unrolled_result = 0;
        unrolled_timer.start();
        for (size_t i = 0; i < iterations; ++i) {
            unrolled_result = simd_dot_product_unrolled(a.get(), b.get(), size);
        }
        unrolled_timer.stop();
        double unrolled_ns = unrolled_timer.elapsed_nanoseconds() / iterations;

        // Calculate metrics
        double speedup = scalar_ns / std::min(simd_ns, unrolled_ns);
        double best_ns = std::min(simd_ns, unrolled_ns);
        double gflops = (2.0 * size) / best_ns;  // 2 ops per element (mul + add)

        // Format size
        std::string size_str;
        if (size >= 1024 * 1024) {
            size_str = std::to_string(size / (1024 * 1024)) + " M";
        } else if (size >= 1024) {
            size_str = std::to_string(size / 1024) + " K";
        } else {
            size_str = std::to_string(size);
        }

        std::cout << std::setw(12) << size_str
                  << std::setw(15) << std::fixed << std::setprecision(1) << scalar_ns
                  << std::setw(15) << simd_ns
                  << std::setw(15) << unrolled_ns
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(2) << gflops
                  << "\n";

        // Verify correctness
        float ref = scalar_result;
        float simd_val = simd_dot_product(a.get(), b.get(), size);
        float unrolled_val = simd_dot_product_unrolled(a.get(), b.get(), size);

        double simd_error = std::abs(simd_val - ref) / (std::abs(ref) + 1e-10);
        double unrolled_error = std::abs(unrolled_val - ref) / (std::abs(ref) + 1e-10);

        if (simd_error > 1e-5 || unrolled_error > 1e-5) {
            std::cout << "  WARNING: Correctness check failed! Rel error: "
                      << simd_error << " (simd), " << unrolled_error << " (unrolled)\n";
        }
    }

    std::cout << "\n";
    print_separator();
    std::cout << "                      Benchmark Complete\n";
    print_separator();
    std::cout << "\n";

    // Roofline analysis for the largest size
    std::cout << "Roofline Analysis (largest size = " << sizes.back() / (1024 * 1024) << " M elements):\n\n";

    size_t large_size = sizes.back();
    double bytes_accessed = 2.0 * large_size * sizeof(float);  // Read 2 arrays
    double flops = 2.0 * large_size;  // mul + add per element
    double arithmetic_intensity = flops / bytes_accessed;

    Timer t;
    t.start();
    volatile float r = simd_dot_product_unrolled(a.get(), b.get(), large_size);
    (void)r;  // Suppress unused variable warning
    t.stop();
    double elapsed_s = t.elapsed_seconds();

    double achieved_gflops = flops / (elapsed_s * 1e9);
    double achieved_bandwidth = bytes_accessed / (elapsed_s * 1e9);

    std::cout << "  Arithmetic Intensity: " << std::setprecision(4) << arithmetic_intensity << " FLOPs/byte\n";
    std::cout << "  Achieved Performance: " << std::setprecision(2) << achieved_gflops << " GFLOPs/s\n";
    std::cout << "  Achieved Bandwidth:   " << std::setprecision(2) << achieved_bandwidth << " GB/s\n";
    std::cout << "  Efficiency Note: Dot product is memory-bound (low arithmetic intensity)\n";
    std::cout << "\n";

    return 0;
}
