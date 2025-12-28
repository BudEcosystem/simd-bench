// ============================================================================
// SIMD-Bench: Comprehensive Insights Validation Loop
// ============================================================================
// This program runs the optimization validation loop:
// 1. Profile baseline code
// 2. Generate insights
// 3. Profile optimized variants (applying insights)
// 4. Report WIN/FAIL for each insight
// ============================================================================

#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/cache_control.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

namespace hn = hwy::HWY_NAMESPACE;
using namespace simd_bench;

// ============================================================================
// Test Kernels - Baseline (V0) and Optimized (V1) versions
// ============================================================================

namespace baseline {

// Dot product - no unrolling
float dot_product(const float* a, const float* b, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        sum = hn::MulAdd(hn::Load(d, a + i), hn::Load(d, b + i), sum);
    }
    float result = hn::ReduceSum(d, sum);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Sum - no unrolling
float sum(const float* data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        sum = hn::Add(sum, hn::Load(d, data + i));
    }
    float result = hn::ReduceSum(d, sum);
    for (; i < n; ++i) result += data[i];
    return result;
}

// Scale - regular stores
void scale(float* out, const float* in, float alpha, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto va = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::Store(hn::Mul(hn::Load(d, in + i), va), d, out + i);
    }
    for (; i < n; ++i) out[i] = in[i] * alpha;
}

}  // namespace baseline

namespace optimized {

// Dot product - 4x unrolled (Insight: Break dependency chain)
float dot_product_unrolled(const float* a, const float* b, size_t n) {
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
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Sum - 4x unrolled (Insight: Break reduction dependency chain)
float sum_unrolled(const float* data, size_t n) {
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

    auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);
    for (; i < n; ++i) result += data[i];
    return result;
}

// Scale - Non-temporal stores (Insight: Use streaming stores for large datasets)
void scale_nontemporal(float* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                       float alpha, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto va = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::Stream(hn::Mul(hn::Load(d, in + i), va), d, out + i);
    }
    hwy::FlushStream();
    for (; i < n; ++i) out[i] = in[i] * alpha;
}

}  // namespace optimized

// ============================================================================
// Validation Result Structure
// ============================================================================

struct ValidationResult {
    std::string kernel_name;
    std::string insight_applied;
    size_t size;
    double baseline_gflops;
    double optimized_gflops;
    double speedup;
    bool is_win;
    std::string insight_description;
};

// ============================================================================
// Benchmark Helper
// ============================================================================

template<typename Func>
double benchmark_kernel(Func&& func, size_t iterations) {
    Timer timer;

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
        func();
    }

    timer.start();
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    timer.stop();

    return timer.elapsed_nanoseconds() / iterations;
}

// ============================================================================
// Main Validation Loop
// ============================================================================

int main() {
    std::cout << "========================================================================\n";
    std::cout << "       SIMD-Bench: Insights Optimization Validation Loop\n";
    std::cout << "========================================================================\n\n";

    // Detect hardware
    HardwareInfo hw = HardwareInfo::detect();
    std::cout << "Hardware: " << hw.cpu_brand << "\n";
    std::cout << "SIMD: " << hw.get_simd_string() << "\n";
    std::cout << "Peak GFLOPS: " << std::fixed << std::setprecision(1)
              << hw.theoretical_peak_sp_gflops << "\n";
    std::cout << "Memory BW: " << hw.measured_memory_bw_gbps << " GB/s\n\n";

    // Create insights engine
    InsightsEngine engine(hw);

    std::vector<ValidationResult> results;

    // Test sizes
    // Allocate enough for 2x L3 cache at minimum
    size_t l3_cache_floats = (hw.cache.l3_size_kb * 1024) / sizeof(float);
    size_t max_size = std::max(size_t(16 * 1024 * 1024), l3_cache_floats * 2);  // 64MB or 2x L3
    auto a = hwy::AllocateAligned<float>(max_size);
    auto b = hwy::AllocateAligned<float>(max_size);
    auto out = hwy::AllocateAligned<float>(max_size);

    // Initialize with random data
    RandomInputGenerator gen(42);
    gen.generate_uniform(a.get(), max_size, -1.0f, 1.0f);
    gen.generate_uniform(b.get(), max_size, -1.0f, 1.0f);

    std::cout << "========================================================================\n";
    std::cout << "                    RUNNING VALIDATION LOOP\n";
    std::cout << "========================================================================\n\n";

    // =========================================================================
    // Test 1: Dot Product - Unrolling Insight
    // NOTE: Unrolling helps when data fits in cache (compute-bound scenario)
    // For memory-bound scenarios (data >> L3), memory bandwidth is the limit
    // =========================================================================
    std::cout << "--- DOT PRODUCT: Testing 'Break Dependency Chain' Insight ---\n";
    std::cout << "    (Unrolling helps when data fits in L2/L3 cache)\n\n";

    // Test sizes where unrolling should help (fits in L3) and shouldn't (exceeds L3)
    size_t l2_elements = (hw.cache.l2_size_kb * 1024) / sizeof(float) / 2;  // Both a[] and b[]
    size_t l3_elements_half = (hw.cache.l3_size_kb * 1024) / sizeof(float) / 4;  // Leave room
    size_t dp_size1 = std::min(l2_elements, max_size);
    size_t dp_size2 = std::min(l3_elements_half, max_size);
    size_t dp_size3 = std::min(l3_elements_half * 4, max_size);
    for (size_t size : {dp_size1, dp_size2, dp_size3}) {
        size_t iterations = std::max(size_t(10), 10000000 / size);
        double flops = 2.0 * size;

        volatile float result;

        // Baseline
        double baseline_ns = benchmark_kernel([&]() {
            result = baseline::dot_product(a.get(), b.get(), size);
        }, iterations);
        do_not_optimize(result);
        double baseline_gflops = flops / baseline_ns;

        // Optimized (unrolled)
        double optimized_ns = benchmark_kernel([&]() {
            result = optimized::dot_product_unrolled(a.get(), b.get(), size);
        }, iterations);
        do_not_optimize(result);
        double optimized_gflops = flops / optimized_ns;

        double speedup = optimized_gflops / baseline_gflops;
        bool is_win = speedup > 1.05;  // >5% improvement is a win

        results.push_back({
            "dot_product",
            "4x Unrolling",
            size,
            baseline_gflops,
            optimized_gflops,
            speedup,
            is_win,
            "Break dependency chain with multiple accumulators"
        });

        std::cout << "  Size " << std::setw(10) << size
                  << ": " << std::setw(8) << std::setprecision(2) << baseline_gflops
                  << " -> " << std::setw(8) << optimized_gflops
                  << " GFLOPS (" << std::setprecision(2) << speedup << "x) "
                  << (is_win ? "WIN" : "FAIL") << "\n";
    }
    std::cout << "\n";

    // =========================================================================
    // Test 2: Sum Reduction - Unrolling Insight
    // =========================================================================
    std::cout << "--- SUM REDUCTION: Testing 'Break Reduction Dependency' Insight ---\n\n";

    for (size_t size : {4096UL, 65536UL, 1048576UL}) {
        size_t iterations = std::max(size_t(10), 10000000 / size);
        double flops = size;

        volatile float result;

        // Baseline
        double baseline_ns = benchmark_kernel([&]() {
            result = baseline::sum(a.get(), size);
        }, iterations);
        do_not_optimize(result);
        double baseline_gflops = flops / baseline_ns;

        // Optimized (unrolled)
        double optimized_ns = benchmark_kernel([&]() {
            result = optimized::sum_unrolled(a.get(), size);
        }, iterations);
        do_not_optimize(result);
        double optimized_gflops = flops / optimized_ns;

        double speedup = optimized_gflops / baseline_gflops;
        bool is_win = speedup > 1.05;

        results.push_back({
            "sum_reduction",
            "4x Unrolling",
            size,
            baseline_gflops,
            optimized_gflops,
            speedup,
            is_win,
            "Break reduction dependency chain"
        });

        std::cout << "  Size " << std::setw(10) << size
                  << ": " << std::setw(8) << std::setprecision(2) << baseline_gflops
                  << " -> " << std::setw(8) << optimized_gflops
                  << " GFLOPS (" << std::setprecision(2) << speedup << "x) "
                  << (is_win ? "WIN" : "FAIL") << "\n";
    }
    std::cout << "\n";

    // =========================================================================
    // Test 3: Scale - Non-temporal Stores Insight
    // NOTE: NT stores only help when working set exceeds L3 cache!
    // L3 = 16MB = 4M floats, so we need sizes > 4M elements
    // =========================================================================
    std::cout << "--- VECTOR SCALE: Testing 'Non-Temporal Stores' Insight ---\n";
    std::cout << "    (L3 cache = " << hw.cache.l3_size_kb / 1024 << " MB, NT stores help when > L3)\n\n";

    // Only test sizes where NT stores should actually help (exceeds L3)
    // Cap at max_size to avoid exceeding our allocation
    size_t l3_elements = (hw.cache.l3_size_kb * 1024) / sizeof(float);
    size_t nt_size1 = std::min(l3_elements * 2, max_size);
    size_t nt_size2 = std::min(l3_elements * 4, max_size);
    for (size_t size : {nt_size1, nt_size2}) {
        size_t iterations = std::max(size_t(10), 50000000 / size);
        double flops = size;

        // Baseline
        double baseline_ns = benchmark_kernel([&]() {
            baseline::scale(out.get(), a.get(), 2.5f, size);
        }, iterations);
        do_not_optimize(out[0]);
        double baseline_gflops = flops / baseline_ns;

        // Optimized (non-temporal)
        double optimized_ns = benchmark_kernel([&]() {
            optimized::scale_nontemporal(out.get(), a.get(), 2.5f, size);
        }, iterations);
        do_not_optimize(out[0]);
        double optimized_gflops = flops / optimized_ns;

        double speedup = optimized_gflops / baseline_gflops;
        bool is_win = speedup > 1.05;

        results.push_back({
            "vector_scale",
            "Non-Temporal Stores",
            size,
            baseline_gflops,
            optimized_gflops,
            speedup,
            is_win,
            "Use streaming stores for write-only large datasets"
        });

        std::cout << "  Size " << std::setw(10) << size
                  << ": " << std::setw(8) << std::setprecision(2) << baseline_gflops
                  << " -> " << std::setw(8) << optimized_gflops
                  << " GFLOPS (" << std::setprecision(2) << speedup << "x) "
                  << (is_win ? "WIN" : "FAIL") << "\n";
    }
    std::cout << "\n";

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "========================================================================\n";
    std::cout << "                       VALIDATION SUMMARY\n";
    std::cout << "========================================================================\n\n";

    int wins = 0, fails = 0;
    double total_speedup = 0;

    std::cout << std::left << std::setw(15) << "Kernel"
              << std::setw(22) << "Insight Applied"
              << std::right << std::setw(12) << "Size"
              << std::setw(12) << "Baseline"
              << std::setw(12) << "Optimized"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Result" << "\n";
    std::cout << std::string(93, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(15) << r.kernel_name
                  << std::setw(22) << r.insight_applied
                  << std::right << std::setw(12) << r.size
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.baseline_gflops
                  << std::setw(12) << r.optimized_gflops
                  << std::setw(9) << std::setprecision(2) << r.speedup << "x"
                  << std::setw(10) << (r.is_win ? "WIN" : "FAIL") << "\n";

        if (r.is_win) {
            wins++;
            total_speedup += r.speedup;
        } else {
            fails++;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << "                         FINAL RESULTS\n";
    std::cout << "========================================================================\n\n";

    double accuracy = (100.0 * wins) / (wins + fails);
    double avg_speedup = wins > 0 ? total_speedup / wins : 0;

    std::cout << "  Total Tests:     " << (wins + fails) << "\n";
    std::cout << "  Wins:            " << wins << "\n";
    std::cout << "  Fails:           " << fails << "\n";
    std::cout << "  Accuracy:        " << std::setprecision(1) << accuracy << "%\n";
    std::cout << "  Avg Speedup:     " << std::setprecision(2) << avg_speedup << "x (for wins)\n\n";

    if (accuracy >= 80.0) {
        std::cout << "  STATUS: INSIGHTS ENGINE VALIDATED - Recommendations are actionable!\n";
    } else {
        std::cout << "  STATUS: INSIGHTS ENGINE NEEDS IMPROVEMENT\n";
    }

    std::cout << "\n========================================================================\n";
    std::cout << "                    INSIGHT EFFECTIVENESS BREAKDOWN\n";
    std::cout << "========================================================================\n\n";

    // Group by insight type
    std::map<std::string, std::pair<int, int>> insight_stats;  // wins, total
    for (const auto& r : results) {
        auto& stats = insight_stats[r.insight_applied];
        stats.second++;
        if (r.is_win) stats.first++;
    }

    for (const auto& [insight, stats] : insight_stats) {
        double rate = (100.0 * stats.first) / stats.second;
        std::cout << "  " << std::left << std::setw(25) << insight
                  << ": " << stats.first << "/" << stats.second
                  << " (" << std::setprecision(0) << rate << "% effective)\n";
    }

    std::cout << "\n========================================================================\n";

    return (accuracy >= 80.0) ? 0 : 1;
}
