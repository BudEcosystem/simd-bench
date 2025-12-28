// Matrix Multiplication Example - Demonstrates SIMD-Bench for matrix operations
#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

namespace hn = hwy::HWY_NAMESPACE;
using namespace simd_bench;

// Naive scalar matrix multiplication (C = A * B)
void scalar_matmul(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Optimized scalar with loop reordering (ikj) for better cache locality
void scalar_matmul_ikj(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    std::memset(C, 0, M * N * sizeof(float));
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// SIMD matrix multiplication with Highway
void simd_matmul(float* HWY_RESTRICT C, const float* HWY_RESTRICT A,
                 const float* HWY_RESTRICT B, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            auto a_ik = hn::Set(d, A[i * K + k]);
            size_t j = 0;

            // SIMD loop
            for (; j + lanes <= N; j += lanes) {
                auto c_ij = hn::Load(d, C + i * N + j);
                auto b_kj = hn::Load(d, B + k * N + j);
                c_ij = hn::MulAdd(a_ik, b_kj, c_ij);
                hn::Store(c_ij, d, C + i * N + j);
            }

            // Remainder
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Blocked/tiled SIMD matrix multiplication for better cache utilization
void simd_matmul_blocked(float* HWY_RESTRICT C, const float* HWY_RESTRICT A,
                         const float* HWY_RESTRICT B, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    // Block sizes tuned for L1 cache (~32KB)
    // Block should fit 3 * BLOCK^2 * 4 bytes in L1
    constexpr size_t BLOCK_M = 32;
    constexpr size_t BLOCK_N = 32;
    constexpr size_t BLOCK_K = 32;

    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += BLOCK_M) {
        size_t imax = std::min(i0 + BLOCK_M, M);

        for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
            size_t kmax = std::min(k0 + BLOCK_K, K);

            for (size_t j0 = 0; j0 < N; j0 += BLOCK_N) {
                size_t jmax = std::min(j0 + BLOCK_N, N);

                // Process block
                for (size_t i = i0; i < imax; ++i) {
                    for (size_t k = k0; k < kmax; ++k) {
                        auto a_ik = hn::Set(d, A[i * K + k]);
                        size_t j = j0;

                        // SIMD inner loop
                        for (; j + lanes <= jmax; j += lanes) {
                            auto c_ij = hn::Load(d, C + i * N + j);
                            auto b_kj = hn::Load(d, B + k * N + j);
                            c_ij = hn::MulAdd(a_ik, b_kj, c_ij);
                            hn::Store(c_ij, d, C + i * N + j);
                        }

                        // Scalar remainder within block
                        for (; j < jmax; ++j) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// SIMD with 4x unrolling in j dimension
void simd_matmul_unrolled(float* HWY_RESTRICT C, const float* HWY_RESTRICT A,
                          const float* HWY_RESTRICT B, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            auto a_ik = hn::Set(d, A[i * K + k]);
            size_t j = 0;

            // 4x unrolled SIMD loop
            for (; j + 4 * lanes <= N; j += 4 * lanes) {
                auto c0 = hn::Load(d, C + i * N + j);
                auto c1 = hn::Load(d, C + i * N + j + lanes);
                auto c2 = hn::Load(d, C + i * N + j + 2 * lanes);
                auto c3 = hn::Load(d, C + i * N + j + 3 * lanes);

                auto b0 = hn::Load(d, B + k * N + j);
                auto b1 = hn::Load(d, B + k * N + j + lanes);
                auto b2 = hn::Load(d, B + k * N + j + 2 * lanes);
                auto b3 = hn::Load(d, B + k * N + j + 3 * lanes);

                c0 = hn::MulAdd(a_ik, b0, c0);
                c1 = hn::MulAdd(a_ik, b1, c1);
                c2 = hn::MulAdd(a_ik, b2, c2);
                c3 = hn::MulAdd(a_ik, b3, c3);

                hn::Store(c0, d, C + i * N + j);
                hn::Store(c1, d, C + i * N + j + lanes);
                hn::Store(c2, d, C + i * N + j + 2 * lanes);
                hn::Store(c3, d, C + i * N + j + 3 * lanes);
            }

            // Single SIMD loop for remainder
            for (; j + lanes <= N; j += lanes) {
                auto c_ij = hn::Load(d, C + i * N + j);
                auto b_kj = hn::Load(d, B + k * N + j);
                c_ij = hn::MulAdd(a_ik, b_kj, c_ij);
                hn::Store(c_ij, d, C + i * N + j);
            }

            // Scalar remainder
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void print_separator() {
    std::cout << std::string(80, '=') << std::endl;
}

bool verify_result(const float* C, const float* C_ref, size_t M, size_t N, double tolerance = 1e-4) {
    for (size_t i = 0; i < M * N; ++i) {
        double diff = std::abs(C[i] - C_ref[i]);
        double ref_val = std::abs(C_ref[i]);
        double rel_error = diff / (ref_val + 1e-10);
        if (rel_error > tolerance && diff > 1e-6) {
            std::cout << "Mismatch at index " << i << ": expected " << C_ref[i]
                      << ", got " << C[i] << " (rel error: " << rel_error << ")\n";
            return false;
        }
    }
    return true;
}

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "\n";
    print_separator();
    std::cout << "          SIMD-Bench: Matrix Multiplication Benchmark Example\n";
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
    std::cout << "  Peak GFLOPS (est): " << std::fixed << std::setprecision(1)
              << hw.theoretical_peak_sp_gflops << " (SP), " << hw.theoretical_peak_dp_gflops << " (DP)\n";
    std::cout << "\n";

    // Test different matrix sizes
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    // Random initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t max_size = sizes.back();
    auto A = hwy::AllocateAligned<float>(max_size * max_size);
    auto B = hwy::AllocateAligned<float>(max_size * max_size);
    auto C = hwy::AllocateAligned<float>(max_size * max_size);
    auto C_ref = hwy::AllocateAligned<float>(max_size * max_size);

    for (size_t i = 0; i < max_size * max_size; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    // Print results header
    std::cout << std::setw(8) << "Size"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "IKJ (ms)"
              << std::setw(14) << "SIMD (ms)"
              << std::setw(14) << "Blocked (ms)"
              << std::setw(14) << "Unrolled (ms)"
              << std::setw(10) << "GFLOPs"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (size_t N : sizes) {
        size_t M = N, K = N;  // Square matrices
        double flops = 2.0 * M * N * K;  // Each element: K muls + K adds

        // Determine iterations
        size_t iterations = std::max(size_t(1), 100000000 / (N * N * N));

        // Benchmark naive scalar
        Timer naive_timer;
        naive_timer.start();
        for (size_t iter = 0; iter < iterations; ++iter) {
            scalar_matmul(C_ref.get(), A.get(), B.get(), M, N, K);
        }
        naive_timer.stop();
        double naive_ms = naive_timer.elapsed_nanoseconds() / (iterations * 1e6);

        // Benchmark IKJ scalar
        Timer ikj_timer;
        ikj_timer.start();
        for (size_t iter = 0; iter < iterations; ++iter) {
            scalar_matmul_ikj(C.get(), A.get(), B.get(), M, N, K);
        }
        ikj_timer.stop();
        double ikj_ms = ikj_timer.elapsed_nanoseconds() / (iterations * 1e6);

        // Benchmark SIMD
        Timer simd_timer;
        simd_timer.start();
        for (size_t iter = 0; iter < iterations; ++iter) {
            simd_matmul(C.get(), A.get(), B.get(), M, N, K);
        }
        simd_timer.stop();
        double simd_ms = simd_timer.elapsed_nanoseconds() / (iterations * 1e6);

        // Benchmark blocked SIMD
        Timer blocked_timer;
        blocked_timer.start();
        for (size_t iter = 0; iter < iterations; ++iter) {
            simd_matmul_blocked(C.get(), A.get(), B.get(), M, N, K);
        }
        blocked_timer.stop();
        double blocked_ms = blocked_timer.elapsed_nanoseconds() / (iterations * 1e6);

        // Benchmark unrolled SIMD
        Timer unrolled_timer;
        unrolled_timer.start();
        for (size_t iter = 0; iter < iterations; ++iter) {
            simd_matmul_unrolled(C.get(), A.get(), B.get(), M, N, K);
        }
        unrolled_timer.stop();
        double unrolled_ms = unrolled_timer.elapsed_nanoseconds() / (iterations * 1e6);

        // Find best time and calculate GFLOPs
        double best_ms = std::min({simd_ms, blocked_ms, unrolled_ms});
        double gflops = flops / (best_ms * 1e6);

        std::cout << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(3) << naive_ms
                  << std::setw(14) << ikj_ms
                  << std::setw(14) << simd_ms
                  << std::setw(14) << blocked_ms
                  << std::setw(14) << unrolled_ms
                  << std::setw(10) << std::setprecision(1) << gflops
                  << "\n";

        // Verify correctness of best SIMD variant
        if (blocked_ms <= simd_ms && blocked_ms <= unrolled_ms) {
            simd_matmul_blocked(C.get(), A.get(), B.get(), M, N, K);
        } else if (unrolled_ms < simd_ms) {
            simd_matmul_unrolled(C.get(), A.get(), B.get(), M, N, K);
        } else {
            simd_matmul(C.get(), A.get(), B.get(), M, N, K);
        }

        if (!verify_result(C.get(), C_ref.get(), M, N)) {
            std::cout << "  CORRECTNESS FAILURE for N=" << N << "\n";
        }
    }

    std::cout << "\n";
    print_separator();
    std::cout << "                        Performance Analysis\n";
    print_separator();
    std::cout << "\n";

    // Detailed analysis for 512x512
    size_t analysis_N = 512;
    if (std::find(sizes.begin(), sizes.end(), analysis_N) != sizes.end()) {
        size_t M = analysis_N, N = analysis_N, K = analysis_N;
        double flops = 2.0 * M * N * K;
        double bytes_A = M * K * sizeof(float);
        double bytes_B = K * N * sizeof(float);
        double bytes_C = M * N * sizeof(float);
        (void)(bytes_A + bytes_B + bytes_C);  // total_bytes for reference

        // Actual memory traffic is much higher due to repeated access
        // For naive: each B element read M times, each A row read N times
        double naive_bytes = M * N * K * sizeof(float) * 2 + M * N * sizeof(float);
        // For blocked: roughly 3 * M * N * K / BLOCK reads
        double blocked_bytes = 3 * M * N * sizeof(float) * (K / 32.0);

        double arithmetic_intensity_naive = flops / naive_bytes;
        double arithmetic_intensity_blocked = flops / blocked_bytes;

        std::cout << "Analysis for " << analysis_N << "x" << analysis_N << " matrix multiplication:\n\n";
        std::cout << "  Total FLOPs: " << std::scientific << std::setprecision(2) << flops << "\n";
        std::cout << "  Matrix sizes: A=" << bytes_A / 1024 << " KB, B=" << bytes_B / 1024
                  << " KB, C=" << bytes_C / 1024 << " KB\n";
        std::cout << "\n";
        std::cout << "  Arithmetic Intensity (naive):   ~" << std::fixed << std::setprecision(2)
                  << arithmetic_intensity_naive << " FLOPs/byte\n";
        std::cout << "  Arithmetic Intensity (blocked): ~" << arithmetic_intensity_blocked
                  << " FLOPs/byte\n";
        std::cout << "\n";
        std::cout << "  Peak theoretical GFLOPs: " << std::setprecision(1) << hw.theoretical_peak_sp_gflops << "\n";

        // Measure actual performance
        Timer t;
        t.start();
        simd_matmul_blocked(C.get(), A.get(), B.get(), M, N, K);
        t.stop();
        double achieved_gflops = flops / (t.elapsed_nanoseconds());
        double efficiency = achieved_gflops / hw.theoretical_peak_sp_gflops * 100;

        std::cout << "  Achieved GFLOPs (blocked):      " << std::setprecision(2) << achieved_gflops << "\n";
        std::cout << "  Efficiency vs peak:             " << std::setprecision(1) << efficiency << "%\n";
        std::cout << "\n";
        std::cout << "  Notes:\n";
        std::cout << "    - Loop reordering (IKJ) improves cache locality significantly\n";
        std::cout << "    - SIMD vectorizes the inner loop across j dimension\n";
        std::cout << "    - Blocking improves cache reuse for larger matrices\n";
        std::cout << "    - For peak performance, consider:\n";
        std::cout << "      * Register blocking (micro-kernels)\n";
        std::cout << "      * Prefetching\n";
        std::cout << "      * Multi-threading\n";
        std::cout << "      * Better blocking factors tuned to cache sizes\n";
    }

    std::cout << "\n";
    print_separator();
    std::cout << "                      Benchmark Complete\n";
    print_separator();
    std::cout << "\n";

    return 0;
}
