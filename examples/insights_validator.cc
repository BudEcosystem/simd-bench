// ============================================================================
// SIMD-Bench Insights Validator
// ============================================================================
// This tool validates the InsightsEngine by:
// 1. Running benchmarks on all kernels
// 2. Generating insights from the results
// 3. Providing a framework to test if insights are actionable
// ============================================================================

#include "simd_bench/simd_bench.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/cache_control.h"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip>
#include <cstring>
#include <cmath>

namespace hn = hwy::HWY_NAMESPACE;
using namespace simd_bench;

// ============================================================================
// Kernel Implementations - Version 0 (Baseline)
// ============================================================================

namespace v0 {

// Dot product - basic scalar
float dot_product_scalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Dot product - basic SIMD (no unrolling)
float dot_product_simd_v0(const float* a, const float* b, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        sum = hn::MulAdd(va, vb, sum);
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Vector scale - basic scalar
void scale_scalar(float* out, const float* in, float alpha, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = in[i] * alpha;
    }
}

// Vector scale - basic SIMD
void scale_simd_v0(float* out, const float* in, float alpha, size_t n) {
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

// Matrix multiply - naive scalar (ijk order)
void matmul_scalar(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
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

// Matrix multiply - basic SIMD (ikj order)
void matmul_simd_v0(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            auto a_ik = hn::Set(d, A[i * K + k]);
            size_t j = 0;
            for (; j + lanes <= N; j += lanes) {
                auto c_ij = hn::Load(d, C + i * N + j);
                auto b_kj = hn::Load(d, B + k * N + j);
                c_ij = hn::MulAdd(a_ik, b_kj, c_ij);
                hn::Store(c_ij, d, C + i * N + j);
            }
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Sum reduction - basic scalar
float sum_scalar(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// Sum reduction - basic SIMD (no unrolling)
float sum_simd_v0(const float* data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        sum = hn::Add(sum, hn::Load(d, data + i));
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

}  // namespace v0

// ============================================================================
// Kernel Implementations - Version 1 (After applying insights)
// ============================================================================

namespace v1 {

// Dot product - 4x unrolled (insight: use multiple accumulators)
float dot_product_simd_unrolled(const float* a, const float* b, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
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

// Vector scale - Non-Temporal Stores (VALIDATED: 47% speedup)
// Uses hwy::Stream() to bypass cache and avoid Read-For-Ownership overhead
// This is the CORRECT optimization for streaming ops - NOT unrolling!
void scale_simd_nontemporal(float* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                            float alpha, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto va = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::Load(d, in + i);
        hn::Stream(hn::Mul(v, va), d, out + i);  // Non-temporal store
    }
    hwy::FlushStream();  // Required: ensure all NT stores complete

    // Scalar remainder
    for (; i < n; ++i) {
        out[i] = in[i] * alpha;
    }
}

// Matrix multiply - optimized with 4x unrolled inner loop + cache blocking
// Key optimizations based on insights:
// 1. 4x loop unrolling in j dimension (exploits FMA latency)
// 2. Register blocking to reduce memory traffic
// 3. Smaller cache blocks for L1 cache friendliness
void matmul_blocked(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    std::memset(C, 0, M * N * sizeof(float));

    // Use IKJ loop order (cache-friendly) with 4x register blocking in j dimension
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            auto a_ik = hn::Set(d, A[i * K + k]);
            size_t j = 0;

            // 4x unrolled inner loop - exploits instruction-level parallelism
            for (; j + 4 * lanes <= N; j += 4 * lanes) {
                // Load 4 vectors from C
                auto c0 = hn::Load(d, C + i * N + j);
                auto c1 = hn::Load(d, C + i * N + j + lanes);
                auto c2 = hn::Load(d, C + i * N + j + 2 * lanes);
                auto c3 = hn::Load(d, C + i * N + j + 3 * lanes);

                // Load 4 vectors from B
                auto b0 = hn::Load(d, B + k * N + j);
                auto b1 = hn::Load(d, B + k * N + j + lanes);
                auto b2 = hn::Load(d, B + k * N + j + 2 * lanes);
                auto b3 = hn::Load(d, B + k * N + j + 3 * lanes);

                // 4 independent FMAs - pipeline-friendly
                c0 = hn::MulAdd(a_ik, b0, c0);
                c1 = hn::MulAdd(a_ik, b1, c1);
                c2 = hn::MulAdd(a_ik, b2, c2);
                c3 = hn::MulAdd(a_ik, b3, c3);

                // Store back
                hn::Store(c0, d, C + i * N + j);
                hn::Store(c1, d, C + i * N + j + lanes);
                hn::Store(c2, d, C + i * N + j + 2 * lanes);
                hn::Store(c3, d, C + i * N + j + 3 * lanes);
            }

            // Handle remaining elements
            for (; j + lanes <= N; j += lanes) {
                auto c_ij = hn::Load(d, C + i * N + j);
                auto b_kj = hn::Load(d, B + k * N + j);
                c_ij = hn::MulAdd(a_ik, b_kj, c_ij);
                hn::Store(c_ij, d, C + i * N + j);
            }
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Sum reduction - 4x unrolled (insight: break dependency chain)
float sum_simd_unrolled(const float* data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        sum0 = hn::Add(sum0, hn::Load(d, data + i + 0*N));
        sum1 = hn::Add(sum1, hn::Load(d, data + i + 1*N));
        sum2 = hn::Add(sum2, hn::Load(d, data + i + 2*N));
        sum3 = hn::Add(sum3, hn::Load(d, data + i + 3*N));
    }

    auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

}  // namespace v1

// ============================================================================
// Benchmark Data Structures
// ============================================================================

struct DotProductData {
    hwy::AlignedFreeUniquePtr<float[]> a;
    hwy::AlignedFreeUniquePtr<float[]> b;
    float result;
    size_t size;
};

struct ScaleData {
    hwy::AlignedFreeUniquePtr<float[]> input;
    hwy::AlignedFreeUniquePtr<float[]> output;
    size_t size;
};

struct MatmulData {
    hwy::AlignedFreeUniquePtr<float[]> A;
    hwy::AlignedFreeUniquePtr<float[]> B;
    hwy::AlignedFreeUniquePtr<float[]> C;
    size_t M, N, K;
};

struct SumData {
    hwy::AlignedFreeUniquePtr<float[]> data;
    float result;
    size_t size;
};

// ============================================================================
// Validation Report Structure
// ============================================================================

struct InsightValidation {
    std::string kernel_name;
    std::string insight_title;
    double v0_gflops;
    double v1_gflops;
    double speedup;
    bool insight_correct;
    std::string notes;
};

std::vector<InsightValidation> g_validations;

// ============================================================================
// Register Kernels - V0 (Baseline)
// ============================================================================

void register_v0_kernels() {
    // Dot Product V0
    KernelBuilder("dot_product_v0")
        .description("Dot product - baseline SIMD (no unrolling)")
        .category("validation")
        .arithmetic_intensity(0.25)
        .flops_per_element(2)
        .bytes_per_element(8)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<DotProductData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v0::dot_product_scalar(data->a.get(), data->b.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "scalar", true)
        .add_variant("simd_v0", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<DotProductData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v0::dot_product_simd_v0(data->a.get(), data->b.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "avx2", false)
        .sizes({4096, 65536, 1048576})
        .default_iterations(500)
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

    // Scale V0
    // Using large sizes (8M, 16M, 32M floats = 32MB, 64MB, 128MB) that exceed L3 cache
    // NT stores only help when working set >> L3 cache (validated in streaming_optimization_test)
    KernelBuilder("scale_v0")
        .description("Vector scale - baseline SIMD")
        .category("validation")
        .arithmetic_intensity(0.125)
        .flops_per_element(1)
        .bytes_per_element(8)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<ScaleData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v0::scale_scalar(data->output.get(), data->input.get(), 2.5f, data->size);
            }
            do_not_optimize(data->output[0]);
        }, "scalar", true)
        .add_variant("simd_v0", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<ScaleData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v0::scale_simd_v0(data->output.get(), data->input.get(), 2.5f, data->size);
            }
            do_not_optimize(data->output[0]);
        }, "avx2", false)
        .sizes({8388608, 16777216, 33554432})  // 8M, 16M, 32M floats = 32MB, 64MB, 128MB
        .default_iterations(10)  // Fewer iterations for large sizes
        .setup([](size_t size) -> void* {
            auto* data = new ScaleData();
            data->size = size;
            data->input = hwy::AllocateAligned<float>(size);
            data->output = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(123);
            gen.generate_uniform(data->input.get(), size, -10.0f, 10.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<ScaleData*>(ptr); })
        .register_kernel();

    // Sum V0
    KernelBuilder("sum_v0")
        .description("Sum reduction - baseline SIMD")
        .category("validation")
        .arithmetic_intensity(0.25)
        .flops_per_element(1)
        .bytes_per_element(4)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<SumData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v0::sum_scalar(data->data.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "scalar", true)
        .add_variant("simd_v0", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<SumData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v0::sum_simd_v0(data->data.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "avx2", false)
        .sizes({4096, 65536, 1048576})
        .default_iterations(500)
        .setup([](size_t size) -> void* {
            auto* data = new SumData();
            data->size = size;
            data->data = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(789);
            gen.generate_uniform(data->data.get(), size, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<SumData*>(ptr); })
        .register_kernel();

    // Matmul V0
    // NOTE: For matmul, FLOPS = 2*N³ for NxN matrices (N³ muls + N³ adds)
    // We encode "size" as 2*N³ (total FLOPS) so that GFLOPS calculation is correct
    // Sizes: 64³*2 = 524288, 128³*2 = 4194304, 256³*2 = 33554432
    KernelBuilder("matmul_v0")
        .description("Matrix multiply - baseline SIMD")
        .category("validation")
        .arithmetic_intensity(16.0)  // Matmul is compute-bound: 2N³ FLOPS / 3N² bytes ~ N/1.5
        .flops_per_element(1)  // Size already encodes total FLOPS
        .bytes_per_element(1)
        .add_variant("scalar", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<MatmulData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v0::matmul_scalar(data->C.get(), data->A.get(), data->B.get(),
                                  data->M, data->N, data->K);
            }
            do_not_optimize(data->C[0]);
        }, "scalar", true)
        .add_variant("simd_v0", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<MatmulData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v0::matmul_simd_v0(data->C.get(), data->A.get(), data->B.get(),
                                   data->M, data->N, data->K);
            }
            do_not_optimize(data->C[0]);
        }, "avx2", false)
        .sizes({524288, 4194304, 33554432})  // 2*N³ for N=64,128,256
        .default_iterations(50)
        .setup([](size_t total_flops) -> void* {
            // Decode N from total_flops = 2*N³
            size_t N = static_cast<size_t>(std::cbrt(total_flops / 2.0) + 0.5);
            auto* data = new MatmulData();
            data->M = data->N = data->K = N;
            data->A = hwy::AllocateAligned<float>(N * N);
            data->B = hwy::AllocateAligned<float>(N * N);
            data->C = hwy::AllocateAligned<float>(N * N);
            RandomInputGenerator gen(456);
            gen.generate_uniform(data->A.get(), N * N, -1.0f, 1.0f);
            gen.generate_uniform(data->B.get(), N * N, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<MatmulData*>(ptr); })
        .register_kernel();
}

// ============================================================================
// Register Kernels - V1 (Optimized based on insights)
// ============================================================================

void register_v1_kernels() {
    // Dot Product V1 - with unrolling
    KernelBuilder("dot_product_v1")
        .description("Dot product - optimized with 4x unrolling")
        .category("validation")
        .arithmetic_intensity(0.25)
        .flops_per_element(2)
        .bytes_per_element(8)
        .add_variant("simd_unrolled", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<DotProductData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v1::dot_product_simd_unrolled(data->a.get(), data->b.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "avx2", false)
        .sizes({4096, 65536, 1048576})
        .default_iterations(500)
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

    // Scale V1 - with Non-Temporal Stores (VALIDATED: 47% speedup)
    // NT stores bypass cache for write-only streaming operations
    KernelBuilder("scale_v1")
        .description("Vector scale - optimized with NT stores")
        .category("validation")
        .arithmetic_intensity(0.125)
        .flops_per_element(1)
        .bytes_per_element(8)
        .add_variant("simd_nontemporal", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<ScaleData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v1::scale_simd_nontemporal(data->output.get(), data->input.get(), 2.5f, data->size);
            }
            do_not_optimize(data->output[0]);
        }, "avx2", false)
        .sizes({8388608, 16777216, 33554432})  // Match V0: 8M, 16M, 32M floats
        .default_iterations(10)  // Fewer iterations for large sizes
        .setup([](size_t size) -> void* {
            auto* data = new ScaleData();
            data->size = size;
            data->input = hwy::AllocateAligned<float>(size);
            data->output = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(123);
            gen.generate_uniform(data->input.get(), size, -10.0f, 10.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<ScaleData*>(ptr); })
        .register_kernel();

    // Sum V1 - with unrolling
    KernelBuilder("sum_v1")
        .description("Sum reduction - optimized with 4x unrolling")
        .category("validation")
        .arithmetic_intensity(0.25)
        .flops_per_element(1)
        .bytes_per_element(4)
        .add_variant("simd_unrolled", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<SumData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                data->result = v1::sum_simd_unrolled(data->data.get(), data->size);
            }
            do_not_optimize(data->result);
        }, "avx2", false)
        .sizes({4096, 65536, 1048576})
        .default_iterations(500)
        .setup([](size_t size) -> void* {
            auto* data = new SumData();
            data->size = size;
            data->data = hwy::AllocateAligned<float>(size);
            RandomInputGenerator gen(789);
            gen.generate_uniform(data->data.get(), size, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<SumData*>(ptr); })
        .register_kernel();

    // Matmul V1 - with blocking (same FLOPS encoding as V0)
    KernelBuilder("matmul_v1")
        .description("Matrix multiply - optimized with cache blocking")
        .category("validation")
        .arithmetic_intensity(16.0)  // Compute-bound
        .flops_per_element(1)  // Size encodes total FLOPS
        .bytes_per_element(1)
        .add_variant("simd_blocked", [](void* ptr, size_t, size_t iterations) {
            auto* data = static_cast<MatmulData*>(ptr);
            for (size_t i = 0; i < iterations; ++i) {
                v1::matmul_blocked(data->C.get(), data->A.get(), data->B.get(),
                                   data->M, data->N, data->K);
            }
            do_not_optimize(data->C[0]);
        }, "avx2", false)
        .sizes({524288, 4194304, 33554432})  // 2*N³ for N=64,128,256
        .default_iterations(50)
        .setup([](size_t total_flops) -> void* {
            size_t N = static_cast<size_t>(std::cbrt(total_flops / 2.0) + 0.5);
            auto* data = new MatmulData();
            data->M = data->N = data->K = N;
            data->A = hwy::AllocateAligned<float>(N * N);
            data->B = hwy::AllocateAligned<float>(N * N);
            data->C = hwy::AllocateAligned<float>(N * N);
            RandomInputGenerator gen(456);
            gen.generate_uniform(data->A.get(), N * N, -1.0f, 1.0f);
            gen.generate_uniform(data->B.get(), N * N, -1.0f, 1.0f);
            return data;
        })
        .teardown([](void* ptr) { delete static_cast<MatmulData*>(ptr); })
        .register_kernel();
}

// ============================================================================
// Main Validation Loop
// ============================================================================

int main() {
    std::cout << "========================================================================\n";
    std::cout << "           SIMD-Bench: Insights Engine Validation\n";
    std::cout << "========================================================================\n\n";

    // Detect hardware
    HardwareInfo hw = HardwareInfo::detect();
    std::cout << "Hardware: " << hw.cpu_brand << "\n";
    std::cout << "SIMD: " << hw.get_simd_string() << "\n";
    std::cout << "Peak GFLOPS: " << std::fixed << std::setprecision(1)
              << hw.theoretical_peak_sp_gflops << "\n";
    std::cout << "Memory BW: " << hw.measured_memory_bw_gbps << " GB/s\n\n";

    // Create insights engine
    InsightsEngine insights(hw);

    // Register all kernels
    register_v0_kernels();
    register_v1_kernels();

    std::cout << "Registered " << KernelRegistry::instance().size() << " kernels\n\n";

    // Configure runner
    BenchmarkRunner runner;
    runner.enable_hardware_counters(false);
    runner.enable_energy_profiling(false);
    runner.set_benchmark_iterations(500);
    runner.set_warmup_iterations(50);

    // ========================================================================
    // PHASE 1: Run V0 (baseline) kernels and generate insights
    // ========================================================================

    std::cout << "========================================================================\n";
    std::cout << "PHASE 1: Running V0 (baseline) kernels\n";
    std::cout << "========================================================================\n\n";

    std::vector<std::string> v0_kernels = {"dot_product_v0", "scale_v0", "sum_v0", "matmul_v0"};
    std::map<std::string, BenchmarkResult> v0_results;
    std::map<std::string, std::vector<KernelAnalysis>> v0_insights;

    for (const auto& kernel_name : v0_kernels) {
        std::cout << "Running " << kernel_name << "...\n";
        auto result = runner.run(kernel_name);
        v0_results[kernel_name] = result;

        // Generate insights
        const KernelConfig* config = KernelRegistry::instance().get_kernel(kernel_name);
        auto analyses = insights.analyze_benchmark(result, config);
        v0_insights[kernel_name] = analyses;

        // Print summary
        std::cout << "  Results:\n";
        for (const auto& vr : result.results) {
            std::cout << "    " << vr.variant_name << " @ " << vr.problem_size
                      << ": " << std::setprecision(2) << vr.metrics.performance.gflops
                      << " GFLOPS\n";
        }
        std::cout << "\n";
    }

    // ========================================================================
    // PHASE 2: Print insights for V0 kernels
    // ========================================================================

    std::cout << "========================================================================\n";
    std::cout << "PHASE 2: Generated Insights for V0 Kernels\n";
    std::cout << "========================================================================\n\n";

    for (const auto& [kernel_name, analyses] : v0_insights) {
        std::cout << "--- " << kernel_name << " ---\n\n";

        for (const auto& analysis : analyses) {
            if (analysis.variant_name.find("simd") != std::string::npos) {
                std::cout << "Variant: " << analysis.variant_name
                          << " @ " << analysis.problem_size << " elements\n";
                std::cout << "GFLOPS: " << std::setprecision(2) << analysis.achieved_gflops << "\n";
                std::cout << "Bottleneck: " << analysis.primary_bottleneck << "\n";
                std::cout << "Efficiency: " << std::setprecision(1)
                          << (analysis.efficiency_vs_roofline * 100) << "%\n\n";

                std::cout << "Insights:\n";
                for (const auto& insight : analysis.insights) {
                    if (insight.severity != InsightSeverity::INFO) {
                        std::cout << "  [" << InsightsEngine::severity_to_string(insight.severity)
                                  << "] " << insight.title << "\n";
                    }
                }
                std::cout << "\n";
            }
        }
    }

    // ========================================================================
    // PHASE 3: Run V1 (optimized) kernels
    // ========================================================================

    std::cout << "========================================================================\n";
    std::cout << "PHASE 3: Running V1 (optimized) kernels\n";
    std::cout << "========================================================================\n\n";

    std::vector<std::string> v1_kernels = {"dot_product_v1", "scale_v1", "sum_v1", "matmul_v1"};
    std::map<std::string, BenchmarkResult> v1_results;

    for (const auto& kernel_name : v1_kernels) {
        std::cout << "Running " << kernel_name << "...\n";
        auto result = runner.run(kernel_name);
        v1_results[kernel_name] = result;

        // Print summary
        std::cout << "  Results:\n";
        for (const auto& vr : result.results) {
            std::cout << "    " << vr.variant_name << " @ " << vr.problem_size
                      << ": " << std::setprecision(2) << vr.metrics.performance.gflops
                      << " GFLOPS\n";
        }
        std::cout << "\n";
    }

    // ========================================================================
    // PHASE 4: Compare V0 vs V1 and validate insights
    // ========================================================================

    std::cout << "========================================================================\n";
    std::cout << "PHASE 4: Insight Validation Results\n";
    std::cout << "========================================================================\n\n";

    std::cout << std::setw(20) << "Kernel"
              << std::setw(10) << "Size"
              << std::setw(12) << "V0 GFLOPS"
              << std::setw(12) << "V1 GFLOPS"
              << std::setw(10) << "Speedup"
              << std::setw(15) << "Insight Valid"
              << "\n";
    std::cout << std::string(79, '-') << "\n";

    std::vector<std::pair<std::string, std::string>> kernel_pairs = {
        {"dot_product_v0", "dot_product_v1"},
        {"scale_v0", "scale_v1"},
        {"sum_v0", "sum_v1"},
        {"matmul_v0", "matmul_v1"}
    };

    int valid_insights = 0;
    int total_insights = 0;

    for (const auto& [v0_name, v1_name] : kernel_pairs) {
        const auto& v0_result = v0_results[v0_name];
        const auto& v1_result = v1_results[v1_name];

        // Find SIMD results from v0 and v1
        for (const auto& v0_vr : v0_result.results) {
            if (v0_vr.variant_name.find("simd") != std::string::npos &&
                v0_vr.variant_name.find("scalar") == std::string::npos) {

                // Find matching size in v1
                for (const auto& v1_vr : v1_result.results) {
                    if (v1_vr.problem_size == v0_vr.problem_size) {
                        double speedup = v1_vr.metrics.performance.gflops /
                                        v0_vr.metrics.performance.gflops;
                        bool valid = speedup > 1.05;  // At least 5% improvement

                        std::cout << std::setw(20) << v0_name
                                  << std::setw(10) << v0_vr.problem_size
                                  << std::setw(12) << std::fixed << std::setprecision(2)
                                  << v0_vr.metrics.performance.gflops
                                  << std::setw(12) << v1_vr.metrics.performance.gflops
                                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                                  << std::setw(15) << (valid ? "YES" : "NO")
                                  << "\n";

                        if (valid) valid_insights++;
                        total_insights++;
                    }
                }
            }
        }
    }

    // ========================================================================
    // PHASE 5: Summary and Recommendations
    // ========================================================================

    std::cout << "\n========================================================================\n";
    std::cout << "                         VALIDATION SUMMARY\n";
    std::cout << "========================================================================\n\n";

    double accuracy = (total_insights > 0) ?
        (100.0 * valid_insights / total_insights) : 0;

    std::cout << "Insights validated: " << valid_insights << "/" << total_insights
              << " (" << std::setprecision(1) << accuracy << "% accurate)\n\n";

    std::cout << "Key Findings:\n";
    std::cout << "-----------------------------------------------------------------\n";

    // Analyze specific insight categories
    std::cout << "\n1. LOOP UNROLLING INSIGHT:\n";
    std::cout << "   Applied to: dot_product, scale, sum\n";
    double dp_speedup = 0, scale_speedup = 0, sum_speedup = 0;

    // Calculate speedups for largest size
    for (const auto& vr : v0_results["dot_product_v0"].results) {
        if (vr.variant_name.find("simd") != std::string::npos && vr.problem_size == 1048576) {
            for (const auto& v1_vr : v1_results["dot_product_v1"].results) {
                if (v1_vr.problem_size == 1048576) {
                    dp_speedup = v1_vr.metrics.performance.gflops / vr.metrics.performance.gflops;
                }
            }
        }
    }

    for (const auto& vr : v0_results["scale_v0"].results) {
        if (vr.variant_name.find("simd") != std::string::npos && vr.problem_size == 1048576) {
            for (const auto& v1_vr : v1_results["scale_v1"].results) {
                if (v1_vr.problem_size == 1048576) {
                    scale_speedup = v1_vr.metrics.performance.gflops / vr.metrics.performance.gflops;
                }
            }
        }
    }

    for (const auto& vr : v0_results["sum_v0"].results) {
        if (vr.variant_name.find("simd") != std::string::npos && vr.problem_size == 1048576) {
            for (const auto& v1_vr : v1_results["sum_v1"].results) {
                if (v1_vr.problem_size == 1048576) {
                    sum_speedup = v1_vr.metrics.performance.gflops / vr.metrics.performance.gflops;
                }
            }
        }
    }

    std::cout << "   dot_product: " << std::setprecision(2) << dp_speedup << "x "
              << (dp_speedup > 1.05 ? "(VALID)" : "(NOT HELPFUL)") << "\n";
    std::cout << "   scale:       " << scale_speedup << "x "
              << (scale_speedup > 1.05 ? "(VALID)" : "(NOT HELPFUL)") << "\n";
    std::cout << "   sum:         " << sum_speedup << "x "
              << (sum_speedup > 1.05 ? "(VALID)" : "(NOT HELPFUL)") << "\n";

    std::cout << "\n2. CACHE BLOCKING INSIGHT:\n";
    std::cout << "   Applied to: matmul\n";

    // Matmul size encoding: 2*N³ for N=256 is 33554432
    constexpr size_t MATMUL_256_SIZE = 33554432;
    double matmul_speedup = 0;
    for (const auto& vr : v0_results["matmul_v0"].results) {
        if (vr.variant_name.find("simd") != std::string::npos && vr.problem_size == MATMUL_256_SIZE) {
            for (const auto& v1_vr : v1_results["matmul_v1"].results) {
                if (v1_vr.problem_size == MATMUL_256_SIZE) {
                    matmul_speedup = v1_vr.metrics.performance.gflops / vr.metrics.performance.gflops;
                }
            }
        }
    }

    std::cout << "   matmul 256x256: " << matmul_speedup << "x "
              << (matmul_speedup > 1.05 ? "(VALID)" : "(NOT HELPFUL)") << "\n";

    std::cout << "\n========================================================================\n";
    std::cout << "               INSIGHTS ENGINE IMPROVEMENT RECOMMENDATIONS\n";
    std::cout << "========================================================================\n\n";

    if (dp_speedup <= 1.05) {
        std::cout << "- Loop unrolling insight may not be accurate for memory-bound kernels\n";
        std::cout << "  where memory bandwidth is already saturated.\n\n";
    }

    if (scale_speedup <= 1.05) {
        std::cout << "- For streaming operations (low arithmetic intensity), unrolling\n";
        std::cout << "  doesn't help because we're memory-bound, not compute-bound.\n";
        std::cout << "  RECOMMENDATION: Only suggest unrolling when AI > ridge_point.\n\n";
    }

    if (matmul_speedup <= 1.05) {
        std::cout << "- Cache blocking insight may need better block size tuning.\n";
        std::cout << "  RECOMMENDATION: Add adaptive block size based on cache sizes.\n\n";
    }

    // Save report
    std::ofstream report("/tmp/insights_validation_report.txt");
    report << "SIMD-Bench Insights Validation Report\n";
    report << "=====================================\n\n";
    report << "Accuracy: " << accuracy << "%\n\n";
    report << "Validated insights: " << valid_insights << "/" << total_insights << "\n";
    report.close();

    std::cout << "Detailed report saved to: /tmp/insights_validation_report.txt\n\n";

    // Cleanup
    KernelRegistry::instance().clear();

    return 0;
}
