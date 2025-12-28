#include <benchmark/benchmark.h>
#include "simd_bench/kernel_registry.h"
#include "simd_bench/correctness.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"

using namespace simd_bench;

namespace hn = hwy::HWY_NAMESPACE;

// Scalar implementations
namespace scalar {

void vector_add(float* result, const float* a, const float* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul(float* result, const float* a, const float* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

float dot_product(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float sum(const float* data, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

void exp_approx(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::exp(input[i]);
    }
}

}  // namespace scalar

// SIMD implementations using Highway
namespace simd {

void vector_add(float* HWY_RESTRICT result, const float* HWY_RESTRICT a,
                const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        hn::Store(hn::Add(va, vb), d, result + i);
    }
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul(float* HWY_RESTRICT result, const float* HWY_RESTRICT a,
                const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        hn::Store(hn::Mul(va, vb), d, result + i);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

float dot_product(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum = hn::Zero(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        sum = hn::MulAdd(va, vb, sum);
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float sum(const float* HWY_RESTRICT data, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto vsum = hn::Zero(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto v = hn::Load(d, data + i);
        vsum = hn::Add(vsum, v);
    }

    float result = hn::ReduceSum(d, vsum);
    for (; i < count; ++i) {
        result += data[i];
    }
    return result;
}

}  // namespace simd

// Benchmarks

static void BM_ScalarVectorAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    auto a = hwy::AllocateAligned<float>(size);
    auto b = hwy::AllocateAligned<float>(size);
    auto result = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    for (auto _ : state) {
        scalar::vector_add(result.get(), a.get(), b.get(), size);
        benchmark::DoNotOptimize(result.get());
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);  // 2 reads + 1 write
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_ScalarVectorAdd)->Range(1024, 1 << 20);

static void BM_SIMDVectorAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    auto a = hwy::AllocateAligned<float>(size);
    auto b = hwy::AllocateAligned<float>(size);
    auto result = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    for (auto _ : state) {
        simd::vector_add(result.get(), a.get(), b.get(), size);
        benchmark::DoNotOptimize(result.get());
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_SIMDVectorAdd)->Range(1024, 1 << 20);

static void BM_ScalarDotProduct(benchmark::State& state) {
    const size_t size = state.range(0);
    auto a = hwy::AllocateAligned<float>(size);
    auto b = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i % 100) / 100.0f;
        b[i] = static_cast<float>(i % 100) / 100.0f;
    }

    for (auto _ : state) {
        float result = scalar::dot_product(a.get(), b.get(), size);
        benchmark::DoNotOptimize(result);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * size * 2);  // 1 mul + 1 add per element
}
BENCHMARK(BM_ScalarDotProduct)->Range(1024, 1 << 20);

static void BM_SIMDDotProduct(benchmark::State& state) {
    const size_t size = state.range(0);
    auto a = hwy::AllocateAligned<float>(size);
    auto b = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i % 100) / 100.0f;
        b[i] = static_cast<float>(i % 100) / 100.0f;
    }

    for (auto _ : state) {
        float result = simd::dot_product(a.get(), b.get(), size);
        benchmark::DoNotOptimize(result);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * size * 2);
}
BENCHMARK(BM_SIMDDotProduct)->Range(1024, 1 << 20);

static void BM_ScalarSum(benchmark::State& state) {
    const size_t size = state.range(0);
    auto data = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    for (auto _ : state) {
        float result = scalar::sum(data.get(), size);
        benchmark::DoNotOptimize(result);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_ScalarSum)->Range(1024, 1 << 20);

static void BM_SIMDSum(benchmark::State& state) {
    const size_t size = state.range(0);
    auto data = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    for (auto _ : state) {
        float result = simd::sum(data.get(), size);
        benchmark::DoNotOptimize(result);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_SIMDSum)->Range(1024, 1 << 20);

// Memory bandwidth benchmark
static void BM_MemoryCopy(benchmark::State& state) {
    const size_t size = state.range(0);
    auto src = hwy::AllocateAligned<float>(size);
    auto dst = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        src[i] = static_cast<float>(i);
    }

    for (auto _ : state) {
        std::memcpy(dst.get(), src.get(), size * sizeof(float));
        benchmark::DoNotOptimize(dst.get());
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);  // read + write
}
BENCHMARK(BM_MemoryCopy)->Range(1024, 1 << 24);

// Cache hierarchy benchmark
static void BM_SequentialAccess(benchmark::State& state) {
    const size_t size = state.range(0);
    auto data = hwy::AllocateAligned<float>(size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}
BENCHMARK(BM_SequentialAccess)->Range(1024, 1 << 24);

static void BM_RandomAccess(benchmark::State& state) {
    const size_t size = state.range(0);
    auto data = hwy::AllocateAligned<float>(size);
    auto indices = hwy::AllocateAligned<size_t>(size);

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
        indices[i] = (i * 7919) % size;  // Pseudo-random access pattern
    }

    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += data[indices[i]];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}
BENCHMARK(BM_RandomAccess)->Range(1024, 1 << 20);

BENCHMARK_MAIN();
