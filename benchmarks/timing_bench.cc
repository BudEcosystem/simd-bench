#include <benchmark/benchmark.h>
#include "simd_bench/timing.h"
#include <vector>
#include <numeric>

using namespace simd_bench;

// Benchmark rdtsc overhead
static void BM_RDTSC(benchmark::State& state) {
    for (auto _ : state) {
        uint64_t cycles = rdtsc();
        benchmark::DoNotOptimize(cycles);
    }
}
BENCHMARK(BM_RDTSC);

// Benchmark rdtscp overhead
static void BM_RDTSCP(benchmark::State& state) {
    for (auto _ : state) {
        uint64_t cycles = rdtscp();
        benchmark::DoNotOptimize(cycles);
    }
}
BENCHMARK(BM_RDTSCP);

// Benchmark Timer class overhead
static void BM_TimerOverhead(benchmark::State& state) {
    Timer timer;
    for (auto _ : state) {
        timer.start();
        timer.stop();
        benchmark::DoNotOptimize(timer.elapsed_nanoseconds());
    }
}
BENCHMARK(BM_TimerOverhead);

// Benchmark chrono vs rdtsc
static void BM_ChronoNow(benchmark::State& state) {
    for (auto _ : state) {
        auto now = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(now);
    }
}
BENCHMARK(BM_ChronoNow);

// Benchmark serialize instruction overhead
static void BM_Serialize(benchmark::State& state) {
    for (auto _ : state) {
        serialize();
    }
}
BENCHMARK(BM_Serialize);

// Benchmark cpuid fence overhead
static void BM_CPUIDFence(benchmark::State& state) {
    for (auto _ : state) {
        cpuid_fence();
    }
}
BENCHMARK(BM_CPUIDFence);

// Benchmark benchmark_timing helper
static void BM_BenchmarkTiming(benchmark::State& state) {
    auto func = []() {
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) {
            x += i;
        }
    };

    for (auto _ : state) {
        auto stats = benchmark_timing(func, 10);
        benchmark::DoNotOptimize(stats);
    }
}
BENCHMARK(BM_BenchmarkTiming);

// Benchmark vector sum to measure timing accuracy
static void BM_VectorSum(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.0f);

    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_VectorSum)->Range(64, 1 << 20);

// Benchmark with ScopedTimer
static void BM_ScopedTimerOverhead(benchmark::State& state) {
    double elapsed_seconds = 0.0;
    for (auto _ : state) {
        {
            ScopedTimer scoped(elapsed_seconds);
            // Empty scope
        }
        benchmark::DoNotOptimize(elapsed_seconds);
    }
}
BENCHMARK(BM_ScopedTimerOverhead);

BENCHMARK_MAIN();
