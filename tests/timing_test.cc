#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/timing.h"
#include <thread>
#include <chrono>
#include <cmath>

namespace simd_bench {
namespace testing {

class TimingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test RDTSC returns non-zero and is monotonic
TEST_F(TimingTest, RdtscReturnsNonZero) {
    uint64_t tsc = rdtsc();
    EXPECT_GT(tsc, 0u);
}

TEST_F(TimingTest, RdtscIsMonotonic) {
    uint64_t tsc1 = rdtsc();
    volatile int dummy = 0;
    for (int i = 0; i < 1000; ++i) dummy += i;
    uint64_t tsc2 = rdtsc();

    EXPECT_GT(tsc2, tsc1);
}

TEST_F(TimingTest, RdtscpReturnsNonZero) {
    uint64_t tsc = rdtscp();
    EXPECT_GT(tsc, 0u);
}

// Test Timer class
TEST_F(TimingTest, TimerMeasuresElapsedTime) {
    Timer timer;
    timer.start();

    // Sleep for ~10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    timer.stop();

    double elapsed = timer.elapsed_seconds();
    EXPECT_GT(elapsed, 0.005);  // At least 5ms
    EXPECT_LT(elapsed, 0.1);    // Less than 100ms
}

TEST_F(TimingTest, TimerMeasuresCycles) {
    Timer timer;
    timer.start();

    volatile int dummy = 0;
    for (int i = 0; i < 10000; ++i) dummy += i;

    timer.stop();

    uint64_t cycles = timer.elapsed_cycles();
    EXPECT_GT(cycles, 0u);
}

TEST_F(TimingTest, TimerElapsedMilliseconds) {
    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop();

    double ms = timer.elapsed_milliseconds();
    EXPECT_GT(ms, 3.0);
    EXPECT_LT(ms, 50.0);
}

TEST_F(TimingTest, TimerElapsedMicroseconds) {
    Timer timer;
    timer.start();
    volatile int dummy = 0;
    for (int i = 0; i < 1000; ++i) dummy += i;
    timer.stop();

    double us = timer.elapsed_microseconds();
    EXPECT_GT(us, 0.0);
}

TEST_F(TimingTest, TimerElapsedNanoseconds) {
    Timer timer;
    timer.start();
    volatile int dummy = 0;
    dummy += 1;
    timer.stop();

    double ns = timer.elapsed_nanoseconds();
    EXPECT_GT(ns, 0.0);
}

// Test ScopedTimer
TEST_F(TimingTest, ScopedTimerMeasuresElapsedTime) {
    double elapsed = 0.0;

    {
        ScopedTimer scoped(elapsed);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    EXPECT_GT(elapsed, 0.003);
    EXPECT_LT(elapsed, 0.1);
}

// Test frequency measurement
TEST_F(TimingTest, MeasureFrequencyReturnsReasonableValue) {
    double freq = Timer::measure_frequency_ghz();

    // Modern CPUs are typically between 1 and 6 GHz
    EXPECT_GT(freq, 0.5);
    EXPECT_LT(freq, 8.0);
}

// Test benchmark_timing function
TEST_F(TimingTest, BenchmarkTimingReturnsValidStats) {
    auto func = []() {
        volatile int dummy = 0;
        for (int i = 0; i < 1000; ++i) dummy += i;
    };

    TimingStats stats = benchmark_timing(func, 3, 10);

    EXPECT_EQ(stats.samples, 10u);
    EXPECT_GT(stats.min_seconds, 0.0);
    EXPECT_GE(stats.max_seconds, stats.min_seconds);
    EXPECT_GE(stats.mean_seconds, stats.min_seconds);
    EXPECT_LE(stats.mean_seconds, stats.max_seconds);
    EXPECT_GE(stats.stddev_seconds, 0.0);
    EXPECT_GT(stats.min_cycles, 0u);
}

TEST_F(TimingTest, BenchmarkTimingMedianIsValid) {
    auto func = []() {
        volatile int dummy = 0;
        for (int i = 0; i < 100; ++i) dummy += i;
    };

    TimingStats stats = benchmark_timing(func, 2, 21);  // Odd number for clear median

    EXPECT_GE(stats.median_seconds, stats.min_seconds);
    EXPECT_LE(stats.median_seconds, stats.max_seconds);
}

// Test measure_latency_ns
TEST_F(TimingTest, MeasureLatencyReturnsPositiveValue) {
    auto func = []() {
        volatile int x = 1 + 1;
        (void)x;
    };

    double latency = measure_latency_ns(func, 10000);
    EXPECT_GT(latency, 0.0);
}

// Test measure_throughput
TEST_F(TimingTest, MeasureThroughputReturnsPositiveValue) {
    int count = 0;
    auto func = [&count]() {
        count++;
    };

    double throughput = measure_throughput(func, 1, 0.01);  // 10ms target
    EXPECT_GT(throughput, 0.0);
}

// Test do_not_optimize
TEST_F(TimingTest, DoNotOptimizePreservesValue) {
    int x = 42;
    do_not_optimize(x);
    EXPECT_EQ(x, 42);

    float f = 3.14f;
    do_not_optimize(f);
    EXPECT_FLOAT_EQ(f, 3.14f);
}

// Test serialization functions don't crash
TEST_F(TimingTest, SerializeFunctionsExecuteWithoutCrash) {
    EXPECT_NO_THROW(serialize());
    EXPECT_NO_THROW(cpuid_fence());
    EXPECT_NO_THROW(compiler_fence());
}

// Test timing consistency
TEST_F(TimingTest, TimingIsConsistent) {
    std::vector<double> times;

    auto func = []() {
        volatile int dummy = 0;
        for (int i = 0; i < 10000; ++i) dummy += i;
    };

    // Run multiple times
    for (int i = 0; i < 10; ++i) {
        Timer timer;
        timer.start();
        func();
        timer.stop();
        times.push_back(timer.elapsed_seconds());
    }

    // Calculate coefficient of variation
    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / times.size();

    double variance = 0.0;
    for (double t : times) variance += (t - mean) * (t - mean);
    variance /= times.size();
    double stddev = std::sqrt(variance);

    double cv = stddev / mean;

    // Coefficient of variation should be less than 50% for consistent timing
    EXPECT_LT(cv, 0.5);
}

}  // namespace testing
}  // namespace simd_bench
