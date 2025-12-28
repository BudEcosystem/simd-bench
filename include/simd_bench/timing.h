#pragma once

#include "types.h"
#include <cstdint>
#include <chrono>
#include <utility>

namespace simd_bench {

// High-precision timing utilities

// Read Time-Stamp Counter (x86)
inline uint64_t rdtsc() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (val));
    return val;
#else
    return 0;
#endif
}

// Read TSC with memory fence (more accurate for timing)
inline uint64_t rdtscp() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi, aux;
    __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi), "=c" (aux));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__ ("isb; mrs %0, cntvct_el0" : "=r" (val));
    return val;
#else
    return 0;
#endif
}

// Memory fence to serialize operations
inline void serialize() {
#if defined(__x86_64__) || defined(_M_X64)
    __asm__ __volatile__ ("mfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ __volatile__ ("dsb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// CPUID fence for x86 (serializes instruction stream)
inline void cpuid_fence() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    __asm__ __volatile__ (
        "cpuid"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "a" (0)
    );
#endif
}

// Timer class for precise measurements
class Timer {
public:
    Timer() = default;

    void start() {
        serialize();
        start_cycles_ = rdtsc();
        start_time_ = Clock::now();
    }

    void stop() {
        stop_time_ = Clock::now();
        stop_cycles_ = rdtscp();
        serialize();
    }

    // Get elapsed time in various units
    double elapsed_seconds() const {
        return std::chrono::duration<double>(stop_time_ - start_time_).count();
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

    double elapsed_microseconds() const {
        return elapsed_seconds() * 1e6;
    }

    double elapsed_nanoseconds() const {
        return elapsed_seconds() * 1e9;
    }

    uint64_t elapsed_cycles() const {
        return stop_cycles_ - start_cycles_;
    }

    // Measure CPU frequency by timing a known workload
    static double measure_frequency_ghz();

private:
    TimePoint start_time_;
    TimePoint stop_time_;
    uint64_t start_cycles_ = 0;
    uint64_t stop_cycles_ = 0;
};

// Scoped timer for RAII-style timing
class ScopedTimer {
public:
    explicit ScopedTimer(double& result_seconds)
        : result_(result_seconds) {
        timer_.start();
    }

    ~ScopedTimer() {
        timer_.stop();
        result_ = timer_.elapsed_seconds();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Timer timer_;
    double& result_;
};

// Statistical timing with multiple runs
struct TimingStats {
    double min_seconds = 0.0;
    double max_seconds = 0.0;
    double mean_seconds = 0.0;
    double median_seconds = 0.0;
    double stddev_seconds = 0.0;
    uint64_t min_cycles = 0;
    uint64_t max_cycles = 0;
    double mean_cycles = 0.0;
    size_t samples = 0;
};

// Benchmark function with multiple iterations and warmup
TimingStats benchmark_timing(
    const std::function<void()>& func,
    size_t warmup_iterations = 5,
    size_t measure_iterations = 100
);

// Measure latency of a single operation
double measure_latency_ns(
    const std::function<void()>& func,
    size_t iterations = 1000
);

// Measure throughput (operations per second)
double measure_throughput(
    const std::function<void()>& func,
    size_t ops_per_call,
    double target_seconds = 1.0
);

// Prevent compiler from optimizing away a value
template<typename T>
inline void do_not_optimize(T const& value) {
#if defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#elif defined(__GNUC__)
    asm volatile("" : : "g"(value) : "memory");
#else
    volatile T sink = value;
    (void)sink;
#endif
}

// Prevent compiler from reordering across this point
inline void compiler_fence() {
    asm volatile("" ::: "memory");
}

}  // namespace simd_bench
