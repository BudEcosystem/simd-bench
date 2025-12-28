#include "simd_bench/timing.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <sstream>
#include <ratio>

namespace simd_bench {

// Timer resolution validation implementation
TimerResolution validate_timer_resolution() {
    TimerResolution result;

    // Check the clock's compile-time period
    using Period = Clock::period;
    double theoretical_resolution_ns = 1e9 * static_cast<double>(Period::num) / Period::den;

    // Empirically measure actual resolution by finding minimum observable time difference
    std::vector<double> deltas;
    deltas.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        auto t1 = Clock::now();
        auto t2 = Clock::now();

        // Keep sampling until we get a non-zero delta
        while (t2 == t1) {
            t2 = Clock::now();
        }

        double delta = std::chrono::duration<double, std::nano>(t2 - t1).count();
        deltas.push_back(delta);
    }

    // Use median as the measured resolution (robust to outliers)
    std::sort(deltas.begin(), deltas.end());
    double measured_resolution_ns = deltas[deltas.size() / 2];

    // Use the larger of theoretical and measured resolution
    result.resolution_ns = std::max(theoretical_resolution_ns, measured_resolution_ns);

    // Categorize resolution quality
    result.is_high_resolution = (result.resolution_ns < 100.0);       // < 100ns
    result.is_suitable_for_microbench = (result.resolution_ns < 1000.0);  // < 1us

    // Generate description
    std::ostringstream oss;
    oss << "Timer resolution: " << result.resolution_ns << " ns";
    if (result.is_high_resolution) {
        oss << " (high-resolution, suitable for nanosecond-level timing)";
    } else if (result.is_suitable_for_microbench) {
        oss << " (suitable for microsecond-level timing)";
    } else {
        oss << " (WARNING: resolution too coarse for accurate microbenchmarks)";
    }
    result.description = oss.str();

    return result;
}

double get_minimum_measurable_duration_ns() {
    static double min_duration = -1.0;

    if (min_duration < 0) {
        auto resolution = validate_timer_resolution();
        // Recommend at least 10x the resolution for reliable measurements
        min_duration = resolution.resolution_ns * 10.0;
    }

    return min_duration;
}

double Timer::measure_frequency_ghz() {
    // Run a calibration loop and measure cycles vs time
    Timer timer;

    // Warmup
    volatile int dummy = 0;
    for (int i = 0; i < 1000000; ++i) dummy += i;

    // Measure
    timer.start();
    const int iterations = 100000000;
    dummy = 0;
    for (int i = 0; i < iterations; ++i) dummy += i;
    timer.stop();

    double seconds = timer.elapsed_seconds();
    uint64_t cycles = timer.elapsed_cycles();

    return static_cast<double>(cycles) / (seconds * 1e9);
}

TimingStats benchmark_timing(
    const std::function<void()>& func,
    size_t warmup_iterations,
    size_t measure_iterations
) {
    TimingStats stats;
    stats.samples = measure_iterations;

    // Warmup
    for (size_t i = 0; i < warmup_iterations; ++i) {
        func();
    }

    // Collect measurements
    std::vector<double> times(measure_iterations);
    std::vector<uint64_t> cycles(measure_iterations);

    for (size_t i = 0; i < measure_iterations; ++i) {
        Timer timer;
        timer.start();
        func();
        timer.stop();
        times[i] = timer.elapsed_seconds();
        cycles[i] = timer.elapsed_cycles();
    }

    // Calculate statistics
    stats.min_seconds = *std::min_element(times.begin(), times.end());
    stats.max_seconds = *std::max_element(times.begin(), times.end());
    stats.min_cycles = *std::min_element(cycles.begin(), cycles.end());
    stats.max_cycles = *std::max_element(cycles.begin(), cycles.end());

    // Mean
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    stats.mean_seconds = sum / measure_iterations;

    uint64_t cycle_sum = std::accumulate(cycles.begin(), cycles.end(), 0ULL);
    stats.mean_cycles = static_cast<double>(cycle_sum) / measure_iterations;

    // Median
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    if (measure_iterations % 2 == 0) {
        stats.median_seconds = (sorted_times[measure_iterations/2 - 1] +
                               sorted_times[measure_iterations/2]) / 2.0;
    } else {
        stats.median_seconds = sorted_times[measure_iterations/2];
    }

    // Standard deviation
    double variance = 0.0;
    for (double t : times) {
        double diff = t - stats.mean_seconds;
        variance += diff * diff;
    }
    variance /= measure_iterations;
    stats.stddev_seconds = std::sqrt(variance);

    return stats;
}

double measure_latency_ns(
    const std::function<void()>& func,
    size_t iterations
) {
    // Warmup
    for (size_t i = 0; i < 100; ++i) {
        func();
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    timer.stop();

    return timer.elapsed_nanoseconds() / static_cast<double>(iterations);
}

double measure_throughput(
    const std::function<void()>& func,
    size_t ops_per_call,
    double target_seconds
) {
    // First, estimate how many iterations we need
    Timer timer;

    // Quick calibration
    timer.start();
    for (int i = 0; i < 100; ++i) {
        func();
    }
    timer.stop();

    double time_per_100 = timer.elapsed_seconds();
    size_t estimated_iterations = static_cast<size_t>(
        (target_seconds / time_per_100) * 100
    );
    estimated_iterations = std::max(estimated_iterations, size_t(1000));

    // Actual measurement
    timer.start();
    for (size_t i = 0; i < estimated_iterations; ++i) {
        func();
    }
    timer.stop();

    double elapsed = timer.elapsed_seconds();
    size_t total_ops = estimated_iterations * ops_per_call;

    return static_cast<double>(total_ops) / elapsed;
}

}  // namespace simd_bench
