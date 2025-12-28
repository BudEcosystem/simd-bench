#pragma once

// SIMD-Bench: Holistic SIMD Kernel Analysis Tool
// Version 1.0.0

#include "simd_bench/types.h"
#include "simd_bench/timing.h"
#include "simd_bench/hardware.h"
#include "simd_bench/kernel_registry.h"
#include "simd_bench/performance_counters.h"
#include "simd_bench/roofline.h"
#include "simd_bench/tma.h"
#include "simd_bench/energy.h"
#include "simd_bench/correctness.h"
#include "simd_bench/report_generator.h"
#include "simd_bench/runner.h"
#include "simd_bench/insights.h"

namespace simd_bench {

// Version information
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

inline std::string get_version_string() {
    return std::to_string(VERSION_MAJOR) + "." +
           std::to_string(VERSION_MINOR) + "." +
           std::to_string(VERSION_PATCH);
}

// Feature availability macros
#ifdef SIMD_BENCH_HAS_PAPI
    #define SIMD_BENCH_PAPI_AVAILABLE 1
#else
    #define SIMD_BENCH_PAPI_AVAILABLE 0
#endif

#ifdef SIMD_BENCH_HAS_LIKWID
    #define SIMD_BENCH_LIKWID_AVAILABLE 1
#else
    #define SIMD_BENCH_LIKWID_AVAILABLE 0
#endif

#ifdef SIMD_BENCH_HAS_RAPL
    #define SIMD_BENCH_RAPL_AVAILABLE 1
#else
    #define SIMD_BENCH_RAPL_AVAILABLE 0
#endif

}  // namespace simd_bench
