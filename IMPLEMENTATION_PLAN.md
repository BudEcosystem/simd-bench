# SIMD-Bench Implementation Plan

## Status Summary

### Already Implemented (Verified)
| Issue | Status | Notes |
|-------|--------|-------|
| All source files exist | ✅ Complete | insights.cc, performance_counters.cc, roofline.cc, tma.cc all present |
| Null-checks after allocations | ✅ Complete | Added in kernel_registry.cc |
| Thread-safe progress callback | ✅ Complete | Mutex protection in runner.h |
| Highway dependency flexible | ✅ Complete | FetchContent fallback in CMakeLists.txt |
| RAPL wraparound handling | ✅ Complete | Multiple wraparound tracking in energy.cc |
| Timer resolution validation | ✅ Complete | validate_timer_resolution() in timing.cc |
| FP comparison scaled | ✅ Complete | sqrt(N) scaling in verify_dot_product() |
| Cache-level ridge points | ✅ Complete | L1/L2/L3/DRAM in roofline.cc |
| AMD TMA support | ✅ Complete | calculate_amd_metrics() in tma.cc |

---

## Remaining Bug Fixes (Priority Order)

### 1. Integer Overflow in FLOPS Calculation
**File**: `src/kernel_registry.cc`, `include/simd_bench/types.h`
**Severity**: Medium
**Description**: For large matrices (N=512+), `2*N³` can overflow `size_t` on 32-bit systems.

**Fix**:
```cpp
// Use uint64_t for FLOPS calculations
struct FlopsCalculator {
    static uint64_t matmul_flops(size_t M, size_t N, size_t K) {
        return static_cast<uint64_t>(2) * M * N * K;
    }

    // Check for overflow before calculation
    static bool would_overflow(size_t M, size_t N, size_t K) {
        // 2*M*N*K > UINT64_MAX ?
        if (M > UINT64_MAX / 2) return true;
        if (N > (UINT64_MAX / 2) / M) return true;
        if (K > (UINT64_MAX / 2 / M) / N) return true;
        return false;
    }
};
```

**Estimated Effort**: 1-2 hours

---

### 2. Peak GFLOPS Ignores AVX-512 Frequency Throttling
**File**: `src/hardware.cc`
**Severity**: Medium
**Description**: AVX-512 causes 15-25% frequency drop. Current peak calculation is unrealistic.

**Fix**:
```cpp
struct FrequencyThrottling {
    static constexpr double AVX512_LIGHT_PENALTY = 0.95;   // L0: 5% slowdown
    static constexpr double AVX512_MEDIUM_PENALTY = 0.85;  // L1: 15% slowdown
    static constexpr double AVX512_HEAVY_PENALTY = 0.75;   // L2: 25% slowdown
};

double HardwareInfo::calculate_peak_gflops(bool double_precision,
                                            AVX512License license) const {
    int lanes = max_vector_bits / (double_precision ? 64 : 32);
    double freq = measured_frequency_ghz;

    // Apply AVX-512 throttling penalty
    if (max_vector_bits >= 512) {
        switch (license) {
            case AVX512License::L0: freq *= 0.95; break;
            case AVX512License::L1: freq *= 0.85; break;
            case AVX512License::L2: freq *= 0.75; break;
        }
    }

    return freq * lanes * fma_units * 2;
}
```

**Estimated Effort**: 2-3 hours

---

### 3. Dynamic Arithmetic Intensity Measurement
**File**: `src/roofline.cc`, `src/metrics_analyzer.cc`
**Severity**: Medium
**Description**: AI is currently hard-coded. Should measure dynamically using hardware counters.

**Fix**:
```cpp
struct DynamicAI {
    // Measure actual AI using IMC counters or cache miss rates
    static double measure(const CounterValues& values, uint64_t total_flops) {
        uint64_t bytes_read = values.get(CounterEvent::OFFCORE_REQUESTS_DEMAND_DATA_RD) * 64;
        uint64_t bytes_written = values.get(CounterEvent::OFFCORE_REQUESTS_DEMAND_RFO) * 64;
        uint64_t total_bytes = bytes_read + bytes_written;

        if (total_bytes == 0) return 0.0;
        return static_cast<double>(total_flops) / total_bytes;
    }

    // Calculate cache amplification factor
    static double cache_amplification(double measured_ai, double theoretical_ai) {
        if (theoretical_ai == 0) return 1.0;
        return theoretical_ai / measured_ai;
    }
};
```

**Estimated Effort**: 3-4 hours

---

### 4. Memory Bandwidth with Non-Temporal Stores
**File**: `src/hardware.cc`
**Severity**: Low
**Description**: Current measurement uses normal SIMD stores. Should use non-temporal for streaming bandwidth.

**Fix**:
```cpp
double measure_streaming_bandwidth_gbps(size_t size_mb) {
    const size_t size = size_mb * 1024 * 1024;
    const size_t count = size / sizeof(float);

    auto src = hwy::AllocateAligned<float>(count);
    auto dst = hwy::AllocateAligned<float>(count);

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    Timer timer;
    timer.start();

    for (size_t i = 0; i + N <= count; i += N) {
        auto v = hn::Load(d, src.get() + i);
        hn::Stream(v, d, dst.get() + i);  // Non-temporal store
    }
    hn::FlushStream();  // Ensure all stores complete

    timer.stop();

    double bytes = 2.0 * size;
    return bytes / timer.elapsed_seconds() / 1e9;
}
```

**Estimated Effort**: 2-3 hours

---

### 5. Windows/macOS Fallback for perf_event
**File**: `src/performance_counters.cc`
**Severity**: Low (platform-specific)
**Description**: Only Linux perf_event is implemented. Windows/macOS use null counters.

**Fix**: Add platform-specific backends:
- **Windows**: Intel PCM library or ETW
- **macOS**: Apple's `kpc` framework

```cpp
#if defined(_WIN32)
class WindowsPCMCounters : public IPerformanceCounters {
    // Use Intel PCM or Windows Performance Counters
};
#elif defined(__APPLE__)
class AppleKPCCounters : public IPerformanceCounters {
    // Use Apple's kpc framework (requires entitlements)
};
#endif
```

**Estimated Effort**: 8-16 hours (per platform)

---

## New Feature Implementation Plan

### Phase 1: High Priority Features (1-2 weeks)

#### A. Memory Traffic Measurement (IMC Counters)
**New Files**: `include/simd_bench/memory_traffic.h`, `src/memory_traffic.cc`

```cpp
class MemoryTrafficAnalyzer {
public:
    struct TrafficMetrics {
        uint64_t bytes_read_dram;
        uint64_t bytes_written_dram;
        double measured_arithmetic_intensity;
        double cache_amplification;
        double bandwidth_utilization;
    };

    bool initialize();  // Setup uncore counters
    void start();
    void stop();
    TrafficMetrics get_metrics(uint64_t total_flops) const;

    static bool is_available();  // Check for uncore counter support
};
```

**Implementation Notes**:
- Use IMC (Integrated Memory Controller) uncore counters
- Intel: `UNC_M_CAS_COUNT.RD/WR`
- AMD: Use L3 miss counters as proxy
- Requires MSR access or PAPI uncore support

**Estimated Effort**: 8-12 hours

---

#### B. Prefetch Analysis
**New Files**: `include/simd_bench/prefetch.h`, `src/prefetch.cc`

```cpp
struct PrefetchMetrics {
    uint64_t sw_prefetch_issued;
    uint64_t hw_prefetch_useful;
    uint64_t hw_prefetch_useless;
    double prefetch_coverage;
    int recommended_prefetch_distance;
    std::string recommendation;
};

class PrefetchAnalyzer {
public:
    PrefetchMetrics analyze(const CounterValues& values,
                            size_t working_set_bytes,
                            double memory_latency_ns);

    // Calculate optimal prefetch distance
    static int calculate_prefetch_distance(
        double memory_latency_ns,
        double loop_iteration_ns,
        size_t stride_bytes);
};
```

**Counter Events Needed**:
- `L2_LINES_IN.ALL` (hardware prefetch useful)
- `L2_LINES_OUT.USELESS_HWPF` (hardware prefetch useless)
- `SW_PREFETCH_ACCESS.*` (software prefetch)

**Estimated Effort**: 6-8 hours

---

#### C. Branch-Free Code Detection
**New Files**: `include/simd_bench/branch_analysis.h`, `src/branch_analysis.cc`

```cpp
struct BranchMetrics {
    uint64_t conditional_branches;
    uint64_t mispredictions;
    double branch_density;  // per 1000 instructions
    double misprediction_rate;
    bool is_branchless;
    std::vector<std::string> branchless_opportunities;
};

class BranchAnalyzer {
public:
    BranchMetrics analyze(const CounterValues& values);

    // Detect SIMD-unfriendly branch patterns
    std::vector<std::string> detect_vectorization_blockers(
        const BranchMetrics& metrics,
        double vectorization_ratio);
};
```

**Thresholds**:
- Branch density < 5 per 1000 instructions = branchless
- Misprediction rate > 5% = optimization opportunity

**Estimated Effort**: 4-6 hours

---

#### D. Auto-Vectorization Analysis
**New Files**: `include/simd_bench/autovec.h`, `src/autovec.cc`

```cpp
struct AutoVecAnalysis {
    bool compiler_vectorized;
    std::string compiler_report;  // From -fopt-info-vec
    double auto_vec_speedup;
    double intrinsic_speedup;
    std::vector<std::string> missed_optimizations;
    std::vector<std::string> vectorization_blockers;
};

class AutoVecAnalyzer {
public:
    // Compile and run auto-vectorized version
    AutoVecAnalysis compare(
        const std::string& source_file,
        const KernelConfig& intrinsic_kernel);

    // Parse compiler optimization report
    static std::vector<std::string> parse_vec_report(
        const std::string& compiler_output);
};
```

**Implementation Notes**:
- Compile kernel with `-fopt-info-vec-all`
- Parse output for vectorization success/failure reasons
- Compare performance with intrinsic version

**Estimated Effort**: 12-16 hours

---

### Phase 2: Medium Priority Features (2-4 weeks)

#### E. Multi-Core Scaling Analysis
```cpp
class ScalingAnalyzer {
public:
    struct ScalingResult {
        std::vector<int> thread_counts;
        std::vector<double> speedups;
        double parallel_efficiency;
        int optimal_thread_count;
        std::string scaling_category;  // "linear", "sublinear", "saturated"
        std::string bottleneck;  // "memory_bandwidth", "false_sharing", etc.
    };

    ScalingResult analyze(const KernelConfig& kernel,
                          std::vector<int> threads = {1,2,4,8,16});
};
```

**Estimated Effort**: 12-16 hours

---

#### F. Register Pressure Detection
```cpp
struct RegisterPressureMetrics {
    uint64_t register_spills;
    uint64_t register_fills;
    double spill_ratio;
    bool has_register_pressure;
    int estimated_live_registers;
    std::vector<std::string> reduction_suggestions;
};
```

**Counter Events**:
- Stack operations (push/pop) rate
- L1 hit rate patterns indicating spills
- Instruction mix analysis

**Estimated Effort**: 6-8 hours

---

#### G. Loop Tiling Advisor
```cpp
struct TilingRecommendation {
    size_t l1_tile_size;
    size_t l2_tile_size;
    size_t l3_tile_size;
    double estimated_speedup;
    std::string code_example;
};

TilingRecommendation recommend_tiling(
    size_t working_set_bytes,
    const CacheInfo& cache,
    size_t element_size,
    size_t access_stride);
```

**Algorithm**:
1. Analyze cache miss patterns
2. Calculate optimal tile sizes for each cache level
3. Generate code template with recommended tiling

**Estimated Effort**: 8-12 hours

---

### Phase 3: Architecture Improvements (4-6 weeks)

#### A. Plugin System for Counter Backends
```cpp
class ICounterBackend {
public:
    virtual ~ICounterBackend() = default;
    virtual std::string name() const = 0;
    virtual bool probe() = 0;
    virtual int priority() const = 0;  // Higher = preferred
    virtual std::vector<CounterEvent> supported_events() = 0;
    virtual bool add_event(CounterEvent event) = 0;
    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual CounterValues read() = 0;
};

class BackendRegistry {
    std::vector<std::unique_ptr<ICounterBackend>> backends_;
public:
    void register_backend(std::unique_ptr<ICounterBackend> backend);
    ICounterBackend* get_best_available();
    std::vector<std::string> list_available();
};

// Built-in backends
class PerfEventBackend : public ICounterBackend { ... };
class PAPIBackend : public ICounterBackend { ... };
class LIKWIDBackend : public ICounterBackend { ... };
class NullBackend : public ICounterBackend { ... };  // Fallback
```

**Estimated Effort**: 16-24 hours

---

#### B. YAML Configuration File Support
```yaml
# simd_bench.yaml
benchmarks:
  - name: dot_product
    sizes: [1024, 4096, 16384, 65536]
    iterations: 1000
    warmup: 10

hardware_counters:
  backend: auto  # "auto", "perf", "papi", "likwid"
  events:
    - CYCLES
    - INSTRUCTIONS
    - FP_ARITH_256B_PACKED_SINGLE
    - L1D_READ_MISS

analysis:
  roofline: true
  tma: true
  insights: true

output:
  formats: [json, html, markdown]
  path: ./reports/

thresholds:
  vectorization_warning: 0.8
  cache_miss_warning: 0.05
```

**Estimated Effort**: 8-12 hours

---

#### C. Differential Benchmarking (Regression Tracking)
```cpp
class RegressionTracker {
public:
    void set_baseline(const std::string& baseline_file);
    void set_baseline(const std::string& git_commit,
                      const std::string& results_dir);

    struct RegressionReport {
        std::vector<std::string> regressions;   // >5% slower
        std::vector<std::string> improvements;  // >5% faster
        std::vector<std::string> unchanged;
        std::map<std::string, double> all_changes;
        bool has_critical_regressions;
    };

    RegressionReport compare(const std::vector<BenchmarkResult>& current);

    // CI integration
    int exit_code() const;  // 0 = pass, 1 = regressions found
    std::string github_comment() const;  // Markdown for PR comments
};
```

**Estimated Effort**: 12-16 hours

---

## Implementation Timeline

| Week | Tasks |
|------|-------|
| 1 | Bug fixes (overflow, AVX-512 throttling, dynamic AI) |
| 2 | Memory Traffic Measurement, Prefetch Analysis |
| 3 | Branch Analysis, Auto-Vec Analysis |
| 4 | Multi-Core Scaling, Register Pressure |
| 5 | Loop Tiling Advisor, Config File Support |
| 6 | Plugin System, Regression Tracking |
| 7 | Testing, Documentation, Examples |

---

## Testing Strategy

### Unit Tests for Each Feature
- Add test files: `tests/memory_traffic_test.cc`, `tests/prefetch_test.cc`, etc.
- Mock counter values for deterministic testing
- Threshold validation tests

### Integration Tests
- End-to-end benchmark with all new metrics
- Cross-architecture validation (if possible)
- Performance regression tests for the framework itself

### Validation Examples
- Create examples demonstrating each new feature
- Document expected output for each metric

---

## File Structure After Implementation

```
simd-bench/
├── include/simd_bench/
│   ├── memory_traffic.h      (NEW)
│   ├── prefetch.h            (NEW)
│   ├── branch_analysis.h     (NEW)
│   ├── autovec.h             (NEW)
│   ├── scaling.h             (NEW)
│   ├── register_pressure.h   (NEW)
│   ├── loop_tiling.h         (NEW)
│   ├── backend_registry.h    (NEW)
│   ├── config.h              (NEW)
│   └── regression.h          (NEW)
├── src/
│   ├── memory_traffic.cc     (NEW)
│   ├── prefetch.cc           (NEW)
│   ├── branch_analysis.cc    (NEW)
│   ├── autovec.cc            (NEW)
│   ├── scaling.cc            (NEW)
│   ├── register_pressure.cc  (NEW)
│   ├── loop_tiling.cc        (NEW)
│   ├── backend_registry.cc   (NEW)
│   ├── config.cc             (NEW)
│   └── regression.cc         (NEW)
├── tests/
│   └── [corresponding test files]
├── examples/
│   └── [feature demonstration examples]
└── simd_bench.yaml.example   (NEW)
```
