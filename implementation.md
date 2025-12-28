# SIMD-Bench: Holistic SIMD Kernel Analysis Tool

## Implementation Plan

A comprehensive framework for profiling, benchmarking, and iteratively improving SIMD kernels across multiple hardware platforms.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals and Objectives](#2-goals-and-objectives)
3. [Architecture Overview](#3-architecture-overview)
4. [Metrics Framework](#4-metrics-framework)
5. [Tool Integration](#5-tool-integration)
6. [Hardware Platform Support](#6-hardware-platform-support)
7. [Implementation Phases](#7-implementation-phases)
8. [Core Components](#8-core-components)
9. [Visualization and Reporting](#9-visualization-and-reporting)
10. [CI/CD Integration](#10-cicd-integration)
11. [References and Resources](#11-references-and-resources)

---

## 1. Executive Summary

SIMD-Bench is a holistic analysis tool designed to evaluate SIMD kernels across multiple dimensions:
- **Performance**: GFLOPS, throughput, latency, bandwidth
- **Quality**: Vectorization ratio, SIMD efficiency, instruction mix
- **Portability**: Cross-platform performance comparison
- **Energy**: Power consumption and energy efficiency
- **Correctness**: Numerical accuracy and floating-point verification

The tool provides actionable insights through the Roofline Model, Top-Down Microarchitecture Analysis (TMA), and automated regression testing.

### Key Differentiators

| Feature | Existing Tools | SIMD-Bench |
|---------|---------------|------------|
| Cross-platform | Limited | x86 (SSE/AVX/AVX-512), ARM (NEON/SVE), RISC-V (RVV) |
| Roofline Integration | Separate tool | Built-in with cache-aware model |
| Automated Optimization | Manual | Guided recommendations |
| Energy Profiling | Optional | Integrated RAPL/PowerAPI |
| CI/CD Ready | No | Native GitHub Actions/GitLab CI support |

---

## 2. Goals and Objectives

### Primary Goals

1. **Measure**: Accurately profile SIMD kernel performance using hardware counters
2. **Analyze**: Identify bottlenecks (memory, compute, frontend, backend)
3. **Compare**: Benchmark across different SIMD widths and architectures
4. **Optimize**: Provide actionable recommendations for improvement
5. **Track**: Detect performance regressions in CI/CD pipelines

### Target Users

- High-performance library developers
- Compiler engineers evaluating vectorization
- Application developers optimizing hot loops
- Researchers studying SIMD efficiency

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SIMD-Bench Framework                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Kernel     │  │   Static     │  │   Dynamic    │  │   Energy    │ │
│  │   Registry   │  │   Analyzer   │  │   Profiler   │  │   Monitor   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                  │        │
│         └─────────────────┴────────┬────────┴──────────────────┘        │
│                                    │                                    │
│                        ┌───────────▼───────────┐                        │
│                        │    Metrics Collector  │                        │
│                        └───────────┬───────────┘                        │
│                                    │                                    │
│         ┌──────────────────────────┼──────────────────────────┐        │
│         │                          │                          │        │
│  ┌──────▼──────┐  ┌────────────────▼────────────────┐  ┌──────▼──────┐ │
│  │  Roofline   │  │  Top-Down Microarchitecture    │  │  Regression │ │
│  │   Model     │  │     Analysis (TMA)             │  │   Detector  │ │
│  └──────┬──────┘  └────────────────┬────────────────┘  └──────┬──────┘ │
│         │                          │                          │        │
│         └──────────────────────────┴──────────────────────────┘        │
│                                    │                                    │
│                        ┌───────────▼───────────┐                        │
│                        │   Report Generator    │                        │
│                        │  (HTML/JSON/Markdown) │                        │
│                        └───────────────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Description |
|-----------|-------------|
| **Kernel Registry** | Manages kernel definitions, variants, and metadata |
| **Static Analyzer** | Parses compiler reports, analyzes assembly |
| **Dynamic Profiler** | Collects hardware performance counters at runtime |
| **Energy Monitor** | Measures power consumption via RAPL/PowerAPI |
| **Metrics Collector** | Aggregates data from all sources |
| **Roofline Model** | Computes and visualizes arithmetic intensity bounds |
| **TMA Engine** | Performs top-down microarchitecture analysis |
| **Regression Detector** | Tracks performance across versions |
| **Report Generator** | Creates human/machine-readable reports |

---

## 4. Metrics Framework

### 4.1 Performance Metrics

| Metric | Description | Unit | Source |
|--------|-------------|------|--------|
| **GFLOPS** | Giga floating-point ops per second | GFLOP/s | HW Counters |
| **GOPS** | Giga integer ops per second | GOP/s | HW Counters |
| **Throughput** | Elements processed per second | elements/s | Timing |
| **Latency** | Time to process single element | ns | RDTSC |
| **IPC** | Instructions per cycle | ratio | HW Counters |
| **CPI** | Cycles per instruction | ratio | HW Counters |

### 4.2 SIMD-Specific Metrics

| Metric | Description | Formula | Target |
|--------|-------------|---------|--------|
| **Vectorization Ratio** | Packed vs scalar instructions | `packed_ops / (packed_ops + scalar_ops)` | >95% |
| **Vector Capacity Usage** | Actual vs maximum SIMD width | `actual_width / max_width` | >80% |
| **SIMD Efficiency** | Theoretical vs actual speedup | `simd_time / (scalar_time / lanes)` | >70% |
| **FMA Utilization** | FMA vs separate MUL+ADD | `fma_ops / (fma_ops + mul_ops + add_ops)` | >80% |
| **Lane Utilization** | Active lanes per operation | `active_lanes / total_lanes` | >90% |

### 4.3 Memory Metrics

| Metric | Description | Unit | Source |
|--------|-------------|------|--------|
| **Memory Bandwidth** | Data transfer rate | GB/s | HW Counters |
| **Arithmetic Intensity** | FLOPs per byte transferred | FLOP/byte | Computed |
| **L1 Hit Rate** | L1 cache efficiency | % | HW Counters |
| **L2 Hit Rate** | L2 cache efficiency | % | HW Counters |
| **LLC Miss Rate** | Last-level cache misses | % | HW Counters |

### 4.4 Microarchitecture Metrics (TMA)

Based on Intel's [Top-down Microarchitecture Analysis Method](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/top-down-microarchitecture-analysis-method.html):

```
Pipeline Slots
├── Retiring (useful work)
│   ├── Base (non-vectorized)
│   └── Vectorized
│       ├── FP Vector (256-bit, 512-bit)
│       └── Int Vector
├── Bad Speculation
│   ├── Branch Mispredicts
│   └── Machine Clears
├── Frontend Bound
│   ├── Fetch Latency
│   └── Fetch Bandwidth
└── Backend Bound
    ├── Core Bound
    │   ├── Divider
    │   ├── Ports Utilization
    │   └── FP Latency
    └── Memory Bound
        ├── L1 Bound
        ├── L2 Bound
        ├── L3 Bound
        └── DRAM Bound
```

### 4.5 Energy Metrics

| Metric | Description | Unit | Source |
|--------|-------------|------|--------|
| **Package Power** | Total CPU power | Watts | RAPL |
| **Core Power** | Core-only power | Watts | RAPL |
| **DRAM Power** | Memory power | Watts | RAPL |
| **Energy per Op** | Energy efficiency | nJ/FLOP | Computed |
| **Energy-Delay Product** | Combined metric | J*s | Computed |

### 4.6 Correctness Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Max Absolute Error** | Maximum deviation from reference | < 1e-6 (float) |
| **Max Relative Error** | Maximum percentage deviation | < 1e-5 |
| **ULP Error** | Units in last place deviation | < 4 ULP |
| **NaN/Inf Count** | Exceptional values produced | 0 |

---

## 5. Tool Integration

### 5.1 Hardware Performance Counters

#### Linux perf
```bash
# SIMD instruction counting
perf stat -e fp_arith_inst_retired.256b_packed_single,\
            fp_arith_inst_retired.scalar_single,\
            instructions,cycles ./benchmark
```

#### LIKWID
```bash
# AVX FLOPS measurement
likwid-perfctr -C 0 -g FLOPS_AVX ./benchmark

# Memory bandwidth
likwid-perfctr -C 0 -g MEM ./benchmark

# Cache analysis
likwid-perfctr -C 0 -g L2CACHE ./benchmark
```

**Key Performance Groups:**
- `FLOPS_SP` / `FLOPS_DP` - Floating-point operations
- `FLOPS_AVX` - AVX-specific operations
- `L2CACHE` / `L3CACHE` - Cache miss analysis
- `MEM` - Memory bandwidth
- `ENERGY` - Power consumption

Reference: [LIKWID GitHub](https://github.com/RRZE-HPC/likwid)

#### PAPI (Cross-Platform)
```c
#include <papi.h>

int events[] = {PAPI_VEC_SP, PAPI_FP_OPS, PAPI_L1_DCM};
PAPI_start_counters(events, 3);
// ... kernel execution ...
PAPI_stop_counters(values, 3);
```

**Key Preset Events:**
- `PAPI_VEC_SP` - Single-precision vector instructions
- `PAPI_VEC_DP` - Double-precision vector instructions
- `PAPI_FP_OPS` - Floating-point operations
- `PAPI_L1_DCM` - L1 data cache misses

Reference: [PAPI Official](https://icl.utk.edu/papi/)

### 5.2 Instruction Analysis

#### Intel SDE (Software Development Emulator)
```bash
# Instruction mix histogram
sde -mix -iform -- ./benchmark

# FLOPS counting
sde -mix -dyn_mask_profile -- ./benchmark

# SSE/AVX transition detection
sde -ast -- ./benchmark
```

**Output includes:**
- Instruction counts by opcode
- ISA extension grouping (SSE, AVX, AVX-512)
- Masked element analysis (AVX-512)
- Basic block hotspots

Reference: [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)

#### objdump/llvm-objdump
```bash
# Disassemble and filter SIMD instructions
objdump -d binary | grep -E "vfmadd|vmulps|vaddps|vmovaps"

# Count instruction types
objdump -d binary | grep -c "vfmadd"
```

### 5.3 Compiler Optimization Reports

#### GCC
```bash
gcc -O3 -march=native -fopt-info-vec-all source.c
# -fopt-info-vec-optimized  # Only successful vectorizations
# -fopt-info-vec-missed     # Failed vectorizations with reasons
```

#### Clang/LLVM
```bash
clang -O3 -march=native \
  -Rpass=loop-vectorize \
  -Rpass-missed=loop-vectorize \
  -Rpass-analysis=loop-vectorize source.c
```

#### Intel Compiler (ICX/ICC)
```bash
icx -O3 -xHost -qopt-report=5 -qopt-report-phase=vec source.c
# Generates .optrpt file with detailed vectorization report
```

Reference: [Cornell Optimization Reports](https://cvw.cac.cornell.edu/vector/compilers/optimization-reports)

### 5.4 Profiling Tools

#### Intel VTune Profiler
```bash
# Microarchitecture exploration
vtune -collect uarch-exploration -- ./benchmark

# Memory access analysis
vtune -collect memory-access -- ./benchmark

# Hotspots with SIMD utilization
vtune -collect hotspots -knob sampling-mode=hw -- ./benchmark
```

**Key Metrics:**
- Vector Capacity Usage
- SIMD Instructions per Cycle
- Memory Bound percentage

Reference: [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

#### Intel Advisor
```bash
# Survey hotspots
advixe-cl -collect survey -- ./benchmark

# Roofline analysis
advixe-cl -collect roofline -- ./benchmark

# Trip counts and FLOPS
advixe-cl -collect tripcounts -flop -- ./benchmark
```

Reference: [Intel Advisor Roofline](https://www.codeproject.com/Articles/1169323/Intel-Advisor-Roofline-Analysis)

### 5.5 Energy Profiling

#### RAPL (Linux)
```c
// Read from powercap interface
FILE* f = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
fscanf(f, "%lu", &energy_before);
// ... kernel execution ...
fscanf(f, "%lu", &energy_after);
double joules = (energy_after - energy_before) / 1e6;
```

Reference: [Daniel Lemire - SIMD Energy](https://lemire.me/blog/2024/02/19/measuring-energy-usage-regular-code-vs-simd-code/)

#### perf with Energy Events
```bash
perf stat -e power/energy-pkg/,power/energy-ram/ ./benchmark
```

### 5.6 Visualization

#### Flame Graphs
```bash
# Record with perf
perf record -F 99 -g ./benchmark
perf script > out.perf

# Generate flame graph
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > flamegraph.svg
```

Reference: [Brendan Gregg Flame Graphs](https://www.brendangregg.com/flamegraphs.html)

---

## 6. Hardware Platform Support

### 6.1 x86-64

| Platform | SIMD Extensions | Vector Width | Tools |
|----------|-----------------|--------------|-------|
| Intel (Pre-Skylake) | SSE4.2, AVX, AVX2 | 256-bit | VTune, Advisor, LIKWID |
| Intel Skylake+ | AVX-512 | 512-bit | VTune, Advisor, LIKWID |
| AMD Zen 1-3 | SSE4.2, AVX2 | 256-bit | LIKWID, perf |
| AMD Zen 4+ | AVX-512 | 512-bit | LIKWID, perf |

**Key Hardware Counters (Intel):**
```
FP_ARITH_INST_RETIRED.SCALAR_SINGLE
FP_ARITH_INST_RETIRED.128B_PACKED_SINGLE
FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE
FP_ARITH_INST_RETIRED.512B_PACKED_SINGLE
```

### 6.2 ARM

| Platform | SIMD Extensions | Vector Width | Tools |
|----------|-----------------|--------------|-------|
| Cortex-A76/A78 | NEON | 128-bit | Streamline, perf |
| Neoverse N1/V1 | NEON, SVE | 128-2048 bit | Streamline, perf |
| Apple M1/M2/M3 | NEON | 128-bit | Instruments |
| NVIDIA Grace | NEON, SVE2 | 128-bit | perf, LIKWID |

**Key ARM PMU Events:**
```
ASE_INST_SPEC      # SIMD instruction speculated
SVE_INST_SPEC      # SVE instruction speculated
INST_SPEC          # Total instructions speculated
STALL_BACKEND_MEM  # Memory stall cycles
```

Reference: [ARM Performance Analysis](https://learn.arm.com/learning-paths/servers-and-cloud-computing/profiling-for-neoverse/performance-analysis-concepts/)

### 6.3 RISC-V

| Platform | SIMD Extensions | Vector Width | Tools |
|----------|-----------------|--------------|-------|
| SiFive P670 | RVV 1.0 | 256-bit | perf |
| Spacemit X60 | RVV 1.0 | 256-bit | perf |

---

## 7. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-4)

**Deliverables:**
- [ ] Project structure and build system (CMake)
- [ ] Kernel registration API
- [ ] Basic timing infrastructure (RDTSC, chrono)
- [ ] Hardware detection (CPUID, /proc/cpuinfo)
- [ ] JSON configuration and output format

**Key Files:**
```
simd-bench/
├── CMakeLists.txt
├── include/
│   ├── simd_bench/core.h
│   ├── simd_bench/timing.h
│   ├── simd_bench/hardware.h
│   └── simd_bench/kernel.h
├── src/
│   ├── core.cc
│   ├── timing.cc
│   └── hardware.cc
└── tests/
    └── core_test.cc
```

### Phase 2: Performance Counters (Weeks 5-8)

**Deliverables:**
- [ ] PAPI integration (cross-platform)
- [ ] LIKWID integration (Linux x86/ARM)
- [ ] perf_event integration (Linux)
- [ ] Windows ETW integration (optional)
- [ ] Unified counter abstraction layer

**Key APIs:**
```cpp
class PerformanceCounters {
public:
    void start(const std::vector<Event>& events);
    void stop();
    CounterValues read();

    // Platform-specific backends
    static std::unique_ptr<PerformanceCounters> create(Backend backend);
};
```

### Phase 3: Analysis Engines (Weeks 9-12)

**Deliverables:**
- [ ] Roofline model implementation
- [ ] Cache-aware roofline (L1, L2, L3, DRAM)
- [ ] Top-down microarchitecture analysis (TMA)
- [ ] Vectorization ratio calculator
- [ ] Memory bandwidth analyzer

**Roofline Model:**
```cpp
struct RooflinePoint {
    double arithmetic_intensity;  // FLOP/byte
    double achieved_gflops;
    std::string bound;           // "compute" | "memory"
};

RooflineModel model(peak_gflops, mem_bandwidth_gbps);
model.addCeiling("L1", l1_bandwidth);
model.addCeiling("L2", l2_bandwidth);
model.addCeiling("L3", l3_bandwidth);
model.plot("roofline.svg");
```

### Phase 4: Static Analysis (Weeks 13-16)

**Deliverables:**
- [ ] Compiler report parser (GCC, Clang, ICC)
- [ ] Assembly analyzer (objdump integration)
- [ ] Intel SDE integration
- [ ] Instruction mix histogram
- [ ] Vectorization opportunity detector

**Compiler Report Parser:**
```cpp
struct VectorizationReport {
    std::string file;
    int line;
    bool vectorized;
    std::string reason;  // If not vectorized
    int vector_width;
    std::string isa;     // SSE, AVX, AVX-512
};

std::vector<VectorizationReport> parseGccReport(const std::string& output);
std::vector<VectorizationReport> parseClangReport(const std::string& output);
```

### Phase 5: Energy Profiling (Weeks 17-18)

**Deliverables:**
- [ ] RAPL integration (Intel/AMD)
- [ ] ARM energy counters
- [ ] Energy-per-operation metrics
- [ ] Power throttling detection

### Phase 6: Correctness Testing (Weeks 19-20)

**Deliverables:**
- [ ] Reference scalar implementation comparison
- [ ] ULP error calculation
- [ ] NaN/Inf detection
- [ ] Random input generation
- [ ] Edge case testing

**Correctness API:**
```cpp
struct CorrectnessResult {
    double max_absolute_error;
    double max_relative_error;
    double max_ulp_error;
    int nan_count;
    int inf_count;
    bool passed;
};

CorrectnessResult verify(
    const std::function<void(float*, size_t)>& simd_kernel,
    const std::function<void(float*, size_t)>& reference_kernel,
    size_t count,
    float tolerance = 1e-6f
);
```

### Phase 7: Reporting and Visualization (Weeks 21-24)

**Deliverables:**
- [ ] HTML report generator
- [ ] Markdown report generator
- [ ] JSON output format
- [ ] Roofline plot (SVG/PNG)
- [ ] Performance comparison tables
- [ ] Flame graph integration

### Phase 8: CI/CD Integration (Weeks 25-26)

**Deliverables:**
- [ ] GitHub Actions workflow
- [ ] GitLab CI configuration
- [ ] Performance regression detection
- [ ] Baseline management
- [ ] Alert configuration

---

## 8. Core Components

### 8.1 Kernel Definition API

```cpp
#include <simd_bench/kernel.h>

// Define a kernel with metadata
SIMD_BENCH_KERNEL(dot_product) {
    .name = "Dot Product",
    .description = "Vector dot product using FMA",
    .category = "BLAS Level 1",
    .arithmetic_intensity = 0.25,  // FLOP/byte
    .flops_per_element = 2,

    .variants = {
        {"scalar", dot_product_scalar},
        {"sse", dot_product_sse},
        {"avx2", dot_product_avx2},
        {"avx512", dot_product_avx512},
        {"neon", dot_product_neon},
    },

    .sizes = {1024, 4096, 16384, 65536, 262144, 1048576},
    .iterations = 1000,
};
```

### 8.2 Benchmark Runner

```cpp
#include <simd_bench/runner.h>

int main() {
    BenchmarkRunner runner;

    // Configure
    runner.setWarmupIterations(5);
    runner.setBenchmarkIterations(100);
    runner.enableEnergyProfiling(true);
    runner.enableHardwareCounters({"FLOPS_SP", "L2CACHE", "MEM"});

    // Run
    auto results = runner.run(dot_product);

    // Report
    HTMLReporter reporter;
    reporter.generate(results, "report.html");

    return 0;
}
```

### 8.3 Hardware Detection

```cpp
#include <simd_bench/hardware.h>

HardwareInfo hw = HardwareInfo::detect();

std::cout << "CPU: " << hw.cpu_name << "\n";
std::cout << "Cores: " << hw.cores << "\n";
std::cout << "SIMD: " << hw.simd_extensions << "\n";  // "SSE4.2, AVX2, FMA"
std::cout << "Max Vector Width: " << hw.max_vector_bits << " bits\n";
std::cout << "L1 Cache: " << hw.l1_cache_kb << " KB\n";
std::cout << "L2 Cache: " << hw.l2_cache_kb << " KB\n";
std::cout << "L3 Cache: " << hw.l3_cache_kb << " KB\n";
std::cout << "Memory Bandwidth: " << hw.measured_mem_bw_gbps << " GB/s\n";
std::cout << "Peak GFLOPS: " << hw.theoretical_peak_gflops << "\n";
```

### 8.4 Performance Counter Abstraction

```cpp
#include <simd_bench/counters.h>

// Create platform-appropriate backend
auto counters = PerformanceCounters::create();

// Define events
counters->addEvent(Event::FP_ARITH_256B_PACKED_SINGLE);
counters->addEvent(Event::FP_ARITH_SCALAR_SINGLE);
counters->addEvent(Event::CYCLES);
counters->addEvent(Event::INSTRUCTIONS);

counters->start();
kernel();
counters->stop();

auto values = counters->read();
double vectorization_ratio = values[0] / (values[0] + values[1]);
double ipc = values[3] / static_cast<double>(values[2]);
```

---

## 9. Visualization and Reporting

### 9.1 Roofline Chart

```
GFLOPS ▲
       │                              ┌────────────────── Peak Compute
  120 ─┤                         ╱────┘
       │                    ╱────
  100 ─┤               ╱────
       │          ╱────
   80 ─┤     ╱────                    ★ MatMul (61.66 GFLOPS)
       │╱────
   60 ─┤         ★ L1 Dot (59.33)
       │
   40 ─┤
       │
   20 ─┤    ★ L2 Dot (17.65)
       │ ★ Memory Dot (4.78)
    0 ─┴──────┬──────┬──────┬──────┬──────▶ Arithmetic Intensity
              0.25   1      4      16     (FLOP/byte)
```

### 9.2 TMA Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Slot Breakdown                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Retiring          ███████████████████████████████████░░░ 85.2% │
│    └─ Vectorized   ██████████████████████████████████░░░░ 82.1% │
│                                                                  │
│  Bad Speculation   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.3% │
│                                                                  │
│  Frontend Bound    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.1% │
│                                                                  │
│  Backend Bound     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 11.4% │
│    └─ Memory Bound ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  9.8% │
│    └─ Core Bound   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.6% │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Performance Comparison Table

```markdown
| Kernel    | Size   | Scalar  | SSE4    | AVX2    | AVX-512 | Best   |
|-----------|--------|---------|---------|---------|---------|--------|
| dot_prod  | 1M     | 0.99    | 3.21    | 4.78    | 8.12    | AVX-512|
| saxpy     | 1M     | 1.24    | 4.12    | 5.89    | 9.23    | AVX-512|
| gemm      | 512²   | 1.45    | 11.71   | 22.78   | 45.12   | AVX-512|
| softmax   | 32K    | 0.08    | 0.42    | 0.63    | 1.02    | AVX-512|

Units: GFLOPS (higher is better)
```

### 9.4 HTML Report Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>SIMD-Bench Report</title>
    <script src="chart.js"></script>
    <style>/* ... */</style>
</head>
<body>
    <header>
        <h1>SIMD Kernel Analysis Report</h1>
        <p>Generated: 2025-12-27 15:30:00</p>
    </header>

    <section id="hardware">
        <h2>Hardware Configuration</h2>
        <!-- CPU, Memory, SIMD capabilities -->
    </section>

    <section id="summary">
        <h2>Executive Summary</h2>
        <!-- Key findings, recommendations -->
    </section>

    <section id="roofline">
        <h2>Roofline Analysis</h2>
        <!-- Interactive roofline chart -->
    </section>

    <section id="kernels">
        <h2>Kernel Results</h2>
        <!-- Per-kernel detailed analysis -->
    </section>

    <section id="recommendations">
        <h2>Optimization Recommendations</h2>
        <!-- Actionable suggestions -->
    </section>
</body>
</html>
```

---

## 10. CI/CD Integration

### 10.1 GitHub Actions Workflow

```yaml
name: SIMD Performance

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y likwid cmake ninja-build

      - name: Build
        run: |
          cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
          cmake --build build

      - name: Run benchmarks
        run: |
          cd build
          ./simd-bench --output=results.json

      - name: Check for regressions
        run: |
          python scripts/check_regression.py \
            --baseline baseline.json \
            --current results.json \
            --threshold 5%

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: build/results.json
```

### 10.2 Regression Detection Algorithm

```python
def detect_regression(baseline: dict, current: dict, threshold: float = 0.05):
    """
    Detect performance regressions.

    Args:
        baseline: Previous benchmark results
        current: Current benchmark results
        threshold: Maximum allowed regression (5% default)

    Returns:
        List of regressions with severity
    """
    regressions = []

    for kernel_name, kernel_results in current['kernels'].items():
        if kernel_name not in baseline['kernels']:
            continue

        baseline_perf = baseline['kernels'][kernel_name]['gflops']
        current_perf = kernel_results['gflops']

        change = (current_perf - baseline_perf) / baseline_perf

        if change < -threshold:
            regressions.append({
                'kernel': kernel_name,
                'baseline': baseline_perf,
                'current': current_perf,
                'change': change * 100,
                'severity': 'critical' if change < -0.15 else 'warning'
            })

    return regressions
```

### 10.3 Baseline Management

```bash
# Create new baseline from current results
simd-bench baseline create --from results.json --name v1.0.0

# List baselines
simd-bench baseline list

# Compare against baseline
simd-bench compare --baseline v1.0.0 --current results.json

# Update baseline (requires approval)
simd-bench baseline update --name v1.0.0 --from results.json
```

---

## 11. References and Resources

### 11.1 Academic Papers

1. **Roofline Model**: Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM.
   - [Berkeley Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

2. **TSVC Benchmark**: Callahan, D., Dongarra, J., & Levine, D. (1988). "Vectorizing Compilers: A Test Suite and Results." Supercomputing '88.
   - [TSVC_2 GitHub](https://github.com/UoB-HPC/TSVC_2)

3. **SimdBench**: (2025). "Benchmarking Large Language Models for SIMD-Intrinsic Code Generation."
   - [arXiv](https://arxiv.org/html/2507.15224v1)

4. **SIMD for Databases**: Polychroniou, O., et al. (2015). "Rethinking SIMD Vectorization for In-Memory Databases." ACM SIGMOD.
   - [ACM DL](https://dl.acm.org/doi/10.1145/2723372.2747645)

5. **RAPL Analysis**: Hähnel, M., et al. (2012). "Measuring Energy Consumption for Short Code Paths Using RAPL."
   - [ACM DL](https://dl.acm.org/doi/10.1145/2425248.2425252)

### 11.2 Tools Documentation

| Tool | Documentation | Purpose |
|------|---------------|---------|
| [LIKWID](https://github.com/RRZE-HPC/likwid) | [Wiki](https://github.com/RRZE-HPC/likwid/wiki) | HW counters, microbenchmarks |
| [PAPI](https://icl.utk.edu/papi/) | [Docs](https://bitbucket.org/icl/papi/wiki/Home) | Cross-platform counters |
| [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) | [User Guide](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-2/overview.html) | Comprehensive profiling |
| [Intel Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html) | [Cookbook](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/2023-0/overview.html) | Vectorization, roofline |
| [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) | [Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/benefits-of-using-intel-software-development-emulator.html) | Instruction mix |
| [perf](https://perf.wiki.kernel.org/) | [Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial) | Linux profiling |
| [FlameGraph](https://github.com/brendangregg/FlameGraph) | [Blog](https://www.brendangregg.com/flamegraphs.html) | Visualization |

### 11.3 SIMD Libraries

| Library | Platforms | Link |
|---------|-----------|------|
| Google Highway | x86, ARM, RISC-V, WASM | [GitHub](https://github.com/google/highway) |
| xsimd | x86, ARM | [GitHub](https://github.com/xtensor-stack/xsimd) |
| SIMDe | All (emulation) | [GitHub](https://github.com/simd-everywhere/simde) |
| Vc | x86 | [GitHub](https://github.com/VcDevel/Vc) |
| VecCore | Abstraction | [GitHub](https://github.com/root-project/veccore) |
| Agner Fog VCL | x86 | [Website](https://www.agner.org/optimize/#vectorclass) |

### 11.4 Microarchitecture References

- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/) - Instruction tables, microarchitecture details
- [uops.info](https://uops.info/) - Instruction latency/throughput measurements
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) - x86 SIMD intrinsics reference
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/) - ARM SIMD reference

### 11.5 Compiler Resources

- [GCC Vectorization](https://gcc.gnu.org/wiki/VectorizationIssueTOC) - GCC auto-vectorization details
- [LLVM Loop Vectorizer](https://llvm.org/docs/Vectorizers.html) - Clang vectorization documentation
- [Intel Compiler Vectorization](https://www.intel.com/content/www/us/en/developer/articles/technical/vectorization-llvm-gcc-cpus-gpus.html) - ICX/ICC vectorization

---

## Appendix A: Quick Start Commands

```bash
# Clone and build
git clone https://github.com/yourorg/simd-bench.git
cd simd-bench
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run basic benchmark
./build/simd-bench --kernel dot_product

# Run with hardware counters (requires sudo or capabilities)
sudo ./build/simd-bench --kernel dot_product --counters FLOPS_SP

# Generate roofline plot
./build/simd-bench --roofline --output roofline.svg

# Full analysis report
./build/simd-bench --full-analysis --output report.html

# CI mode (JSON output, regression check)
./build/simd-bench --ci --baseline baseline.json --threshold 5%
```

---

## Appendix B: Example Output

```json
{
  "metadata": {
    "tool": "simd-bench",
    "version": "1.0.0",
    "timestamp": "2025-12-27T15:30:00Z"
  },
  "hardware": {
    "cpu": "Intel Core i7-10700KF",
    "cores": 8,
    "threads": 16,
    "frequency_ghz": 3.8,
    "turbo_ghz": 5.1,
    "simd_extensions": ["SSE4.2", "AVX2", "FMA"],
    "max_vector_bits": 256,
    "l1_cache_kb": 32,
    "l2_cache_kb": 256,
    "l3_cache_mb": 16,
    "memory_bandwidth_gbps": 12.7,
    "theoretical_peak_gflops": 121.4
  },
  "kernels": {
    "dot_product": {
      "variants": {
        "scalar": {"gflops": 0.99, "efficiency": 0.008},
        "avx2": {"gflops": 4.78, "efficiency": 0.039},
        "avx2_8x": {"gflops": 66.21, "efficiency": 0.546}
      },
      "best_variant": "avx2_8x",
      "vectorization_ratio": 0.9997,
      "arithmetic_intensity": 0.25,
      "bound": "memory",
      "recommendations": [
        "Consider blocking to improve cache utilization",
        "Use non-temporal stores for streaming access patterns"
      ]
    }
  },
  "summary": {
    "total_kernels": 10,
    "compute_bound": 3,
    "memory_bound": 7,
    "avg_vectorization_ratio": 0.952,
    "energy_efficiency_nj_flop": 2.3
  }
}
```

---

*Document Version: 1.0.0*
*Last Updated: 2025-12-27*
*Authors: Generated with assistance from Claude*
