// ============================================================================
// SIMD-Bench Insights Engine
// ============================================================================
// Comprehensive kernel tuning insights based on industry best practices from:
// - Intel Optimization Manual & VTune methodology
// - Roofline Model (Berkeley/NERSC)
// - Top-Down Microarchitecture Analysis (TMA)
// - Cache optimization literature
// - SIMD vectorization best practices
// ============================================================================

#include "simd_bench/insights.h"
#include "simd_bench/roofline.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <regex>

namespace simd_bench {

// ============================================================================
// InsightsEngine Implementation
// ============================================================================

InsightsEngine::InsightsEngine() : hw_(HardwareInfo::detect()) {}

InsightsEngine::InsightsEngine(const HardwareInfo& hw) : hw_(hw) {}

void InsightsEngine::set_thresholds(const InsightThresholds& thresholds) {
    thresholds_ = thresholds;
}

std::string InsightsEngine::category_to_string(InsightCategory cat) {
    switch (cat) {
        case InsightCategory::MEMORY_BOUND: return "Memory Bound";
        case InsightCategory::COMPUTE_BOUND: return "Compute Bound";
        case InsightCategory::VECTORIZATION: return "Vectorization";
        case InsightCategory::CACHE_EFFICIENCY: return "Cache Efficiency";
        case InsightCategory::DATA_ALIGNMENT: return "Data Alignment";
        case InsightCategory::LOOP_OPTIMIZATION: return "Loop Optimization";
        case InsightCategory::BRANCH_PREDICTION: return "Branch Prediction";
        case InsightCategory::INSTRUCTION_MIX: return "Instruction Mix";
        case InsightCategory::REGISTER_PRESSURE: return "Register Pressure";
        case InsightCategory::MEMORY_ACCESS_PATTERN: return "Memory Access Pattern";
        case InsightCategory::ARITHMETIC_INTENSITY: return "Arithmetic Intensity";
        case InsightCategory::PARALLELISM: return "Parallelism";
        case InsightCategory::LATENCY_HIDING: return "Latency Hiding";
        case InsightCategory::GENERAL: return "General";
        default: return "Unknown";
    }
}

std::string InsightsEngine::severity_to_string(InsightSeverity sev) {
    switch (sev) {
        case InsightSeverity::CRITICAL: return "CRITICAL";
        case InsightSeverity::HIGH: return "HIGH";
        case InsightSeverity::MEDIUM: return "MEDIUM";
        case InsightSeverity::LOW: return "LOW";
        case InsightSeverity::INFO: return "INFO";
        default: return "UNKNOWN";
    }
}

std::string InsightsEngine::confidence_to_string(InsightConfidence conf) {
    switch (conf) {
        case InsightConfidence::HIGH: return "High";
        case InsightConfidence::MEDIUM: return "Medium";
        case InsightConfidence::LOW: return "Low";
        default: return "Unknown";
    }
}

std::string InsightsEngine::classify_bottleneck(double ai, double efficiency) const {
    double ridge_point = hw_.theoretical_peak_sp_gflops / hw_.measured_memory_bw_gbps;

    if (ai < ridge_point * 0.5) {
        return "Memory Bandwidth";
    } else if (ai < ridge_point) {
        if (efficiency < 0.5) {
            return "Memory Latency";
        }
        return "Memory/Compute Balanced";
    } else {
        if (efficiency < 0.3) {
            return "Instruction Throughput";
        } else if (efficiency < 0.6) {
            return "Compute (Vectorization)";
        }
        return "Compute (Near Optimal)";
    }
}

double InsightsEngine::calculate_theoretical_max(double ai) const {
    double memory_ceiling = ai * hw_.measured_memory_bw_gbps;
    double compute_ceiling = hw_.theoretical_peak_sp_gflops;
    return std::min(memory_ceiling, compute_ceiling);
}

// ============================================================================
// Main Analysis Functions
// ============================================================================

KernelAnalysis InsightsEngine::analyze_variant(
    const VariantResult& result,
    const KernelConfig* config
) const {
    KernelAnalysis analysis;
    analysis.kernel_name = "unknown";
    analysis.variant_name = result.variant_name;
    analysis.problem_size = result.problem_size;

    // Extract metrics
    analysis.achieved_gflops = result.metrics.performance.gflops;
    analysis.arithmetic_intensity = config ? config->arithmetic_intensity : 0.25;
    analysis.theoretical_max_gflops = calculate_theoretical_max(analysis.arithmetic_intensity);
    analysis.efficiency_vs_roofline = analysis.achieved_gflops / analysis.theoretical_max_gflops;
    analysis.efficiency_vs_peak = analysis.achieved_gflops / hw_.theoretical_peak_sp_gflops;

    // Calculate memory bandwidth utilization
    double bytes_per_second = analysis.achieved_gflops * 1e9 / analysis.arithmetic_intensity;
    analysis.memory_bandwidth_utilization = bytes_per_second / (hw_.measured_memory_bw_gbps * 1e9);

    // Vectorization ratio (if available)
    analysis.vectorization_ratio = result.metrics.simd.packed_256_ops > 0 ?
        static_cast<double>(result.metrics.simd.packed_256_ops) /
        (result.metrics.simd.scalar_ops + result.metrics.simd.packed_256_ops + 1) : 0;

    // IPC (if available from hardware counters)
    analysis.ipc = result.metrics.performance.cycles > 0 ?
        static_cast<double>(result.metrics.performance.instructions) /
        result.metrics.performance.cycles : 0;

    // Cache miss rate
    uint64_t total_accesses = result.metrics.memory.l1_hits + result.metrics.memory.l1_misses;
    analysis.cache_miss_rate = total_accesses > 0 ?
        static_cast<double>(result.metrics.memory.l1_misses) / total_accesses : 0;

    // Classify bottleneck
    analysis.primary_bottleneck = classify_bottleneck(
        analysis.arithmetic_intensity, analysis.efficiency_vs_roofline);

    // Generate insights
    bool is_scalar = result.variant_name.find("scalar") != std::string::npos;
    analysis.insights = generate_insights(
        analysis.achieved_gflops,
        analysis.arithmetic_intensity,
        analysis.memory_bandwidth_utilization * hw_.measured_memory_bw_gbps,
        analysis.ipc,
        analysis.cache_miss_rate,
        analysis.vectorization_ratio,
        result.problem_size,
        is_scalar
    );

    // Add roofline-specific insights
    auto roofline_insights = analyze_roofline_position(
        analysis.arithmetic_intensity, analysis.achieved_gflops);
    analysis.insights.insert(analysis.insights.end(),
        roofline_insights.begin(), roofline_insights.end());

    // Generate prioritized next steps
    analysis.next_steps = generate_next_steps(analysis.insights);

    return analysis;
}

std::vector<KernelAnalysis> InsightsEngine::analyze_benchmark(
    const BenchmarkResult& result,
    const KernelConfig* config
) const {
    std::vector<KernelAnalysis> analyses;

    for (const auto& vr : result.results) {
        auto analysis = analyze_variant(vr, config);
        analysis.kernel_name = result.kernel_name;
        analyses.push_back(std::move(analysis));
    }

    return analyses;
}

InsightsReport InsightsEngine::analyze_all(
    const std::vector<BenchmarkResult>& results,
    const std::map<std::string, const KernelConfig*>& configs
) const {
    InsightsReport report;
    report.hardware = hw_;

    // Get current time
    auto now = std::time(nullptr);
    report.timestamp = std::ctime(&now);

    // Analyze each benchmark
    for (const auto& br : results) {
        const KernelConfig* config = nullptr;
        auto it = configs.find(br.kernel_name);
        if (it != configs.end()) {
            config = it->second;
        }

        auto analyses = analyze_benchmark(br, config);
        report.kernel_analyses.insert(report.kernel_analyses.end(),
            analyses.begin(), analyses.end());
    }

    // Cross-kernel analysis
    report.global_insights = analyze_cross_kernel_patterns(report.kernel_analyses);

    // Count insights by severity
    report.total_insights = 0;
    report.critical_count = 0;
    report.high_count = 0;
    report.medium_count = 0;
    report.low_count = 0;

    for (const auto& ka : report.kernel_analyses) {
        for (const auto& insight : ka.insights) {
            report.total_insights++;
            switch (insight.severity) {
                case InsightSeverity::CRITICAL: report.critical_count++; break;
                case InsightSeverity::HIGH: report.high_count++; break;
                case InsightSeverity::MEDIUM: report.medium_count++; break;
                case InsightSeverity::LOW: report.low_count++; break;
                default: break;
            }
        }
    }

    return report;
}

// ============================================================================
// Insight Generation - Main Entry Point
// ============================================================================

std::vector<OptimizationInsight> InsightsEngine::generate_insights(
    double gflops,
    double arithmetic_intensity,
    double memory_bandwidth_gbps,
    double ipc,
    double cache_miss_rate,
    double vectorization_ratio,
    size_t problem_size,
    bool is_scalar_baseline
) const {
    std::vector<OptimizationInsight> insights;

    double ridge_point = hw_.theoretical_peak_sp_gflops / hw_.measured_memory_bw_gbps;
    double theoretical_max = calculate_theoretical_max(arithmetic_intensity);
    double efficiency = gflops / theoretical_max;

    // ==== ROOFLINE-BASED INSIGHTS ====

    // Rule 1: Memory bandwidth limited (improved: fires for all memory-bound cases)
    if (arithmetic_intensity < ridge_point) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_BOUND;
        // Higher severity if also low efficiency (poorly optimized memory-bound)
        insight.severity = efficiency < 0.3 ? InsightSeverity::CRITICAL : InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Memory Bandwidth Limited";
        insight.description =
            "This kernel is memory-bound with arithmetic intensity (" +
            std::to_string(arithmetic_intensity) + " FLOP/byte) below the ridge point (" +
            std::to_string(ridge_point) + " FLOP/byte). Performance is limited by memory bandwidth.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", Ridge=" + std::to_string(ridge_point) +
            ", BW Utilization=" + std::to_string(memory_bandwidth_gbps / hw_.measured_memory_bw_gbps * 100) + "%";
        insight.recommendation =
            "To improve performance:\n"
            "1. Increase arithmetic intensity through data reuse (loop tiling/blocking)\n"
            "2. Reduce memory traffic (data compression, quantization)\n"
            "3. Improve cache utilization with better data locality\n"
            "4. Consider kernel fusion to amortize memory access costs\n"
            "5. Use prefetching to hide memory latency";
        insight.potential_speedup = std::min(ridge_point / arithmetic_intensity, 10.0);
        insight.references.push_back("https://docs.nersc.gov/tools/performance/roofline/");
        insights.push_back(insight);
    }

    // Rule 2: Low arithmetic intensity with room for improvement
    if (arithmetic_intensity < 0.5 && efficiency < 0.3) {
        OptimizationInsight insight;
        insight.category = InsightCategory::ARITHMETIC_INTENSITY;
        insight.severity = InsightSeverity::CRITICAL;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Very Low Arithmetic Intensity";
        insight.description =
            "Arithmetic intensity is very low (" + std::to_string(arithmetic_intensity) +
            " FLOP/byte). The kernel does very little compute per memory access.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "Consider these optimizations:\n"
            "1. **Loop Tiling**: Block data to fit in L1/L2 cache and reuse\n"
            "   - Tile size for L1: ~" + std::to_string(hw_.cache.l1d_size_kb / 4) + " KB\n"
            "   - Tile size for L2: ~" + std::to_string(hw_.cache.l2_size_kb / 2) + " KB\n"
            "2. **Kernel Fusion**: Combine multiple passes over data\n"
            "3. **Batching**: Process multiple inputs together\n"
            "4. **Blocking for Matrix Operations**: Use cache-oblivious algorithms";
        insight.code_example =
            "// Before: streaming access\n"
            "for (int i = 0; i < N; i++) c[i] = a[i] + b[i];\n\n"
            "// After: tiled access for better reuse\n"
            "for (int ii = 0; ii < N; ii += TILE) {\n"
            "    for (int i = ii; i < min(ii+TILE, N); i++) {\n"
            "        c[i] = a[i] + b[i];\n"
            "    }\n"
            "}";
        insight.potential_speedup = 2.0;
        insights.push_back(insight);
    }

    // Rule 3: Good efficiency - near optimal
    if (efficiency > thresholds_.excellent_efficiency) {
        OptimizationInsight insight;
        insight.category = InsightCategory::GENERAL;
        insight.severity = InsightSeverity::INFO;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Near-Optimal Performance";
        insight.description =
            "This kernel achieves " + std::to_string(efficiency * 100) +
            "% of theoretical maximum. Performance is excellent.";
        insight.evidence = "Achieved=" + std::to_string(gflops) +
            " GFLOPS, Theoretical Max=" + std::to_string(theoretical_max) + " GFLOPS";
        insight.recommendation =
            "Performance is already good. Further optimization options:\n"
            "1. Try AVX-512 if available (current: " + hw_.get_simd_string() + ")\n"
            "2. Explore algorithmic improvements\n"
            "3. Consider multi-threading if single-threaded";
        insight.potential_speedup = 1.0;
        insights.push_back(insight);
    }

    // ==== VECTORIZATION INSIGHTS ====

    // Rule 4: Scalar baseline with no SIMD
    if (is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::VECTORIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Scalar Implementation - SIMD Opportunity";
        insight.description =
            "This is a scalar baseline. SIMD vectorization can provide significant speedup.";
        double expected_speedup = hw_.max_vector_bits / 32.0 * 0.5;  // 50% efficiency assumption
        insight.evidence = "Vector width: " + std::to_string(hw_.max_vector_bits) +
            " bits, Expected speedup: " + std::to_string(expected_speedup) + "x";
        insight.recommendation =
            "SIMD vectorization strategies:\n"
            "1. **Auto-vectorization**: Use `-O3 -march=native` and check compiler reports\n"
            "2. **Pragmas**: Add `#pragma omp simd` or `#pragma GCC ivdep`\n"
            "3. **Intrinsics**: Use AVX2 intrinsics for explicit control\n"
            "4. **Libraries**: Use Highway, ISPC, or Vc for portable SIMD\n\n"
            "Prerequisites for vectorization:\n"
            "- No loop-carried dependencies\n"
            "- Contiguous memory access (unit stride)\n"
            "- Aligned data (32-byte for AVX2, 64-byte for AVX-512)\n"
            "- No function calls in hot loop";
        insight.code_example =
            "// Enable auto-vectorization:\n"
            "#pragma omp simd\n"
            "for (int i = 0; i < N; i++) {\n"
            "    c[i] = a[i] + b[i];\n"
            "}\n\n"
            "// Or use intrinsics:\n"
            "for (int i = 0; i < N; i += 8) {\n"
            "    __m256 va = _mm256_load_ps(&a[i]);\n"
            "    __m256 vb = _mm256_load_ps(&b[i]);\n"
            "    _mm256_store_ps(&c[i], _mm256_add_ps(va, vb));\n"
            "}";
        insight.potential_speedup = expected_speedup;
        insight.references.push_back("https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-simd-vectorization-when-targeting-intel-xeon-processor-scalable-family.html");
        insights.push_back(insight);
    }

    // Rule 5: Low vectorization ratio (for non-scalar variants)
    if (!is_scalar_baseline && vectorization_ratio > 0 &&
        vectorization_ratio < thresholds_.poor_vectorization_ratio) {
        OptimizationInsight insight;
        insight.category = InsightCategory::VECTORIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Low Vectorization Ratio";
        insight.description =
            "Only " + std::to_string(vectorization_ratio * 100) +
            "% of operations are vectorized. Many scalar operations remain.";
        insight.evidence = "Vectorization ratio: " + std::to_string(vectorization_ratio * 100) + "%";
        insight.recommendation =
            "Common vectorization blockers and solutions:\n"
            "1. **Dependencies**: Check for loop-carried dependencies\n"
            "   - Use `#pragma GCC ivdep` if safe to ignore\n"
            "   - Restructure algorithm to remove dependencies\n"
            "2. **Non-contiguous access**: Use stride-1 access patterns\n"
            "   - Convert AoS to SoA data layout\n"
            "3. **Aliasing**: Add `restrict` keyword to pointers\n"
            "4. **Conditionals**: Replace branches with masked operations\n"
            "5. **Function calls**: Inline or use vectorized math libraries";
        insight.potential_speedup = 1.0 / vectorization_ratio;
        insights.push_back(insight);
    }

    // ==== CACHE EFFICIENCY INSIGHTS ====

    // Rule 6: High cache miss rate
    if (cache_miss_rate > thresholds_.high_l1_miss_rate) {
        OptimizationInsight insight;
        insight.category = InsightCategory::CACHE_EFFICIENCY;
        insight.severity = cache_miss_rate > 0.2 ? InsightSeverity::CRITICAL : InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "High Cache Miss Rate";
        insight.description =
            "Cache miss rate is " + std::to_string(cache_miss_rate * 100) +
            "%. Significant time is spent waiting for memory.";
        insight.evidence = "Cache miss rate: " + std::to_string(cache_miss_rate * 100) + "%";
        insight.recommendation =
            "Cache optimization strategies:\n"
            "1. **Loop Tiling/Blocking**:\n"
            "   - L1 tile: " + std::to_string(static_cast<int>(sqrt(hw_.cache.l1d_size_kb * 1024 / sizeof(float) / 2))) + " elements\n"
            "   - L2 tile: " + std::to_string(static_cast<int>(sqrt(hw_.cache.l2_size_kb * 1024 / sizeof(float) / 2))) + " elements\n"
            "2. **Prefetching**: Add software prefetch hints\n"
            "   `_mm_prefetch(&data[i+64], _MM_HINT_T0);`\n"
            "3. **Data Layout**: Ensure sequential access patterns\n"
            "4. **Working Set**: Reduce data size if possible";
        insight.code_example =
            "// Loop tiling example\n"
            "const int TILE = 64;  // Tune based on cache size\n"
            "for (int ii = 0; ii < N; ii += TILE) {\n"
            "    for (int jj = 0; jj < M; jj += TILE) {\n"
            "        for (int i = ii; i < min(ii+TILE,N); i++) {\n"
            "            for (int j = jj; j < min(jj+TILE,M); j++) {\n"
            "                // Process tile\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}";
        insight.potential_speedup = 1.0 + cache_miss_rate * 5;  // Rough estimate
        insight.references.push_back("https://www.intel.com/content/www/us/en/developer/articles/technical/loop-optimizations-where-blocks-are-required.html");
        insights.push_back(insight);
    }

    // Rule 7: Working set size vs cache analysis
    size_t working_set_bytes = problem_size * sizeof(float) * 2;  // Estimate
    if (working_set_bytes > hw_.cache.l1d_size_kb * 1024) {
        OptimizationInsight insight;
        insight.category = InsightCategory::CACHE_EFFICIENCY;

        if (working_set_bytes > hw_.cache.l3_size_kb * 1024) {
            insight.severity = InsightSeverity::HIGH;
            insight.title = "Working Set Exceeds L3 Cache";
            insight.description =
                "Working set (~" + std::to_string(working_set_bytes / 1024 / 1024) +
                " MB) exceeds L3 cache (" + std::to_string(hw_.cache.l3_size_kb / 1024) + " MB).";
        } else if (working_set_bytes > hw_.cache.l2_size_kb * 1024) {
            insight.severity = InsightSeverity::MEDIUM;
            insight.title = "Working Set Exceeds L2 Cache";
            insight.description =
                "Working set (~" + std::to_string(working_set_bytes / 1024) +
                " KB) exceeds L2 cache (" + std::to_string(hw_.cache.l2_size_kb) + " KB).";
        } else {
            insight.severity = InsightSeverity::LOW;
            insight.title = "Working Set Exceeds L1 Cache";
            insight.description =
                "Working set (~" + std::to_string(working_set_bytes / 1024) +
                " KB) exceeds L1 cache (" + std::to_string(hw_.cache.l1d_size_kb) + " KB).";
        }

        insight.confidence = InsightConfidence::MEDIUM;
        insight.evidence = "Working set: " + std::to_string(working_set_bytes) +
            " bytes, Problem size: " + std::to_string(problem_size);
        insight.recommendation =
            "Consider:\n"
            "1. **Streaming stores**: Use non-temporal stores to bypass cache\n"
            "   `_mm256_stream_ps(ptr, vec);`\n"
            "2. **Cache blocking**: Process data in cache-sized chunks\n"
            "3. **Data compression**: Reduce working set size\n"
            "4. **Out-of-core algorithms**: For very large datasets";
        insight.potential_speedup = 1.2;
        insights.push_back(insight);
    }

    // ==== IPC AND INSTRUCTION MIX INSIGHTS ====

    // Rule 8: Low IPC
    if (ipc > 0 && ipc < thresholds_.poor_ipc) {
        OptimizationInsight insight;
        insight.category = InsightCategory::INSTRUCTION_MIX;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Low Instructions Per Cycle (IPC)";
        insight.description =
            "IPC is " + std::to_string(ipc) +
            ", well below expected ~3.0 for vectorized code. The CPU is underutilized.";
        insight.evidence = "IPC: " + std::to_string(ipc);
        insight.recommendation =
            "Low IPC causes and solutions:\n"
            "1. **Memory stalls**: Reduce cache misses (see cache insights)\n"
            "2. **Branch mispredictions**: Use branchless code\n"
            "3. **Long dependency chains**: Break chains with multiple accumulators\n"
            "4. **Instruction latency**: Use FMA and low-latency operations\n"
            "5. **Frontend stalls**: Reduce code size, improve instruction cache";
        insight.code_example =
            "// Break dependency chain with multiple accumulators\n"
            "// Before:\n"
            "float sum = 0;\n"
            "for (int i = 0; i < N; i++) sum += a[i];\n\n"
            "// After (4x unroll with independent accumulators):\n"
            "float sum0=0, sum1=0, sum2=0, sum3=0;\n"
            "for (int i = 0; i < N; i += 4) {\n"
            "    sum0 += a[i];   sum1 += a[i+1];\n"
            "    sum2 += a[i+2]; sum3 += a[i+3];\n"
            "}\n"
            "sum = sum0 + sum1 + sum2 + sum3;";
        insight.potential_speedup = thresholds_.good_ipc / ipc;
        insights.push_back(insight);
    }

    // ==== LOOP OPTIMIZATION INSIGHTS ====

    // Rule 9: Recommend loop unrolling for compute-bound kernels
    // Unrolling only helps when compute-bound (AI >= ridge_point) - for memory-bound
    // workloads, the bottleneck is memory bandwidth, not instruction throughput.
    // FIXED: Previously triggered at AI > 0.25, causing false positives for streaming ops.
    bool is_compute_bound = arithmetic_intensity >= ridge_point * 0.8;  // Allow near-ridge cases
    if (efficiency < 0.5 && !is_scalar_baseline && is_compute_bound) {
        // Only suggest if we don't have IPC data (otherwise rule 14 handles it)
        if (ipc == 0) {
            OptimizationInsight insight;
            insight.category = InsightCategory::LOOP_OPTIMIZATION;
            insight.severity = InsightSeverity::MEDIUM;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Consider Loop Unrolling";
            insight.description =
                "Low efficiency (" + std::to_string(efficiency * 100) +
                "%) suggests potential for improving instruction-level parallelism.";
            insight.evidence = "Efficiency: " + std::to_string(efficiency * 100) +
                "%, Problem size: " + std::to_string(problem_size);
            insight.recommendation =
                "Loop unrolling guidelines:\n"
                "1. **Unroll factor**: Start with 4x (matches FMA latency)\n"
                "2. **Multiple accumulators**: Break dependency chains\n"
                "3. **Register limits**: AVX2 has 16 YMM registers\n"
                "4. **Compiler flags**: `-funroll-loops` or `#pragma unroll(N)`\n"
                "5. **Test empirically**: Optimal factor varies by kernel";
            insight.code_example =
                "// 4x unrolled with independent accumulators\n"
                "__m256 sum0 = _mm256_setzero_ps();\n"
                "__m256 sum1 = _mm256_setzero_ps();\n"
                "__m256 sum2 = _mm256_setzero_ps();\n"
                "__m256 sum3 = _mm256_setzero_ps();\n\n"
                "for (int i = 0; i < N; i += 32) {\n"
                "    sum0 = _mm256_fmadd_ps(load(&a[i+0]), load(&b[i+0]), sum0);\n"
                "    sum1 = _mm256_fmadd_ps(load(&a[i+8]), load(&b[i+8]), sum1);\n"
                "    sum2 = _mm256_fmadd_ps(load(&a[i+16]), load(&b[i+16]), sum2);\n"
                "    sum3 = _mm256_fmadd_ps(load(&a[i+24]), load(&b[i+24]), sum3);\n"
                "}";
            insight.potential_speedup = 1.5;
            insight.references.push_back("https://en.algorithmica.org/hpc/simd/reduction/");
            insights.push_back(insight);
        }
    }

    // ==== MEMORY ACCESS PATTERN INSIGHTS ====

    // Rule 10: Recommend data alignment
    if (!is_scalar_baseline && efficiency < 0.7) {
        OptimizationInsight insight;
        insight.category = InsightCategory::DATA_ALIGNMENT;
        insight.severity = InsightSeverity::LOW;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "Ensure Data Alignment";
        insight.description =
            "SIMD performance depends on data alignment. Ensure " +
            std::to_string(hw_.max_vector_bits / 8) + "-byte alignment.";
        insight.evidence = "Vector width: " + std::to_string(hw_.max_vector_bits) + " bits";
        insight.recommendation =
            "Data alignment best practices:\n"
            "1. **Allocation**: Use `aligned_alloc(64, size)` or `_mm_malloc`\n"
            "2. **Declarations**: Use `alignas(64)` for stack arrays\n"
            "3. **Compiler hints**: `__attribute__((aligned(64)))`\n"
            "4. **Cache lines**: Align to 64 bytes for optimal cache behavior";
        insight.code_example =
            "// Aligned allocation\n"
            "float* data = (float*)aligned_alloc(64, N * sizeof(float));\n\n"
            "// Or with Highway:\n"
            "auto data = hwy::AllocateAligned<float>(N);\n\n"
            "// Stack array:\n"
            "alignas(64) float local_data[256];";
        insight.potential_speedup = 1.1;
        insights.push_back(insight);
    }

    // ==== NEW RULES BASED ON EXPERT RESEARCH ====

    // Rule 11: FMA Utilization Opportunity
    // FMA can provide 2x throughput for multiply-add operations
    if (!is_scalar_baseline && arithmetic_intensity > 1.0 && efficiency < 0.6) {
        OptimizationInsight insight;
        insight.category = InsightCategory::INSTRUCTION_MIX;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Check FMA Instruction Usage";
        insight.description =
            "Compute-intensive code may benefit from Fused Multiply-Add (FMA) instructions. "
            "FMA performs a*b+c in a single operation with better throughput and precision.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "FMA optimization strategies:\n"
            "1. **Compiler flags**: Ensure `-mfma` is enabled\n"
            "2. **Explicit intrinsics**: Use `_mm256_fmadd_ps(a, b, c)`\n"
            "3. **Code structure**: Expose multiply-add patterns:\n"
            "   - `result = a*b + c` instead of `tmp = a*b; result = tmp + c`\n"
            "4. **Latency**: FMA has 4-5 cycle latency - use 4+ accumulators";
        insight.code_example =
            "// Before: separate multiply and add (2 instructions)\n"
            "__m256 result = _mm256_add_ps(_mm256_mul_ps(a, b), c);\n\n"
            "// After: fused multiply-add (1 instruction)\n"
            "__m256 result = _mm256_fmadd_ps(a, b, c);  // a*b + c\n\n"
            "// Also available:\n"
            "// _mm256_fmsub_ps(a, b, c)  // a*b - c\n"
            "// _mm256_fnmadd_ps(a, b, c) // -a*b + c\n"
            "// _mm256_fnmsub_ps(a, b, c) // -a*b - c";
        insight.potential_speedup = 1.3;
        insight.references.push_back("https://momentsingraphics.de/FMA.html");
        insights.push_back(insight);
    }

    // Rule 12: Memory Latency vs Bandwidth Bound Classification
    // Low IPC with low bandwidth utilization suggests latency-bound
    double bw_utilization = memory_bandwidth_gbps / hw_.measured_memory_bw_gbps;
    if (ipc > 0 && ipc < 1.0 && bw_utilization < 0.3 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_BOUND;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Memory Latency Bound (Not Bandwidth)";
        insight.description =
            "Low IPC (" + std::to_string(ipc) + ") with low bandwidth utilization (" +
            std::to_string(bw_utilization * 100) + "%) indicates the kernel is stalled on "
            "memory latency rather than bandwidth saturation.";
        insight.evidence = "IPC=" + std::to_string(ipc) +
            ", BW Utilization=" + std::to_string(bw_utilization * 100) + "%";
        insight.recommendation =
            "Memory latency mitigation strategies:\n"
            "1. **Software prefetching**: Prefetch data before it's needed\n"
            "   `_mm_prefetch(&data[i+DISTANCE], _MM_HINT_T0);`\n"
            "2. **Increase ILP**: Process multiple independent streams\n"
            "3. **Data structure optimization**: Improve spatial locality\n"
            "4. **Pointer chasing**: Convert linked structures to arrays\n"
            "5. **Cache blocking**: Ensure working set fits in cache";
        insight.code_example =
            "// Software prefetching example\n"
            "const int PREFETCH_DIST = 64;  // ~64 elements ahead\n"
            "for (int i = 0; i < N; i += 8) {\n"
            "    _mm_prefetch(&data[i + PREFETCH_DIST], _MM_HINT_T0);\n"
            "    __m256 v = _mm256_load_ps(&data[i]);\n"
            "    // ... process v ...\n"
            "}";
        insight.potential_speedup = 2.0;
        insights.push_back(insight);
    }

    // Rule 13: Recommend Software Prefetching for Large Working Sets
    size_t estimated_ws = problem_size * sizeof(float);
    if (estimated_ws > hw_.cache.l2_size_kb * 1024 && efficiency < 0.5 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LATENCY_HIDING;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Consider Software Prefetching";
        insight.description =
            "Working set (" + std::to_string(estimated_ws / 1024) + " KB) exceeds L2 cache. "
            "Software prefetching can hide memory latency by fetching data before it's needed.";
        insight.evidence = "Working set: " + std::to_string(estimated_ws / 1024) +
            " KB, L2 cache: " + std::to_string(hw_.cache.l2_size_kb) + " KB";
        insight.recommendation =
            "Prefetch distance calculation:\n"
            "  distance = (memory_latency_cycles / loop_body_cycles) * elements_per_iteration\n\n"
            "Typical values:\n"
            "- L2 prefetch distance: 16-64 cache lines ahead\n"
            "- L1 prefetch distance: 4-16 cache lines ahead\n\n"
            "Prefetch hints:\n"
            "- `_MM_HINT_T0`: Prefetch to all cache levels (L1)\n"
            "- `_MM_HINT_T1`: Prefetch to L2 and below\n"
            "- `_MM_HINT_T2`: Prefetch to L3 and below\n"
            "- `_MM_HINT_NTA`: Non-temporal, minimize cache pollution";
        insight.code_example =
            "// Prefetch with distance tuning\n"
            "#define L2_PREFETCH_DIST 32\n"
            "#define L1_PREFETCH_DIST 8\n\n"
            "for (int i = 0; i < N; i += 8) {\n"
            "    // Prefetch to L2 (further ahead)\n"
            "    _mm_prefetch(&a[i + L2_PREFETCH_DIST], _MM_HINT_T1);\n"
            "    // Prefetch to L1 (closer)\n"
            "    _mm_prefetch(&a[i + L1_PREFETCH_DIST], _MM_HINT_T0);\n"
            "    \n"
            "    __m256 va = _mm256_load_ps(&a[i]);\n"
            "    // ... process ...\n"
            "}";
        insight.potential_speedup = 1.6;
        insight.references.push_back("https://www.sciencedirect.com/topics/computer-science/prefetch-distance");
        insights.push_back(insight);
    }

    // Rule 14: Loop Unrolling Guidelines Based on FMA Latency
    // FMA latency is ~4-5 cycles, so 4-8x unroll is optimal
    if (!is_scalar_baseline && efficiency < 0.6 && arithmetic_intensity > 0.5) {
        // Check if current performance suggests insufficient ILP
        if (ipc > 0 && ipc < 2.0) {
            OptimizationInsight insight;
            insight.category = InsightCategory::LOOP_OPTIMIZATION;
            insight.severity = InsightSeverity::MEDIUM;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Optimize Loop Unrolling Factor";
            insight.description =
                "IPC of " + std::to_string(ipc) + " suggests insufficient instruction-level "
                "parallelism. Optimal unroll factor depends on instruction latency and "
                "available registers.";
            insight.evidence = "IPC=" + std::to_string(ipc) +
                ", Efficiency=" + std::to_string(efficiency * 100) + "%";
            insight.recommendation =
                "Loop unrolling guidelines:\n"
                "1. **Unroll factor = ceil(latency / throughput)**\n"
                "   - FMA: latency=4-5, throughput=2/cycle -> unroll 2-3x minimum\n"
                "   - For hiding latency: 4-8x unroll is typical\n"
                "2. **Register pressure**: Don't exceed ~12-14 vector registers\n"
                "   - AVX2: 16 YMM registers available\n"
                "   - Use 4 for accumulators, rest for data\n"
                "3. **Code size**: Keep hot loop in DSB (µop cache ~1.5K µops)\n"
                "4. **Compiler hints**: `#pragma unroll(N)` or let compiler decide";
            insight.code_example =
                "// 4x unrolled reduction (matches FMA latency)\n"
                "__m256 sum0 = _mm256_setzero_ps();\n"
                "__m256 sum1 = _mm256_setzero_ps();\n"
                "__m256 sum2 = _mm256_setzero_ps();\n"
                "__m256 sum3 = _mm256_setzero_ps();\n\n"
                "for (int i = 0; i < N; i += 32) {\n"
                "    sum0 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+0]),\n"
                "                           _mm256_load_ps(&b[i+0]), sum0);\n"
                "    sum1 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+8]),\n"
                "                           _mm256_load_ps(&b[i+8]), sum1);\n"
                "    sum2 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+16]),\n"
                "                           _mm256_load_ps(&b[i+16]), sum2);\n"
                "    sum3 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+24]),\n"
                "                           _mm256_load_ps(&b[i+24]), sum3);\n"
                "}\n"
                "// Combine accumulators\n"
                "__m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1),\n"
                "                             _mm256_add_ps(sum2, sum3));";
            insight.potential_speedup = 1.5;
            insight.references.push_back("https://en.algorithmica.org/hpc/simd/reduction/");
            insights.push_back(insight);
        }
    }

    // Rule 15: AVX-512 Frequency Throttling Warning
    bool has_avx512 = hw_.max_vector_bits >= 512 ||
                      has_extension(hw_.simd_extensions, SIMDExtension::AVX512F);
    if (has_avx512 && !is_scalar_baseline && gflops > 10) {
        OptimizationInsight insight;
        insight.category = InsightCategory::GENERAL;
        insight.severity = InsightSeverity::LOW;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "AVX-512 Frequency Throttling Awareness";
        insight.description =
            "This system supports AVX-512. Be aware that heavy AVX-512 usage can trigger "
            "frequency downclocking (3-20% reduction) on some Intel processors.";
        insight.evidence = "AVX-512 available: " + std::string(has_avx512 ? "yes" : "no");
        insight.recommendation =
            "AVX-512 frequency throttling considerations:\n"
            "1. **License levels**: Intel uses L0 (100%), L1 (~97%), L2 (~85%)\n"
            "2. **Light instructions** (512-bit loads/stores): L1 throttling\n"
            "3. **Heavy instructions** (512-bit FMA, multiply): L2 throttling\n"
            "4. **Mixed workloads**: Even 2.5% AVX-512 can trigger throttling\n\n"
            "Mitigation strategies:\n"
            "- Profile with and without AVX-512 to measure actual impact\n"
            "- Consider AVX2 for mixed workloads\n"
            "- Use AVX-512 for dedicated compute-heavy sections only\n"
            "- Rocket Lake and newer: No license-based throttling";
        insight.potential_speedup = 1.0;  // Informational
        insight.references.push_back("https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html");
        insights.push_back(insight);
    }

    // Rule 16: Branch Misprediction Recommendation
    // For low-IPC code that's not memory-bound, branches may be the issue
    if (ipc > 0 && ipc < 1.5 && bw_utilization > 0.3 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::BRANCH_PREDICTION;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "Check for Branch Mispredictions";
        insight.description =
            "Low IPC (" + std::to_string(ipc) + ") with moderate bandwidth suggests possible "
            "branch mispredictions. Modern CPUs have 15-20 cycle misprediction penalty.";
        insight.evidence = "IPC=" + std::to_string(ipc) +
            ", BW Utilization=" + std::to_string(bw_utilization * 100) + "%";
        insight.recommendation =
            "Branch optimization strategies:\n"
            "1. **Threshold**: Branches with <75% prediction accuracy hurt performance\n"
            "2. **Branchless code**: Use arithmetic/SIMD masking instead\n"
            "3. **Profile branches**: Use `perf stat -e branch-misses`\n"
            "4. **CMOV**: Compiler may use conditional moves for ternary\n"
            "5. **Loop restructuring**: Separate predictable from unpredictable branches\n\n"
            "When branchless helps:\n"
            "- Unpredictable conditions (data-dependent, random)\n"
            "- SIMD code (no scalar branches in vector loops)\n"
            "- Low cache miss rates (data dependency penalty is acceptable)";
        insight.code_example =
            "// Branchy (bad if unpredictable)\n"
            "if (x > 0) result = a; else result = b;\n\n"
            "// Branchless with arithmetic\n"
            "int mask = -(x > 0);  // -1 if true, 0 if false\n"
            "result = (a & mask) | (b & ~mask);\n\n"
            "// SIMD branchless\n"
            "__m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);\n"
            "__m256 result = _mm256_blendv_ps(b, a, mask);";
        insight.potential_speedup = 1.5;
        insight.references.push_back("https://en.algorithmica.org/hpc/pipelining/branchless/");
        insights.push_back(insight);
    }

    // Rule 17: Memory Bandwidth Saturation Detection
    if (bw_utilization > thresholds_.high_bw_utilization) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_BOUND;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Memory Bandwidth Saturated";
        insight.description =
            "Memory bandwidth utilization is " + std::to_string(bw_utilization * 100) +
            "% (" + std::to_string(memory_bandwidth_gbps) + " GB/s). "
            "Further compute optimizations won't significantly improve performance.";
        insight.evidence = "Achieved BW: " + std::to_string(memory_bandwidth_gbps) +
            " GB/s, Peak BW: " + std::to_string(hw_.measured_memory_bw_gbps) + " GB/s";
        insight.recommendation =
            "When memory bandwidth is saturated:\n"
            "1. **Reduce memory traffic**:\n"
            "   - Use compression or lower precision (FP16, INT8)\n"
            "   - Eliminate redundant loads/stores\n"
            "2. **Increase arithmetic intensity**:\n"
            "   - Kernel fusion (combine multiple passes)\n"
            "   - Cache blocking for data reuse\n"
            "3. **Non-temporal stores**: For write-only data\n"
            "   `_mm256_stream_ps(ptr, vec);`\n"
            "4. **Hardware solutions**:\n"
            "   - Higher memory bandwidth (HBM, more channels)\n"
            "   - Multi-socket for more aggregate bandwidth";
        insight.potential_speedup = 1.0;  // Limited improvement possible
        insights.push_back(insight);
    }

    // Rule 18: TMA-Style Frontend Bound Check
    // Very high efficiency but low absolute performance may indicate frontend issues
    if (efficiency > 0.5 && gflops < hw_.theoretical_peak_sp_gflops * 0.3 && !is_scalar_baseline) {
        // This is unusual - high relative efficiency but low absolute performance
        // Could indicate frontend bottleneck or instruction cache issues
        OptimizationInsight insight;
        insight.category = InsightCategory::INSTRUCTION_MIX;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "Check Frontend Efficiency";
        insight.description =
            "Performance pattern suggests possible frontend bottleneck. "
            "DSB (µop cache) misses or instruction cache issues may be limiting throughput.";
        insight.evidence = "Achieved: " + std::to_string(gflops) +
            " GFLOPS, Efficiency vs roofline: " + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "Frontend optimization strategies:\n"
            "1. **Reduce code size**: Keep hot loops small (<1.5K µops for DSB)\n"
            "2. **Loop alignment**: Align hot loops to 32-byte boundary\n"
            "   `__attribute__((aligned(32)))`\n"
            "3. **Limit unrolling**: Over-unrolling causes DSB eviction\n"
            "4. **Check I-cache misses**: Use `perf stat -e L1-icache-load-misses`\n"
            "5. **Profile frontend**: TMA Frontend_Bound metric in VTune\n\n"
            "DSB vs MITE:\n"
            "- DSB (Decoded Stream Buffer): 6 µops/cycle, preferred\n"
            "- MITE (legacy decoder): 4-5 µops/cycle, power hungry";
        insight.potential_speedup = 1.3;
        insight.references.push_back("https://paweldziepak.dev/2019/06/21/avoiding-icache-misses/");
        insights.push_back(insight);
    }

    // Rule 19: Multi-threaded False Sharing Warning (for future extension)
    // This is a placeholder - actual detection requires thread-level metrics
    if (problem_size > 1000000 && hw_.logical_cores > 1) {
        // Large problem size on multi-core system
        OptimizationInsight insight;
        insight.category = InsightCategory::PARALLELISM;
        insight.severity = InsightSeverity::LOW;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "False Sharing Prevention (Multi-threaded)";
        insight.description =
            "For multi-threaded execution, ensure thread-local data is separated by "
            "at least one cache line (64 bytes) to prevent false sharing.";
        insight.evidence = "Cores: " + std::to_string(hw_.logical_cores) +
            ", Problem size: " + std::to_string(problem_size);
        insight.recommendation =
            "False sharing prevention:\n"
            "1. **Padding**: Add 64 bytes between thread-local variables\n"
            "   ```cpp\n"
            "   struct alignas(64) ThreadData {\n"
            "       float local_sum;\n"
            "       char padding[60];  // Ensure 64-byte separation\n"
            "   };\n"
            "   ```\n"
            "2. **Detection**: Use `perf c2c` to find contended cache lines\n"
            "3. **HITM metric**: High Remote-HITM indicates cross-socket contention\n"
            "4. **Thread-local storage**: Use `thread_local` for per-thread data";
        insight.potential_speedup = 1.2;
        insight.references.push_back("https://joemario.github.io/blog/2016/09/01/c2c-blog/");
        insights.push_back(insight);
    }

    // Rule 20: NUMA Awareness for Large Working Sets
    size_t l3_bytes = hw_.cache.l3_size_kb * 1024;
    if (estimated_ws > l3_bytes * 2 && hw_.logical_cores > 8) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "NUMA Awareness for Large Working Set";
        insight.description =
            "Working set (" + std::to_string(estimated_ws / 1024 / 1024) + " MB) significantly "
            "exceeds L3 cache. On multi-socket systems, ensure NUMA-aware memory allocation.";
        insight.evidence = "Working set: " + std::to_string(estimated_ws / 1024 / 1024) +
            " MB, L3 cache: " + std::to_string(hw_.cache.l3_size_kb / 1024) + " MB";
        insight.recommendation =
            "NUMA optimization strategies:\n"
            "1. **First-touch policy**: Initialize data on the thread that uses it\n"
            "2. **numactl**: Run with `numactl --localalloc` or `--interleave=all`\n"
            "3. **libnuma**: Use `numa_alloc_onnode()` for explicit placement\n"
            "4. **Thread affinity**: Pin threads to NUMA nodes with `pthread_setaffinity_np`\n"
            "5. **Data partitioning**: Divide work so each thread accesses local memory\n\n"
            "Detection:\n"
            "- `numastat` to see memory distribution\n"
            "- VTune NUMA analysis for cross-socket traffic";
        insight.potential_speedup = 1.5;
        insights.push_back(insight);
    }

    // Rule 21: Streaming Store Opportunity (VALIDATED: 47% speedup for scale)
    // VALIDATED FINDINGS from streaming_optimization_test.cc:
    // - Scale with NT stores: 1.47x faster
    // - Add with NT stores: 1.30x faster
    // - Unrolling streaming ops: 0.95x (SLOWER)
    //
    // NT stores help by avoiding Read-For-Ownership (RFO) overhead.
    // When CPU writes to a cache line it doesn't own, it must:
    // 1. Read the entire cache line from memory
    // 2. Modify the portion being written
    // 3. Write back (or mark dirty)
    // NT stores bypass this, writing directly to memory.
    size_t working_set = problem_size * sizeof(float);
    bool working_set_exceeds_l3 = working_set > hw_.cache.l3_size_kb * 1024;  // At least 1x L3
    bool is_pure_streaming = arithmetic_intensity < 0.15;  // Very low AI = streaming write pattern

    // VALIDATED: NT stores help even at moderate BW utilization - removed >50% requirement
    if (is_pure_streaming && working_set_exceeds_l3 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
        insight.severity = InsightSeverity::HIGH;  // Increased - validated significant benefit
        insight.confidence = InsightConfidence::HIGH;  // Increased - validated 47% improvement
        insight.title = "Use Non-Temporal Stores for Streaming Writes";
        insight.description =
            "Pure streaming operation (AI=" + std::to_string(arithmetic_intensity) + ") with "
            "working set (" + std::to_string(working_set / 1024 / 1024) + " MB) exceeding cache. "
            "VALIDATED: NT stores give 30-47% speedup by avoiding Read-For-Ownership (RFO). "
            "NOTE: Loop unrolling does NOT help streaming ops (measured 5% slower).";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            " (pure streaming), Working set=" + std::to_string(working_set / 1024 / 1024) + " MB" +
            ", L3=" + std::to_string(hw_.cache.l3_size_kb / 1024) + " MB";
        insight.recommendation =
            "Streaming write optimization (VALIDATED):\n"
            "1. **Use NT stores**: hwy::Stream() or _mm256_stream_ps()\n"
            "2. **Add memory fence**: hwy::FlushStream() or _mm_sfence() at end\n"
            "3. **DON'T unroll**: Unrolling streaming ops is counterproductive\n"
            "4. **Alignment**: Ensure 32-byte alignment (AVX2)\n"
            "5. **Expected speedup**: 1.3-1.5x for pure streaming writes\n\n"
            "Why NT stores help:\n"
            "- Avoids Read-For-Ownership (RFO) cache line fetch\n"
            "- Reduces memory bus traffic by 2x for write-only patterns\n"
            "- Bypasses cache pollution for data not reused soon";
        insight.code_example =
            "// Highway: Use Stream() instead of Store()\n"
            "for (size_t i = 0; i < N; i += Lanes(d)) {\n"
            "    auto v = Mul(Load(d, in + i), alpha);\n"
            "    Stream(v, d, out + i);  // Non-temporal store\n"
            "}\n"
            "hwy::FlushStream();  // Required after streaming stores\n\n"
            "// Intrinsics equivalent:\n"
            "_mm256_stream_ps(&dst[i], v);\n"
            "_mm_sfence();  // After all stores complete";
        insight.potential_speedup = 1.4;  // Validated average
        insight.references.push_back("https://blogs.fau.de/hager/archives/2103");
        insight.references.push_back("https://sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/");
        insights.push_back(insight);
    }

    // Rule 22: Reduction Operation Dependency Chain
    // Special case: Reductions appear memory-bound by AI but are often bottlenecked
    // by dependency chains in the accumulator. Unrolling with multiple accumulators
    // helps even for memory-bound workloads.
    //
    // CRITICAL FIX: Must distinguish between:
    // - REDUCTION (dot product, sum): AI >= 0.2, has accumulator, unrolling helps
    // - STREAMING (scale, copy): AI < 0.15, no accumulator, NT stores help
    //
    // Reduction pattern: reads 2 inputs (8B), 2 FLOPs (mul+add) -> AI = 0.25
    // Streaming pattern: reads 1 (4B), writes 1 (4B), 1 FLOP -> AI = 0.125
    // VALIDATED: Unrolling scale gave 0.95x (SLOWER), but dot_product got 2.19x
    //
    // EDGE CASE FIX: If BW efficiency > 300%, memory is already cache-efficient
    // (data in L2/L3) and reduction chain isn't the bottleneck. Don't suggest.
    // VALIDATED: dot_product @ 65536 (256KB L2) had 635% efficiency, unrolling gave 0.98x
    bool is_reduction_pattern = arithmetic_intensity >= 0.2 && arithmetic_intensity < 0.5;
    bool is_pure_streaming_op = arithmetic_intensity < 0.15;  // Scale, copy, etc.
    double peak_efficiency = gflops / hw_.theoretical_peak_sp_gflops;
    // Check if we're far from peak (room for improvement via ILP)
    bool has_ilp_headroom = peak_efficiency < 0.15 && gflops > 0;
    // If efficiency > 500%, data is L2/L3 resident and already running optimally
    // Using 500% threshold because sum @ 323% still benefits from unrolling (3.34x)
    bool already_cache_optimized = efficiency > 5.0;  // >500% means very fast cache hits
    // Only apply to ACTUAL reductions when reduction chain is the bottleneck
    if (is_reduction_pattern && has_ilp_headroom && !is_scalar_baseline &&
        !is_pure_streaming_op && !already_cache_optimized && ipc == 0) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Break Reduction Dependency Chain";
        insight.description =
            "This kernel has reduction-like characteristics (low arithmetic intensity: " +
            std::to_string(arithmetic_intensity) + " FLOP/byte). "
            "Even for memory-bound workloads, reductions are often bottlenecked by dependency "
            "chains in the accumulator, not memory bandwidth. Multiple accumulators enable "
            "instruction-level parallelism by breaking the serial dependency.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            " (reduction-like), Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "Reduction optimization strategies:\n"
            "1. **Multiple accumulators**: Use 4-8 independent sum variables\n"
            "2. **Matches FMA latency**: 4-5 cycle latency requires ~4 independent ops\n"
            "3. **SIMD lanes**: Each SIMD lane can have its own accumulator\n"
            "4. **Final reduction**: Combine accumulators at the end\n"
            "5. **Expected speedup**: 2-4x for dependency-chain-bound reductions";
        insight.code_example =
            "// Breaking dependency chain with 4 accumulators:\n"
            "__m256 sum0 = _mm256_setzero_ps();\n"
            "__m256 sum1 = _mm256_setzero_ps();\n"
            "__m256 sum2 = _mm256_setzero_ps();\n"
            "__m256 sum3 = _mm256_setzero_ps();\n\n"
            "for (int i = 0; i < N; i += 32) {\n"
            "    sum0 = _mm256_add_ps(sum0, _mm256_load_ps(&a[i+0]));\n"
            "    sum1 = _mm256_add_ps(sum1, _mm256_load_ps(&a[i+8]));\n"
            "    sum2 = _mm256_add_ps(sum2, _mm256_load_ps(&a[i+16]));\n"
            "    sum3 = _mm256_add_ps(sum3, _mm256_load_ps(&a[i+24]));\n"
            "}\n"
            "sum = hadd(sum0 + sum1 + sum2 + sum3);";
        insight.potential_speedup = 3.0;
        insight.references.push_back("https://en.algorithmica.org/hpc/simd/reduction/");
        insights.push_back(insight);
    }

    // Rule 23: Fast Polynomial Approximations for Transcendentals (VALIDATED: 6x speedup)
    // When efficiency is low and arithmetic intensity suggests transcendental functions,
    // fast polynomial approximations provide massive speedups (5-10x typical)
    // Validated on: Sigmoid (6.47x), Softmax (5.65x)
    bool likely_transcendental = arithmetic_intensity > 0.8 && arithmetic_intensity < 3.0;
    bool low_absolute_perf = gflops < hw_.theoretical_peak_sp_gflops * 0.15;
    if (likely_transcendental && low_absolute_perf && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::INSTRUCTION_MIX;
        insight.severity = InsightSeverity::CRITICAL;  // High impact - validated 6x speedup
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Use Fast Polynomial Approximations for Transcendentals";
        insight.description =
            "Low performance (" + std::to_string(gflops) + " GFLOPS, " +
            std::to_string(gflops / hw_.theoretical_peak_sp_gflops * 100) + "% of peak) with "
            "moderate arithmetic intensity suggests expensive transcendental functions "
            "(exp, log, tanh, etc.). SIMD polynomial approximations provide 5-10x speedup.";
        insight.evidence = "GFLOPS=" + std::to_string(gflops) +
            ", AI=" + std::to_string(arithmetic_intensity) +
            ", Peak efficiency=" + std::to_string(gflops / hw_.theoretical_peak_sp_gflops * 100) + "%";
        insight.recommendation =
            "Transcendental function optimization:\n"
            "1. **Replace std::exp/log with polynomial approximations**\n"
            "   - 4-8 term polynomials give <0.01% error for most ML applications\n"
            "2. **Use range reduction + polynomial**:\n"
            "   - exp(x) = 2^(x*log2(e)) = 2^int * 2^frac\n"
            "   - Use polynomial for 2^frac, bit manipulation for 2^int\n"
            "3. **Leverage FMA**: Horner's method with FMA for polynomials\n"
            "4. **VALIDATED**: Sigmoid with fast exp gives 6.5x speedup\n"
            "5. **VALIDATED**: Softmax with fast exp gives 5.6x speedup\n\n"
            "Accuracy vs Speed tradeoff:\n"
            "- 4-term: ~0.1% error, fastest\n"
            "- 6-term: ~0.01% error, good balance\n"
            "- 8-term: ~0.001% error, near std::exp";
        insight.code_example =
            "// Fast exp approximation (validated 6x speedup for sigmoid)\n"
            "template<class D, class V>\n"
            "V FastExp(D d, V x) {\n"
            "    x = Clamp(x, Set(d, -88.f), Set(d, 88.f));\n"
            "    auto scaled = Mul(x, Set(d, 1.442695041f));  // log2(e)\n"
            "    auto rounded = Round(scaled);\n"
            "    auto frac = Sub(scaled, rounded);\n"
            "    \n"
            "    // Polynomial for 2^frac (Horner's method)\n"
            "    auto poly = MulAdd(Set(d,0.0096f), frac, Set(d,0.0555f));\n"
            "    poly = MulAdd(poly, frac, Set(d, 0.2402f));\n"
            "    poly = MulAdd(poly, frac, Set(d, 0.6931f));\n"
            "    poly = MulAdd(poly, frac, Set(d, 1.0f));\n"
            "    \n"
            "    // 2^int via bit manipulation\n"
            "    auto pow2_int = BitCast(d, ShiftLeft<23>(int_part + 127));\n"
            "    return Mul(poly, pow2_int);\n"
            "}";
        insight.potential_speedup = 6.0;  // Validated
        insight.references.push_back("https://www.researchgate.net/publication/272178514_Fast_Exponential_Computation_on_SIMD_Architectures");
        insights.push_back(insight);
    }

    // Rule 24: Multi-row Parallelism for GEMV (VALIDATED: 1.57x speedup)
    // GEMV is critical for LLM inference. Processing multiple rows simultaneously
    // amortizes the cost of loading the input vector x.
    // Validated on: GEMV [768x768] with 4-row parallel
    bool is_gemv_like = arithmetic_intensity > 0.4 && arithmetic_intensity < 0.6;
    bool moderate_perf = gflops > 5 && gflops < hw_.theoretical_peak_sp_gflops * 0.5;
    if (is_gemv_like && moderate_perf && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Use Multi-row Parallelism for GEMV";
        insight.description =
            "Arithmetic intensity (~0.5) suggests GEMV or similar matrix-vector operation. "
            "Processing 4 output rows simultaneously amortizes input vector loads and "
            "improves memory bandwidth utilization. VALIDATED: 1.57x speedup.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", GFLOPS=" + std::to_string(gflops) +
            ", Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "GEMV optimization for LLM inference:\n"
            "1. **4-row parallel**: Process 4 output elements per loop iteration\n"
            "   - Loads input vector x once, uses for 4 weight rows\n"
            "   - 4 independent accumulators hide FMA latency\n"
            "2. **Register allocation**: 4 sum vectors + 1 x vector = 5 registers\n"
            "3. **Memory pattern**: Weight rows are contiguous (good for prefetch)\n"
            "4. **VALIDATED**: 1.57x speedup on 768x768 GEMV\n\n"
            "Why this works:\n"
            "- GEMV is memory-bound (AI ≈ 0.5)\n"
            "- Loading x is expensive; reuse across multiple rows\n"
            "- 4 accumulators = 4 independent FMA streams";
        insight.code_example =
            "// 4-row parallel GEMV (validated 1.57x speedup)\n"
            "for (size_t m = 0; m + 4 <= M; m += 4) {\n"
            "    Vec sum0 = Zero(), sum1 = Zero();\n"
            "    Vec sum2 = Zero(), sum3 = Zero();\n"
            "    \n"
            "    for (size_t k = 0; k + N <= K; k += N) {\n"
            "        Vec vx = Load(x + k);  // Load x once\n"
            "        sum0 = MulAdd(Load(W + m*K + k), vx, sum0);\n"
            "        sum1 = MulAdd(Load(W + (m+1)*K + k), vx, sum1);\n"
            "        sum2 = MulAdd(Load(W + (m+2)*K + k), vx, sum2);\n"
            "        sum3 = MulAdd(Load(W + (m+3)*K + k), vx, sum3);\n"
            "    }\n"
            "    y[m+0] = ReduceSum(sum0);\n"
            "    y[m+1] = ReduceSum(sum1);\n"
            "    y[m+2] = ReduceSum(sum2);\n"
            "    y[m+3] = ReduceSum(sum3);\n"
            "}";
        insight.potential_speedup = 1.6;  // Validated
        insights.push_back(insight);
    }

    // Rule 25: Register Pressure Warning for Unrolled Transcendentals
    // Unrolling complex operations (like polynomial approximations) can cause
    // register spilling and performance regression. VALIDATED: Sigmoid 4x unroll = 13% slower
    if (likely_transcendental && !is_scalar_baseline && efficiency < 0.5) {
        OptimizationInsight insight;
        insight.category = InsightCategory::REGISTER_PRESSURE;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Warning: Aggressive Unrolling May Hurt Transcendentals";
        insight.description =
            "Transcendental function implementations use many registers for polynomial "
            "coefficients and intermediate values. Aggressive loop unrolling (4x+) can "
            "cause register spilling and DECREASE performance. VALIDATED: 4x unrolled "
            "sigmoid was 13% SLOWER than non-unrolled.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            " (transcendental-like), Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "Unrolling guidelines for transcendentals:\n"
            "1. **DON'T unroll by default**: Polynomial approximations already use\n"
            "   many registers (coefficients, intermediates, masks)\n"
            "2. **Register count**: AVX2 has 16 YMM registers\n"
            "   - FastExp uses ~8 registers per element\n"
            "   - 4x unroll = 32 registers needed = guaranteed spilling\n"
            "3. **ALWAYS BENCHMARK**: Unrolling can help OR hurt\n"
            "4. **Alternative**: Increase SIMD width instead of unrolling\n\n"
            "When unrolling DOES help:\n"
            "- Simple operations (add, max, FMA)\n"
            "- Low register pressure per element\n"
            "- Memory-bound operations (hide latency)";
        insight.potential_speedup = 1.0;  // Warning, not optimization
        insight.references.push_back("https://www.capsl.udel.edu/conferences/open64/2008/Papers/104.pdf");
        insights.push_back(insight);
    }

    // Rule 26: Sequential Dependency Detection (VALIDATED in Iteration 3)
    // Some operations have inherent sequential dependencies that limit SIMD benefit:
    // - Prefix sum/scan: each element depends on all previous elements
    // - Running statistics: mean/variance accumulate sequentially
    // - Recursive filters: IIR, EMA with feedback
    // VALIDATED: Prefix sum SIMD was 45% SLOWER due to parallel scan overhead
    bool very_low_ai = arithmetic_intensity < 0.2;  // Very simple operation
    bool low_efficiency = efficiency < 0.3;
    if (very_low_ai && low_efficiency && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Check for Sequential Dependencies";
        insight.description =
            "Very low arithmetic intensity (" + std::to_string(arithmetic_intensity) +
            ") with low efficiency may indicate sequential dependencies that limit SIMD benefit. "
            "Operations like prefix sum, running statistics, or recursive filters have inherent "
            "dependencies that prevent straightforward SIMD parallelization.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", Efficiency=" + std::to_string(efficiency * 100) + "%";
        insight.recommendation =
            "Sequential dependency patterns:\n"
            "1. **Prefix sum/scan**: y[i] = y[i-1] + x[i]\n"
            "   - SIMD parallel scan exists but has overhead\n"
            "   - Only beneficial for very large arrays (>1M elements)\n"
            "2. **Running statistics**: mean, variance with accumulator\n"
            "   - Use Welford's algorithm with multiple accumulators\n"
            "3. **Recursive filters**: IIR, EMA with feedback\n"
            "   - May need algorithmic reformulation\n\n"
            "VALIDATED: Prefix sum with SIMD was 45% SLOWER than scalar.\n"
            "Consider keeping scalar for inherently sequential operations.";
        insight.potential_speedup = 1.0;  // Warning only
        insights.push_back(insight);
    }

    // Rule 27: Small Problem Size Warning (VALIDATED in Iteration 4)
    // For very small arrays (N < 4 * vector_lanes), SIMD reduction overhead
    // can dominate and make scalar faster.
    // VALIDATED: Max reduction on N=16 with 8 lanes = 0.17x (83% SLOWER)
    bool very_small_problem = problem_size < 64;  // Typical: 4 * 16 lanes
    if (very_small_problem && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Warning: SIMD Overhead Dominates for Small Problems";
        insight.description =
            "Problem size (" + std::to_string(problem_size) + ") is very small. "
            "SIMD setup/reduction overhead can exceed the compute savings. "
            "VALIDATED: Max on 16 elements was 83% SLOWER with SIMD than scalar.";
        insight.evidence = "Problem size=" + std::to_string(problem_size) +
            " (< 64 elements), Vector lanes likely 8-16";
        insight.recommendation =
            "Small problem handling:\n"
            "1. **Use scalar fallback**: if (n < SIMD_THRESHOLD) use_scalar();\n"
            "   - Threshold ~ 4 * vector_lanes (32-64 elements)\n"
            "2. **Reduction cost**: ReduceSum/Max adds 10-20 cycles overhead\n"
            "   - For N=16, this is 1-2 cycles per element = huge\n"
            "3. **Batch multiple small operations**: Process many small vectors\n"
            "   together to amortize SIMD overhead\n"
            "4. **VALIDATED**: For N < 4*lanes, scalar is often faster\n\n"
            "When SIMD still helps for small N:\n"
            "- Expensive operations (transcendentals)\n"
            "- Batched processing (many small vectors)";
        insight.code_example =
            "// Scalar fallback for small problems\n"
            "float max_value(const float* data, size_t n) {\n"
            "    const size_t N = Lanes(d);\n"
            "    if (n < 4 * N) {\n"
            "        // Scalar for small arrays\n"
            "        float m = data[0];\n"
            "        for (size_t i = 1; i < n; ++i)\n"
            "            if (data[i] > m) m = data[i];\n"
            "        return m;\n"
            "    }\n"
            "    // SIMD for larger arrays\n"
            "    ...\n"
            "}";
        insight.potential_speedup = 1.0;  // Warning only
        insights.push_back(insight);
    }

    // Rule 28: Inter-Operation Batching (VALIDATED in Iteration 4)
    // When processing many small independent operations (e.g., softmax per row),
    // batch them together to better utilize SIMD.
    // VALIDATED: BatchedSoftmax with 4-batch parallel = 7.37x speedup
    bool medium_problem = problem_size >= 64 && problem_size < 4096;
    bool is_batch_candidate = arithmetic_intensity > 0.3 && arithmetic_intensity < 2.0;
    if (medium_problem && is_batch_candidate && !is_scalar_baseline && efficiency < 0.5) {
        OptimizationInsight insight;
        insight.category = InsightCategory::PARALLELISM;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Use Inter-Operation Batching for Small Sequences";
        insight.description =
            "Medium-sized operations may benefit from inter-operation batching. "
            "Instead of processing one row at a time, process 4+ rows together to "
            "improve register utilization and amortize reduction overhead. "
            "VALIDATED: BatchedSoftmax (1024x64) with 4-batch = 7.37x speedup.";
        insight.evidence = "Problem size=" + std::to_string(problem_size) +
            ", AI=" + std::to_string(arithmetic_intensity);
        insight.recommendation =
            "Inter-operation batching strategies:\n"
            "1. **Process N rows simultaneously**: 4 rows is often optimal\n"
            "   - Keeps N accumulators active (hides FMA latency)\n"
            "   - Better register utilization\n"
            "2. **Share intermediate results**: For softmax, find max of 4 rows\n"
            "   together if possible\n"
            "3. **Reduce branching overhead**: Loop over batch dimension outermost\n"
            "4. **VALIDATED**: 7.37x speedup on 1024 rows x 64 columns\n\n"
            "Batch size selection:\n"
            "- 4 batches: Good balance of parallelism and register usage\n"
            "- 8 batches: May cause register spilling\n"
            "- 2 batches: Suboptimal ILP";
        insight.code_example =
            "// Inter-batch parallel softmax (validated 7.37x)\n"
            "void batched_softmax(float* data, size_t B, size_t N) {\n"
            "    for (size_t b = 0; b + 4 <= B; b += 4) {\n"
            "        float* rows[4] = {data+b*N, data+(b+1)*N,\n"
            "                          data+(b+2)*N, data+(b+3)*N};\n"
            "        // Find max for all 4 rows\n"
            "        // Apply exp and sum for all 4 rows\n"
            "        // Normalize all 4 rows\n"
            "    }\n"
            "}";
        insight.potential_speedup = 4.0;  // Validated 7.37x
        insights.push_back(insight);
    }

    // ========================================================================
    // Rules 29-34: Validated from Research Report Heuristics
    // Source: "SIMD kernel optimization: quantitative heuristics"
    // ========================================================================

    // Rule 29: Trip Count Threshold (VALIDATED: clear threshold at 4×lanes)
    // Below this threshold, SIMD overhead dominates; above, clear speedup
    size_t vector_lanes = hw_.max_vector_bits / 32;  // Approximate for float
    size_t trip_count_threshold = 4 * vector_lanes;
    if (problem_size < trip_count_threshold && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Trip Count Below Vectorization Threshold";
        insight.description =
            "Problem size (" + std::to_string(problem_size) + ") is below the "
            "vectorization threshold of " + std::to_string(trip_count_threshold) +
            " (4 × " + std::to_string(vector_lanes) + " vector lanes). "
            "SIMD overhead may exceed compute savings. VALIDATED: Clear speedup "
            "only above 4×vector_lanes threshold.";
        insight.evidence = "Problem size=" + std::to_string(problem_size) +
            ", Threshold=" + std::to_string(trip_count_threshold);
        insight.recommendation =
            "Trip count threshold guidelines:\n"
            "1. **Minimum trip count**: 4 × vector_lanes for profitable SIMD\n"
            "   - SSE (128-bit): ≥16 for float, ≥8 for double\n"
            "   - AVX2 (256-bit): ≥32 for float, ≥16 for double\n"
            "   - AVX-512 (512-bit): ≥64 for float, ≥32 for double\n"
            "2. **Remainder handling**: If remainder > 50% of main loop, avoid SIMD\n"
            "3. **Use scalar fallback**: if (n < threshold) use_scalar();\n"
            "4. **Batch small operations**: Process multiple small arrays together";
        insight.potential_speedup = 1.0;  // Warning only
        insights.push_back(insight);
    }

    // Rule 30: Gather/Scatter Penalty (VALIDATED: 13.57x slower)
    // Report claims 5-20x; we validated 13.57x (later 14.94x)
    //
    // IMPROVED DETECTION: Gather/scatter has specific characteristics:
    // 1. Very low BW efficiency (< 10%) - random access wastes cache lines
    // 2. Low IPC - gather has high latency (multiple cache accesses per instruction)
    // 3. High cache miss rate - random access defeats prefetching
    // 4. Low arithmetic intensity - just loading data, not much compute
    //
    // Previous trigger was too generic (any slow SIMD code would match).
    // Now require multiple indicators specific to gather/scatter pattern.
    bool very_low_bw_efficiency = bw_utilization < 0.1;  // < 10% (was 20%)
    bool low_ipc_pattern = ipc < 0.5;  // Gather has high latency -> low IPC
    bool high_cache_miss = cache_miss_rate > 0.1;  // Random access -> cache misses
    bool low_ai_gather = arithmetic_intensity < 0.5;  // Just loading, not computing
    bool moderate_problem = problem_size > 1000 && problem_size < 10000000;

    // Require at least 3 of 4 indicators for higher confidence
    int gather_indicators = (very_low_bw_efficiency ? 1 : 0) +
                           (low_ipc_pattern ? 1 : 0) +
                           (high_cache_miss ? 1 : 0) +
                           (low_ai_gather ? 1 : 0);

    if (gather_indicators >= 3 && moderate_problem && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
        insight.severity = InsightSeverity::CRITICAL;
        insight.confidence = (gather_indicators == 4) ? InsightConfidence::HIGH : InsightConfidence::MEDIUM;
        insight.title = "Likely Gather/Scatter or Random Access Pattern";
        insight.description =
            "Multiple indicators suggest gather/scatter or random memory access:\n"
            "- BW efficiency: " + std::to_string(bw_utilization * 100) + "% (< 10% typical for gather)\n"
            "- IPC: " + std::to_string(ipc) + " (low due to memory latency)\n"
            "- Cache miss rate: " + std::to_string(cache_miss_rate * 100) + "%\n\n"
            "VALIDATED: Gather operations are 10-15x slower than contiguous loads. "
            "If using indirect indexing (arr[indices[i]]), this is the cause.";
        insight.evidence = "Indicators matched: " + std::to_string(gather_indicators) + "/4 "
            "(BW<10%: " + std::string(very_low_bw_efficiency ? "Y" : "N") +
            ", IPC<0.5: " + std::string(low_ipc_pattern ? "Y" : "N") +
            ", CacheMiss>10%: " + std::string(high_cache_miss ? "Y" : "N") +
            ", AI<0.5: " + std::string(low_ai_gather ? "Y" : "N") + ")";
        insight.recommendation =
            "**Gather/scatter is inherently slow - restructure for contiguous access:**\n\n"
            "1. **Restructure data layout**: Convert AoS to SoA\n"
            "   ```cpp\n"
            "   // Bad: struct Particle { float x,y,z; }; particles[indices[i]].x\n"
            "   // Good: float* x; float* y; float* z; // Access x[i] contiguously\n"
            "   ```\n\n"
            "2. **Pre-sort or bin data**: If indices are needed, sort by access pattern\n\n"
            "3. **Use scalar fallback**: For small arrays with random access,\n"
            "   scalar may be faster than SIMD gather\n\n"
            "4. **VALIDATED speedup**: 10-15x by switching to contiguous access";
        insight.potential_speedup = 10.0;  // Validated 13.57x - 14.94x
        insight.references.push_back("https://www.intel.com/content/www/us/en/docs/intrinsics-guide/");
        insights.push_back(insight);
    }

    // Rule 31: AoS to SoA Transformation (VALIDATED: 4.46x speedup)
    // Report claims 3.7x; we validated 4.46x
    // Detect by checking for memory-bound with low GFLOPS despite high problem size
    bool memory_bound_pattern = arithmetic_intensity < ridge_point && arithmetic_intensity > 0.1;
    bool unexpectedly_low_perf = gflops < hw_.theoretical_peak_sp_gflops * 0.1;
    if (memory_bound_pattern && unexpectedly_low_perf && problem_size > 10000 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Consider Array-of-Structures to Structure-of-Arrays Transformation";
        insight.description =
            "Memory-bound workload with low performance may indicate AoS (Array-of-"
            "Structures) memory layout. Converting to SoA (Structure-of-Arrays) "
            "enables contiguous SIMD loads. VALIDATED: 4.46x speedup for particle "
            "simulation with SoA vs AoS (research claimed 3.7x).";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            ", GFLOPS=" + std::to_string(gflops) +
            " (expected higher for size " + std::to_string(problem_size) + ")";
        insight.recommendation =
            "AoS to SoA transformation:\n"
            "1. **Convert interleaved to planar layout**:\n"
            "   Before (AoS): struct { float x,y,z; } particles[N];\n"
            "   After (SoA):  float x[N], y[N], z[N];\n"
            "2. **Benefits**:\n"
            "   - Single vector load per component vs gather for AoS\n"
            "   - Better cache line utilization\n"
            "   - VALIDATED: 4.46x speedup\n"
            "3. **When to use AoS**: Irregular/random access patterns (database)\n"
            "4. **When to use SoA**: Regular/streaming patterns (simulations)";
        insight.code_example =
            "// AoS (slow - requires gather)\n"
            "struct Particle { float x, y, z; };\n"
            "Particle particles[N];\n"
            "// Access: particles[i].x (stride = sizeof(Particle))\n\n"
            "// SoA (fast - contiguous SIMD load)\n"
            "float* x = aligned_alloc(N);\n"
            "float* y = aligned_alloc(N);\n"
            "float* z = aligned_alloc(N);\n"
            "// Access: x[i] (stride = sizeof(float))";
        insight.potential_speedup = 4.0;  // Validated 4.46x
        insights.push_back(insight);
    }

    // Rule 32: Optimal Accumulator Count for L1-Resident Data (VALIDATED: 2-5x)
    // Formula: Accumulators = Latency / Throughput
    // For Skylake+ FMA: 4 cycles / 0.5 = 8 accumulators optimal
    // Validated: 2 acc = 2.02x, 4 acc = 3.68x, 8 acc = 4.91x
    bool compute_bound = arithmetic_intensity > ridge_point;
    bool l1_resident = problem_size * sizeof(float) < 32 * 1024;  // < 32KB
    if (compute_bound && l1_resident && efficiency < 0.5 && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Use Multiple Accumulators to Hide FMA Latency";
        insight.description =
            "L1-resident compute-bound workload with low efficiency suggests "
            "insufficient instruction-level parallelism. Adding multiple independent "
            "accumulators breaks loop-carried dependencies. VALIDATED: 2 accumulators "
            "give 2.02x speedup, 4 give 3.68x, 8 give 4.91x for L1-resident data.";
        insight.evidence = "Problem fits in L1 (" + std::to_string(problem_size * sizeof(float) / 1024) +
            " KB), Compute-bound (AI=" + std::to_string(arithmetic_intensity) + ")";
        insight.recommendation =
            "Accumulator count formula: Latency ÷ Reciprocal_Throughput\n"
            "1. **Skylake+ FMA**: 4 cycles / 0.5 throughput = 8 accumulators\n"
            "2. **Haswell FMA**: 5 cycles / 0.5 throughput = 10 accumulators\n"
            "3. **Integer ADD**: 1 cycle latency, 3-4 ports = 6-8 accumulators\n\n"
            "VALIDATED speedups (L1-resident FMA reduction):\n"
            "- 1 accumulator: baseline\n"
            "- 2 accumulators: 2.02x (claimed 2x - exact match!)\n"
            "- 4 accumulators: 3.68x\n"
            "- 8 accumulators: 4.91x\n\n"
            "Note: For memory-bound workloads, more accumulators won't help.";
        insight.code_example =
            "// Single accumulator (slow - dependency chain)\n"
            "Vec sum = Zero();\n"
            "for (i = 0; i < n; i += N)\n"
            "    sum = MulAdd(Load(a+i), Load(a+i), sum);\n\n"
            "// 4 accumulators (validated 3.68x faster)\n"
            "Vec s0=Zero(), s1=Zero(), s2=Zero(), s3=Zero();\n"
            "for (i = 0; i + 4*N <= n; i += 4*N) {\n"
            "    s0 = MulAdd(Load(a+i), Load(a+i), s0);\n"
            "    s1 = MulAdd(Load(a+i+N), Load(a+i+N), s1);\n"
            "    s2 = MulAdd(Load(a+i+2*N), Load(a+i+2*N), s2);\n"
            "    s3 = MulAdd(Load(a+i+3*N), Load(a+i+3*N), s3);\n"
            "}\n"
            "Vec total = Add(Add(s0,s1), Add(s2,s3));";
        insight.potential_speedup = 4.0;  // Validated up to 4.91x
        insight.references.push_back("https://en.algorithmica.org/hpc/simd/reduction/");
        insights.push_back(insight);
    }

    // Rule 33: Stride Pattern Warning (VALIDATED: Stride-1 SIMD = 4.65x)
    // Non-unit stride kills SIMD benefit for large arrays
    // Report: "stride-2 gains zero benefit for arrays >2M elements"
    if (bw_utilization < 0.3 && problem_size > 100000 && !is_scalar_baseline &&
        arithmetic_intensity < 0.3) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::LOW;
        insight.title = "Check for Non-Unit Stride Memory Access";
        insight.description =
            "Low bandwidth utilization with simple arithmetic intensity may indicate "
            "non-unit stride access pattern. VALIDATED: Stride-1 SIMD gives 4.65x "
            "speedup over scalar. Non-unit stride wastes cache line bandwidth.";
        insight.evidence = "BW utilization=" + std::to_string(bw_utilization * 100) +
            "%, AI=" + std::to_string(arithmetic_intensity);
        insight.recommendation =
            "Stride pattern impact on cache line utilization:\n"
            "- Stride-1: 100% utilization (all bytes used)\n"
            "- Stride-2: ~50% utilization (half bytes wasted)\n"
            "- Stride-4: ~25% utilization (3/4 bytes wasted)\n"
            "- Stride-16+: ~6% utilization (nearly all wasted)\n\n"
            "Fixes:\n"
            "1. **Restructure loops**: Access data sequentially\n"
            "2. **Loop interchange**: Make stride-1 dimension innermost\n"
            "3. **Data transposition**: If access pattern is fixed\n"
            "4. **VALIDATED**: Stride-1 SIMD = 4.65x speedup";
        insight.potential_speedup = 4.0;  // Validated 4.65x
        insights.push_back(insight);
    }

    // ========================================================================
    // Rules 34-36: Validated from Cache & Memory Heuristics Report
    // ========================================================================

    // Rule 34: Loop Tiling/Blocking for Cache Locality (VALIDATED: 24.72x)
    // For matrix operations that exceed cache, tiling dramatically improves performance
    // by ensuring data reuse within cache before eviction.
    // VALIDATED: MatMul 512x512 with 64x64 tiling + SIMD = 24.72x speedup
    bool large_working_set = problem_size * sizeof(float) > hw_.cache.l2_size_kb * 1024;
    bool matrix_like_ai = arithmetic_intensity > 1.0 && arithmetic_intensity < 100.0;
    if (large_working_set && matrix_like_ai && !is_scalar_baseline && efficiency < 0.3) {
        OptimizationInsight insight;
        insight.category = InsightCategory::CACHE_EFFICIENCY;
        insight.severity = InsightSeverity::CRITICAL;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Apply Loop Tiling/Blocking for Cache Locality";
        insight.description =
            "Working set (" + std::to_string(problem_size * sizeof(float) / 1024) +
            " KB) exceeds L2 cache (" + std::to_string(hw_.cache.l2_size_kb) +
            " KB). For matrix-like operations, loop tiling can provide order-of-magnitude "
            "speedups by ensuring data stays in cache for reuse. "
            "VALIDATED: Matrix multiply with 64x64 tiling + SIMD = 24.72x speedup.";
        insight.evidence = "Working set=" + std::to_string(problem_size * sizeof(float) / 1024) +
            " KB, L2=" + std::to_string(hw_.cache.l2_size_kb) +
            " KB, AI=" + std::to_string(arithmetic_intensity);
        insight.recommendation =
            "Loop tiling optimization:\n"
            "1. **Choose tile size to fit in L1/L2 cache**:\n"
            "   - L1: ~32KB → tiles of ~64x64 floats for 3 matrices\n"
            "   - L2: ~256KB → larger tiles possible\n"
            "2. **Tile all dimensions that cause cache thrashing**:\n"
            "   - MatMul: tile i, j, and k dimensions\n"
            "   - Transpose: tile both row and column\n"
            "3. **Combine with SIMD for maximum benefit**:\n"
            "   - VALIDATED: Tiling alone = 2.32x\n"
            "   - VALIDATED: Tiling + SIMD = 24.72x\n"
            "4. **Consider cache line size (64 bytes)**:\n"
            "   - Tile dimensions should be multiples of 16 floats";
        insight.code_example =
            "// Blocked matrix multiply (validated 24.72x)\n"
            "for (size_t ii = 0; ii < N; ii += TILE) {\n"
            "  for (size_t jj = 0; jj < N; jj += TILE) {\n"
            "    for (size_t kk = 0; kk < N; kk += TILE) {\n"
            "      // Process TILE x TILE x TILE block\n"
            "      for (i = ii; i < ii+TILE; ++i)\n"
            "        for (k = kk; k < kk+TILE; ++k) {\n"
            "          Vec a_ik = Set(A[i*N + k]);\n"
            "          for (j = jj; j + VL <= jj+TILE; j += VL)\n"
            "            C[i*N+j] = MulAdd(a_ik, Load(B+k*N+j), Load(C+i*N+j));\n"
            "        }\n"
            "    }\n"
            "  }\n"
            "}";
        insight.potential_speedup = 20.0;  // Validated 24.72x
        insight.references.push_back("https://www.intel.com/content/www/us/en/developer/articles/technical/how-to-use-loop-blocking-to-optimize-memory-use-on-32-bit-intel-architecture.html");
        insights.push_back(insight);
    }

    // Rule 35: Branchless SIMD for Unpredictable Branches (VALIDATED: 5x)
    // When branches are data-dependent and unpredictable (~50% taken rate),
    // using SIMD blend/select operations eliminates misprediction penalty.
    // VALIDATED: 5.08x speedup for branchless scalar, 4.96x for SIMD blend
    // (Branch misprediction costs 15-30 cycles per mispredict)
    bool low_ipc_indicator = ipc > 0 && ipc < 1.0;  // Very low IPC suggests stalls
    bool simple_compute = arithmetic_intensity < 1.0;
    if (low_ipc_indicator && simple_compute && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::BRANCH_PREDICTION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Use Branchless SIMD (Blend/Select) for Unpredictable Conditions";
        insight.description =
            "Very low IPC (" + std::to_string(ipc) + ") with simple arithmetic may indicate "
            "branch misprediction overhead. For data-dependent conditions (like filtering, "
            "thresholding, conditional operations), SIMD blend/select operations eliminate "
            "branches entirely. VALIDATED: 5.08x speedup for unpredictable 50/50 branch.";
        insight.evidence = "IPC=" + std::to_string(ipc) +
            ", AI=" + std::to_string(arithmetic_intensity);
        insight.recommendation =
            "Branchless SIMD patterns:\n"
            "1. **Replace if/else with SIMD blend**:\n"
            "   - Compute BOTH branches for all elements\n"
            "   - Use mask from comparison to select results\n"
            "2. **When to use branchless**:\n"
            "   - Unpredictable conditions (~50% true/false)\n"
            "   - Data-dependent branches in tight loops\n"
            "   - NOT for highly predictable branches (>95% one direction)\n"
            "3. **VALIDATED speedups**:\n"
            "   - Branchless scalar: 5.08x\n"
            "   - Branchless SIMD blend: 4.96x\n"
            "4. **Branch misprediction cost**: 15-30 cycles per mispredict";
        insight.code_example =
            "// Branchy version (slow for unpredictable data)\n"
            "if (input[i] > thresh) output[i] = input[i] * 2;\n"
            "else output[i] = input[i] * 0.5;\n\n"
            "// Branchless SIMD (validated 5x faster)\n"
            "auto v = Load(d, input + i);\n"
            "auto mask = Gt(v, thresh);  // comparison mask\n"
            "auto result_true = Mul(v, two);\n"
            "auto result_false = Mul(v, half);\n"
            "auto result = IfThenElse(mask, result_true, result_false);\n"
            "Store(result, d, output + i);";
        insight.potential_speedup = 5.0;  // Validated 5.08x
        insight.references.push_back("https://en.algorithmica.org/hpc/pipelining/branchless/");
        insights.push_back(insight);
    }

    // Rule 36: Kernel Fusion to Reduce Memory Traffic (VALIDATED: 2.06x)
    // Multiple passes over data multiply memory traffic. Fusing kernels into
    // a single pass reduces memory bandwidth usage significantly.
    // VALIDATED: Normalize+ReLU fusion = 2.06x speedup
    bool memory_bandwidth_limited = bw_utilization > 0.5;
    bool low_ai_streaming = arithmetic_intensity < 0.5;
    if (memory_bandwidth_limited && low_ai_streaming && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_BOUND;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Apply Kernel Fusion to Reduce Memory Traffic";
        insight.description =
            "High bandwidth utilization (" + std::to_string(bw_utilization * 100) +
            "%) with low arithmetic intensity suggests multiple passes over data. "
            "Fusing multiple operations into a single pass reduces memory traffic by "
            "eliminating intermediate storage. VALIDATED: Normalize+ReLU fusion = 2.06x speedup.";
        insight.evidence = "BW utilization=" + std::to_string(bw_utilization * 100) +
            "%, AI=" + std::to_string(arithmetic_intensity);
        insight.recommendation =
            "Kernel fusion strategies:\n"
            "1. **Identify multi-pass patterns**:\n"
            "   - Separate normalize then activate loops\n"
            "   - Scale then bias then clamp in sequence\n"
            "   - Multiple elementwise operations on same data\n"
            "2. **Fuse into single loop**:\n"
            "   - Eliminates intermediate arrays (saves memory)\n"
            "   - Halves or more memory traffic\n"
            "   - Data stays in registers between operations\n"
            "3. **VALIDATED: 2.06x speedup** for Normalize+ReLU fusion\n"
            "4. **Memory traffic reduction**:\n"
            "   - Unfused: Read + Write + Read + Write (4N)\n"
            "   - Fused: Read + Write (2N) = 50% reduction";
        insight.code_example =
            "// Unfused (slow - 2 passes over data)\n"
            "for (i) temp[i] = (input[i] - mean) / std;  // normalize\n"
            "for (i) output[i] = max(0, temp[i]);        // ReLU\n\n"
            "// Fused (validated 2.06x faster)\n"
            "for (i) {\n"
            "    auto v = Load(input + i);\n"
            "    v = Mul(Sub(v, mean), inv_std);  // normalize\n"
            "    v = Max(v, zero);                 // ReLU\n"
            "    Store(v, output + i);\n"
            "}";
        insight.potential_speedup = 2.0;  // Validated 2.06x
        insights.push_back(insight);
    }

    // Rule 37: Auto-Vectorization Parity Check
    // When manual SIMD provides minimal speedup over scalar, the compiler may
    // already be auto-vectorizing. Simple element-wise operations with no
    // dependencies are prime candidates for auto-vectorization.
    // VALIDATED: Threshold ops showed 0.9x (scalar faster) due to auto-vectorization
    bool simple_operation = arithmetic_intensity < 1.0;  // Simple ops have low AI
    bool minimal_speedup = efficiency > 0.8;  // Already close to peak
    // Trigger for scalar baselines that are already fast - indicates auto-vec

    if (simple_operation && minimal_speedup && is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::VECTORIZATION;
        insight.severity = InsightSeverity::INFO;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Compiler Auto-Vectorization May Be Active";
        insight.description =
            "This simple operation may already be auto-vectorized by the compiler "
            "with -O3 -march=native. Manual SIMD overhead (function calls, masking) "
            "could actually slow down simple operations that compilers vectorize well.";
        insight.recommendation =
            "**Auto-Vectorization Check:**\n"
            "1. Check compiler vectorization reports:\n"
            "   - GCC: `-fopt-info-vec-optimized -fopt-info-vec-missed`\n"
            "   - Clang: `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize`\n"
            "2. Simple element-wise ops (max, min, clamp, threshold) auto-vectorize well\n"
            "3. Manual SIMD benefits most when:\n"
            "   - Complex patterns (gather/scatter, horizontal reductions)\n"
            "   - The compiler misses vectorization opportunities\n"
            "   - You need specific instruction sequences (FMA, fused ops)\n"
            "4. If scalar is faster, trust the compiler for simple ops";
        insight.code_example =
            "// Simple ops - compiler auto-vectorizes well\n"
            "// Manual SIMD may add overhead with no benefit\n"
            "for (i) output[i] = (input[i] > thresh) ? 255 : 0;  // Auto-vec!\n"
            "for (i) output[i] = std::max(input[i], 0.0f);       // Auto-vec!\n\n"
            "// Verify with: gcc -O3 -fopt-info-vec-optimized mycode.cc\n"
            "// Example output: 'mycode.cc:12: optimized: loop vectorized'";
        insight.potential_speedup = 1.0;  // No speedup expected
        insight.references.push_back("https://llvm.org/docs/Vectorizers.html");
        insights.push_back(insight);
    }

    // Rule 38: Loop-Carried Dependency Detection
    // Some algorithms have inherent serial dependencies that prevent effective
    // SIMD parallelization. Prefix sum is the classic example.
    // VALIDATED: Prefix sum showed 1.0x speedup (no benefit) due to serial dependency
    bool very_low_speedup = efficiency < 0.2;  // Much lower than expected
    bool could_be_vectorized = problem_size > 1000;  // Large enough for SIMD
    bool streaming_access = arithmetic_intensity < 0.5;  // Simple memory pattern
    // Trigger for SIMD code (not scalar baseline) that still has low efficiency

    if (very_low_speedup && could_be_vectorized && streaming_access && !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Possible Loop-Carried Dependency";
        insight.description =
            "Despite vectorization, performance is far below expected. This may indicate "
            "a loop-carried dependency where each iteration depends on the previous result. "
            "Classic examples: prefix sum, running average, recursive filters.";
        insight.recommendation =
            "**Dependency Analysis:**\n"
            "1. Check for loop-carried dependencies:\n"
            "   - `y[i] = y[i-1] + x[i]` (prefix sum - serial dependency)\n"
            "   - `y[i] = alpha * y[i-1] + x[i]` (IIR filter - serial)\n"
            "2. If dependency exists, consider:\n"
            "   - **Block-based parallel prefix sum**: O(log N) depth\n"
            "   - **Segmented operations**: Parallelize within independent segments\n"
            "   - **Algorithmic transformation**: Some serial algorithms have parallel variants\n"
            "3. Accept limited speedup for inherently serial algorithms\n"
            "4. Consider hybrid: SIMD within blocks, serial between blocks";
        insight.code_example =
            "// Serial prefix sum (can't parallelize naively)\n"
            "for (i = 1; i < N; i++)\n"
            "    prefix[i] = prefix[i-1] + data[i];  // Depends on i-1!\n\n"
            "// Parallel prefix sum (block-based)\n"
            "// Step 1: Parallel sum within blocks\n"
            "for (block in blocks) sum_within_block_SIMD(block);\n"
            "// Step 2: Serial scan of block sums\n"
            "for (i = 1; i < num_blocks; i++)\n"
            "    block_prefix[i] = block_prefix[i-1] + block_sums[i-1];\n"
            "// Step 3: Parallel add block offsets\n"
            "for (block in blocks) add_offset_SIMD(block, block_prefix[block_id]);";
        insight.potential_speedup = 2.0;  // Block-based can give ~2x
        insight.references.push_back("https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda");
        insights.push_back(insight);
    }

    // Rule 39: Horizontal Reduction Inside Loop Anti-Pattern (VALIDATED: 5.40x speedup)
    // Research document: "Avoid horizontal operations inside loops. Use vertical
    // additions to maintain several partial sum vectors and only perform the
    // horizontal reduction once after the loop concludes."
    //
    // Detection: Reduction-like kernel (low AI, accumulation pattern) with
    // unexpectedly low performance despite being memory-bound.
    bool is_reduction_kernel = arithmetic_intensity >= 0.2 && arithmetic_intensity < 0.5;
    bool low_reduction_efficiency = efficiency < 0.4 && gflops > 0;
    bool not_memory_saturated = bw_utilization < 0.7;  // If BW is saturated, that's the limit

    if (is_reduction_kernel && low_reduction_efficiency && not_memory_saturated &&
        !is_scalar_baseline) {
        OptimizationInsight insight;
        insight.category = InsightCategory::LOOP_OPTIMIZATION;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "Avoid Horizontal Reduction Inside Loop";
        insight.description =
            "Reduction-like kernel with low efficiency (" +
            std::to_string(static_cast<int>(efficiency * 100)) + "%) may be performing "
            "horizontal reductions (ReduceSum/ReduceMax) inside the hot loop. "
            "VALIDATED: Deferring horizontal reduction to loop end gives 5.40x speedup.";
        insight.evidence = "AI=" + std::to_string(arithmetic_intensity) +
            " (reduction), Efficiency=" + std::to_string(efficiency * 100) +
            "%, BW Util=" + std::to_string(bw_utilization * 100) + "%";
        insight.recommendation =
            "**Horizontal Reduction Anti-Pattern (VALIDATED: 5.40x speedup)**\n\n"
            "BAD PATTERN - Horizontal sum inside loop:\n"
            "```\n"
            "float result = 0;\n"
            "for (i = 0; i < N; i += Lanes) {\n"
            "    result += ReduceSum(Load(a+i) * Load(b+i));  // SLOW!\n"
            "}\n"
            "```\n\n"
            "GOOD PATTERN - Vertical accumulation, horizontal at end:\n"
            "```\n"
            "auto sum = Zero();\n"
            "for (i = 0; i < N; i += Lanes) {\n"
            "    sum = MulAdd(Load(a+i), Load(b+i), sum);  // Vertical!\n"
            "}\n"
            "return ReduceSum(sum);  // Horizontal only ONCE\n"
            "```\n\n"
            "BEST - Multi-accumulator + deferred horizontal:\n"
            "```\n"
            "auto s0=Zero(), s1=Zero(), s2=Zero(), s3=Zero();\n"
            "for (i = 0; i < N; i += 4*Lanes) {\n"
            "    s0 = MulAdd(Load(a+i+0*N), Load(b+i+0*N), s0);\n"
            "    s1 = MulAdd(Load(a+i+1*N), Load(b+i+1*N), s1);\n"
            "    // ... more accumulators\n"
            "}\n"
            "return ReduceSum(s0 + s1 + s2 + s3);  // Combine then reduce\n"
            "```";
        insight.code_example =
            "// VALIDATED: 5.40x speedup from vertical accumulation\n"
            "auto sum = Zero(d);\n"
            "for (size_t i = 0; i + Lanes(d) <= n; i += Lanes(d)) {\n"
            "    auto va = Load(d, a + i);\n"
            "    auto vb = Load(d, b + i);\n"
            "    sum = MulAdd(va, vb, sum);  // Vertical accumulation\n"
            "}\n"
            "return ReduceSum(d, sum);  // Horizontal only at end";
        insight.potential_speedup = 5.0;  // Validated 5.40x
        insight.references.push_back("Google Highway documentation on reductions");
        insights.push_back(insight);
    }

    return insights;
}

// ============================================================================
// Roofline Analysis
// ============================================================================

std::vector<OptimizationInsight> InsightsEngine::analyze_roofline_position(
    double ai, double achieved_gflops
) const {
    std::vector<OptimizationInsight> insights;

    double memory_ceiling = ai * hw_.measured_memory_bw_gbps;
    double compute_ceiling = hw_.theoretical_peak_sp_gflops;
    double theoretical_max = std::min(memory_ceiling, compute_ceiling);
    double ridge_point = compute_ceiling / hw_.measured_memory_bw_gbps;

    double efficiency = achieved_gflops / theoretical_max;

    // Detailed roofline insight
    OptimizationInsight insight;
    insight.category = InsightCategory::GENERAL;
    insight.severity = InsightSeverity::INFO;
    insight.confidence = InsightConfidence::HIGH;
    insight.title = "Roofline Analysis Summary";

    std::ostringstream desc;
    desc << std::fixed << std::setprecision(2);
    desc << "Arithmetic Intensity: " << ai << " FLOP/byte\n";
    desc << "Ridge Point: " << ridge_point << " FLOP/byte\n";
    desc << "Achieved: " << achieved_gflops << " GFLOPS\n";
    desc << "Theoretical Max: " << theoretical_max << " GFLOPS\n";
    desc << "Efficiency: " << (efficiency * 100) << "%\n";
    desc << "Regime: " << (ai < ridge_point ? "Memory-Bound" : "Compute-Bound");
    insight.description = desc.str();

    insight.evidence = "Memory ceiling=" + std::to_string(memory_ceiling) +
        " GFLOPS, Compute ceiling=" + std::to_string(compute_ceiling) + " GFLOPS";

    if (ai < ridge_point) {
        insight.recommendation =
            "**Memory-Bound Optimization Path:**\n"
            "1. Increase data reuse (blocking/tiling)\n"
            "2. Reduce memory traffic (compression, quantization)\n"
            "3. Improve memory access patterns\n"
            "4. Use cache more effectively\n"
            "5. Consider kernel fusion";
    } else {
        insight.recommendation =
            "**Compute-Bound Optimization Path:**\n"
            "1. Improve vectorization\n"
            "2. Use FMA instructions\n"
            "3. Reduce instruction count\n"
            "4. Increase ILP with unrolling\n"
            "5. Consider AVX-512 if available";
    }

    insight.potential_speedup = 1.0 / efficiency;
    insights.push_back(insight);

    return insights;
}

// ============================================================================
// Next Steps Generation
// ============================================================================

std::vector<std::string> InsightsEngine::generate_next_steps(
    const std::vector<OptimizationInsight>& insights
) const {
    std::vector<std::string> steps;

    // Sort insights by severity and potential speedup
    std::vector<const OptimizationInsight*> sorted;
    for (const auto& i : insights) {
        sorted.push_back(&i);
    }

    std::sort(sorted.begin(), sorted.end(),
        [](const OptimizationInsight* a, const OptimizationInsight* b) {
            if (a->severity != b->severity) {
                return static_cast<int>(a->severity) < static_cast<int>(b->severity);
            }
            return a->potential_speedup > b->potential_speedup;
        });

    int count = 0;
    for (const auto* insight : sorted) {
        if (insight->severity == InsightSeverity::INFO) continue;
        if (count >= 5) break;

        std::ostringstream step;
        step << count + 1 << ". [" << severity_to_string(insight->severity) << "] "
             << insight->title;
        if (insight->potential_speedup > 1.1) {
            step << " (potential " << std::fixed << std::setprecision(1)
                 << insight->potential_speedup << "x speedup)";
        }
        steps.push_back(step.str());
        count++;
    }

    return steps;
}

// ============================================================================
// Cross-Kernel Analysis
// ============================================================================

std::vector<OptimizationInsight> InsightsEngine::analyze_cross_kernel_patterns(
    const std::vector<KernelAnalysis>& analyses
) const {
    std::vector<OptimizationInsight> insights;

    if (analyses.empty()) return insights;

    // Check if all kernels are memory-bound
    int memory_bound_count = 0;
    for (const auto& a : analyses) {
        if (a.primary_bottleneck.find("Memory") != std::string::npos) {
            memory_bound_count++;
        }
    }

    if (memory_bound_count == static_cast<int>(analyses.size())) {
        OptimizationInsight insight;
        insight.category = InsightCategory::MEMORY_BOUND;
        insight.severity = InsightSeverity::HIGH;
        insight.confidence = InsightConfidence::HIGH;
        insight.title = "All Kernels Memory-Bound";
        insight.description =
            "All benchmarked kernels are memory-bound. Consider system-level optimizations.";
        insight.recommendation =
            "System-level recommendations:\n"
            "1. Enable huge pages for large allocations\n"
            "2. Consider NUMA-aware allocation\n"
            "3. Check memory channel utilization\n"
            "4. Consider kernel fusion across operations";
        insights.push_back(insight);
    }

    // Check for consistent low vectorization
    size_t low_vec_count = 0;
    for (const auto& a : analyses) {
        if (a.vectorization_ratio > 0 && a.vectorization_ratio < 0.5) {
            low_vec_count++;
        }
    }

    if (low_vec_count > analyses.size() / 2) {
        OptimizationInsight insight;
        insight.category = InsightCategory::VECTORIZATION;
        insight.severity = InsightSeverity::MEDIUM;
        insight.confidence = InsightConfidence::MEDIUM;
        insight.title = "Systematic Vectorization Issues";
        insight.description =
            "Multiple kernels show poor vectorization. Check for common blockers.";
        insight.recommendation =
            "Review codebase for:\n"
            "1. Pointer aliasing issues (add `restrict`)\n"
            "2. Non-contiguous data structures (AoS vs SoA)\n"
            "3. Global vectorization settings (`-march=native`)";
        insights.push_back(insight);
    }

    return insights;
}

// ============================================================================
// Output Formatting
// ============================================================================

std::string InsightsEngine::format_insights_text(const KernelAnalysis& analysis) const {
    std::ostringstream oss;

    oss << "=== Kernel Analysis: " << analysis.kernel_name
        << " (" << analysis.variant_name << ") ===\n\n";

    oss << "Problem Size: " << analysis.problem_size << " elements\n";
    oss << "Primary Bottleneck: " << analysis.primary_bottleneck << "\n";
    oss << "Achieved: " << std::fixed << std::setprecision(2)
        << analysis.achieved_gflops << " GFLOPS\n";
    oss << "Theoretical Max: " << analysis.theoretical_max_gflops << " GFLOPS\n";
    oss << "Efficiency: " << (analysis.efficiency_vs_roofline * 100) << "%\n\n";

    oss << "--- Insights ---\n\n";

    for (const auto& insight : analysis.insights) {
        oss << "[" << severity_to_string(insight.severity) << "] "
            << insight.title << "\n";
        oss << insight.description << "\n\n";
        oss << "Recommendation:\n" << insight.recommendation << "\n\n";
        if (!insight.code_example.empty()) {
            oss << "Example:\n" << insight.code_example << "\n\n";
        }
        oss << "---\n\n";
    }

    oss << "--- Prioritized Next Steps ---\n\n";
    for (const auto& step : analysis.next_steps) {
        oss << step << "\n";
    }

    return oss.str();
}

std::string InsightsEngine::format_insights_markdown(const KernelAnalysis& analysis) const {
    std::ostringstream oss;

    oss << "# Kernel Analysis: " << analysis.kernel_name << "\n\n";
    oss << "**Variant:** " << analysis.variant_name << "\n";
    oss << "**Problem Size:** " << analysis.problem_size << " elements\n\n";

    oss << "## Performance Summary\n\n";
    oss << "| Metric | Value |\n";
    oss << "|--------|-------|\n";
    oss << "| Primary Bottleneck | " << analysis.primary_bottleneck << " |\n";
    oss << "| Achieved GFLOPS | " << std::fixed << std::setprecision(2)
        << analysis.achieved_gflops << " |\n";
    oss << "| Theoretical Max | " << analysis.theoretical_max_gflops << " |\n";
    oss << "| Efficiency | " << (analysis.efficiency_vs_roofline * 100) << "% |\n\n";

    oss << "## Insights\n\n";

    for (const auto& insight : analysis.insights) {
        std::string badge;
        switch (insight.severity) {
            case InsightSeverity::CRITICAL: badge = "🔴 CRITICAL"; break;
            case InsightSeverity::HIGH: badge = "🟠 HIGH"; break;
            case InsightSeverity::MEDIUM: badge = "🟡 MEDIUM"; break;
            case InsightSeverity::LOW: badge = "🟢 LOW"; break;
            case InsightSeverity::INFO: badge = "ℹ️ INFO"; break;
        }

        oss << "### " << badge << " - " << insight.title << "\n\n";
        oss << insight.description << "\n\n";
        oss << "**Recommendation:**\n\n" << insight.recommendation << "\n\n";
        if (!insight.code_example.empty()) {
            oss << "**Example:**\n\n```cpp\n" << insight.code_example << "\n```\n\n";
        }
        if (insight.potential_speedup > 1.1) {
            oss << "*Potential speedup: " << std::setprecision(1)
                << insight.potential_speedup << "x*\n\n";
        }
    }

    oss << "## Prioritized Next Steps\n\n";
    for (const auto& step : analysis.next_steps) {
        oss << "- " << step << "\n";
    }

    return oss.str();
}

std::string InsightsEngine::format_insights_json(const KernelAnalysis& analysis) const {
    nlohmann::json j;

    j["kernel_name"] = analysis.kernel_name;
    j["variant_name"] = analysis.variant_name;
    j["problem_size"] = analysis.problem_size;
    j["primary_bottleneck"] = analysis.primary_bottleneck;
    j["achieved_gflops"] = analysis.achieved_gflops;
    j["theoretical_max_gflops"] = analysis.theoretical_max_gflops;
    j["efficiency_vs_roofline"] = analysis.efficiency_vs_roofline;
    j["efficiency_vs_peak"] = analysis.efficiency_vs_peak;

    j["insights"] = nlohmann::json::array();
    for (const auto& insight : analysis.insights) {
        nlohmann::json ins;
        ins["category"] = category_to_string(insight.category);
        ins["severity"] = severity_to_string(insight.severity);
        ins["confidence"] = confidence_to_string(insight.confidence);
        ins["title"] = insight.title;
        ins["description"] = insight.description;
        ins["recommendation"] = insight.recommendation;
        ins["potential_speedup"] = insight.potential_speedup;
        if (!insight.code_example.empty()) {
            ins["code_example"] = insight.code_example;
        }
        j["insights"].push_back(ins);
    }

    j["next_steps"] = analysis.next_steps;

    return j.dump(2);
}

std::string InsightsEngine::format_report_markdown(const InsightsReport& report) const {
    std::ostringstream oss;

    oss << "# SIMD-Bench Performance Insights Report\n\n";
    oss << "*Generated: " << report.timestamp << "*\n\n";

    oss << "## Hardware Configuration\n\n";
    oss << "- **CPU:** " << report.hardware.cpu_brand << "\n";
    oss << "- **Cores:** " << report.hardware.physical_cores << " physical, "
        << report.hardware.logical_cores << " logical\n";
    oss << "- **SIMD:** " << report.hardware.get_simd_string() << "\n";
    oss << "- **Peak SP GFLOPS:** " << report.hardware.theoretical_peak_sp_gflops << "\n";
    oss << "- **Memory BW:** " << report.hardware.measured_memory_bw_gbps << " GB/s\n\n";

    oss << "## Summary\n\n";
    oss << "| Severity | Count |\n";
    oss << "|----------|-------|\n";
    oss << "| Critical | " << report.critical_count << " |\n";
    oss << "| High | " << report.high_count << " |\n";
    oss << "| Medium | " << report.medium_count << " |\n";
    oss << "| Low | " << report.low_count << " |\n";
    oss << "| **Total** | **" << report.total_insights << "** |\n\n";

    // Global insights
    if (!report.global_insights.empty()) {
        oss << "## Global Insights\n\n";
        for (const auto& insight : report.global_insights) {
            oss << "### " << insight.title << "\n\n";
            oss << insight.description << "\n\n";
            oss << insight.recommendation << "\n\n";
        }
    }

    // Per-kernel analysis
    oss << "## Per-Kernel Analysis\n\n";
    for (const auto& analysis : report.kernel_analyses) {
        oss << format_insights_markdown(analysis) << "\n---\n\n";
    }

    return oss.str();
}

std::string InsightsEngine::format_report_json(const InsightsReport& report) const {
    nlohmann::json j;

    j["timestamp"] = report.timestamp;
    j["hardware"] = {
        {"cpu", report.hardware.cpu_brand},
        {"cores", report.hardware.physical_cores},
        {"simd", report.hardware.get_simd_string()},
        {"peak_gflops", report.hardware.theoretical_peak_sp_gflops},
        {"memory_bw_gbps", report.hardware.measured_memory_bw_gbps}
    };

    j["summary"] = {
        {"total_insights", report.total_insights},
        {"critical", report.critical_count},
        {"high", report.high_count},
        {"medium", report.medium_count},
        {"low", report.low_count}
    };

    j["kernel_analyses"] = nlohmann::json::array();
    for (const auto& analysis : report.kernel_analyses) {
        j["kernel_analyses"].push_back(nlohmann::json::parse(format_insights_json(analysis)));
    }

    return j.dump(2);
}

// ============================================================================
// Insight Templates Implementation
// ============================================================================

namespace insight_templates {

OptimizationInsight memory_bandwidth_limited(double utilization, double achieved_bw) {
    OptimizationInsight insight;
    insight.category = InsightCategory::MEMORY_BOUND;
    insight.severity = InsightSeverity::HIGH;
    insight.confidence = InsightConfidence::HIGH;
    insight.title = "Memory Bandwidth Saturated";
    insight.description =
        "Memory bandwidth utilization is " + std::to_string(utilization * 100) +
        "% (" + std::to_string(achieved_bw) + " GB/s). Further compute optimization won't help.";
    insight.recommendation =
        "Focus on reducing memory traffic:\n"
        "1. Improve temporal locality with blocking\n"
        "2. Use compression or lower precision\n"
        "3. Consider streaming stores for write-only data";
    insight.potential_speedup = 1.0 / utilization;
    return insight;
}

OptimizationInsight recommend_loop_tiling(size_t tile_size) {
    OptimizationInsight insight;
    insight.category = InsightCategory::LOOP_OPTIMIZATION;
    insight.severity = InsightSeverity::MEDIUM;
    insight.confidence = InsightConfidence::MEDIUM;
    insight.title = "Recommend Loop Tiling";
    insight.description = "Data access pattern suggests loop tiling would improve cache utilization.";
    insight.recommendation =
        "Apply loop tiling with tile size ~" + std::to_string(tile_size) +
        " elements to fit working set in cache.";
    insight.code_example =
        "// Tiled loop\n"
        "for (int ii = 0; ii < N; ii += TILE) {\n"
        "    for (int i = ii; i < min(ii+TILE, N); i++) {\n"
        "        // Process\n"
        "    }\n"
        "}";
    insight.potential_speedup = 1.5;
    return insight;
}

OptimizationInsight recommend_fma_usage() {
    OptimizationInsight insight;
    insight.category = InsightCategory::INSTRUCTION_MIX;
    insight.severity = InsightSeverity::MEDIUM;
    insight.confidence = InsightConfidence::MEDIUM;
    insight.title = "Use Fused Multiply-Add (FMA)";
    insight.description =
        "FMA instructions perform multiply-add in one operation with better precision and throughput.";
    insight.recommendation =
        "Use FMA intrinsics or ensure compiler generates them:\n"
        "1. `_mm256_fmadd_ps(a, b, c)` for a*b+c\n"
        "2. Compile with `-mfma` flag\n"
        "3. Restructure code to expose multiply-add patterns";
    insight.code_example =
        "// Before: separate multiply and add\n"
        "__m256 result = _mm256_add_ps(_mm256_mul_ps(a, b), c);\n\n"
        "// After: fused multiply-add\n"
        "__m256 result = _mm256_fmadd_ps(a, b, c);";
    insight.potential_speedup = 1.3;
    insight.references.push_back("https://momentsingraphics.de/FMA.html");
    return insight;
}

OptimizationInsight recommend_branchless_code() {
    OptimizationInsight insight;
    insight.category = InsightCategory::BRANCH_PREDICTION;
    insight.severity = InsightSeverity::MEDIUM;
    insight.confidence = InsightConfidence::MEDIUM;
    insight.title = "Consider Branchless Code";
    insight.description =
        "Unpredictable branches cause pipeline stalls. Branchless alternatives may help.";
    insight.recommendation =
        "Replace branches with:\n"
        "1. Conditional moves: `result = cond ? a : b;` (compiler may use cmov)\n"
        "2. Arithmetic: `result = cond * a + (1-cond) * b;`\n"
        "3. SIMD masking: `_mm256_blendv_ps(a, b, mask)`";
    insight.code_example =
        "// Branchy code\n"
        "if (x > 0) result = a; else result = b;\n\n"
        "// Branchless alternative\n"
        "__m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);\n"
        "__m256 result = _mm256_blendv_ps(b, a, mask);";
    insight.potential_speedup = 2.0;
    insight.references.push_back("https://en.algorithmica.org/hpc/pipelining/branchless/");
    return insight;
}

OptimizationInsight recommend_aos_to_soa() {
    OptimizationInsight insight;
    insight.category = InsightCategory::MEMORY_ACCESS_PATTERN;
    insight.severity = InsightSeverity::HIGH;
    insight.confidence = InsightConfidence::MEDIUM;
    insight.title = "Convert AoS to SoA Data Layout";
    insight.description =
        "Array-of-Structures (AoS) layout causes non-contiguous SIMD loads. "
        "Structure-of-Arrays (SoA) enables efficient vectorization.";
    insight.recommendation =
        "Restructure data from AoS to SoA:\n"
        "- Before: `struct Point { float x, y, z; } points[N];`\n"
        "- After: `float x[N], y[N], z[N];`\n\n"
        "This enables contiguous loads for each component.";
    insight.code_example =
        "// AoS: non-contiguous access\n"
        "for (int i = 0; i < N; i++) {\n"
        "    result += points[i].x * points[i].y;\n"
        "}\n\n"
        "// SoA: contiguous access, vectorizable\n"
        "for (int i = 0; i < N; i += 8) {\n"
        "    __m256 vx = _mm256_load_ps(&x[i]);\n"
        "    __m256 vy = _mm256_load_ps(&y[i]);\n"
        "    sum = _mm256_fmadd_ps(vx, vy, sum);\n"
        "}";
    insight.potential_speedup = 4.0;
    return insight;
}

}  // namespace insight_templates

// ============================================================================
// InsightRuleEngine Implementation
// ============================================================================

InsightRuleEngine::InsightRuleEngine() {
    // Register default rules
    for (const auto& [name, rule] : get_default_rules()) {
        rules_[name] = rule;
    }
}

void InsightRuleEngine::add_rule(const std::string& name, InsightRule rule) {
    rules_[name] = std::move(rule);
}

void InsightRuleEngine::remove_rule(const std::string& name) {
    rules_.erase(name);
}

std::vector<OptimizationInsight> InsightRuleEngine::evaluate(
    const KernelMetrics& metrics,
    const HardwareInfo& hw,
    const InsightThresholds& thresholds,
    const KernelConfig* config
) const {
    std::vector<OptimizationInsight> insights;

    for (const auto& [name, rule] : rules_) {
        auto result = rule(metrics, hw, thresholds, config);
        if (result.has_value()) {
            insights.push_back(std::move(result.value()));
        }
    }

    return insights;
}

std::vector<std::pair<std::string, InsightRule>> InsightRuleEngine::get_default_rules() {
    std::vector<std::pair<std::string, InsightRule>> rules;

    // Rule: High cache miss rate
    rules.push_back({"high_cache_miss", [](
        const KernelMetrics& metrics,
        const HardwareInfo& /*hw*/,
        const InsightThresholds& thresholds,
        const KernelConfig* /*config*/
    ) -> std::optional<OptimizationInsight> {
        uint64_t total = metrics.memory.l1_hits + metrics.memory.l1_misses;
        if (total == 0) return std::nullopt;

        double miss_rate = static_cast<double>(metrics.memory.l1_misses) / total;

        if (miss_rate > thresholds.high_l1_miss_rate) {
            OptimizationInsight insight;
            insight.category = InsightCategory::CACHE_EFFICIENCY;
            insight.severity = miss_rate > 0.2 ? InsightSeverity::CRITICAL : InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::HIGH;
            insight.title = "High L1 Cache Miss Rate";
            insight.description = "L1 cache miss rate: " + std::to_string(miss_rate * 100) + "%";
            insight.recommendation =
                "Apply cache blocking and improve data locality:\n"
                "1. Tile loops to fit working set in L1 cache\n"
                "2. Ensure sequential access patterns\n"
                "3. Consider prefetching for irregular access";
            insight.potential_speedup = 1.0 + miss_rate * 3;
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: Low vectorization ratio
    rules.push_back({"low_vectorization", [](
        const KernelMetrics& metrics,
        const HardwareInfo& hw,
        const InsightThresholds& thresholds,
        const KernelConfig* /*config*/
    ) -> std::optional<OptimizationInsight> {
        uint64_t total_ops = metrics.simd.scalar_ops + metrics.simd.packed_256_ops +
                            metrics.simd.packed_512_ops;
        if (total_ops == 0) return std::nullopt;

        double vec_ratio = static_cast<double>(metrics.simd.packed_256_ops + metrics.simd.packed_512_ops) /
                          total_ops;

        if (vec_ratio < thresholds.poor_vectorization_ratio) {
            OptimizationInsight insight;
            insight.category = InsightCategory::VECTORIZATION;
            insight.severity = InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Poor Vectorization";
            insight.description = "Only " + std::to_string(vec_ratio * 100) + "% of operations vectorized.";
            insight.recommendation = "Check for vectorization blockers: pointer aliasing, non-contiguous access, branches in loops.";
            insight.potential_speedup = hw.max_vector_bits / 32.0 * 0.5;
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: Low IPC (Instructions Per Cycle)
    rules.push_back({"low_ipc", [](
        const KernelMetrics& metrics,
        const HardwareInfo& /*hw*/,
        const InsightThresholds& thresholds,
        const KernelConfig* /*config*/
    ) -> std::optional<OptimizationInsight> {
        if (metrics.performance.cycles == 0) return std::nullopt;

        double ipc = static_cast<double>(metrics.performance.instructions) / metrics.performance.cycles;

        if (ipc > 0 && ipc < thresholds.poor_ipc) {
            OptimizationInsight insight;
            insight.category = InsightCategory::INSTRUCTION_MIX;
            insight.severity = ipc < 0.5 ? InsightSeverity::CRITICAL : InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Low Instructions Per Cycle";
            insight.description = "IPC: " + std::to_string(ipc) + " (expected >2.0 for optimized code)";
            insight.recommendation =
                "Low IPC indicates stalls. Check for:\n"
                "1. Memory latency (cache misses)\n"
                "2. Dependency chains (use multiple accumulators)\n"
                "3. Branch mispredictions\n"
                "4. Frontend stalls (instruction cache misses)";
            insight.potential_speedup = thresholds.good_ipc / ipc;
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: High L2 miss rate
    rules.push_back({"high_l2_miss", [](
        const KernelMetrics& metrics,
        const HardwareInfo& /*hw*/,
        const InsightThresholds& thresholds,
        const KernelConfig* /*config*/
    ) -> std::optional<OptimizationInsight> {
        uint64_t total = metrics.memory.l2_hits + metrics.memory.l2_misses;
        if (total == 0) return std::nullopt;

        double miss_rate = static_cast<double>(metrics.memory.l2_misses) / total;

        if (miss_rate > thresholds.high_l2_miss_rate) {
            OptimizationInsight insight;
            insight.category = InsightCategory::CACHE_EFFICIENCY;
            insight.severity = InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::HIGH;
            insight.title = "High L2 Cache Miss Rate";
            insight.description = "L2 cache miss rate: " + std::to_string(miss_rate * 100) + "%";
            insight.recommendation =
                "Working set exceeds L2 cache:\n"
                "1. Tile loops to fit in L2 cache\n"
                "2. Consider loop fusion to reduce memory passes\n"
                "3. Use software prefetching for L3/memory";
            insight.potential_speedup = 1.0 + miss_rate * 2;
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: Memory bandwidth utilization
    rules.push_back({"bandwidth_analysis", [](
        const KernelMetrics& metrics,
        const HardwareInfo& hw,
        const InsightThresholds& thresholds,
        const KernelConfig* config
    ) -> std::optional<OptimizationInsight> {
        if (!config || metrics.performance.gflops == 0) return std::nullopt;

        double ai = config->arithmetic_intensity;
        if (ai == 0) return std::nullopt;

        double bytes_per_second = metrics.performance.gflops * 1e9 / ai;
        double bw_utilization = bytes_per_second / (hw.measured_memory_bw_gbps * 1e9);

        if (bw_utilization > thresholds.high_bw_utilization) {
            OptimizationInsight insight;
            insight.category = InsightCategory::MEMORY_BOUND;
            insight.severity = InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::HIGH;
            insight.title = "Memory Bandwidth Saturated";
            insight.description = "Bandwidth utilization: " + std::to_string(bw_utilization * 100) + "%";
            insight.recommendation =
                "Memory bandwidth is limiting performance. Consider:\n"
                "1. Increase arithmetic intensity (data reuse)\n"
                "2. Reduce data precision (FP16, INT8)\n"
                "3. Use non-temporal stores for write-only data\n"
                "4. Kernel fusion to reduce memory passes";
            insight.potential_speedup = 1.0;  // Already at limit
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: Compute bound with low efficiency
    rules.push_back({"compute_inefficiency", [](
        const KernelMetrics& metrics,
        const HardwareInfo& hw,
        const InsightThresholds& thresholds,
        const KernelConfig* config
    ) -> std::optional<OptimizationInsight> {
        if (!config || metrics.performance.gflops == 0) return std::nullopt;

        double ai = config->arithmetic_intensity;
        double ridge_point = hw.theoretical_peak_sp_gflops / hw.measured_memory_bw_gbps;

        // Only for compute-bound kernels
        if (ai < ridge_point) return std::nullopt;

        double efficiency = metrics.performance.gflops / hw.theoretical_peak_sp_gflops;

        if (efficiency < thresholds.poor_efficiency) {
            OptimizationInsight insight;
            insight.category = InsightCategory::COMPUTE_BOUND;
            insight.severity = InsightSeverity::HIGH;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Compute-Bound with Low Efficiency";
            insight.description = "Achieving only " + std::to_string(efficiency * 100) +
                "% of peak compute throughput";
            insight.recommendation =
                "Compute-bound but inefficient. Check:\n"
                "1. Vectorization quality (use SIMD intrinsics)\n"
                "2. FMA instruction usage\n"
                "3. Instruction-level parallelism (unrolling)\n"
                "4. Execution port saturation";
            insight.potential_speedup = 1.0 / efficiency;
            return insight;
        }
        return std::nullopt;
    }});

    // Rule: Excessive scalar operations in SIMD code
    rules.push_back({"scalar_ops_in_simd", [](
        const KernelMetrics& metrics,
        const HardwareInfo& hw,
        const InsightThresholds& /*thresholds*/,
        const KernelConfig* /*config*/
    ) -> std::optional<OptimizationInsight> {
        uint64_t vector_ops = metrics.simd.packed_256_ops + metrics.simd.packed_512_ops;
        uint64_t scalar_ops = metrics.simd.scalar_ops;

        if (vector_ops == 0 || scalar_ops == 0) return std::nullopt;

        double scalar_ratio = static_cast<double>(scalar_ops) / (scalar_ops + vector_ops);

        // If we have vector ops but also significant scalar ops
        if (scalar_ratio > 0.2 && vector_ops > 1000) {
            OptimizationInsight insight;
            insight.category = InsightCategory::VECTORIZATION;
            insight.severity = InsightSeverity::MEDIUM;
            insight.confidence = InsightConfidence::MEDIUM;
            insight.title = "Mixed Scalar/Vector Operations";
            insight.description = std::to_string(scalar_ratio * 100) +
                "% of operations are scalar in otherwise vectorized code";
            insight.recommendation =
                "Reduce scalar operations in SIMD code:\n"
                "1. Vectorize reduction operations\n"
                "2. Handle loop tails with masked operations\n"
                "3. Use gather/scatter for irregular access\n"
                "4. Inline small functions to enable vectorization";
            insight.potential_speedup = 1.0 + scalar_ratio * (hw.max_vector_bits / 32.0 - 1);
            return insight;
        }
        return std::nullopt;
    }});

    return rules;
}

// ============================================================================
// CodePatternAnalyzer Implementation (Basic)
// ============================================================================

std::vector<CodePattern> CodePatternAnalyzer::analyze_source(const std::string& source_code) const {
    std::vector<CodePattern> patterns;

    // Check for restrict keyword
    if (!uses_restrict_qualifier(source_code)) {
        CodePattern pattern;
        pattern.pattern_name = "missing_restrict";
        pattern.description = "Pointers lack 'restrict' qualifier, may prevent vectorization";
        pattern.is_antipattern = true;
        pattern.related_insight.category = InsightCategory::VECTORIZATION;
        pattern.related_insight.severity = InsightSeverity::MEDIUM;
        pattern.related_insight.title = "Add restrict Qualifier";
        pattern.related_insight.recommendation =
            "Add 'restrict' to pointer parameters to enable vectorization.";
        patterns.push_back(pattern);
    }

    // Check for alignment attributes
    if (!has_alignment_attributes(source_code)) {
        CodePattern pattern;
        pattern.pattern_name = "missing_alignment";
        pattern.description = "No explicit alignment attributes found";
        pattern.is_antipattern = true;
        pattern.related_insight.category = InsightCategory::DATA_ALIGNMENT;
        pattern.related_insight.severity = InsightSeverity::LOW;
        pattern.related_insight.title = "Add Alignment Attributes";
        pattern.related_insight.recommendation =
            "Use alignas(64) or __attribute__((aligned(64))) for SIMD data.";
        patterns.push_back(pattern);
    }

    return patterns;
}

bool CodePatternAnalyzer::uses_restrict_qualifier(const std::string& code) const {
    return code.find("restrict") != std::string::npos ||
           code.find("__restrict") != std::string::npos;
}

bool CodePatternAnalyzer::has_alignment_attributes(const std::string& code) const {
    return code.find("alignas") != std::string::npos ||
           code.find("aligned(") != std::string::npos ||
           code.find("AllocateAligned") != std::string::npos;
}

bool CodePatternAnalyzer::has_loop_carried_dependency(const std::string& loop_code) const {
    // Simple heuristic: look for patterns like a[i] = ... a[i-1] ...
    std::regex dep_pattern(R"(\w+\[i\]\s*=.*\w+\[i\s*-\s*\d+\])");
    return std::regex_search(loop_code, dep_pattern);
}

bool CodePatternAnalyzer::has_non_unit_stride(const std::string& loop_code) const {
    // Look for array access with non-unit stride
    std::regex stride_pattern(R"(\w+\[\s*\d+\s*\*\s*i\s*\])");
    return std::regex_search(loop_code, stride_pattern);
}

bool CodePatternAnalyzer::has_function_call_in_loop(const std::string& loop_code) const {
    // Look for function calls (simplistic)
    std::regex call_pattern(R"(\w+\s*\([^)]*\))");
    return std::regex_search(loop_code, call_pattern);
}

bool CodePatternAnalyzer::has_conditional_in_loop(const std::string& loop_code) const {
    return loop_code.find("if") != std::string::npos ||
           loop_code.find("?") != std::string::npos;
}

double CodePatternAnalyzer::estimate_vectorization_potential(const std::string& loop_code) const {
    double score = 1.0;

    if (has_loop_carried_dependency(loop_code)) score *= 0.1;
    if (has_non_unit_stride(loop_code)) score *= 0.5;
    if (has_function_call_in_loop(loop_code)) score *= 0.3;
    if (has_conditional_in_loop(loop_code)) score *= 0.7;
    if (!uses_restrict_qualifier(loop_code)) score *= 0.8;

    return score;
}

}  // namespace simd_bench
