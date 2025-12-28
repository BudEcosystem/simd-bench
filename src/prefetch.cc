#include "simd_bench/prefetch.h"
#include <cmath>
#include <sstream>
#include <algorithm>

namespace simd_bench {

// ============================================================================
// PrefetchAnalyzer implementation
// ============================================================================

PrefetchMetrics PrefetchAnalyzer::analyze(
    uint64_t l2_lines_in_all,
    uint64_t l2_lines_out_useless_hwpf,
    uint64_t sw_prefetch_access,
    uint64_t dram_requests,
    uint64_t total_instructions
) {
    PrefetchMetrics metrics;

    metrics.sw_prefetch_issued = sw_prefetch_access;
    metrics.hw_prefetch_useful = l2_lines_in_all;
    metrics.hw_prefetch_useless = l2_lines_out_useless_hwpf;

    // Calculate prefetch accuracy
    uint64_t total_hw_prefetch = metrics.hw_prefetch_useful + metrics.hw_prefetch_useless;
    if (total_hw_prefetch > 0) {
        metrics.prefetch_accuracy =
            static_cast<double>(metrics.hw_prefetch_useful) / total_hw_prefetch;
    }

    // Calculate prefetch coverage (how much DRAM traffic was prefetched)
    if (dram_requests > 0) {
        // Useful prefetches that would have been demand misses
        metrics.prefetch_coverage =
            static_cast<double>(metrics.hw_prefetch_useful) /
            (dram_requests + metrics.hw_prefetch_useful);
    }

    // Timeliness estimate: based on useless ratio (useless = too early or wrong)
    metrics.prefetch_timeliness = metrics.prefetch_accuracy;  // Approximation

    // Generate recommendations
    if (metrics.prefetch_accuracy < 0.5) {
        metrics.recommendation = "Hardware prefetcher has low accuracy - irregular access pattern";
        metrics.suggestions.push_back("Consider software prefetching with explicit pattern");
        metrics.suggestions.push_back("Reorganize data layout for sequential access");
    } else if (metrics.prefetch_coverage < 0.3) {
        metrics.recommendation = "Low prefetch coverage - many cache misses not anticipated";
        metrics.suggestions.push_back("Add software prefetching to cover gaps");
        metrics.suggestions.push_back("Increase prefetch distance for latency hiding");
    } else if (metrics.prefetch_accuracy > 0.9 && metrics.prefetch_coverage > 0.7) {
        metrics.recommendation = "Prefetching is effective - hardware prefetch working well";
    } else {
        metrics.recommendation = "Prefetching is partially effective - consider tuning";
    }

    // Recommend prefetch distance based on coverage gaps
    if (metrics.prefetch_coverage < 0.7) {
        // Rough estimate: need to prefetch further ahead
        metrics.recommended_prefetch_distance = static_cast<int>(
            8 * (1.0 - metrics.prefetch_coverage) + 4);
    } else {
        metrics.recommended_prefetch_distance = 4;  // Default for good coverage
    }

    return metrics;
}

int PrefetchAnalyzer::calculate_prefetch_distance(
    double memory_latency_ns,
    double loop_iteration_ns,
    size_t stride_bytes
) {
    if (loop_iteration_ns <= 0 || stride_bytes == 0) {
        return 8;  // Safe default
    }

    // Prefetch distance = memory_latency / iteration_time
    // Need to prefetch far enough ahead to hide memory latency
    double iterations_to_hide = memory_latency_ns / loop_iteration_ns;

    // Convert to cache lines
    size_t cache_line_size = 64;
    size_t cache_lines_per_iteration = (stride_bytes + cache_line_size - 1) / cache_line_size;

    int distance = static_cast<int>(std::ceil(iterations_to_hide * cache_lines_per_iteration));

    // Clamp to reasonable range
    distance = std::max(1, std::min(distance, 64));

    return distance;
}

int PrefetchAnalyzer::calculate_prefetch_distance_detailed(
    double l3_latency_ns,
    double dram_latency_ns,
    double loop_iteration_ns,
    size_t stride_bytes,
    size_t l2_size_kb,
    size_t cache_line_bytes
) {
    if (loop_iteration_ns <= 0 || stride_bytes == 0) {
        return 8;
    }

    // Use DRAM latency for L2 prefetch, L3 latency for L1 prefetch
    double effective_latency = dram_latency_ns;

    // Account for L2 capacity constraints
    size_t l2_lines = (l2_size_kb * 1024) / cache_line_bytes;
    size_t cache_lines_per_iteration = (stride_bytes + cache_line_bytes - 1) / cache_line_bytes;

    // Can't prefetch more than fits in L2
    size_t max_distance = l2_lines / (cache_lines_per_iteration * 2);  // Leave room for data

    double iterations_to_hide = effective_latency / loop_iteration_ns;
    int distance = static_cast<int>(std::ceil(iterations_to_hide * cache_lines_per_iteration));

    // Clamp to L2 capacity and reasonable limits
    distance = std::max(1, std::min(distance, static_cast<int>(max_distance)));
    distance = std::min(distance, 64);

    return distance;
}

std::string PrefetchAnalyzer::generate_prefetch_code(
    int prefetch_distance,
    size_t stride_bytes,
    const std::string& pointer_name,
    bool use_nta
) {
    std::ostringstream code;

    std::string hint = use_nta ? "_MM_HINT_NTA" : "_MM_HINT_T0";

    code << "// Prefetch " << prefetch_distance << " iterations ahead\n";
    code << "_mm_prefetch(reinterpret_cast<const char*>(\n";
    code << "    " << pointer_name << " + " << prefetch_distance * stride_bytes;
    code << "), " << hint << ");\n";

    if (stride_bytes >= 64 && !use_nta) {
        code << "\n// For large strides, consider L2 prefetch:\n";
        code << "_mm_prefetch(reinterpret_cast<const char*>(\n";
        code << "    " << pointer_name << " + " << (prefetch_distance + 4) * stride_bytes;
        code << "), _MM_HINT_T1);  // L2 prefetch\n";
    }

    return code.str();
}

PrefetchAnalyzer::HardwarePrefetchAnalysis PrefetchAnalyzer::analyze_hw_prefetch(
    double l1_miss_rate,
    double l2_miss_rate,
    double l3_miss_rate,
    size_t stride_bytes,
    size_t working_set_bytes
) {
    HardwarePrefetchAnalysis analysis;

    // HW prefetchers typically handle:
    // - Sequential access (stride = 1-2 cache lines)
    // - Small regular strides up to ~2KB on Intel
    constexpr size_t MAX_HW_PREFETCH_STRIDE = 2048;

    if (stride_bytes <= 64) {
        // Sequential or near-sequential: HW prefetch should work
        if (l2_miss_rate < 0.05) {
            analysis.hw_prefetch_sufficient = true;
            analysis.hw_prefetch_efficiency = 1.0 - l2_miss_rate;
            analysis.reason = "Sequential pattern - hardware prefetch effective";
        } else {
            analysis.hw_prefetch_sufficient = false;
            analysis.needs_sw_prefetch = true;
            analysis.hw_prefetch_efficiency = 1.0 - l2_miss_rate;
            analysis.reason = "Sequential pattern but high miss rate - may need SW assist";
        }
    } else if (stride_bytes <= MAX_HW_PREFETCH_STRIDE) {
        // Moderate stride: HW prefetch may work
        analysis.hw_prefetch_efficiency = 1.0 - l2_miss_rate;

        if (l2_miss_rate < 0.1) {
            analysis.hw_prefetch_sufficient = true;
            analysis.reason = "Regular stride detected - hardware prefetch handling it";
        } else {
            analysis.hw_prefetch_sufficient = false;
            analysis.needs_sw_prefetch = true;
            analysis.reason = "Stride too large for efficient HW prefetch";
            analysis.recommendations.push_back(
                "Add software prefetch with distance = latency / iteration_time");
        }
    } else {
        // Large stride: HW prefetch won't help
        analysis.hw_prefetch_sufficient = false;
        analysis.hw_prefetch_efficiency = 0.0;
        analysis.needs_sw_prefetch = (l3_miss_rate > 0.01);
        analysis.reason = "Stride exceeds hardware prefetcher capability";
        analysis.recommendations.push_back(
            "Consider data layout restructuring to reduce stride");
        analysis.recommendations.push_back(
            "Use explicit software prefetching if restructuring not possible");
    }

    // Check for working set issues
    if (working_set_bytes > 16 * 1024 * 1024 && l3_miss_rate > 0.1) {  // > 16MB
        analysis.recommendations.push_back(
            "Large working set causing L3 misses - consider tiling");
    }

    return analysis;
}

// ============================================================================
// Pattern detection
// ============================================================================

PrefetchPattern detect_prefetch_pattern(
    uint64_t sequential_reads,
    uint64_t strided_reads,
    uint64_t random_reads,
    size_t detected_stride,
    double memory_latency_ns,
    double iteration_time_ns
) {
    PrefetchPattern pattern;

    uint64_t total_reads = sequential_reads + strided_reads + random_reads;
    if (total_reads == 0) {
        pattern.type = PrefetchPattern::Type::SEQUENTIAL;
        pattern.description = "No read pattern data available";
        return pattern;
    }

    double seq_ratio = static_cast<double>(sequential_reads) / total_reads;
    double strided_ratio = static_cast<double>(strided_reads) / total_reads;
    double random_ratio = static_cast<double>(random_reads) / total_reads;

    pattern.stride = detected_stride;
    pattern.regularity = 1.0 - random_ratio;

    if (seq_ratio > 0.8) {
        pattern.type = PrefetchPattern::Type::SEQUENTIAL;
        pattern.hw_prefetch_effective = true;
        pattern.optimal_sw_prefetch_distance = 0;  // HW prefetch sufficient
        pattern.description = "Sequential access - hardware prefetch effective";
    } else if (strided_ratio > 0.6) {
        pattern.type = PrefetchPattern::Type::STRIDED;
        pattern.stride = detected_stride;

        // HW prefetch effective for small strides
        pattern.hw_prefetch_effective = (detected_stride <= 512);

        if (!pattern.hw_prefetch_effective) {
            pattern.optimal_sw_prefetch_distance =
                PrefetchAnalyzer::calculate_prefetch_distance(
                    memory_latency_ns, iteration_time_ns, detected_stride);
        }

        pattern.description = "Strided access with stride " +
            std::to_string(detected_stride) + " bytes";
    } else if (random_ratio > 0.5) {
        pattern.type = PrefetchPattern::Type::RANDOM;
        pattern.hw_prefetch_effective = false;
        pattern.optimal_sw_prefetch_distance = 0;  // Prefetch won't help
        pattern.description = "Random access pattern - prefetching not effective";
    } else if (seq_ratio > 0.3 && detected_stride == 0) {
        pattern.type = PrefetchPattern::Type::INDIRECT;
        pattern.hw_prefetch_effective = false;
        pattern.description = "Indirect/pointer-chasing pattern";
    } else {
        // Large sequential streaming
        if (seq_ratio > 0.9 && total_reads > 1000000) {
            pattern.type = PrefetchPattern::Type::STREAMING;
            pattern.hw_prefetch_effective = true;
            pattern.description = "Streaming access - consider non-temporal operations";
        } else {
            pattern.type = PrefetchPattern::Type::SEQUENTIAL;
            pattern.hw_prefetch_effective = true;
            pattern.description = "Mixed access pattern";
        }
    }

    return pattern;
}

// ============================================================================
// Prefetch tuning recommendations
// ============================================================================

PrefetchTuningResult recommend_prefetch_tuning(
    const PrefetchMetrics& metrics,
    const PrefetchPattern& pattern,
    const HardwareInfo& hw,
    size_t working_set_bytes,
    size_t stride_bytes
) {
    PrefetchTuningResult result;

    // Estimate memory latency
    double dram_latency_ns = DRAMLatency::DDR4_3200_NS;
    if (hw.measured_memory_bw_gbps > 50) {
        dram_latency_ns = DRAMLatency::DDR5_4800_NS;  // Fast memory = probably DDR5
    }

    // Estimate loop iteration time from frequency
    double freq_ghz = (hw.measured_frequency_ghz > 0) ? hw.measured_frequency_ghz : 2.5;  // Default 2.5 GHz
    double cycle_time_ns = 1.0 / freq_ghz;  // ns per cycle
    double estimated_iteration_ns = cycle_time_ns * 10;  // Rough estimate: 10 cycles/iteration

    switch (pattern.type) {
        case PrefetchPattern::Type::SEQUENTIAL:
            if (metrics.prefetch_coverage > 0.8) {
                result.needs_prefetch = false;
                result.explanation.push_back(
                    "Hardware prefetch is effective for sequential access");
            } else {
                result.needs_prefetch = true;
                result.l2_prefetch_distance = 8;
                result.explanation.push_back(
                    "Sequential access but coverage low - add light prefetching");
            }
            break;

        case PrefetchPattern::Type::STRIDED:
            result.needs_prefetch = (stride_bytes > 256);
            if (result.needs_prefetch) {
                result.l1_prefetch_distance = PrefetchAnalyzer::calculate_prefetch_distance(
                    dram_latency_ns * 0.3,  // L3 latency
                    estimated_iteration_ns,
                    stride_bytes
                );
                result.l2_prefetch_distance = PrefetchAnalyzer::calculate_prefetch_distance(
                    dram_latency_ns,
                    estimated_iteration_ns,
                    stride_bytes
                );
                result.expected_improvement = 0.3;  // ~30% improvement typical
                result.explanation.push_back(
                    "Strided access with stride > 256 bytes - software prefetch recommended");
            }
            break;

        case PrefetchPattern::Type::STREAMING:
            result.needs_prefetch = true;
            result.use_nta = true;
            result.l2_prefetch_distance = 16;
            result.expected_improvement = 0.4;
            result.explanation.push_back(
                "Streaming access - use non-temporal prefetch to avoid cache pollution");
            break;

        case PrefetchPattern::Type::INDIRECT:
            result.needs_prefetch = false;
            result.explanation.push_back(
                "Indirect access pattern - prefetching has limited benefit");
            result.explanation.push_back(
                "Consider restructuring data for better locality");
            break;

        case PrefetchPattern::Type::RANDOM:
            result.needs_prefetch = false;
            result.explanation.push_back(
                "Random access pattern - prefetching will not help");
            result.explanation.push_back(
                "Focus on reducing working set size or improving data layout");
            break;
    }

    // Generate code suggestion if prefetch is recommended
    if (result.needs_prefetch) {
        int distance = std::max(result.l1_prefetch_distance, result.l2_prefetch_distance);
        if (distance == 0) distance = 8;

        result.code_suggestion = PrefetchAnalyzer::generate_prefetch_code(
            distance, stride_bytes, "data", result.use_nta);
    }

    // Check for over-prefetching
    if (metrics.hw_prefetch_useless > metrics.hw_prefetch_useful * 2) {
        result.explanation.push_back(
            "Warning: High useless prefetch rate - access pattern may be unpredictable");
    }

    return result;
}

}  // namespace simd_bench
