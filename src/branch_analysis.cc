#include "simd_bench/branch_analysis.h"
#include <cmath>
#include <sstream>
#include <algorithm>

namespace simd_bench {

// ============================================================================
// BranchAnalyzer implementation
// ============================================================================

BranchMetrics BranchAnalyzer::analyze(
    uint64_t branch_instructions,
    uint64_t branch_misses,
    uint64_t total_instructions
) {
    BranchMetrics metrics;

    metrics.conditional_branches = branch_instructions;
    metrics.mispredictions = branch_misses;
    metrics.total_instructions = total_instructions;

    // Calculate derived metrics
    if (total_instructions > 0) {
        metrics.branch_density =
            (static_cast<double>(branch_instructions) / total_instructions) * 1000.0;
        metrics.misprediction_density =
            (static_cast<double>(branch_misses) / total_instructions) * 1000.0;
    }

    if (branch_instructions > 0) {
        metrics.misprediction_rate =
            static_cast<double>(branch_misses) / branch_instructions;
    }

    // Classification
    metrics.is_branchless = (metrics.branch_density < BranchThresholds::BRANCHLESS);
    metrics.has_branch_issues = (metrics.misprediction_rate > BranchThresholds::MISP_ACCEPTABLE);

    // Calculate vectorization friendliness
    // Low branches + low misprediction = high friendliness
    double branch_factor = std::max(0.0,
        1.0 - (metrics.branch_density / BranchThresholds::MODERATE));
    double misp_factor = std::max(0.0,
        1.0 - (metrics.misprediction_rate / BranchThresholds::MISP_POOR));

    metrics.vectorization_friendliness = branch_factor * 0.6 + misp_factor * 0.4;

    // Generate recommendations
    if (metrics.is_branchless) {
        metrics.recommendations.push_back(
            "Code is essentially branchless - excellent for SIMD vectorization");
    } else if (metrics.branch_density < BranchThresholds::LOW) {
        metrics.recommendations.push_back(
            "Low branch density - vectorization should be effective");
    } else if (metrics.has_branch_issues) {
        metrics.recommendations.push_back(
            "High misprediction rate (" +
            std::to_string(static_cast<int>(metrics.misprediction_rate * 100)) +
            "%) - consider branchless alternatives");
        metrics.branchless_opportunities.push_back(
            "Use conditional moves (cmov) instead of branches");
        metrics.branchless_opportunities.push_back(
            "Use SIMD blend operations with masks");
    }

    if (metrics.branch_density > BranchThresholds::HIGH) {
        metrics.recommendations.push_back(
            "Very high branch density - code structure limits vectorization");
        metrics.branchless_opportunities.push_back(
            "Consider data-parallel restructuring");
    }

    return metrics;
}

BranchMetrics BranchAnalyzer::analyze_detailed(
    uint64_t conditional_branches,
    uint64_t unconditional_branches,
    uint64_t mispredictions,
    uint64_t total_instructions
) {
    BranchMetrics metrics;

    metrics.conditional_branches = conditional_branches;
    metrics.unconditional_branches = unconditional_branches;
    metrics.mispredictions = mispredictions;
    metrics.total_instructions = total_instructions;

    uint64_t total_branches = conditional_branches + unconditional_branches;

    if (total_instructions > 0) {
        // Only conditional branches affect vectorization
        metrics.branch_density =
            (static_cast<double>(conditional_branches) / total_instructions) * 1000.0;
        metrics.misprediction_density =
            (static_cast<double>(mispredictions) / total_instructions) * 1000.0;
    }

    if (conditional_branches > 0) {
        metrics.misprediction_rate =
            static_cast<double>(mispredictions) / conditional_branches;
    }

    // Classification
    metrics.is_branchless = (metrics.branch_density < BranchThresholds::BRANCHLESS);
    metrics.has_branch_issues = (metrics.misprediction_rate > BranchThresholds::MISP_ACCEPTABLE);

    // Vectorization friendliness
    double branch_factor = std::max(0.0,
        1.0 - (metrics.branch_density / BranchThresholds::MODERATE));
    double misp_factor = std::max(0.0,
        1.0 - (metrics.misprediction_rate / BranchThresholds::MISP_POOR));

    metrics.vectorization_friendliness = branch_factor * 0.6 + misp_factor * 0.4;

    // Detailed recommendations based on branch types
    if (unconditional_branches > conditional_branches * 2) {
        metrics.recommendations.push_back(
            "Many unconditional branches (calls/returns) - function call overhead may dominate");
    }

    return metrics;
}

std::vector<std::string> BranchAnalyzer::detect_vectorization_blockers(
    const BranchMetrics& metrics,
    double vectorization_ratio
) {
    std::vector<std::string> blockers;

    // High branch density blocks vectorization
    if (metrics.branch_density > BranchThresholds::MODERATE) {
        blockers.push_back(
            "High conditional branch density (" +
            std::to_string(static_cast<int>(metrics.branch_density)) +
            "/1000 instructions) prevents efficient vectorization");
    }

    // Mispredictions indicate data-dependent branches
    if (metrics.misprediction_rate > BranchThresholds::MISP_ACCEPTABLE) {
        blockers.push_back(
            "Data-dependent branches with " +
            std::to_string(static_cast<int>(metrics.misprediction_rate * 100)) +
            "% misprediction rate");
    }

    // Low vectorization despite low branches = other issues
    if (metrics.branch_density < BranchThresholds::LOW && vectorization_ratio < 0.5) {
        blockers.push_back(
            "Low vectorization despite few branches - check for aliasing or dependencies");
    }

    // Mismatch between branch density and vectorization
    if (metrics.vectorization_friendliness > 0.7 && vectorization_ratio < 0.3) {
        blockers.push_back(
            "Code appears branchless but vectorization is low - investigate loop structure");
    }

    return blockers;
}

bool BranchAnalyzer::is_vectorization_friendly(const BranchMetrics& metrics) {
    return metrics.vectorization_friendliness >= BranchThresholds::SIMD_FRIENDLY;
}

std::vector<std::string> BranchAnalyzer::suggest_branchless_alternatives(
    const BranchMetrics& metrics,
    bool has_floating_point
) {
    std::vector<std::string> suggestions;

    if (metrics.is_branchless) {
        suggestions.push_back("Code is already essentially branchless");
        return suggestions;
    }

    // General suggestions based on misprediction rate
    if (metrics.misprediction_rate > BranchThresholds::MISP_POOR) {
        suggestions.push_back(
            "Pattern: if (cond) x = a; else x = b;\n"
            "Replace with: x = cond ? a : b; (allows cmov)\n"
            "Or SIMD: _mm256_blendv_ps(b, a, mask);");
    }

    if (metrics.branch_density > BranchThresholds::MODERATE) {
        suggestions.push_back(
            "Pattern: if (x < threshold) process(x);\n"
            "Replace with: SIMD comparison + masked operation\n"
            "mask = _mm256_cmp_ps(x, threshold, _CMP_LT_OQ);\n"
            "result = _mm256_blendv_ps(zero, processed, mask);");
    }

    if (has_floating_point) {
        // FP-specific suggestions
        suggestions.push_back(
            "Pattern: if (x > 0) x = sqrt(x);\n"
            "Replace with: x = x > 0 ? sqrt(x) : x; (branchless)\n"
            "Or: Use _mm256_blendv_ps with sqrt result");

        suggestions.push_back(
            "Pattern: y = abs(x) < epsilon ? 0 : x/y;\n"
            "Replace with: Masked divide with blend to zero");
    }

    suggestions.push_back(
        "Pattern: count += (condition);\n"
        "Replace with: _mm256_add_epi32(count, _mm256_and_si256(one, mask));");

    suggestions.push_back(
        "Pattern: min/max with branches\n"
        "Replace with: _mm256_min_ps / _mm256_max_ps");

    return suggestions;
}

// ============================================================================
// Pattern detection
// ============================================================================

BranchPattern detect_branch_pattern(
    const BranchMetrics& metrics,
    double loop_trip_count_estimate
) {
    BranchPattern pattern;

    // Classify based on density and misprediction
    if (metrics.is_branchless) {
        pattern.type = BranchPatternType::BRANCHLESS;
        pattern.confidence = 0.95;
        pattern.simd_compatibility = 1.0;
        pattern.description = "Branchless code - ideal for vectorization";
        pattern.optimization_strategy = "Focus on SIMD width utilization";
    } else if (metrics.misprediction_rate < BranchThresholds::MISP_EXCELLENT) {
        pattern.type = BranchPatternType::PREDICTABLE;
        pattern.confidence = 0.85;
        pattern.simd_compatibility = 0.8;
        pattern.description = "Highly predictable branches - speculative execution effective";
        pattern.optimization_strategy = "Branch predictor handles well, but SIMD masking may help";
    } else if (metrics.misprediction_rate > BranchThresholds::MISP_POOR) {
        pattern.type = BranchPatternType::DATA_DEPENDENT;
        pattern.confidence = 0.9;
        pattern.simd_compatibility = 0.3;
        pattern.description = "Data-dependent branches - hard to predict";
        pattern.optimization_strategy = "Convert to branchless or use SIMD masking";
    } else if (loop_trip_count_estimate > 0 &&
               metrics.branch_density < BranchThresholds::LOW) {
        pattern.type = BranchPatternType::LOOP_BOUND;
        pattern.confidence = 0.7;
        pattern.simd_compatibility = 0.9;
        pattern.description = "Branches mainly for loop control";
        pattern.optimization_strategy = "Loop vectorization should be effective";
    } else if (metrics.branch_density > BranchThresholds::HIGH) {
        pattern.type = BranchPatternType::MIXED;
        pattern.confidence = 0.6;
        pattern.simd_compatibility = 0.2;
        pattern.description = "Complex branching pattern";
        pattern.optimization_strategy = "Consider algorithmic restructuring";
    } else {
        pattern.type = BranchPatternType::MIXED;
        pattern.confidence = 0.5;
        pattern.simd_compatibility = 0.5;
        pattern.description = "Mixed branch pattern";
        pattern.optimization_strategy = "Profile to identify hot branches";
    }

    return pattern;
}

// ============================================================================
// SIMD masking analysis
// ============================================================================

SIMDMaskingAnalysis analyze_simd_masking(
    const BranchMetrics& metrics,
    double true_ratio,
    int simd_width_bits
) {
    SIMDMaskingAnalysis analysis;

    int lanes = simd_width_bits / 32;  // Assuming 32-bit elements

    // Masking is beneficial when:
    // 1. Branches are data-dependent (high misprediction)
    // 2. True ratio is not extreme (0.1-0.9 range)
    // 3. Branch density is moderate

    bool high_misprediction = (metrics.misprediction_rate > BranchThresholds::MISP_ACCEPTABLE);
    bool balanced_ratio = (true_ratio > 0.1 && true_ratio < 0.9);
    bool moderate_branches = (metrics.branch_density > BranchThresholds::BRANCHLESS &&
                              metrics.branch_density < BranchThresholds::HIGH);

    analysis.masking_beneficial = high_misprediction && balanced_ratio && moderate_branches;

    // Expected lane utilization based on true ratio
    analysis.lane_utilization = true_ratio;

    // Expected speedup from masking
    if (analysis.masking_beneficial) {
        // Speedup = SIMD_width * efficiency - overhead
        double efficiency = std::min(true_ratio, 1.0 - true_ratio);  // Work ratio
        efficiency = std::max(efficiency, 0.1);  // Minimum efficiency

        // Misprediction cost saved
        double misp_cost = metrics.misprediction_rate * 15.0;  // ~15 cycles per misprediction

        analysis.expected_speedup = (lanes * efficiency * 0.8) +
                                     (misp_cost * metrics.branch_density / 1000.0);
        analysis.expected_speedup = std::max(1.0, analysis.expected_speedup);
    } else {
        analysis.expected_speedup = 1.0;
    }

    // Recommend approach based on ratio
    if (true_ratio > 0.8) {
        analysis.recommended_approach = "blend";
        analysis.code_example =
            "// Most elements active - use blend\n"
            "auto mask = condition_check(data);\n"
            "auto result = blend(fallback, computed, mask);";
    } else if (true_ratio < 0.2) {
        analysis.recommended_approach = "compress";
        analysis.code_example =
            "// Few elements active - use compress/gather\n"
            "auto indices = compress_indices(condition_check(data));\n"
            "process_sparse(data, indices);";
    } else {
        analysis.recommended_approach = "blend";
        analysis.code_example =
            "// Balanced ratio - standard masking\n"
            "auto mask = _mm256_cmp_ps(data, threshold, _CMP_LT_OQ);\n"
            "auto result = _mm256_blendv_ps(else_val, then_val, mask);";
    }

    return analysis;
}

// ============================================================================
// Branch to SIMD conversion suggestions
// ============================================================================

std::vector<BranchToSIMDConversion> suggest_branch_to_simd(
    const BranchMetrics& metrics,
    const BranchPattern& pattern,
    int available_simd_width
) {
    std::vector<BranchToSIMDConversion> conversions;

    if (metrics.is_branchless) {
        return conversions;  // Already branchless
    }

    // Conditional assignment pattern
    {
        BranchToSIMDConversion conv;
        conv.can_convert = true;
        conv.scalar_pattern = "Conditional assignment: x = cond ? a : b";
        conv.simd_alternative = "SIMD blend with comparison mask";
        conv.code_before =
            "for (int i = 0; i < n; ++i) {\n"
            "    result[i] = data[i] > threshold ? a[i] : b[i];\n"
            "}";
        conv.code_after =
            "for (size_t i = 0; i < n; i += 8) {\n"
            "    auto d = _mm256_loadu_ps(&data[i]);\n"
            "    auto mask = _mm256_cmp_ps(d, thresh_v, _CMP_GT_OQ);\n"
            "    auto r = _mm256_blendv_ps(\n"
            "        _mm256_loadu_ps(&b[i]),\n"
            "        _mm256_loadu_ps(&a[i]), mask);\n"
            "    _mm256_storeu_ps(&result[i], r);\n"
            "}";
        conv.expected_speedup = 4.0;
        conversions.push_back(conv);
    }

    // Conditional increment pattern
    if (metrics.misprediction_rate > BranchThresholds::MISP_GOOD) {
        BranchToSIMDConversion conv;
        conv.can_convert = true;
        conv.scalar_pattern = "Conditional increment: if (cond) count++";
        conv.simd_alternative = "Masked addition";
        conv.code_before =
            "int count = 0;\n"
            "for (int i = 0; i < n; ++i) {\n"
            "    if (data[i] > threshold) count++;\n"
            "}";
        conv.code_after =
            "__m256i count_v = _mm256_setzero_si256();\n"
            "auto ones = _mm256_set1_epi32(1);\n"
            "for (size_t i = 0; i < n; i += 8) {\n"
            "    auto mask = _mm256_cmp_ps(data_v, thresh, _CMP_GT_OQ);\n"
            "    auto inc = _mm256_and_si256(\n"
            "        ones, _mm256_castps_si256(mask));\n"
            "    count_v = _mm256_add_epi32(count_v, inc);\n"
            "}\n"
            "// Horizontal sum at end";
        conv.expected_speedup = 6.0;
        conversions.push_back(conv);
    }

    // Min/max pattern
    {
        BranchToSIMDConversion conv;
        conv.can_convert = true;
        conv.scalar_pattern = "Min/max: if (a < b) min = a";
        conv.simd_alternative = "SIMD min/max intrinsics";
        conv.code_before =
            "for (int i = 0; i < n; ++i) {\n"
            "    if (a[i] < b[i]) result[i] = a[i];\n"
            "    else result[i] = b[i];\n"
            "}";
        conv.code_after =
            "for (size_t i = 0; i < n; i += 8) {\n"
            "    auto av = _mm256_loadu_ps(&a[i]);\n"
            "    auto bv = _mm256_loadu_ps(&b[i]);\n"
            "    auto r = _mm256_min_ps(av, bv);\n"
            "    _mm256_storeu_ps(&result[i], r);\n"
            "}";
        conv.expected_speedup = 8.0;
        conversions.push_back(conv);
    }

    // Clamp pattern
    {
        BranchToSIMDConversion conv;
        conv.can_convert = true;
        conv.scalar_pattern = "Clamp: x = max(min(x, hi), lo)";
        conv.simd_alternative = "Chained min/max";
        conv.code_before =
            "for (int i = 0; i < n; ++i) {\n"
            "    if (data[i] > hi) data[i] = hi;\n"
            "    else if (data[i] < lo) data[i] = lo;\n"
            "}";
        conv.code_after =
            "for (size_t i = 0; i < n; i += 8) {\n"
            "    auto v = _mm256_loadu_ps(&data[i]);\n"
            "    v = _mm256_min_ps(v, hi_v);\n"
            "    v = _mm256_max_ps(v, lo_v);\n"
            "    _mm256_storeu_ps(&data[i], v);\n"
            "}";
        conv.expected_speedup = 8.0;
        conversions.push_back(conv);
    }

    return conversions;
}

}  // namespace simd_bench
