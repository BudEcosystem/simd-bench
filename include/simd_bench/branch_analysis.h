#pragma once

#include "types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace simd_bench {

// Branch behavior metrics
struct BranchMetrics {
    // Raw counts
    uint64_t conditional_branches = 0;    // Total conditional branches
    uint64_t unconditional_branches = 0;  // Jump, call, ret
    uint64_t mispredictions = 0;          // Branch mispredictions
    uint64_t total_instructions = 0;      // For normalization

    // Derived metrics
    double branch_density = 0.0;          // Branches per 1000 instructions
    double misprediction_rate = 0.0;      // Mispredictions / conditional branches
    double misprediction_density = 0.0;   // Mispredictions per 1000 instructions

    // Classification
    bool is_branchless = false;           // < 5 branches per 1000 instructions
    bool has_branch_issues = false;       // Misprediction rate > 5%

    // SIMD-friendliness indicators
    double vectorization_friendliness = 0.0;  // 0-1, higher = more SIMD-friendly
    std::vector<std::string> branchless_opportunities;
    std::vector<std::string> recommendations;
};

// Branch pattern analysis
class BranchAnalyzer {
public:
    // Analyze branch behavior from counter values
    static BranchMetrics analyze(
        uint64_t branch_instructions,      // BRANCH_INSTRUCTIONS
        uint64_t branch_misses,            // BRANCH_MISSES
        uint64_t total_instructions        // INSTRUCTIONS
    );

    // Enhanced analysis with conditional branch breakdown
    static BranchMetrics analyze_detailed(
        uint64_t conditional_branches,     // BR_INST_RETIRED.CONDITIONAL
        uint64_t unconditional_branches,   // BR_INST_RETIRED.NEAR_CALL + NEAR_RETURN
        uint64_t mispredictions,           // BR_MISP_RETIRED.ALL_BRANCHES
        uint64_t total_instructions
    );

    // Detect vectorization blockers from branch patterns
    static std::vector<std::string> detect_vectorization_blockers(
        const BranchMetrics& metrics,
        double vectorization_ratio         // From SIMD metrics
    );

    // Check if loop is vectorization-friendly based on branches
    static bool is_vectorization_friendly(const BranchMetrics& metrics);

    // Suggest branchless alternatives
    static std::vector<std::string> suggest_branchless_alternatives(
        const BranchMetrics& metrics,
        bool has_floating_point = true
    );
};

// Branch pattern types
enum class BranchPatternType {
    BRANCHLESS,          // Almost no branches, ideal for SIMD
    PREDICTABLE,         // Regular pattern, branch predictor effective
    DATA_DEPENDENT,      // Branches depend on data, hard to predict
    LOOP_BOUND,          // Branches mainly for loop control
    EARLY_EXIT,          // Early termination pattern
    ERROR_CHECK,         // Rare branch for error handling
    MIXED                // Mixed pattern
};

// Detailed branch pattern info
struct BranchPattern {
    BranchPatternType type = BranchPatternType::MIXED;
    double confidence = 0.0;              // Confidence in pattern detection
    double simd_compatibility = 0.0;      // 0-1, compatibility with SIMD
    std::string description;
    std::string optimization_strategy;
};

// Detect branch pattern from metrics
BranchPattern detect_branch_pattern(
    const BranchMetrics& metrics,
    double loop_trip_count_estimate = 0.0
);

// SIMD masking analysis
struct SIMDMaskingAnalysis {
    bool masking_beneficial = false;      // Would predicated SIMD help?
    double expected_speedup = 0.0;        // Expected speedup from masking
    double lane_utilization = 0.0;        // Expected average lane utilization
    std::string recommended_approach;     // "blend", "compress", "scatter-gather"
    std::string code_example;
};

// Analyze if SIMD masking would help
SIMDMaskingAnalysis analyze_simd_masking(
    const BranchMetrics& metrics,
    double true_ratio,                    // Fraction of time branch is taken
    int simd_width_bits = 256
);

// Branch to SIMD conversion suggestions
struct BranchToSIMDConversion {
    bool can_convert = false;
    std::string scalar_pattern;           // Detected scalar pattern
    std::string simd_alternative;         // Suggested SIMD approach
    std::string code_before;              // Example scalar code
    std::string code_after;               // Example SIMD code
    double expected_speedup = 0.0;
};

// Suggest branch-to-SIMD conversions
std::vector<BranchToSIMDConversion> suggest_branch_to_simd(
    const BranchMetrics& metrics,
    const BranchPattern& pattern,
    int available_simd_width = 256       // AVX2 default
);

// Branch density thresholds
struct BranchThresholds {
    // Branches per 1000 instructions
    static constexpr double BRANCHLESS = 5.0;      // < 5 = branchless
    static constexpr double LOW = 20.0;            // < 20 = low branch density
    static constexpr double MODERATE = 50.0;       // < 50 = moderate
    static constexpr double HIGH = 100.0;          // >= 100 = branch-heavy

    // Misprediction rate thresholds
    static constexpr double MISP_EXCELLENT = 0.01; // < 1% = excellent
    static constexpr double MISP_GOOD = 0.03;      // < 3% = good
    static constexpr double MISP_ACCEPTABLE = 0.05;// < 5% = acceptable
    static constexpr double MISP_POOR = 0.10;      // >= 10% = poor

    // SIMD friendliness thresholds
    static constexpr double SIMD_FRIENDLY = 0.8;   // > 80% = SIMD-friendly
    static constexpr double SIMD_POSSIBLE = 0.5;   // > 50% = possible with masking
};

}  // namespace simd_bench
