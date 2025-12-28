#include "simd_bench/register_pressure.h"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace simd_bench {

// ============================================================================
// RegisterPressureAnalyzer implementation
// ============================================================================

RegisterPressureMetrics RegisterPressureAnalyzer::analyze(
    uint64_t cycles,
    uint64_t instructions,
    uint64_t l1d_read_hits,
    uint64_t l1d_read_misses,
    uint64_t stack_references,
    int max_simd_width
) {
    RegisterPressureMetrics metrics;

    // Set max registers based on SIMD width
    metrics.max_simd_registers = (max_simd_width >= 512) ?
        RegisterPressureThresholds::AVX512_SIMD_REGS :
        RegisterPressureThresholds::AVX2_SIMD_REGS;
    metrics.max_gp_registers = RegisterPressureThresholds::X86_64_GP_REGS;

    // Estimate spills from stack references
    // Stack references that aren't function calls are likely spills
    if (instructions > 0) {
        // Rough heuristic: stack refs beyond expected call overhead
        uint64_t expected_stack_refs = instructions / 100;  // ~1% for calls
        if (stack_references > expected_stack_refs) {
            metrics.register_spills = (stack_references - expected_stack_refs) / 2;
            metrics.register_fills = metrics.register_spills;
        }

        metrics.spill_ratio =
            (static_cast<double>(metrics.register_spills) / instructions) * 1000.0;
    }

    // Estimate register pressure from IPC and memory patterns
    if (instructions > 0 && cycles > 0) {
        double ipc = static_cast<double>(instructions) / cycles;

        // Low IPC with high L1 hit rate suggests register pressure
        double l1_hit_rate = (l1d_read_hits + l1d_read_misses > 0) ?
            static_cast<double>(l1d_read_hits) / (l1d_read_hits + l1d_read_misses) : 0.0;

        if (ipc < 1.5 && l1_hit_rate > 0.95) {
            // High L1 hit rate but low IPC = likely spilling to L1
            metrics.has_register_pressure = true;
            metrics.simd_pressure = std::min(1.0, (2.0 - ipc) / 1.5);
        }
    }

    // Estimate from spill ratio
    if (metrics.spill_ratio > RegisterPressureThresholds::HIGH_SPILL_RATE) {
        metrics.has_register_pressure = true;
        metrics.simd_pressure = std::max(metrics.simd_pressure, 0.8);
        metrics.gp_pressure = 0.6;
    } else if (metrics.spill_ratio > RegisterPressureThresholds::ACCEPTABLE_SPILL_RATE) {
        metrics.simd_pressure = std::max(metrics.simd_pressure, 0.5);
        metrics.gp_pressure = 0.4;
    }

    // Generate suggestions
    metrics.reduction_suggestions = suggest_reductions(metrics, true);

    return metrics;
}

RegisterPressureMetrics RegisterPressureAnalyzer::analyze_from_memory(
    uint64_t mem_loads,
    uint64_t mem_stores,
    uint64_t instructions,
    double ipc,
    int max_simd_width
) {
    RegisterPressureMetrics metrics;

    metrics.max_simd_registers = (max_simd_width >= 512) ?
        RegisterPressureThresholds::AVX512_SIMD_REGS :
        RegisterPressureThresholds::AVX2_SIMD_REGS;
    metrics.max_gp_registers = RegisterPressureThresholds::X86_64_GP_REGS;

    if (instructions == 0) return metrics;

    // Memory operations per instruction
    double mem_ratio = static_cast<double>(mem_loads + mem_stores) / instructions;

    // High memory ratio with low IPC suggests spilling
    if (mem_ratio > 0.5 && ipc < 2.0) {
        // Estimate spill operations (paired loads/stores)
        uint64_t paired_ops = std::min(mem_loads, mem_stores);
        double excess_ratio = mem_ratio - 0.3;  // Expected ratio for compute code

        if (excess_ratio > 0) {
            metrics.register_spills = static_cast<uint64_t>(instructions * excess_ratio * 0.3);
            metrics.register_fills = metrics.register_spills;
            metrics.spill_ratio = excess_ratio * 300;  // Per 1000 instructions
        }
    }

    // Pressure estimation
    metrics.simd_pressure = std::min(1.0, mem_ratio * ipc / 3.0);
    metrics.gp_pressure = std::min(1.0, mem_ratio * 0.5);

    metrics.has_register_pressure =
        (metrics.simd_pressure > RegisterPressureThresholds::MODERATE_PRESSURE) ||
        (metrics.spill_ratio > RegisterPressureThresholds::ACCEPTABLE_SPILL_RATE);

    metrics.reduction_suggestions = suggest_reductions(metrics, true);

    return metrics;
}

RegisterPressureMetrics RegisterPressureAnalyzer::estimate_pressure(
    int loop_carried_dependencies,
    int temporary_values,
    int array_pointers,
    int constants,
    bool uses_fma,
    int simd_width
) {
    RegisterPressureMetrics metrics;

    metrics.max_simd_registers = (simd_width >= 512) ? 32 : 16;
    metrics.max_gp_registers = 16;

    // Estimate GP register usage
    int gp_used = 0;
    gp_used += array_pointers;                    // Base pointers
    gp_used += 2;                                  // Loop counter + limit
    gp_used += std::min(4, constants);            // Some constants in GP regs
    gp_used += 2;                                  // Stack pointer, return address

    metrics.estimated_gp_registers_used = gp_used;
    metrics.gp_pressure = static_cast<double>(gp_used) / metrics.max_gp_registers;

    // Estimate SIMD register usage
    int simd_used = 0;
    simd_used += loop_carried_dependencies;       // Accumulators
    simd_used += temporary_values;                // Intermediates
    simd_used += array_pointers;                  // Current vector values
    if (uses_fma) {
        simd_used += temporary_values / 2;        // FMA needs operands simultaneously
    }
    simd_used += std::min(4, constants);          // Broadcast constants

    metrics.estimated_simd_registers_used = simd_used;
    metrics.simd_pressure = static_cast<double>(simd_used) / metrics.max_simd_registers;

    metrics.estimated_live_registers = simd_used;

    // Determine if pressure is problematic
    metrics.has_register_pressure =
        (metrics.simd_pressure > RegisterPressureThresholds::MODERATE_PRESSURE) ||
        (metrics.gp_pressure > RegisterPressureThresholds::HIGH_PRESSURE);

    // Identify pressure sources
    if (metrics.simd_pressure > 0.8) {
        if (temporary_values > 8) {
            metrics.pressure_sources.push_back("High temporary value count");
        }
        if (loop_carried_dependencies > 4) {
            metrics.pressure_sources.push_back("Many loop-carried dependencies");
        }
        if (array_pointers > 6) {
            metrics.pressure_sources.push_back("Many array operands");
        }
    }

    metrics.reduction_suggestions = suggest_reductions(metrics, true);

    return metrics;
}

bool RegisterPressureAnalyzer::is_simd_pressure_limiting(
    const RegisterPressureMetrics& metrics,
    double vectorization_ratio
) {
    // High register pressure with low vectorization suggests registers are the issue
    if (metrics.simd_pressure > RegisterPressureThresholds::HIGH_PRESSURE &&
        vectorization_ratio < 0.7) {
        return true;
    }

    // High spill ratio with moderate vectorization
    if (metrics.spill_ratio > RegisterPressureThresholds::HIGH_SPILL_RATE &&
        vectorization_ratio < 0.9) {
        return true;
    }

    return false;
}

std::vector<std::string> RegisterPressureAnalyzer::suggest_reductions(
    const RegisterPressureMetrics& metrics,
    bool is_simd_code
) {
    std::vector<std::string> suggestions;

    if (!metrics.has_register_pressure) {
        suggestions.push_back("Register pressure is not a significant issue");
        return suggestions;
    }

    // SIMD-specific suggestions
    if (is_simd_code && metrics.simd_pressure > RegisterPressureThresholds::MODERATE_PRESSURE) {
        suggestions.push_back(
            "Reduce loop unroll factor to decrease live register count");

        if (metrics.estimated_simd_registers_used > 12) {
            suggestions.push_back(
                "Split loop into multiple passes to reduce simultaneous live values");
        }

        suggestions.push_back(
            "Recompute values instead of storing in registers if computation is cheap");

        if (metrics.max_simd_registers == 16) {
            suggestions.push_back(
                "Consider AVX-512 which provides 32 SIMD registers");
        }
    }

    // General suggestions
    if (metrics.gp_pressure > RegisterPressureThresholds::MODERATE_PRESSURE) {
        suggestions.push_back(
            "Reduce number of array pointers by using index calculations");

        suggestions.push_back(
            "Move loop-invariant calculations outside the loop");
    }

    if (metrics.spill_ratio > RegisterPressureThresholds::HIGH_SPILL_RATE) {
        suggestions.push_back(
            "High spill rate detected - consider smaller working set per iteration");

        suggestions.push_back(
            "Use compiler flags like -O3 -march=native for better register allocation");
    }

    return suggestions;
}

// ============================================================================
// Loop register estimation
// ============================================================================

LoopRegisterEstimate estimate_loop_registers(
    int num_arrays,
    int ops_per_iteration,
    int reduction_vars,
    int unroll_factor,
    bool uses_fma
) {
    LoopRegisterEstimate est;

    // GP registers
    est.induction_variables = 1 + (unroll_factor > 1 ? 1 : 0);  // Counter + optional step
    est.array_pointers = num_arrays;

    // SIMD registers
    est.accumulators = reduction_vars * unroll_factor;

    // Temporaries depend on operation count and FMA usage
    if (uses_fma) {
        // FMA: a = a + b * c, needs 3 operands but can reuse
        est.temporaries = (ops_per_iteration * unroll_factor + 1) / 2;
    } else {
        est.temporaries = ops_per_iteration * unroll_factor;
    }

    // Constants (typically 2-4 for common operations)
    est.constants = std::min(4, 1 + (uses_fma ? 1 : 0));

    // Totals
    est.total_gp = est.induction_variables + est.array_pointers + 2;  // +2 for stack/return
    est.total_simd = est.accumulators + est.temporaries + est.constants + num_arrays;

    // Determine limiting factor
    double gp_usage = static_cast<double>(est.total_gp) / RegisterPressureThresholds::X86_64_GP_REGS;
    double simd_usage = static_cast<double>(est.total_simd) / RegisterPressureThresholds::AVX2_SIMD_REGS;

    if (simd_usage > gp_usage && simd_usage > 0.8) {
        est.limiting_factor = "simd";
    } else if (gp_usage > 0.8) {
        est.limiting_factor = "gp";
    } else {
        est.limiting_factor = "none";
    }

    return est;
}

// ============================================================================
// AVX-512 register benefit analysis
// ============================================================================

AVX512RegisterAnalysis analyze_avx512_register_benefit(
    const RegisterPressureMetrics& current_metrics,
    int current_simd_width
) {
    AVX512RegisterAnalysis analysis;

    // Only relevant if currently using AVX2 or less
    if (current_simd_width >= 512) {
        analysis.benefits_from_avx512_regs = false;
        analysis.recommendation = "Already using AVX-512 registers";
        return analysis;
    }

    // Check if current pressure would benefit from more registers
    if (current_metrics.simd_pressure > RegisterPressureThresholds::MODERATE_PRESSURE) {
        // Calculate how many registers would be freed
        int current_max = RegisterPressureThresholds::AVX2_SIMD_REGS;
        int avx512_max = RegisterPressureThresholds::AVX512_SIMD_REGS;

        int current_used = current_metrics.estimated_simd_registers_used;

        // If current usage would fit comfortably in 32 regs
        if (current_used > current_max * 0.7 && current_used < avx512_max * 0.8) {
            analysis.benefits_from_avx512_regs = true;

            // Estimate spill reduction
            analysis.estimated_spill_reduction =
                static_cast<int>(current_metrics.register_spills *
                (1.0 - current_metrics.simd_pressure));

            // Estimate speedup (rough: 2-5% per eliminated spill pair)
            double spill_overhead = current_metrics.spill_ratio * 0.003;  // ~3 cycles per spill
            analysis.expected_speedup_from_regs = spill_overhead;

            analysis.recommendation =
                "AVX-512's 32 registers would reduce register pressure from " +
                std::to_string(static_cast<int>(current_metrics.simd_pressure * 100)) +
                "% to ~" +
                std::to_string(static_cast<int>(current_used * 100.0 / avx512_max)) +
                "%";
        } else {
            analysis.benefits_from_avx512_regs = false;
            analysis.recommendation =
                "Register pressure is too high - consider algorithm restructuring";
        }
    } else {
        analysis.benefits_from_avx512_regs = false;
        analysis.recommendation =
            "Current register pressure is acceptable - AVX-512 registers not critical";
    }

    return analysis;
}

}  // namespace simd_bench
