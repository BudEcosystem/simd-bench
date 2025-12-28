#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>
#include <cstdint>

namespace simd_bench {

// Register pressure metrics
struct RegisterPressureMetrics {
    // Estimated register usage
    int estimated_gp_registers_used = 0;     // General purpose registers
    int estimated_simd_registers_used = 0;   // Vector registers (XMM/YMM/ZMM)
    int max_gp_registers = 16;               // Available GP registers
    int max_simd_registers = 16;             // Available SIMD registers (32 for AVX-512)

    // Spill indicators
    uint64_t register_spills = 0;            // Estimated spill count
    uint64_t register_fills = 0;             // Estimated fill count
    double spill_ratio = 0.0;                // Spills per 1000 instructions

    // Derived metrics
    double gp_pressure = 0.0;                // 0-1, pressure on GP registers
    double simd_pressure = 0.0;              // 0-1, pressure on SIMD registers
    bool has_register_pressure = false;      // True if pressure is significant

    // Analysis results
    int estimated_live_registers = 0;        // Peak live registers
    std::vector<std::string> pressure_sources;
    std::vector<std::string> reduction_suggestions;
};

// Register pressure analyzer
class RegisterPressureAnalyzer {
public:
    // Analyze register pressure from counter values
    static RegisterPressureMetrics analyze(
        uint64_t cycles,
        uint64_t instructions,
        uint64_t l1d_read_hits,           // L1D hits (may indicate register reuse)
        uint64_t l1d_read_misses,
        uint64_t stack_references,        // Stack memory references (spill indicator)
        int max_simd_width = 256          // AVX2 by default
    );

    // Analyze from memory access patterns
    static RegisterPressureMetrics analyze_from_memory(
        uint64_t mem_loads,
        uint64_t mem_stores,
        uint64_t instructions,
        double ipc,
        int max_simd_width = 256
    );

    // Estimate register pressure from code characteristics
    static RegisterPressureMetrics estimate_pressure(
        int loop_carried_dependencies,    // Variables carried across iterations
        int temporary_values,             // Intermediate computation values
        int array_pointers,               // Pointers to arrays
        int constants,                    // Constant values
        bool uses_fma = true,
        int simd_width = 256
    );

    // Check if SIMD register pressure is limiting vectorization
    static bool is_simd_pressure_limiting(
        const RegisterPressureMetrics& metrics,
        double vectorization_ratio
    );

    // Suggest register pressure reduction techniques
    static std::vector<std::string> suggest_reductions(
        const RegisterPressureMetrics& metrics,
        bool is_simd_code = true
    );
};

// Register usage estimation from loop analysis
struct LoopRegisterEstimate {
    int induction_variables = 1;         // Loop counters
    int array_pointers = 0;              // Base pointers
    int accumulators = 0;                // Reduction variables
    int temporaries = 0;                 // Intermediate values
    int constants = 0;                   // Loop-invariant values
    int total_gp = 0;
    int total_simd = 0;
    std::string limiting_factor;         // "gp", "simd", or "none"
};

// Estimate register usage for a loop
LoopRegisterEstimate estimate_loop_registers(
    int num_arrays,                      // Number of input/output arrays
    int ops_per_iteration,               // Operations per iteration
    int reduction_vars,                  // Reduction variables (sum, max, etc.)
    int unroll_factor,                   // Loop unroll factor
    bool uses_fma = true
);

// AVX-512 register pressure (32 registers available)
struct AVX512RegisterAnalysis {
    bool benefits_from_avx512_regs = false;  // Would 32 regs help?
    int estimated_spill_reduction = 0;        // Spills saved with AVX-512
    double expected_speedup_from_regs = 0.0;  // Speedup from more registers
    std::string recommendation;
};

AVX512RegisterAnalysis analyze_avx512_register_benefit(
    const RegisterPressureMetrics& current_metrics,
    int current_simd_width = 256
);

// Register pressure thresholds
struct RegisterPressureThresholds {
    // Pressure thresholds (0-1)
    static constexpr double LOW_PRESSURE = 0.5;
    static constexpr double MODERATE_PRESSURE = 0.75;
    static constexpr double HIGH_PRESSURE = 0.9;
    static constexpr double CRITICAL_PRESSURE = 1.0;

    // Spill rate thresholds (spills per 1000 instructions)
    static constexpr double ACCEPTABLE_SPILL_RATE = 5.0;
    static constexpr double HIGH_SPILL_RATE = 20.0;
    static constexpr double CRITICAL_SPILL_RATE = 50.0;

    // Register counts
    static constexpr int X86_64_GP_REGS = 16;
    static constexpr int AVX2_SIMD_REGS = 16;
    static constexpr int AVX512_SIMD_REGS = 32;
};

}  // namespace simd_bench
