#pragma once

#include "types.h"
#include "kernel_registry.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <filesystem>
#include <chrono>

namespace simd_bench {

// Vectorization status from compiler
enum class VectorizationStatus {
    UNKNOWN,
    VECTORIZED,           // Loop was vectorized
    NOT_VECTORIZED,       // Loop was not vectorized
    PARTIALLY_VECTORIZED, // Only part of the loop vectorized
    INTERLEAVED,          // Loop was interleaved
    UNROLLED,             // Loop was unrolled but not vectorized
    ALREADY_VECTORIZED    // Contains intrinsics, compiler didn't change
};

// Reason why vectorization failed
enum class VectorizationBlocker {
    NONE,
    UNKNOWN_REASON,
    DATA_DEPENDENCY,           // Loop-carried dependency
    CONTROL_FLOW,              // Complex control flow
    NON_CONTIGUOUS_ACCESS,     // Scattered/gather access pattern
    ALIASING,                  // Pointer aliasing
    MIXED_TYPES,               // Mixed data types
    FUNCTION_CALL,             // Call to non-inlinable function
    REDUCTION_CHAIN,           // Complex reduction
    SMALL_TRIP_COUNT,          // Loop trip count too small
    ALIGNMENT_UNKNOWN,         // Unknown alignment
    COST_MODEL,                // Cost model says not profitable
    USER_DIRECTIVE,            // User disabled vectorization
    UNSUPPORTED_OPERATION,     // Operation not supported in SIMD
    INDIRECT_MEMORY_ACCESS,    // Indirect array access
    EXCEPTION_HANDLING         // Exception handling in loop
};

// Single loop analysis result
struct LoopVectorizationInfo {
    std::string source_location;        // File:line
    int loop_line = 0;
    VectorizationStatus status = VectorizationStatus::UNKNOWN;
    VectorizationBlocker blocker = VectorizationBlocker::NONE;
    std::string blocker_description;

    int vector_width = 0;               // Width used (128, 256, 512)
    int unroll_factor = 0;
    int interleave_factor = 0;

    std::optional<double> estimated_speedup;
    std::string compiler_message;       // Raw compiler output

    bool is_critical = false;           // In hot path
};

// Compiler vectorization report
struct VectorizationReport {
    std::string source_file;
    std::string compiler;               // "gcc", "clang", "icc"
    std::string compiler_version;
    std::string optimization_flags;

    std::vector<LoopVectorizationInfo> loops;

    // Summary statistics
    int total_loops = 0;
    int vectorized_loops = 0;
    int failed_loops = 0;
    double vectorization_rate = 0.0;

    // Blocker breakdown
    std::unordered_map<VectorizationBlocker, int> blocker_counts;

    std::vector<std::string> raw_output;  // Full compiler output
};

// Performance comparison between auto-vectorized and intrinsic versions
struct AutoVecComparison {
    std::string kernel_name;
    size_t problem_size;

    // Auto-vectorized version
    double autovec_gflops = 0.0;
    double autovec_elapsed_seconds = 0.0;
    bool autovec_compiled = false;
    std::string autovec_error;

    // Intrinsic/manual version
    double intrinsic_gflops = 0.0;
    double intrinsic_elapsed_seconds = 0.0;

    // Comparison
    double speedup_intrinsic_vs_auto = 0.0;  // >1 means intrinsic is faster
    double efficiency_ratio = 0.0;           // intrinsic_gflops / autovec_gflops

    // Vectorization info
    VectorizationReport autovec_report;

    // Recommendations
    std::vector<std::string> recommendations;
};

// Complete auto-vectorization analysis result
struct AutoVecAnalysis {
    bool compiler_vectorized = false;
    VectorizationReport compiler_report;

    double auto_vec_speedup = 0.0;      // vs scalar
    double intrinsic_speedup = 0.0;     // vs scalar
    double intrinsic_vs_autovec = 0.0;  // intrinsic / autovec

    std::vector<std::string> missed_optimizations;
    std::vector<std::string> vectorization_blockers;
    std::vector<std::string> suggestions;

    // Per-loop analysis
    std::vector<LoopVectorizationInfo> critical_loops;
};

// Compiler configuration for auto-vec testing
struct CompilerConfig {
    std::string compiler_path;          // Path to compiler
    std::string compiler_id;            // "gcc", "clang", "icc"
    std::vector<std::string> base_flags;
    std::vector<std::string> autovec_flags;
    std::vector<std::string> no_vec_flags;
    std::vector<std::string> report_flags;
    std::string target_arch;            // "-march=native"

    // Default configurations
    static CompilerConfig gcc_default();
    static CompilerConfig clang_default();
    static CompilerConfig icc_default();
    static CompilerConfig detect_system_compiler();
};

// Source code snippet for testing
struct SourceSnippet {
    std::string code;
    std::string function_name;
    std::string includes;
    std::vector<std::string> dependencies;  // Other source files needed
};

// Auto-vectorization analyzer
class AutoVecAnalyzer {
public:
    AutoVecAnalyzer();
    explicit AutoVecAnalyzer(const CompilerConfig& config);
    ~AutoVecAnalyzer();

    // Set compiler configuration
    void set_compiler(const CompilerConfig& config);

    // Get current compiler info
    const CompilerConfig& get_compiler() const { return config_; }

    // Analyze a source file
    VectorizationReport analyze_source(const std::string& source_path);

    // Analyze a code snippet
    VectorizationReport analyze_snippet(const SourceSnippet& snippet);

    // Compare auto-vectorized vs intrinsic kernel
    AutoVecComparison compare_with_intrinsic(
        const std::string& source_file,
        const KernelConfig& intrinsic_kernel,
        size_t problem_size,
        size_t iterations = 100
    );

    // Full analysis with comparison
    AutoVecAnalysis full_analysis(
        const std::string& source_file,
        const KernelConfig& intrinsic_kernel,
        const std::vector<size_t>& sizes = {}
    );

    // Parse compiler optimization report
    VectorizationReport parse_compiler_output(
        const std::string& compiler_output,
        const std::string& source_file
    );

    // Generate code fix suggestions
    std::vector<std::string> generate_suggestions(
        const VectorizationReport& report
    );

    // Compile and run a kernel from source
    std::optional<double> compile_and_benchmark(
        const std::string& source_file,
        const std::string& function_name,
        size_t problem_size,
        size_t iterations,
        bool enable_autovec = true
    );

    // Check if compiler supports vectorization reports
    bool supports_vec_reports() const;

    // Get temporary directory for compilation
    std::filesystem::path get_temp_dir() const;

private:
    CompilerConfig config_;
    std::filesystem::path temp_dir_;
    bool initialized_ = false;

    // Internal methods
    bool initialize();
    void cleanup();

    bool compile_source(
        const std::string& source_file,
        const std::string& output_file,
        const std::vector<std::string>& extra_flags,
        std::string& compiler_output
    );

    VectorizationReport parse_gcc_output(
        const std::string& output,
        const std::string& source_file
    );

    VectorizationReport parse_clang_output(
        const std::string& output,
        const std::string& source_file
    );

    VectorizationReport parse_icc_output(
        const std::string& output,
        const std::string& source_file
    );

    VectorizationBlocker classify_blocker(const std::string& message);

    std::string generate_wrapper_source(
        const std::string& original_source,
        const std::string& function_name
    );
};

// Utility functions
std::string vectorization_status_to_string(VectorizationStatus status);
std::string vectorization_blocker_to_string(VectorizationBlocker blocker);

// Generate a summary report
std::string generate_autovec_summary(const AutoVecAnalysis& analysis);
std::string generate_autovec_markdown(const AutoVecAnalysis& analysis);

// Vectorization hint annotations (for source code generation)
struct VectorizationHint {
    enum class Type {
        ALIGN,              // __attribute__((aligned(N)))
        ASSUME_ALIGNED,     // __builtin_assume_aligned
        RESTRICT,           // __restrict
        IVDEP,              // #pragma ivdep
        VECTOR_ALWAYS,      // #pragma omp simd / #pragma vector always
        NONTEMPORAL,        // Streaming stores
        PREFETCH            // Prefetch hint
    };

    Type type;
    std::string code_before;
    std::string code_after;
    std::string explanation;
    int estimated_impact;   // 0-100
};

std::vector<VectorizationHint> suggest_hints(
    const LoopVectorizationInfo& loop,
    const std::string& source_code
);

// Check if a function contains intrinsics
bool contains_simd_intrinsics(const std::string& source_code);

// Detect ISA level used in source (SSE, AVX, AVX-512, etc.)
std::string detect_isa_level(const std::string& source_code);

}  // namespace simd_bench
