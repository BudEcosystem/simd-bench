#include "simd_bench/autovec.h"
#include "simd_bench/timing.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#include <sys/wait.h>
#include <dlfcn.h>
#endif

namespace simd_bench {

namespace {

// Execute a command and capture output
std::pair<int, std::string> execute_command(const std::string& cmd) {
    std::string output;
    std::array<char, 4096> buffer;

    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        return {-1, "Failed to execute command"};
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }

    int status = pclose(pipe);
#ifndef _WIN32
    if (WIFEXITED(status)) {
        status = WEXITSTATUS(status);
    }
#endif

    return {status, output};
}

// Check if a compiler exists
bool compiler_exists(const std::string& compiler) {
    auto [status, output] = execute_command(compiler + " --version");
    return status == 0;
}

// Get compiler version
std::string get_compiler_version(const std::string& compiler) {
    auto [status, output] = execute_command(compiler + " --version");
    if (status != 0) return "unknown";

    // Extract first line
    size_t newline = output.find('\n');
    if (newline != std::string::npos) {
        return output.substr(0, newline);
    }
    return output;
}

// Generate unique temp filename
std::string generate_temp_filename(const std::string& prefix, const std::string& suffix) {
    static int counter = 0;
    std::ostringstream oss;
    oss << prefix << "_" << getpid() << "_" << (++counter) << suffix;
    return oss.str();
}

// Read file contents
std::string read_file_contents(const std::string& path) {
    std::ifstream file(path);
    if (!file) return "";

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

// Write file contents
bool write_file_contents(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file) return false;

    file << content;
    return file.good();
}

}  // anonymous namespace

// ============================================================================
// CompilerConfig Implementation
// ============================================================================

CompilerConfig CompilerConfig::gcc_default() {
    CompilerConfig config;
    config.compiler_path = "g++";
    config.compiler_id = "gcc";
    config.base_flags = {"-std=c++17", "-O3", "-DNDEBUG"};
    config.autovec_flags = {"-ftree-vectorize", "-ffast-math"};
    config.no_vec_flags = {"-fno-tree-vectorize"};
    config.report_flags = {
        "-fopt-info-vec-all",
        "-fopt-info-loop-all"
    };
    config.target_arch = "-march=native";
    return config;
}

CompilerConfig CompilerConfig::clang_default() {
    CompilerConfig config;
    config.compiler_path = "clang++";
    config.compiler_id = "clang";
    config.base_flags = {"-std=c++17", "-O3", "-DNDEBUG"};
    config.autovec_flags = {"-fvectorize", "-ffast-math"};
    config.no_vec_flags = {"-fno-vectorize", "-fno-slp-vectorize"};
    config.report_flags = {
        "-Rpass=loop-vectorize",
        "-Rpass-missed=loop-vectorize",
        "-Rpass-analysis=loop-vectorize"
    };
    config.target_arch = "-march=native";
    return config;
}

CompilerConfig CompilerConfig::icc_default() {
    CompilerConfig config;
    config.compiler_path = "icpc";
    config.compiler_id = "icc";
    config.base_flags = {"-std=c++17", "-O3", "-DNDEBUG"};
    config.autovec_flags = {"-xHost", "-qopt-report=5"};
    config.no_vec_flags = {"-no-vec"};
    config.report_flags = {"-qopt-report=5", "-qopt-report-phase=vec"};
    config.target_arch = "-xHost";
    return config;
}

CompilerConfig CompilerConfig::detect_system_compiler() {
    // Try compilers in order of preference
    if (compiler_exists("g++")) {
        auto config = gcc_default();
        config.compiler_path = "g++";
        return config;
    }
    if (compiler_exists("clang++")) {
        auto config = clang_default();
        config.compiler_path = "clang++";
        return config;
    }
    if (compiler_exists("icpc")) {
        auto config = icc_default();
        config.compiler_path = "icpc";
        return config;
    }
    if (compiler_exists("icpx")) {
        auto config = icc_default();
        config.compiler_path = "icpx";
        config.compiler_id = "icpx";
        return config;
    }

    // Fallback to GCC config
    return gcc_default();
}

// ============================================================================
// AutoVecAnalyzer Implementation
// ============================================================================

AutoVecAnalyzer::AutoVecAnalyzer()
    : config_(CompilerConfig::detect_system_compiler()) {
    initialize();
}

AutoVecAnalyzer::AutoVecAnalyzer(const CompilerConfig& config)
    : config_(config) {
    initialize();
}

AutoVecAnalyzer::~AutoVecAnalyzer() {
    cleanup();
}

bool AutoVecAnalyzer::initialize() {
    if (initialized_) return true;

    // Create temporary directory
    temp_dir_ = std::filesystem::temp_directory_path() / "simd_bench_autovec";
    std::error_code ec;
    std::filesystem::create_directories(temp_dir_, ec);
    if (ec) {
        return false;
    }

    initialized_ = true;
    return true;
}

void AutoVecAnalyzer::cleanup() {
    if (!initialized_) return;

    // Clean up temp directory
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
    initialized_ = false;
}

void AutoVecAnalyzer::set_compiler(const CompilerConfig& config) {
    config_ = config;
}

bool AutoVecAnalyzer::supports_vec_reports() const {
    return config_.compiler_id == "gcc" ||
           config_.compiler_id == "clang" ||
           config_.compiler_id == "icc" ||
           config_.compiler_id == "icpx";
}

std::filesystem::path AutoVecAnalyzer::get_temp_dir() const {
    return temp_dir_;
}

bool AutoVecAnalyzer::compile_source(
    const std::string& source_file,
    const std::string& output_file,
    const std::vector<std::string>& extra_flags,
    std::string& compiler_output
) {
    std::ostringstream cmd;
    cmd << config_.compiler_path;

    // Add base flags
    for (const auto& flag : config_.base_flags) {
        cmd << " " << flag;
    }

    // Add target arch
    if (!config_.target_arch.empty()) {
        cmd << " " << config_.target_arch;
    }

    // Add extra flags
    for (const auto& flag : extra_flags) {
        cmd << " " << flag;
    }

    // Add output and input
    cmd << " -o " << output_file << " " << source_file;

    auto [status, output] = execute_command(cmd.str());
    compiler_output = output;
    return status == 0;
}

VectorizationReport AutoVecAnalyzer::analyze_source(const std::string& source_path) {
    VectorizationReport report;
    report.source_file = source_path;
    report.compiler = config_.compiler_id;
    report.compiler_version = get_compiler_version(config_.compiler_path);

    // Build optimization flags string
    std::ostringstream flags;
    for (const auto& flag : config_.base_flags) flags << flag << " ";
    for (const auto& flag : config_.autovec_flags) flags << flag << " ";
    flags << config_.target_arch;
    report.optimization_flags = flags.str();

    // Compile with vectorization reports
    std::vector<std::string> compile_flags = config_.autovec_flags;
    for (const auto& flag : config_.report_flags) {
        compile_flags.push_back(flag);
    }

    std::string temp_output = (temp_dir_ / generate_temp_filename("autovec", ".o")).string();
    compile_flags.push_back("-c");  // Compile only

    std::string compiler_output;
    compile_source(source_path, temp_output, compile_flags, compiler_output);

    report.raw_output.push_back(compiler_output);

    // Parse the output based on compiler
    if (config_.compiler_id == "gcc") {
        report = parse_gcc_output(compiler_output, source_path);
    } else if (config_.compiler_id == "clang") {
        report = parse_clang_output(compiler_output, source_path);
    } else if (config_.compiler_id == "icc" || config_.compiler_id == "icpx") {
        report = parse_icc_output(compiler_output, source_path);
    }

    // Restore metadata
    report.source_file = source_path;
    report.compiler = config_.compiler_id;
    report.compiler_version = get_compiler_version(config_.compiler_path);
    report.optimization_flags = flags.str();
    report.raw_output.push_back(compiler_output);

    // Calculate summary
    report.total_loops = static_cast<int>(report.loops.size());
    report.vectorized_loops = 0;
    report.failed_loops = 0;

    for (const auto& loop : report.loops) {
        if (loop.status == VectorizationStatus::VECTORIZED ||
            loop.status == VectorizationStatus::INTERLEAVED) {
            report.vectorized_loops++;
        } else if (loop.status == VectorizationStatus::NOT_VECTORIZED) {
            report.failed_loops++;
            report.blocker_counts[loop.blocker]++;
        }
    }

    if (report.total_loops > 0) {
        report.vectorization_rate =
            static_cast<double>(report.vectorized_loops) / report.total_loops;
    }

    // Cleanup temp file
    std::filesystem::remove(temp_output);

    return report;
}

VectorizationReport AutoVecAnalyzer::analyze_snippet(const SourceSnippet& snippet) {
    // Write snippet to temp file
    std::string temp_source = (temp_dir_ / generate_temp_filename("snippet", ".cc")).string();

    std::ostringstream source;
    source << snippet.includes << "\n\n";
    source << snippet.code << "\n";

    if (!write_file_contents(temp_source, source.str())) {
        VectorizationReport empty;
        empty.source_file = "<snippet>";
        return empty;
    }

    auto report = analyze_source(temp_source);
    report.source_file = "<snippet:" + snippet.function_name + ">";

    // Cleanup
    std::filesystem::remove(temp_source);

    return report;
}

VectorizationReport AutoVecAnalyzer::parse_gcc_output(
    const std::string& output,
    const std::string& /* source_file */
) {
    VectorizationReport report;

    std::istringstream iss(output);
    std::string line;

    // GCC optimization info patterns
    // e.g., "file.cc:42:5: note: loop vectorized"
    // e.g., "file.cc:42:5: note: not vectorized: data dependency"
    std::regex vectorized_pattern(
        R"((.+):(\d+):\d+: note: (?:loop |LOOP )?vectorized)"
    );
    std::regex not_vectorized_pattern(
        R"((.+):(\d+):\d+: note: not vectorized:?\s*(.*))"
    );
    std::regex interleaved_pattern(
        R"((.+):(\d+):\d+: note: (?:loop )?interleaved)"
    );
    std::regex unrolled_pattern(
        R"((.+):(\d+):\d+: note: (?:loop )?unrolled)"
    );
    std::regex vect_width_pattern(
        R"(using (\d+)-byte vectors)"
    );

    while (std::getline(iss, line)) {
        std::smatch match;

        LoopVectorizationInfo info;

        if (std::regex_search(line, match, vectorized_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::VECTORIZED;
            info.compiler_message = line;

            // Try to extract vector width
            std::smatch width_match;
            if (std::regex_search(line, width_match, vect_width_pattern)) {
                int bytes = std::stoi(width_match[1].str());
                info.vector_width = bytes * 8;
            }

            report.loops.push_back(info);
        }
        else if (std::regex_search(line, match, not_vectorized_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::NOT_VECTORIZED;
            info.blocker_description = match[3].str();
            info.blocker = classify_blocker(info.blocker_description);
            info.compiler_message = line;

            report.loops.push_back(info);
        }
        else if (std::regex_search(line, match, interleaved_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::INTERLEAVED;
            info.compiler_message = line;

            report.loops.push_back(info);
        }
        else if (std::regex_search(line, match, unrolled_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::UNROLLED;
            info.compiler_message = line;

            report.loops.push_back(info);
        }
    }

    return report;
}

VectorizationReport AutoVecAnalyzer::parse_clang_output(
    const std::string& output,
    const std::string& /* source_file */
) {
    VectorizationReport report;

    std::istringstream iss(output);
    std::string line;

    // Clang patterns
    // e.g., "file.cc:42:5: remark: vectorized loop (vectorization width: 8)"
    // e.g., "file.cc:42:5: remark: loop not vectorized: ..."
    std::regex vectorized_pattern(
        R"((.+):(\d+):\d+: remark: vectorized loop.*width:\s*(\d+))"
    );
    std::regex not_vectorized_pattern(
        R"((.+):(\d+):\d+: remark: (?:loop )?not vectorized:?\s*(.*))"
    );
    std::regex analysis_pattern(
        R"((.+):(\d+):\d+: remark: .*vectoriz.*)"
    );

    while (std::getline(iss, line)) {
        std::smatch match;

        LoopVectorizationInfo info;

        if (std::regex_search(line, match, vectorized_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::VECTORIZED;
            info.vector_width = std::stoi(match[3].str()) * 32; // Assuming 32-bit elements
            info.compiler_message = line;

            report.loops.push_back(info);
        }
        else if (std::regex_search(line, match, not_vectorized_pattern)) {
            info.source_location = match[1].str();
            info.loop_line = std::stoi(match[2].str());
            info.status = VectorizationStatus::NOT_VECTORIZED;
            info.blocker_description = match[3].str();
            info.blocker = classify_blocker(info.blocker_description);
            info.compiler_message = line;

            report.loops.push_back(info);
        }
    }

    return report;
}

VectorizationReport AutoVecAnalyzer::parse_icc_output(
    const std::string& output,
    const std::string& /* source_file */
) {
    VectorizationReport report;

    std::istringstream iss(output);
    std::string line;

    // ICC patterns (from qopt-report)
    // e.g., "LOOP BEGIN at file.cc(42,5)"
    // e.g., "remark #15300: LOOP WAS VECTORIZED"
    // e.g., "remark #15344: loop was not vectorized: ..."
    std::regex loop_begin_pattern(R"(LOOP BEGIN at (.+)\((\d+),)");
    std::regex vectorized_pattern(R"(remark #\d+:.*LOOP WAS VECTORIZED)");
    std::regex not_vectorized_pattern(R"(remark #\d+:.*not vectorized:?\s*(.*))");

    std::string current_file;
    int current_line = 0;

    while (std::getline(iss, line)) {
        std::smatch match;

        if (std::regex_search(line, match, loop_begin_pattern)) {
            current_file = match[1].str();
            current_line = std::stoi(match[2].str());
        }
        else if (std::regex_search(line, vectorized_pattern)) {
            LoopVectorizationInfo info;
            info.source_location = current_file;
            info.loop_line = current_line;
            info.status = VectorizationStatus::VECTORIZED;
            info.compiler_message = line;
            report.loops.push_back(info);
        }
        else if (std::regex_search(line, match, not_vectorized_pattern)) {
            LoopVectorizationInfo info;
            info.source_location = current_file;
            info.loop_line = current_line;
            info.status = VectorizationStatus::NOT_VECTORIZED;
            info.blocker_description = match[1].str();
            info.blocker = classify_blocker(info.blocker_description);
            info.compiler_message = line;
            report.loops.push_back(info);
        }
    }

    return report;
}

VectorizationBlocker AutoVecAnalyzer::classify_blocker(const std::string& message) {
    std::string lower = message;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("depend") != std::string::npos) {
        return VectorizationBlocker::DATA_DEPENDENCY;
    }
    if (lower.find("alias") != std::string::npos) {
        return VectorizationBlocker::ALIASING;
    }
    if (lower.find("control flow") != std::string::npos ||
        lower.find("conditional") != std::string::npos) {
        return VectorizationBlocker::CONTROL_FLOW;
    }
    if (lower.find("call") != std::string::npos ||
        lower.find("function") != std::string::npos) {
        return VectorizationBlocker::FUNCTION_CALL;
    }
    if (lower.find("scatter") != std::string::npos ||
        lower.find("gather") != std::string::npos ||
        lower.find("non-contiguous") != std::string::npos ||
        lower.find("strided") != std::string::npos) {
        return VectorizationBlocker::NON_CONTIGUOUS_ACCESS;
    }
    if (lower.find("reduction") != std::string::npos) {
        return VectorizationBlocker::REDUCTION_CHAIN;
    }
    if (lower.find("trip count") != std::string::npos ||
        lower.find("too few") != std::string::npos) {
        return VectorizationBlocker::SMALL_TRIP_COUNT;
    }
    if (lower.find("align") != std::string::npos) {
        return VectorizationBlocker::ALIGNMENT_UNKNOWN;
    }
    if (lower.find("cost") != std::string::npos ||
        lower.find("profitable") != std::string::npos) {
        return VectorizationBlocker::COST_MODEL;
    }
    if (lower.find("mixed") != std::string::npos ||
        lower.find("type") != std::string::npos) {
        return VectorizationBlocker::MIXED_TYPES;
    }
    if (lower.find("indirect") != std::string::npos) {
        return VectorizationBlocker::INDIRECT_MEMORY_ACCESS;
    }
    if (lower.find("unsupported") != std::string::npos) {
        return VectorizationBlocker::UNSUPPORTED_OPERATION;
    }

    return VectorizationBlocker::UNKNOWN_REASON;
}

AutoVecComparison AutoVecAnalyzer::compare_with_intrinsic(
    const std::string& source_file,
    const KernelConfig& intrinsic_kernel,
    size_t problem_size,
    size_t iterations
) {
    AutoVecComparison result;
    result.kernel_name = intrinsic_kernel.name;
    result.problem_size = problem_size;

    // Get vectorization report
    result.autovec_report = analyze_source(source_file);

    // Compile and benchmark auto-vectorized version
    auto autovec_gflops = compile_and_benchmark(
        source_file,
        intrinsic_kernel.name,
        problem_size,
        iterations,
        true  // Enable autovec
    );

    if (autovec_gflops) {
        result.autovec_compiled = true;
        result.autovec_gflops = *autovec_gflops;
    } else {
        result.autovec_compiled = false;
        result.autovec_error = "Failed to compile auto-vectorized version";
    }

    // Benchmark intrinsic version
    if (!intrinsic_kernel.variants.empty() && intrinsic_kernel.setup) {
        void* data = intrinsic_kernel.setup(problem_size);
        if (data) {
            Timer timer;

            // Find best variant
            const KernelVariant* best_variant = nullptr;
            for (const auto& variant : intrinsic_kernel.variants) {
                if (variant.isa != "scalar") {
                    best_variant = &variant;
                    break;
                }
            }

            if (!best_variant && !intrinsic_kernel.variants.empty()) {
                best_variant = &intrinsic_kernel.variants[0];
            }

            if (best_variant) {
                // Warmup
                for (size_t i = 0; i < 5; ++i) {
                    best_variant->func(data, problem_size, 1);
                }

                // Benchmark
                timer.start();
                best_variant->func(data, problem_size, iterations);
                timer.stop();

                uint64_t flops = intrinsic_kernel.flops_per_element * problem_size * iterations;
                result.intrinsic_gflops = FlopsCalculator::to_gflops(flops, timer.elapsed_seconds());
                result.intrinsic_elapsed_seconds = timer.elapsed_seconds();
            }

            if (intrinsic_kernel.teardown) {
                intrinsic_kernel.teardown(data);
            }
        }
    }

    // Calculate speedup
    if (result.autovec_gflops > 0) {
        result.speedup_intrinsic_vs_auto = result.intrinsic_gflops / result.autovec_gflops;
        result.efficiency_ratio = result.intrinsic_gflops / result.autovec_gflops;
    }

    // Generate recommendations
    if (result.speedup_intrinsic_vs_auto > 1.2) {
        result.recommendations.push_back(
            "Intrinsic version is significantly faster (" +
            std::to_string(static_cast<int>((result.speedup_intrinsic_vs_auto - 1.0) * 100)) +
            "% speedup). Consider using manual vectorization."
        );
    } else if (result.speedup_intrinsic_vs_auto < 0.9) {
        result.recommendations.push_back(
            "Auto-vectorized version is faster. Consider relying on compiler."
        );
    } else {
        result.recommendations.push_back(
            "Performance is similar. Auto-vectorization is effective for this code."
        );
    }

    // Add blocker-specific recommendations
    for (const auto& loop : result.autovec_report.loops) {
        if (loop.status == VectorizationStatus::NOT_VECTORIZED) {
            switch (loop.blocker) {
                case VectorizationBlocker::ALIASING:
                    result.recommendations.push_back(
                        "Line " + std::to_string(loop.loop_line) +
                        ": Add __restrict to pointer arguments"
                    );
                    break;
                case VectorizationBlocker::ALIGNMENT_UNKNOWN:
                    result.recommendations.push_back(
                        "Line " + std::to_string(loop.loop_line) +
                        ": Use __builtin_assume_aligned() or alignas()"
                    );
                    break;
                case VectorizationBlocker::DATA_DEPENDENCY:
                    result.recommendations.push_back(
                        "Line " + std::to_string(loop.loop_line) +
                        ": Review loop-carried dependencies; consider loop splitting"
                    );
                    break;
                case VectorizationBlocker::FUNCTION_CALL:
                    result.recommendations.push_back(
                        "Line " + std::to_string(loop.loop_line) +
                        ": Inline function calls or use intrinsics"
                    );
                    break;
                default:
                    break;
            }
        }
    }

    return result;
}

AutoVecAnalysis AutoVecAnalyzer::full_analysis(
    const std::string& source_file,
    const KernelConfig& intrinsic_kernel,
    const std::vector<size_t>& sizes
) {
    AutoVecAnalysis analysis;

    // Get vectorization report
    analysis.compiler_report = analyze_source(source_file);
    analysis.compiler_vectorized = analysis.compiler_report.vectorized_loops > 0;

    // Use provided sizes or kernel defaults
    std::vector<size_t> test_sizes = sizes.empty() ? intrinsic_kernel.sizes : sizes;
    if (test_sizes.empty()) {
        test_sizes = {1024, 4096, 16384, 65536};
    }

    // Test at largest size for detailed comparison
    size_t test_size = test_sizes.back();
    auto comparison = compare_with_intrinsic(
        source_file,
        intrinsic_kernel,
        test_size,
        100
    );

    // Calculate speedups
    // Compile scalar version for baseline
    auto scalar_gflops = compile_and_benchmark(
        source_file,
        intrinsic_kernel.name,
        test_size,
        100,
        false  // Disable autovec
    );

    if (scalar_gflops && *scalar_gflops > 0) {
        analysis.auto_vec_speedup = comparison.autovec_gflops / *scalar_gflops;
        analysis.intrinsic_speedup = comparison.intrinsic_gflops / *scalar_gflops;
    }

    analysis.intrinsic_vs_autovec = comparison.speedup_intrinsic_vs_auto;

    // Collect missed optimizations
    for (const auto& loop : analysis.compiler_report.loops) {
        if (loop.status == VectorizationStatus::NOT_VECTORIZED) {
            analysis.vectorization_blockers.push_back(
                "Line " + std::to_string(loop.loop_line) + ": " +
                loop.blocker_description
            );
            analysis.critical_loops.push_back(loop);
        }
    }

    // Generate suggestions
    analysis.suggestions = generate_suggestions(analysis.compiler_report);

    return analysis;
}

std::optional<double> AutoVecAnalyzer::compile_and_benchmark(
    const std::string& source_file,
    const std::string& function_name,
    size_t problem_size,
    size_t iterations,
    bool enable_autovec
) {
    // Generate wrapper source
    std::string wrapper = generate_wrapper_source(source_file, function_name);
    std::string wrapper_file = (temp_dir_ / generate_temp_filename("bench", ".cc")).string();
    std::string output_file = (temp_dir_ / generate_temp_filename("bench", "")).string();

    if (!write_file_contents(wrapper_file, wrapper)) {
        return std::nullopt;
    }

    // Compile
    std::vector<std::string> flags = enable_autovec ?
        config_.autovec_flags : config_.no_vec_flags;
    flags.push_back("-shared");
    flags.push_back("-fPIC");

    std::string compiler_output;
    if (!compile_source(wrapper_file, output_file + ".so", flags, compiler_output)) {
        std::filesystem::remove(wrapper_file);
        return std::nullopt;
    }

    // Load and benchmark
    double gflops = 0.0;

#ifndef _WIN32
    void* handle = dlopen((output_file + ".so").c_str(), RTLD_NOW);
    if (handle) {
        using BenchFunc = double (*)(size_t, size_t);
        auto bench_func = reinterpret_cast<BenchFunc>(dlsym(handle, "benchmark_kernel"));

        if (bench_func) {
            gflops = bench_func(problem_size, iterations);
        }

        dlclose(handle);
    }
#endif

    // Cleanup
    std::filesystem::remove(wrapper_file);
    std::filesystem::remove(output_file + ".so");

    return gflops > 0 ? std::make_optional(gflops) : std::nullopt;
}

std::string AutoVecAnalyzer::generate_wrapper_source(
    const std::string& original_source,
    const std::string& function_name
) {
    std::ostringstream oss;

    oss << "#include <chrono>\n";
    oss << "#include <cstddef>\n";
    oss << "#include <cstdint>\n\n";

    // Include original source
    oss << "#include \"" << original_source << "\"\n\n";

    // Generate benchmark wrapper
    oss << "extern \"C\" double benchmark_kernel(size_t size, size_t iterations) {\n";
    oss << "    using Clock = std::chrono::high_resolution_clock;\n";
    oss << "    auto start = Clock::now();\n";
    oss << "    for (size_t i = 0; i < iterations; ++i) {\n";
    oss << "        " << function_name << "(size);\n";
    oss << "    }\n";
    oss << "    auto end = Clock::now();\n";
    oss << "    double seconds = std::chrono::duration<double>(end - start).count();\n";
    oss << "    uint64_t flops = 2 * size * iterations;  // Adjust based on kernel\n";
    oss << "    return static_cast<double>(flops) / (seconds * 1e9);\n";
    oss << "}\n";

    return oss.str();
}

std::vector<std::string> AutoVecAnalyzer::generate_suggestions(
    const VectorizationReport& report
) {
    std::vector<std::string> suggestions;

    // Analyze blocker counts
    for (const auto& [blocker, count] : report.blocker_counts) {
        switch (blocker) {
            case VectorizationBlocker::ALIASING:
                suggestions.push_back(
                    "Use '__restrict' keyword on pointer parameters to help "
                    "compiler prove non-aliasing (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::ALIGNMENT_UNKNOWN:
                suggestions.push_back(
                    "Use 'alignas(64)' for arrays and '__builtin_assume_aligned()' "
                    "for pointers (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::DATA_DEPENDENCY:
                suggestions.push_back(
                    "Consider loop splitting or scalar expansion to break "
                    "dependencies (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::FUNCTION_CALL:
                suggestions.push_back(
                    "Move function calls outside loops or use 'inline' / "
                    "'__attribute__((always_inline))' (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::CONTROL_FLOW:
                suggestions.push_back(
                    "Replace conditionals with branchless operations or use "
                    "'#pragma omp simd' (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::NON_CONTIGUOUS_ACCESS:
                suggestions.push_back(
                    "Restructure data for contiguous access or use gather/scatter "
                    "intrinsics (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::REDUCTION_CHAIN:
                suggestions.push_back(
                    "Use '#pragma omp simd reduction(+:var)' for parallel "
                    "reductions (" + std::to_string(count) + " cases)"
                );
                break;

            case VectorizationBlocker::SMALL_TRIP_COUNT:
                suggestions.push_back(
                    "Increase loop trip count or use '#pragma unroll' for "
                    "small loops (" + std::to_string(count) + " cases)"
                );
                break;

            default:
                break;
        }
    }

    // General suggestions based on vectorization rate
    if (report.vectorization_rate < 0.5) {
        suggestions.push_back(
            "Consider using OpenMP SIMD directives (#pragma omp simd) to guide "
            "the compiler"
        );
        suggestions.push_back(
            "Review data structures for SoA (Structure of Arrays) layout"
        );
    }

    if (report.vectorization_rate > 0.8) {
        suggestions.push_back(
            "Good vectorization rate achieved. Focus on memory access optimization."
        );
    }

    return suggestions;
}

VectorizationReport AutoVecAnalyzer::parse_compiler_output(
    const std::string& compiler_output,
    const std::string& source_file
) {
    if (config_.compiler_id == "gcc") {
        return parse_gcc_output(compiler_output, source_file);
    } else if (config_.compiler_id == "clang") {
        return parse_clang_output(compiler_output, source_file);
    } else if (config_.compiler_id == "icc" || config_.compiler_id == "icpx") {
        return parse_icc_output(compiler_output, source_file);
    }

    VectorizationReport empty;
    empty.source_file = source_file;
    return empty;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string vectorization_status_to_string(VectorizationStatus status) {
    switch (status) {
        case VectorizationStatus::VECTORIZED: return "vectorized";
        case VectorizationStatus::NOT_VECTORIZED: return "not vectorized";
        case VectorizationStatus::PARTIALLY_VECTORIZED: return "partially vectorized";
        case VectorizationStatus::INTERLEAVED: return "interleaved";
        case VectorizationStatus::UNROLLED: return "unrolled only";
        case VectorizationStatus::ALREADY_VECTORIZED: return "contains intrinsics";
        default: return "unknown";
    }
}

std::string vectorization_blocker_to_string(VectorizationBlocker blocker) {
    switch (blocker) {
        case VectorizationBlocker::NONE: return "none";
        case VectorizationBlocker::DATA_DEPENDENCY: return "data dependency";
        case VectorizationBlocker::CONTROL_FLOW: return "control flow";
        case VectorizationBlocker::NON_CONTIGUOUS_ACCESS: return "non-contiguous access";
        case VectorizationBlocker::ALIASING: return "pointer aliasing";
        case VectorizationBlocker::MIXED_TYPES: return "mixed types";
        case VectorizationBlocker::FUNCTION_CALL: return "function call";
        case VectorizationBlocker::REDUCTION_CHAIN: return "reduction chain";
        case VectorizationBlocker::SMALL_TRIP_COUNT: return "small trip count";
        case VectorizationBlocker::ALIGNMENT_UNKNOWN: return "unknown alignment";
        case VectorizationBlocker::COST_MODEL: return "cost model";
        case VectorizationBlocker::USER_DIRECTIVE: return "user directive";
        case VectorizationBlocker::UNSUPPORTED_OPERATION: return "unsupported operation";
        case VectorizationBlocker::INDIRECT_MEMORY_ACCESS: return "indirect memory access";
        case VectorizationBlocker::EXCEPTION_HANDLING: return "exception handling";
        default: return "unknown reason";
    }
}

std::string generate_autovec_summary(const AutoVecAnalysis& analysis) {
    std::ostringstream oss;

    oss << "Auto-Vectorization Analysis Summary\n";
    oss << "====================================\n\n";

    oss << "Compiler: " << analysis.compiler_report.compiler << " "
        << analysis.compiler_report.compiler_version << "\n";
    oss << "Vectorization Rate: "
        << static_cast<int>(analysis.compiler_report.vectorization_rate * 100) << "%\n";
    oss << "Vectorized Loops: " << analysis.compiler_report.vectorized_loops
        << " / " << analysis.compiler_report.total_loops << "\n\n";

    oss << "Speedups:\n";
    oss << "  Auto-vec vs Scalar: " << analysis.auto_vec_speedup << "x\n";
    oss << "  Intrinsic vs Scalar: " << analysis.intrinsic_speedup << "x\n";
    oss << "  Intrinsic vs Auto-vec: " << analysis.intrinsic_vs_autovec << "x\n\n";

    if (!analysis.vectorization_blockers.empty()) {
        oss << "Vectorization Blockers:\n";
        for (const auto& blocker : analysis.vectorization_blockers) {
            oss << "  - " << blocker << "\n";
        }
        oss << "\n";
    }

    if (!analysis.suggestions.empty()) {
        oss << "Suggestions:\n";
        for (const auto& suggestion : analysis.suggestions) {
            oss << "  - " << suggestion << "\n";
        }
    }

    return oss.str();
}

std::string generate_autovec_markdown(const AutoVecAnalysis& analysis) {
    std::ostringstream oss;

    oss << "# Auto-Vectorization Analysis\n\n";

    oss << "## Summary\n\n";
    oss << "| Metric | Value |\n";
    oss << "|--------|-------|\n";
    oss << "| Compiler | " << analysis.compiler_report.compiler << " |\n";
    oss << "| Vectorization Rate | "
        << static_cast<int>(analysis.compiler_report.vectorization_rate * 100) << "% |\n";
    oss << "| Loops Vectorized | " << analysis.compiler_report.vectorized_loops
        << " / " << analysis.compiler_report.total_loops << " |\n";
    oss << "| Auto-vec Speedup | " << analysis.auto_vec_speedup << "x |\n";
    oss << "| Intrinsic Speedup | " << analysis.intrinsic_speedup << "x |\n\n";

    if (!analysis.critical_loops.empty()) {
        oss << "## Failed Vectorizations\n\n";
        oss << "| Line | Status | Blocker | Message |\n";
        oss << "|------|--------|---------|----------|\n";

        for (const auto& loop : analysis.critical_loops) {
            oss << "| " << loop.loop_line
                << " | " << vectorization_status_to_string(loop.status)
                << " | " << vectorization_blocker_to_string(loop.blocker)
                << " | " << loop.blocker_description << " |\n";
        }
        oss << "\n";
    }

    if (!analysis.suggestions.empty()) {
        oss << "## Recommendations\n\n";
        for (const auto& suggestion : analysis.suggestions) {
            oss << "- " << suggestion << "\n";
        }
    }

    return oss.str();
}

std::vector<VectorizationHint> suggest_hints(
    const LoopVectorizationInfo& loop,
    const std::string& /* source_code */
) {
    std::vector<VectorizationHint> hints;

    switch (loop.blocker) {
        case VectorizationBlocker::ALIASING: {
            VectorizationHint hint;
            hint.type = VectorizationHint::Type::RESTRICT;
            hint.code_before = "void func(float* a, float* b)";
            hint.code_after = "void func(float* __restrict a, float* __restrict b)";
            hint.explanation = "Add __restrict to guarantee no aliasing";
            hint.estimated_impact = 80;
            hints.push_back(hint);
            break;
        }

        case VectorizationBlocker::ALIGNMENT_UNKNOWN: {
            VectorizationHint hint;
            hint.type = VectorizationHint::Type::ASSUME_ALIGNED;
            hint.code_before = "float* data;";
            hint.code_after = "float* data = static_cast<float*>(__builtin_assume_aligned(ptr, 64));";
            hint.explanation = "Tell compiler pointer is 64-byte aligned";
            hint.estimated_impact = 60;
            hints.push_back(hint);

            VectorizationHint hint2;
            hint2.type = VectorizationHint::Type::ALIGN;
            hint2.code_before = "float data[N];";
            hint2.code_after = "alignas(64) float data[N];";
            hint2.explanation = "Align array to cache line boundary";
            hint2.estimated_impact = 50;
            hints.push_back(hint2);
            break;
        }

        case VectorizationBlocker::DATA_DEPENDENCY: {
            VectorizationHint hint;
            hint.type = VectorizationHint::Type::IVDEP;
            hint.code_before = "for (int i = 0; i < n; ++i)";
            hint.code_after = "#pragma ivdep\nfor (int i = 0; i < n; ++i)";
            hint.explanation = "Assert no loop-carried dependencies";
            hint.estimated_impact = 70;
            hints.push_back(hint);
            break;
        }

        case VectorizationBlocker::CONTROL_FLOW: {
            VectorizationHint hint;
            hint.type = VectorizationHint::Type::VECTOR_ALWAYS;
            hint.code_before = "for (int i = 0; i < n; ++i)";
            hint.code_after = "#pragma omp simd\nfor (int i = 0; i < n; ++i)";
            hint.explanation = "Force vectorization with masking";
            hint.estimated_impact = 50;
            hints.push_back(hint);
            break;
        }

        default:
            break;
    }

    return hints;
}

bool contains_simd_intrinsics(const std::string& source_code) {
    // Check for common intrinsic patterns
    static const std::vector<std::string> intrinsic_patterns = {
        "_mm_", "_mm256_", "_mm512_",  // Intel intrinsics
        "vld", "vst", "vmul", "vadd",   // ARM NEON
        "svld", "svst", "svadd",        // ARM SVE
        "hn::", "HWY_",                 // Highway
        "__builtin_ia32_"               // GCC builtins
    };

    for (const auto& pattern : intrinsic_patterns) {
        if (source_code.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
}

std::string detect_isa_level(const std::string& source_code) {
    if (source_code.find("_mm512_") != std::string::npos) {
        return "AVX-512";
    }
    if (source_code.find("_mm256_") != std::string::npos) {
        return "AVX2";
    }
    if (source_code.find("_mm_") != std::string::npos) {
        return "SSE";
    }
    if (source_code.find("svld") != std::string::npos ||
        source_code.find("svst") != std::string::npos) {
        return "SVE";
    }
    if (source_code.find("vld") != std::string::npos ||
        source_code.find("vst") != std::string::npos) {
        return "NEON";
    }
    if (source_code.find("hn::") != std::string::npos ||
        source_code.find("HWY_") != std::string::npos) {
        return "Highway (portable)";
    }

    return "Scalar/Unknown";
}

}  // namespace simd_bench
