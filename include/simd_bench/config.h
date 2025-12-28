#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>

namespace simd_bench {

// Configuration value types
using ConfigValue = std::variant<
    bool,
    int64_t,
    double,
    std::string,
    std::vector<std::string>,
    std::vector<int64_t>,
    std::vector<double>
>;

// Benchmark configuration from file
struct FileConfig {
    // Benchmark parameters
    struct BenchmarkParams {
        std::string name;
        std::vector<size_t> sizes;
        size_t iterations = 1000;
        size_t warmup = 10;
        bool enabled = true;
    };

    std::vector<BenchmarkParams> benchmarks;

    // Hardware counter configuration
    struct CounterConfig {
        std::string backend = "auto";  // "auto", "perf", "papi", "likwid"
        std::vector<std::string> events;
        bool enabled = true;
    };

    CounterConfig counters;

    // Analysis configuration
    struct AnalysisConfig {
        bool roofline = true;
        bool tma = true;
        bool insights = true;
        bool scaling = false;
        bool prefetch = false;
        bool register_pressure = false;
    };

    AnalysisConfig analysis;

    // Output configuration
    struct OutputConfig {
        std::vector<std::string> formats = {"json"};
        std::string path = "./";
        bool console = true;
        bool verbose = false;
    };

    OutputConfig output;

    // Threshold configuration
    struct ThresholdConfig {
        double vectorization_warning = 0.8;
        double cache_miss_warning = 0.05;
        double efficiency_warning = 0.5;
        double regression_threshold = 0.05;
    };

    ThresholdConfig thresholds;

    // Environment overrides
    std::map<std::string, ConfigValue> custom;
};

// Configuration file parser
class ConfigParser {
public:
    // Parse configuration from YAML file
    static FileConfig parse_yaml(const std::string& filepath);

    // Parse configuration from JSON file
    static FileConfig parse_json(const std::string& filepath);

    // Parse configuration from environment variables
    static FileConfig parse_environment();

    // Merge configurations (later overrides earlier)
    static FileConfig merge(const FileConfig& base, const FileConfig& overlay);

    // Validate configuration
    static bool validate(const FileConfig& config, std::string& error_message);

    // Generate default configuration file
    static std::string generate_default_yaml();
    static std::string generate_default_json();

    // Write configuration to file
    static bool write_yaml(const FileConfig& config, const std::string& filepath);
    static bool write_json(const FileConfig& config, const std::string& filepath);
};

// Configuration manager
class ConfigManager {
public:
    ConfigManager();

    // Load configuration from file
    bool load(const std::string& filepath);

    // Load from default locations
    bool load_default();

    // Get current configuration
    const FileConfig& config() const { return config_; }
    FileConfig& config() { return config_; }

    // Apply configuration to benchmark
    BenchmarkConfig to_benchmark_config() const;

    // Get value with default
    template<typename T>
    T get(const std::string& key, const T& default_value) const;

    // Set value
    template<typename T>
    void set(const std::string& key, const T& value);

    // Check if key exists
    bool has(const std::string& key) const;

    // Get all custom keys
    std::vector<std::string> custom_keys() const;

private:
    FileConfig config_;
    std::string loaded_path_;
};

// Command-line argument parser for configuration
struct CommandLineConfig {
    std::string config_file;
    std::vector<std::string> benchmarks;
    std::vector<size_t> sizes;
    size_t iterations = 0;
    std::string output_format;
    std::string output_path;
    bool verbose = false;
    bool help = false;
    bool list_benchmarks = false;
    bool list_counters = false;
    std::map<std::string, std::string> overrides;
};

CommandLineConfig parse_command_line(int argc, char* argv[]);

// Apply command-line overrides to file config
FileConfig apply_overrides(const FileConfig& base, const CommandLineConfig& cli);

}  // namespace simd_bench
