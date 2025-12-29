#include "simd_bench/config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <regex>

namespace simd_bench {

// ============================================================================
// Simple YAML/JSON-like parser (minimal implementation without external deps)
// For full YAML support, consider yaml-cpp library
// ============================================================================

namespace {

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

}  // anonymous namespace

// ============================================================================
// ConfigParser implementation
// ============================================================================

FileConfig ConfigParser::parse_yaml(const std::string& filepath) {
    FileConfig config;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        return config;
    }

    std::string line;
    std::string current_section;
    FileConfig::BenchmarkParams current_benchmark;
    bool in_benchmark = false;

    while (std::getline(file, line)) {
        line = trim(line);

        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Check for section headers
        if (line == "benchmarks:") {
            current_section = "benchmarks";
            continue;
        } else if (line == "hardware_counters:") {
            current_section = "counters";
            continue;
        } else if (line == "analysis:") {
            current_section = "analysis";
            continue;
        } else if (line == "output:") {
            current_section = "output";
            continue;
        } else if (line == "thresholds:") {
            current_section = "thresholds";
            continue;
        }

        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, colon_pos));
        std::string value = trim(line.substr(colon_pos + 1));

        // Remove quotes from string values
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        // Handle list items (- prefix)
        if (starts_with(key, "- ")) {
            key = key.substr(2);
        }

        // Parse based on current section
        if (current_section == "benchmarks") {
            if (key == "name" || key == "- name") {
                if (in_benchmark) {
                    config.benchmarks.push_back(current_benchmark);
                }
                current_benchmark = FileConfig::BenchmarkParams();
                current_benchmark.name = value;
                in_benchmark = true;
            } else if (key == "iterations" && in_benchmark) {
                try { current_benchmark.iterations = std::stoull(value); }
                catch (...) { current_benchmark.iterations = 100; }
            } else if (key == "warmup" && in_benchmark) {
                try { current_benchmark.warmup = std::stoull(value); }
                catch (...) { current_benchmark.warmup = 10; }
            } else if (key == "enabled" && in_benchmark) {
                current_benchmark.enabled = (value == "true" || value == "1");
            }
        } else if (current_section == "counters") {
            if (key == "backend") {
                config.counters.backend = value;
            } else if (key == "enabled") {
                config.counters.enabled = (value == "true" || value == "1");
            }
        } else if (current_section == "analysis") {
            if (key == "roofline") config.analysis.roofline = (value == "true");
            else if (key == "tma") config.analysis.tma = (value == "true");
            else if (key == "insights") config.analysis.insights = (value == "true");
            else if (key == "scaling") config.analysis.scaling = (value == "true");
            else if (key == "prefetch") config.analysis.prefetch = (value == "true");
            else if (key == "register_pressure") config.analysis.register_pressure = (value == "true");
        } else if (current_section == "output") {
            if (key == "path") config.output.path = value;
            else if (key == "console") config.output.console = (value == "true");
            else if (key == "verbose") config.output.verbose = (value == "true");
        } else if (current_section == "thresholds") {
            try {
                if (key == "vectorization_warning") config.thresholds.vectorization_warning = std::stod(value);
                else if (key == "cache_miss_warning") config.thresholds.cache_miss_warning = std::stod(value);
                else if (key == "efficiency_warning") config.thresholds.efficiency_warning = std::stod(value);
                else if (key == "regression_threshold") config.thresholds.regression_threshold = std::stod(value);
            } catch (...) {
                // Keep default values on parse error
            }
        }
    }

    // Don't forget the last benchmark
    if (in_benchmark) {
        config.benchmarks.push_back(current_benchmark);
    }

    return config;
}

FileConfig ConfigParser::parse_json(const std::string& filepath) {
    FileConfig config;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        return config;
    }

    // Simple JSON parsing - for production use, consider nlohmann/json
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Very basic JSON parsing (not robust)
    auto extract_string = [&content](const std::string& key) -> std::string {
        std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
        std::smatch match;
        if (std::regex_search(content, match, pattern)) {
            return match[1].str();
        }
        return "";
    };

    auto extract_bool = [&content](const std::string& key) -> bool {
        std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
        std::smatch match;
        if (std::regex_search(content, match, pattern)) {
            return match[1].str() == "true";
        }
        return false;
    };

    auto extract_double = [&content](const std::string& key) -> double {
        std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9.]+)");
        std::smatch match;
        if (std::regex_search(content, match, pattern)) {
            return std::stod(match[1].str());
        }
        return 0.0;
    };

    // Parse config values
    config.counters.backend = extract_string("backend");
    if (config.counters.backend.empty()) config.counters.backend = "auto";

    config.output.path = extract_string("path");
    config.output.verbose = extract_bool("verbose");

    config.analysis.roofline = extract_bool("roofline");
    config.analysis.tma = extract_bool("tma");
    config.analysis.insights = extract_bool("insights");

    config.thresholds.regression_threshold = extract_double("regression_threshold");
    if (config.thresholds.regression_threshold == 0.0) {
        config.thresholds.regression_threshold = 0.05;
    }

    return config;
}

FileConfig ConfigParser::parse_environment() {
    FileConfig config;

    // Read configuration from environment variables
    auto get_env = [](const char* name) -> std::optional<std::string> {
        const char* value = std::getenv(name);
        if (value) return std::string(value);
        return std::nullopt;
    };

    auto get_env_bool = [&get_env](const char* name) -> std::optional<bool> {
        auto value = get_env(name);
        if (!value) return std::nullopt;
        return (*value == "1" || *value == "true" || *value == "yes");
    };

    auto get_env_double = [&get_env](const char* name) -> std::optional<double> {
        auto value = get_env(name);
        if (!value) return std::nullopt;
        try {
            return std::stod(*value);
        } catch (...) {
            return std::nullopt;
        }
    };

    // SIMD_BENCH_* environment variables
    if (auto v = get_env("SIMD_BENCH_COUNTER_BACKEND")) {
        config.counters.backend = *v;
    }

    if (auto v = get_env_bool("SIMD_BENCH_ROOFLINE")) {
        config.analysis.roofline = *v;
    }

    if (auto v = get_env_bool("SIMD_BENCH_TMA")) {
        config.analysis.tma = *v;
    }

    if (auto v = get_env_bool("SIMD_BENCH_VERBOSE")) {
        config.output.verbose = *v;
    }

    if (auto v = get_env("SIMD_BENCH_OUTPUT_PATH")) {
        config.output.path = *v;
    }

    if (auto v = get_env_double("SIMD_BENCH_REGRESSION_THRESHOLD")) {
        config.thresholds.regression_threshold = *v;
    }

    return config;
}

FileConfig ConfigParser::merge(const FileConfig& base, const FileConfig& overlay) {
    FileConfig result = base;

    // Merge benchmarks (overlay adds to base)
    for (const auto& bench : overlay.benchmarks) {
        result.benchmarks.push_back(bench);
    }

    // Merge counters (overlay overrides)
    if (!overlay.counters.backend.empty() && overlay.counters.backend != "auto") {
        result.counters.backend = overlay.counters.backend;
    }
    for (const auto& event : overlay.counters.events) {
        result.counters.events.push_back(event);
    }

    // Merge analysis (overlay overrides only if explicitly set differently)
    // For simplicity, just use overlay values
    result.analysis = overlay.analysis;

    // Merge output
    if (!overlay.output.path.empty()) {
        result.output.path = overlay.output.path;
    }
    if (overlay.output.verbose) {
        result.output.verbose = true;
    }

    // Merge custom values
    for (const auto& [key, value] : overlay.custom) {
        result.custom[key] = value;
    }

    return result;
}

bool ConfigParser::validate(const FileConfig& config, std::string& error_message) {
    // Validate counter backend
    std::vector<std::string> valid_backends = {"auto", "perf", "papi", "likwid", "none"};
    bool valid_backend = false;
    for (const auto& b : valid_backends) {
        if (config.counters.backend == b) {
            valid_backend = true;
            break;
        }
    }
    if (!valid_backend) {
        error_message = "Invalid counter backend: " + config.counters.backend;
        return false;
    }

    // Validate thresholds
    if (config.thresholds.regression_threshold <= 0 ||
        config.thresholds.regression_threshold > 1.0) {
        error_message = "Regression threshold must be between 0 and 1";
        return false;
    }

    // Validate output formats
    std::vector<std::string> valid_formats = {"json", "html", "markdown", "csv"};
    for (const auto& format : config.output.formats) {
        bool valid = false;
        for (const auto& f : valid_formats) {
            if (format == f) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            error_message = "Invalid output format: " + format;
            return false;
        }
    }

    return true;
}

std::string ConfigParser::generate_default_yaml() {
    std::ostringstream yaml;

    yaml << "# SIMD-Bench Configuration File\n\n";

    yaml << "benchmarks:\n";
    yaml << "  - name: dot_product\n";
    yaml << "    sizes: [1024, 4096, 16384, 65536]\n";
    yaml << "    iterations: 1000\n";
    yaml << "    warmup: 10\n\n";

    yaml << "hardware_counters:\n";
    yaml << "  backend: auto  # auto, perf, papi, likwid\n";
    yaml << "  enabled: true\n";
    yaml << "  events:\n";
    yaml << "    - CYCLES\n";
    yaml << "    - INSTRUCTIONS\n";
    yaml << "    - FP_ARITH_256B_PACKED_SINGLE\n";
    yaml << "    - L1D_READ_MISS\n\n";

    yaml << "analysis:\n";
    yaml << "  roofline: true\n";
    yaml << "  tma: true\n";
    yaml << "  insights: true\n";
    yaml << "  scaling: false\n";
    yaml << "  prefetch: false\n";
    yaml << "  register_pressure: false\n\n";

    yaml << "output:\n";
    yaml << "  formats: [json, html, markdown]\n";
    yaml << "  path: ./reports/\n";
    yaml << "  console: true\n";
    yaml << "  verbose: false\n\n";

    yaml << "thresholds:\n";
    yaml << "  vectorization_warning: 0.8\n";
    yaml << "  cache_miss_warning: 0.05\n";
    yaml << "  efficiency_warning: 0.5\n";
    yaml << "  regression_threshold: 0.05\n";

    return yaml.str();
}

std::string ConfigParser::generate_default_json() {
    std::ostringstream json;

    json << "{\n";
    json << "  \"benchmarks\": [\n";
    json << "    {\n";
    json << "      \"name\": \"dot_product\",\n";
    json << "      \"sizes\": [1024, 4096, 16384, 65536],\n";
    json << "      \"iterations\": 1000,\n";
    json << "      \"warmup\": 10\n";
    json << "    }\n";
    json << "  ],\n";
    json << "  \"hardware_counters\": {\n";
    json << "    \"backend\": \"auto\",\n";
    json << "    \"enabled\": true\n";
    json << "  },\n";
    json << "  \"analysis\": {\n";
    json << "    \"roofline\": true,\n";
    json << "    \"tma\": true,\n";
    json << "    \"insights\": true\n";
    json << "  },\n";
    json << "  \"output\": {\n";
    json << "    \"formats\": [\"json\"],\n";
    json << "    \"path\": \"./\",\n";
    json << "    \"verbose\": false\n";
    json << "  },\n";
    json << "  \"thresholds\": {\n";
    json << "    \"regression_threshold\": 0.05\n";
    json << "  }\n";
    json << "}\n";

    return json.str();
}

bool ConfigParser::write_yaml(const FileConfig& config, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "# SIMD-Bench Configuration\n\n";

    file << "hardware_counters:\n";
    file << "  backend: " << config.counters.backend << "\n";
    file << "  enabled: " << (config.counters.enabled ? "true" : "false") << "\n\n";

    file << "analysis:\n";
    file << "  roofline: " << (config.analysis.roofline ? "true" : "false") << "\n";
    file << "  tma: " << (config.analysis.tma ? "true" : "false") << "\n";
    file << "  insights: " << (config.analysis.insights ? "true" : "false") << "\n\n";

    file << "output:\n";
    file << "  path: " << config.output.path << "\n";
    file << "  verbose: " << (config.output.verbose ? "true" : "false") << "\n\n";

    file << "thresholds:\n";
    file << "  regression_threshold: " << config.thresholds.regression_threshold << "\n";

    return true;
}

bool ConfigParser::write_json(const FileConfig& config, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "{\n";
    file << "  \"hardware_counters\": {\n";
    file << "    \"backend\": \"" << config.counters.backend << "\",\n";
    file << "    \"enabled\": " << (config.counters.enabled ? "true" : "false") << "\n";
    file << "  },\n";
    file << "  \"analysis\": {\n";
    file << "    \"roofline\": " << (config.analysis.roofline ? "true" : "false") << ",\n";
    file << "    \"tma\": " << (config.analysis.tma ? "true" : "false") << ",\n";
    file << "    \"insights\": " << (config.analysis.insights ? "true" : "false") << "\n";
    file << "  },\n";
    file << "  \"thresholds\": {\n";
    file << "    \"regression_threshold\": " << config.thresholds.regression_threshold << "\n";
    file << "  }\n";
    file << "}\n";

    return true;
}

// ============================================================================
// ConfigManager implementation
// ============================================================================

ConfigManager::ConfigManager() {
    // Set defaults
    config_.counters.backend = "auto";
    config_.counters.enabled = true;
    config_.analysis.roofline = true;
    config_.analysis.tma = true;
    config_.analysis.insights = true;
    config_.output.console = true;
    config_.thresholds.regression_threshold = 0.05;
}

bool ConfigManager::load(const std::string& filepath) {
    FileConfig loaded;

    if (filepath.find(".yaml") != std::string::npos ||
        filepath.find(".yml") != std::string::npos) {
        loaded = ConfigParser::parse_yaml(filepath);
    } else if (filepath.find(".json") != std::string::npos) {
        loaded = ConfigParser::parse_json(filepath);
    } else {
        return false;
    }

    config_ = ConfigParser::merge(config_, loaded);
    loaded_path_ = filepath;
    return true;
}

bool ConfigManager::load_default() {
    // Try common config file locations
    std::vector<std::string> paths = {
        "./simd_bench.yaml",
        "./simd_bench.yml",
        "./simd_bench.json",
        "./config/simd_bench.yaml",
        "~/.config/simd_bench/config.yaml"
    };

    for (const auto& path : paths) {
        if (load(path)) {
            return true;
        }
    }

    // Also load from environment
    auto env_config = ConfigParser::parse_environment();
    config_ = ConfigParser::merge(config_, env_config);

    return true;
}

BenchmarkConfig ConfigManager::to_benchmark_config() const {
    BenchmarkConfig bc;

    bc.enable_hardware_counters = config_.counters.enabled;
    bc.enable_roofline = config_.analysis.roofline;
    bc.enable_tma = config_.analysis.tma;
    bc.regression_threshold = config_.thresholds.regression_threshold;

    // Map output format
    if (!config_.output.formats.empty()) {
        bc.output_format = config_.output.formats[0];
    }
    bc.output_path = config_.output.path;

    return bc;
}

bool ConfigManager::has(const std::string& key) const {
    return config_.custom.find(key) != config_.custom.end();
}

std::vector<std::string> ConfigManager::custom_keys() const {
    std::vector<std::string> keys;
    for (const auto& [key, _] : config_.custom) {
        keys.push_back(key);
    }
    return keys;
}

// Template specializations for ConfigManager::get
template<>
bool ConfigManager::get<bool>(const std::string& key, const bool& default_value) const {
    auto it = config_.custom.find(key);
    if (it == config_.custom.end()) return default_value;
    if (auto* val = std::get_if<bool>(&it->second)) return *val;
    // Try string conversion
    if (auto* str = std::get_if<std::string>(&it->second)) {
        return (*str == "true" || *str == "1" || *str == "yes");
    }
    return default_value;
}

template<>
int64_t ConfigManager::get<int64_t>(const std::string& key, const int64_t& default_value) const {
    auto it = config_.custom.find(key);
    if (it == config_.custom.end()) return default_value;
    if (auto* val = std::get_if<int64_t>(&it->second)) return *val;
    if (auto* str = std::get_if<std::string>(&it->second)) {
        try { return std::stoll(*str); } catch (...) { return default_value; }
    }
    return default_value;
}

template<>
double ConfigManager::get<double>(const std::string& key, const double& default_value) const {
    auto it = config_.custom.find(key);
    if (it == config_.custom.end()) return default_value;
    if (auto* val = std::get_if<double>(&it->second)) return *val;
    if (auto* i = std::get_if<int64_t>(&it->second)) return static_cast<double>(*i);
    if (auto* str = std::get_if<std::string>(&it->second)) {
        try { return std::stod(*str); } catch (...) { return default_value; }
    }
    return default_value;
}

template<>
std::string ConfigManager::get<std::string>(const std::string& key, const std::string& default_value) const {
    auto it = config_.custom.find(key);
    if (it == config_.custom.end()) return default_value;
    if (auto* val = std::get_if<std::string>(&it->second)) return *val;
    // Convert other types to string
    if (auto* b = std::get_if<bool>(&it->second)) return *b ? "true" : "false";
    if (auto* i = std::get_if<int64_t>(&it->second)) return std::to_string(*i);
    if (auto* d = std::get_if<double>(&it->second)) return std::to_string(*d);
    return default_value;
}

// Template specializations for ConfigManager::set
template<>
void ConfigManager::set<bool>(const std::string& key, const bool& value) {
    config_.custom[key] = value;
}

template<>
void ConfigManager::set<int64_t>(const std::string& key, const int64_t& value) {
    config_.custom[key] = value;
}

template<>
void ConfigManager::set<double>(const std::string& key, const double& value) {
    config_.custom[key] = value;
}

template<>
void ConfigManager::set<std::string>(const std::string& key, const std::string& value) {
    config_.custom[key] = value;
}

// ============================================================================
// Command-line parsing
// ============================================================================

CommandLineConfig parse_command_line(int argc, char* argv[]) {
    CommandLineConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            config.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--list-benchmarks") {
            config.list_benchmarks = true;
        } else if (arg == "--list-counters") {
            config.list_counters = true;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config.config_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if ((arg == "-f" || arg == "--format") && i + 1 < argc) {
            config.output_format = argv[++i];
        } else if ((arg == "-i" || arg == "--iterations") && i + 1 < argc) {
            config.iterations = std::stoull(argv[++i]);
        } else if ((arg == "-b" || arg == "--benchmark") && i + 1 < argc) {
            config.benchmarks.push_back(argv[++i]);
        } else if ((arg == "-s" || arg == "--size") && i + 1 < argc) {
            config.sizes.push_back(std::stoull(argv[++i]));
        } else if (arg.find("--set-") == 0 && i + 1 < argc) {
            std::string key = arg.substr(6);
            config.overrides[key] = argv[++i];
        }
    }

    return config;
}

FileConfig apply_overrides(const FileConfig& base, const CommandLineConfig& cli) {
    FileConfig result = base;

    // Apply command-line overrides
    if (cli.verbose) {
        result.output.verbose = true;
    }

    if (!cli.output_path.empty()) {
        result.output.path = cli.output_path;
    }

    if (!cli.output_format.empty()) {
        result.output.formats = {cli.output_format};
    }

    // Apply custom overrides
    for (const auto& [key, value] : cli.overrides) {
        result.custom[key] = value;
    }

    return result;
}

}  // namespace simd_bench
