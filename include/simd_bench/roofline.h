#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>

namespace simd_bench {

// Cache-aware roofline model ceiling
struct RooflineCeiling {
    std::string name;           // "L1", "L2", "L3", "DRAM", "Peak"
    double bandwidth_gbps;      // Bandwidth limit for memory ceilings
    double gflops;              // Compute limit for peak ceiling
    bool is_compute_ceiling;    // true for peak, false for memory
};

// Roofline model for performance analysis
class RooflineModel {
public:
    RooflineModel();
    explicit RooflineModel(const HardwareInfo& hw);

    // Configure model
    void set_peak_gflops(double peak);
    void set_peak_gflops_sp(double peak_sp);
    void set_peak_gflops_dp(double peak_dp);

    void add_ceiling(const std::string& name, double bandwidth_gbps);
    void clear_ceilings();

    // Default ceilings from hardware info
    void configure_from_hardware(const HardwareInfo& hw);

    // Calculate theoretical maximum GFLOPS for given arithmetic intensity
    double get_theoretical_max(double arithmetic_intensity, bool double_precision = false) const;

    // Determine which ceiling is the bottleneck
    std::string get_limiting_ceiling(double arithmetic_intensity, bool double_precision = false) const;

    // Calculate ridge point (where compute and memory lines intersect)
    double get_ridge_point(const std::string& ceiling_name = "DRAM", bool double_precision = false) const;

    // Analyze a kernel measurement
    RooflinePoint analyze(double arithmetic_intensity, double achieved_gflops, bool double_precision = false) const;

    // Get all ceilings
    const std::vector<RooflineCeiling>& get_ceilings() const { return ceilings_; }

    // Generate SVG plot
    std::string generate_svg(
        const std::vector<RooflinePoint>& points,
        int width = 800,
        int height = 600
    ) const;

    // Generate data for external plotting
    struct PlotData {
        std::vector<double> x_values;  // Arithmetic intensity
        std::vector<std::vector<double>> ceiling_lines;  // Y values for each ceiling
        std::vector<std::string> ceiling_names;
    };
    PlotData get_plot_data(double min_ai = 0.01, double max_ai = 100.0, int num_points = 100) const;

private:
    double peak_gflops_sp_ = 0.0;
    double peak_gflops_dp_ = 0.0;
    std::vector<RooflineCeiling> ceilings_;

    double calculate_ceiling_gflops(const RooflineCeiling& ceiling, double ai) const;
};

// Empirical roofline - measures actual bandwidth at different cache levels
class EmpiricalRoofline {
public:
    EmpiricalRoofline();

    // Run bandwidth measurements
    void measure_bandwidths();

    // Get measured values
    double get_l1_bandwidth_gbps() const { return l1_bw_; }
    double get_l2_bandwidth_gbps() const { return l2_bw_; }
    double get_l3_bandwidth_gbps() const { return l3_bw_; }
    double get_dram_bandwidth_gbps() const { return dram_bw_; }
    double get_peak_gflops() const { return peak_gflops_; }

    // Create roofline model from measurements
    RooflineModel create_model() const;

private:
    double l1_bw_ = 0.0;
    double l2_bw_ = 0.0;
    double l3_bw_ = 0.0;
    double dram_bw_ = 0.0;
    double peak_gflops_ = 0.0;

    double measure_bandwidth_at_size(size_t size_bytes);
    double measure_peak_compute();
};

// Dynamic Arithmetic Intensity measurement from hardware counters
struct DynamicAIResult {
    double measured_ai = 0.0;           // Actual FLOP/byte from counters
    double theoretical_ai = 0.0;        // Theoretical minimum AI
    double cache_amplification = 1.0;   // How much extra memory traffic vs theoretical
    uint64_t total_flops = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    std::string ai_classification;      // "streaming", "memory-bound", "compute-bound"
    std::vector<std::string> insights;  // Observations about cache behavior
};

// Counter-based dynamic AI calculator
class DynamicAIAnalyzer {
public:
    // Calculate AI from offcore/IMC counter values
    // bytes_per_cacheline typically 64
    static DynamicAIResult analyze(
        uint64_t total_flops,
        uint64_t offcore_demand_data_rd,   // CounterEvent::OFFCORE_REQUESTS_DEMAND_DATA_RD
        uint64_t offcore_demand_rfo,       // CounterEvent::OFFCORE_REQUESTS_DEMAND_RFO
        size_t bytes_per_cacheline = 64
    );

    // Calculate AI from L3 miss data (fallback when IMC unavailable)
    static DynamicAIResult analyze_from_l3_misses(
        uint64_t total_flops,
        uint64_t l3_read_misses,
        uint64_t l3_write_misses,
        size_t bytes_per_cacheline = 64
    );

    // Calculate AI from IMC CAS counts (most accurate)
    static DynamicAIResult analyze_from_imc(
        uint64_t total_flops,
        uint64_t imc_cas_reads,
        uint64_t imc_cas_writes,
        size_t bytes_per_transaction = 64   // DDR4: 64 bytes per CAS
    );

    // Compare measured vs theoretical AI
    static void set_theoretical_ai(DynamicAIResult& result, double theoretical_ai);

    // Generate insights about cache behavior
    static std::vector<std::string> generate_insights(const DynamicAIResult& result);
};

// Enhanced roofline point with dynamic AI
struct EnhancedRooflinePoint {
    RooflinePoint base;
    DynamicAIResult dynamic_ai;
    double efficiency_with_measured_ai = 0.0;  // Using measured AI
    double efficiency_with_theoretical_ai = 0.0; // Using theoretical AI
    std::string cache_behavior;  // "efficient", "thrashing", "streaming"
};

// Roofline recommendation generator
struct RooflineRecommendation {
    std::string category;     // "optimization", "bottleneck", "info"
    std::string message;
    double potential_speedup; // Estimated improvement if implemented
};

std::vector<RooflineRecommendation> generate_recommendations(
    const RooflinePoint& point,
    const RooflineModel& model
);

// Enhanced recommendations with dynamic AI insights
std::vector<RooflineRecommendation> generate_enhanced_recommendations(
    const EnhancedRooflinePoint& point,
    const RooflineModel& model
);

}  // namespace simd_bench
