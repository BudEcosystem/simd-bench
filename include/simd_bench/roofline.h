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

}  // namespace simd_bench
