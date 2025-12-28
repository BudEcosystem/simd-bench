#include "simd_bench/roofline.h"
#include "simd_bench/timing.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace hn = hwy::HWY_NAMESPACE;

namespace simd_bench {

RooflineModel::RooflineModel() {
    // Add default peak ceiling
    RooflineCeiling peak;
    peak.name = "Peak";
    peak.gflops = 100.0;  // Default, will be overwritten
    peak.is_compute_ceiling = true;
    ceilings_.push_back(peak);
}

RooflineModel::RooflineModel(const HardwareInfo& hw) {
    configure_from_hardware(hw);
}

void RooflineModel::set_peak_gflops(double peak) {
    peak_gflops_sp_ = peak;
    peak_gflops_dp_ = peak / 2.0;

    // Update peak ceiling
    for (auto& ceiling : ceilings_) {
        if (ceiling.is_compute_ceiling) {
            ceiling.gflops = peak;
            break;
        }
    }
}

void RooflineModel::set_peak_gflops_sp(double peak_sp) {
    peak_gflops_sp_ = peak_sp;

    for (auto& ceiling : ceilings_) {
        if (ceiling.is_compute_ceiling) {
            ceiling.gflops = peak_sp;
            break;
        }
    }
}

void RooflineModel::set_peak_gflops_dp(double peak_dp) {
    peak_gflops_dp_ = peak_dp;
}

void RooflineModel::add_ceiling(const std::string& name, double bandwidth_gbps) {
    RooflineCeiling ceiling;
    ceiling.name = name;
    ceiling.bandwidth_gbps = bandwidth_gbps;
    ceiling.is_compute_ceiling = false;
    ceilings_.push_back(ceiling);
}

void RooflineModel::clear_ceilings() {
    ceilings_.clear();

    // Re-add peak ceiling
    RooflineCeiling peak;
    peak.name = "Peak";
    peak.gflops = peak_gflops_sp_;
    peak.is_compute_ceiling = true;
    ceilings_.push_back(peak);
}

void RooflineModel::configure_from_hardware(const HardwareInfo& hw) {
    ceilings_.clear();

    // Peak compute ceiling
    RooflineCeiling peak;
    peak.name = "Peak";
    peak.gflops = hw.theoretical_peak_sp_gflops;
    peak.is_compute_ceiling = true;
    ceilings_.push_back(peak);

    peak_gflops_sp_ = hw.theoretical_peak_sp_gflops;
    peak_gflops_dp_ = hw.theoretical_peak_dp_gflops;

    // Add memory ceilings based on cache hierarchy
    // Estimate bandwidths based on typical ratios
    double dram_bw = hw.measured_memory_bw_gbps > 0 ? hw.measured_memory_bw_gbps : 20.0;

    add_ceiling("DRAM", dram_bw);
    add_ceiling("L3", dram_bw * 3.0);   // L3 typically 3x DRAM
    add_ceiling("L2", dram_bw * 6.0);   // L2 typically 6x DRAM
    add_ceiling("L1", dram_bw * 15.0);  // L1 typically 15x DRAM
}

double RooflineModel::get_theoretical_max(double arithmetic_intensity, bool double_precision) const {
    double peak = double_precision ? peak_gflops_dp_ : peak_gflops_sp_;
    double max_gflops = peak;

    // Check all memory ceilings
    for (const auto& ceiling : ceilings_) {
        if (!ceiling.is_compute_ceiling) {
            double ceiling_gflops = arithmetic_intensity * ceiling.bandwidth_gbps;
            max_gflops = std::min(max_gflops, ceiling_gflops);
        }
    }

    return std::min(max_gflops, peak);
}

std::string RooflineModel::get_limiting_ceiling(double arithmetic_intensity, bool double_precision) const {
    double peak = double_precision ? peak_gflops_dp_ : peak_gflops_sp_;
    double min_gflops = peak;
    std::string limiting = "Peak";

    for (const auto& ceiling : ceilings_) {
        double ceiling_gflops;
        if (ceiling.is_compute_ceiling) {
            ceiling_gflops = peak;
        } else {
            ceiling_gflops = arithmetic_intensity * ceiling.bandwidth_gbps;
        }

        if (ceiling_gflops < min_gflops) {
            min_gflops = ceiling_gflops;
            limiting = ceiling.name;
        }
    }

    return limiting;
}

double RooflineModel::get_ridge_point(const std::string& ceiling_name, bool double_precision) const {
    double peak = double_precision ? peak_gflops_dp_ : peak_gflops_sp_;

    for (const auto& ceiling : ceilings_) {
        if (ceiling.name == ceiling_name && !ceiling.is_compute_ceiling) {
            return peak / ceiling.bandwidth_gbps;
        }
    }

    // Default: use DRAM if specified ceiling not found
    for (const auto& ceiling : ceilings_) {
        if (ceiling.name == "DRAM" && !ceiling.is_compute_ceiling) {
            return peak / ceiling.bandwidth_gbps;
        }
    }

    return 10.0;  // Default ridge point
}

RooflinePoint RooflineModel::analyze(double arithmetic_intensity, double achieved_gflops,
                                     bool double_precision) const {
    RooflinePoint point;
    point.arithmetic_intensity = arithmetic_intensity;
    point.achieved_gflops = achieved_gflops;

    double theoretical_max = get_theoretical_max(arithmetic_intensity, double_precision);
    point.efficiency = achieved_gflops / theoretical_max;

    // Determine bound
    std::string limiting = get_limiting_ceiling(arithmetic_intensity, double_precision);
    point.bound = (limiting == "Peak") ? "compute" : limiting;

    return point;
}

double RooflineModel::calculate_ceiling_gflops(const RooflineCeiling& ceiling, double ai) const {
    if (ceiling.is_compute_ceiling) {
        return ceiling.gflops;
    }
    return ai * ceiling.bandwidth_gbps;
}

std::string RooflineModel::generate_svg(const std::vector<RooflinePoint>& points,
                                        int width, int height) const {
    std::ostringstream svg;

    // SVG header
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << width << "\" height=\"" << height << "\" "
        << "viewBox=\"0 0 " << width << " " << height << "\">\n";

    // Background
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    // Margins
    int margin_left = 80;
    int margin_right = 40;
    int margin_top = 40;
    int margin_bottom = 60;

    int plot_width = width - margin_left - margin_right;
    int plot_height = height - margin_top - margin_bottom;

    // Log scale parameters
    double min_ai = 0.01;
    double max_ai = 100.0;
    double min_gflops = 0.1;
    double max_gflops = peak_gflops_sp_ * 1.2;

    auto log_x = [&](double ai) {
        return margin_left + plot_width * (std::log10(ai) - std::log10(min_ai)) /
               (std::log10(max_ai) - std::log10(min_ai));
    };

    auto log_y = [&](double gflops) {
        return margin_top + plot_height * (1.0 - (std::log10(gflops) - std::log10(min_gflops)) /
               (std::log10(max_gflops) - std::log10(min_gflops)));
    };

    // Draw axes
    svg << "<line x1=\"" << margin_left << "\" y1=\"" << margin_top + plot_height
        << "\" x2=\"" << margin_left + plot_width << "\" y2=\"" << margin_top + plot_height
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    svg << "<line x1=\"" << margin_left << "\" y1=\"" << margin_top
        << "\" x2=\"" << margin_left << "\" y2=\"" << margin_top + plot_height
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";

    // Draw roofline ceilings
    std::vector<std::string> colors = {"#FF0000", "#00AA00", "#0000FF", "#FF8800", "#8800FF"};
    int color_idx = 0;

    for (const auto& ceiling : ceilings_) {
        std::string color = colors[color_idx % colors.size()];
        color_idx++;

        if (ceiling.is_compute_ceiling) {
            // Horizontal line at peak
            double y = log_y(ceiling.gflops);
            svg << "<line x1=\"" << margin_left << "\" y1=\"" << y
                << "\" x2=\"" << margin_left + plot_width << "\" y2=\"" << y
                << "\" stroke=\"" << color << "\" stroke-width=\"2\" stroke-dasharray=\"5,5\"/>\n";
        } else {
            // Sloped line for memory ceiling
            double x1 = log_x(min_ai);
            double y1 = log_y(min_ai * ceiling.bandwidth_gbps);
            double x2 = log_x(max_ai);
            double y2 = log_y(max_ai * ceiling.bandwidth_gbps);

            // Clip to plot area
            y1 = std::max(y1, static_cast<double>(margin_top));
            y2 = std::max(y2, static_cast<double>(margin_top));

            svg << "<line x1=\"" << x1 << "\" y1=\"" << y1
                << "\" x2=\"" << x2 << "\" y2=\"" << y2
                << "\" stroke=\"" << color << "\" stroke-width=\"2\"/>\n";
        }

        // Label
        svg << "<text x=\"" << margin_left + 10 << "\" y=\"" << margin_top + color_idx * 15
            << "\" fill=\"" << color << "\" font-size=\"12\">" << ceiling.name << "</text>\n";
    }

    // Draw data points
    for (const auto& point : points) {
        double x = log_x(point.arithmetic_intensity);
        double y = log_y(point.achieved_gflops);

        svg << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"6\" "
            << "fill=\"#FF0000\" stroke=\"black\" stroke-width=\"1\"/>\n";
    }

    // Axis labels
    svg << "<text x=\"" << margin_left + plot_width / 2 << "\" y=\"" << height - 10
        << "\" text-anchor=\"middle\" font-size=\"14\">Arithmetic Intensity (FLOP/byte)</text>\n";
    svg << "<text x=\"15\" y=\"" << margin_top + plot_height / 2
        << "\" text-anchor=\"middle\" font-size=\"14\" "
        << "transform=\"rotate(-90 15 " << margin_top + plot_height / 2 << ")\">GFLOPS</text>\n";

    // Title
    svg << "<text x=\"" << width / 2 << "\" y=\"25\" text-anchor=\"middle\" "
        << "font-size=\"16\" font-weight=\"bold\">Roofline Model</text>\n";

    svg << "</svg>\n";

    return svg.str();
}

RooflineModel::PlotData RooflineModel::get_plot_data(double min_ai, double max_ai,
                                                      int num_points) const {
    PlotData data;

    // Generate X values (log scale)
    double log_min = std::log10(min_ai);
    double log_max = std::log10(max_ai);
    double step = (log_max - log_min) / (num_points - 1);

    for (int i = 0; i < num_points; ++i) {
        data.x_values.push_back(std::pow(10.0, log_min + i * step));
    }

    // Generate Y values for each ceiling
    for (const auto& ceiling : ceilings_) {
        data.ceiling_names.push_back(ceiling.name);
        std::vector<double> y_values;

        for (double ai : data.x_values) {
            y_values.push_back(calculate_ceiling_gflops(ceiling, ai));
        }

        data.ceiling_lines.push_back(y_values);
    }

    return data;
}

// EmpiricalRoofline implementation
EmpiricalRoofline::EmpiricalRoofline() {}

void EmpiricalRoofline::measure_bandwidths() {
    // Measure L1 bandwidth (16 KB working set)
    l1_bw_ = measure_bandwidth_at_size(16 * 1024);

    // Measure L2 bandwidth (128 KB working set)
    l2_bw_ = measure_bandwidth_at_size(128 * 1024);

    // Measure L3 bandwidth (4 MB working set)
    l3_bw_ = measure_bandwidth_at_size(4 * 1024 * 1024);

    // Measure DRAM bandwidth (64 MB working set)
    dram_bw_ = measure_bandwidth_at_size(64 * 1024 * 1024);

    // Measure peak compute
    peak_gflops_ = measure_peak_compute();
}

double EmpiricalRoofline::measure_bandwidth_at_size(size_t size_bytes) {
    size_t count = size_bytes / sizeof(float);
    auto src = hwy::AllocateAligned<float>(count);
    auto dst = hwy::AllocateAligned<float>(count);

    // Initialize
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<float>(i);
    }

    // Warmup
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (int w = 0; w < 3; ++w) {
        for (size_t i = 0; i + N <= count; i += N) {
            hn::Store(hn::Load(d, src.get() + i), d, dst.get() + i);
        }
    }

    // Measure
    Timer timer;
    timer.start();

    const int iterations = 100;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + N <= count; i += N) {
            hn::Store(hn::Load(d, src.get() + i), d, dst.get() + i);
        }
    }

    timer.stop();

    double bytes = 2.0 * size_bytes * iterations;  // Read + Write
    return bytes / timer.elapsed_seconds() / 1e9;   // GB/s
}

double EmpiricalRoofline::measure_peak_compute() {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Initialize registers
    auto r0 = hn::Set(d, 1.0f), r1 = hn::Set(d, 1.1f);
    auto r2 = hn::Set(d, 1.2f), r3 = hn::Set(d, 1.3f);
    auto r4 = hn::Set(d, 1.4f), r5 = hn::Set(d, 1.5f);
    auto r6 = hn::Set(d, 1.6f), r7 = hn::Set(d, 1.7f);
    const auto one = hn::Set(d, 1.0f);

    // Warmup
    for (size_t i = 0; i < 1000; ++i) {
        r0 = hn::MulAdd(r0, r4, one);
        r1 = hn::MulAdd(r1, r5, one);
        r2 = hn::MulAdd(r2, r6, one);
        r3 = hn::MulAdd(r3, r7, one);
    }

    // Measure
    Timer timer;
    timer.start();

    const size_t iterations = 100000000;
    for (size_t i = 0; i < iterations; ++i) {
        r0 = hn::MulAdd(r0, r4, one);
        r1 = hn::MulAdd(r1, r5, one);
        r2 = hn::MulAdd(r2, r6, one);
        r3 = hn::MulAdd(r3, r7, one);
        r4 = hn::MulAdd(r4, r0, one);
        r5 = hn::MulAdd(r5, r1, one);
        r6 = hn::MulAdd(r6, r2, one);
        r7 = hn::MulAdd(r7, r3, one);
    }

    timer.stop();

    // Prevent optimization
    volatile float sink = hn::ReduceSum(d, r0) + hn::ReduceSum(d, r7);
    (void)sink;

    // 8 FMAs per iteration, 2 FLOPS per FMA, N lanes per FMA
    double flops = iterations * 8.0 * 2.0 * N;
    return flops / timer.elapsed_seconds() / 1e9;
}

RooflineModel EmpiricalRoofline::create_model() const {
    RooflineModel model;

    model.set_peak_gflops(peak_gflops_);
    model.add_ceiling("L1", l1_bw_);
    model.add_ceiling("L2", l2_bw_);
    model.add_ceiling("L3", l3_bw_);
    model.add_ceiling("DRAM", dram_bw_);

    return model;
}

// Recommendation generation
std::vector<RooflineRecommendation> generate_recommendations(
    const RooflinePoint& point,
    const RooflineModel& model
) {
    std::vector<RooflineRecommendation> recommendations;

    double theoretical_max = model.get_theoretical_max(point.arithmetic_intensity);

    if (point.bound == "compute" || point.bound == "Peak") {
        // Compute bound
        if (point.efficiency < 0.7) {
            recommendations.push_back({
                "optimization",
                "Increase instruction-level parallelism with more independent operations",
                1.0 / point.efficiency
            });
        }
        if (point.efficiency < 0.9) {
            recommendations.push_back({
                "optimization",
                "Consider using FMA instructions to maximize throughput",
                theoretical_max / point.achieved_gflops
            });
        }
    } else {
        // Memory bound
        recommendations.push_back({
            "bottleneck",
            "Kernel is memory-bound on " + point.bound,
            0.0
        });

        if (point.arithmetic_intensity < 1.0) {
            recommendations.push_back({
                "optimization",
                "Increase arithmetic intensity through loop tiling or cache blocking",
                2.0
            });
        }

        recommendations.push_back({
            "optimization",
            "Consider using non-temporal stores for streaming access patterns",
            1.3
        });

        recommendations.push_back({
            "optimization",
            "Prefetch data to hide memory latency",
            1.2
        });
    }

    if (point.efficiency < 0.5) {
        recommendations.push_back({
            "info",
            "Current efficiency is " + std::to_string(static_cast<int>(point.efficiency * 100)) +
            "% of theoretical peak",
            0.0
        });
    }

    return recommendations;
}

// DynamicAIAnalyzer implementation
DynamicAIResult DynamicAIAnalyzer::analyze(
    uint64_t total_flops,
    uint64_t offcore_demand_data_rd,
    uint64_t offcore_demand_rfo,
    size_t bytes_per_cacheline
) {
    DynamicAIResult result;
    result.total_flops = total_flops;

    // Each offcore request fetches a full cache line
    result.bytes_read = offcore_demand_data_rd * bytes_per_cacheline;
    result.bytes_written = offcore_demand_rfo * bytes_per_cacheline;

    uint64_t total_bytes = result.bytes_read + result.bytes_written;

    if (total_bytes > 0) {
        result.measured_ai = static_cast<double>(total_flops) / static_cast<double>(total_bytes);
    } else {
        result.measured_ai = 0.0;
    }

    // Classify AI level
    result.ai_classification = ArithmeticIntensityCalculator::classify(result.measured_ai);

    result.insights = generate_insights(result);

    return result;
}

DynamicAIResult DynamicAIAnalyzer::analyze_from_l3_misses(
    uint64_t total_flops,
    uint64_t l3_read_misses,
    uint64_t l3_write_misses,
    size_t bytes_per_cacheline
) {
    DynamicAIResult result;
    result.total_flops = total_flops;

    // L3 misses go to DRAM
    result.bytes_read = l3_read_misses * bytes_per_cacheline;
    result.bytes_written = l3_write_misses * bytes_per_cacheline;

    uint64_t total_bytes = result.bytes_read + result.bytes_written;

    if (total_bytes > 0) {
        result.measured_ai = static_cast<double>(total_flops) / static_cast<double>(total_bytes);
    } else {
        result.measured_ai = 0.0;
    }

    result.ai_classification = ArithmeticIntensityCalculator::classify(result.measured_ai);
    result.insights = generate_insights(result);

    return result;
}

DynamicAIResult DynamicAIAnalyzer::analyze_from_imc(
    uint64_t total_flops,
    uint64_t imc_cas_reads,
    uint64_t imc_cas_writes,
    size_t bytes_per_transaction
) {
    DynamicAIResult result;
    result.total_flops = total_flops;

    // IMC CAS counts are direct memory controller transactions
    result.bytes_read = imc_cas_reads * bytes_per_transaction;
    result.bytes_written = imc_cas_writes * bytes_per_transaction;

    uint64_t total_bytes = result.bytes_read + result.bytes_written;

    if (total_bytes > 0) {
        result.measured_ai = static_cast<double>(total_flops) / static_cast<double>(total_bytes);
    } else {
        result.measured_ai = 0.0;
    }

    result.ai_classification = ArithmeticIntensityCalculator::classify(result.measured_ai);
    result.insights = generate_insights(result);

    return result;
}

void DynamicAIAnalyzer::set_theoretical_ai(DynamicAIResult& result, double theoretical_ai) {
    result.theoretical_ai = theoretical_ai;

    if (result.measured_ai > 0) {
        result.cache_amplification = ArithmeticIntensityCalculator::cache_amplification(
            theoretical_ai, result.measured_ai);
    } else {
        result.cache_amplification = 1.0;
    }

    // Re-generate insights with theoretical AI context
    result.insights = generate_insights(result);
}

std::vector<std::string> DynamicAIAnalyzer::generate_insights(const DynamicAIResult& result) {
    std::vector<std::string> insights;

    // Basic AI classification insight
    if (result.measured_ai < 0.25) {
        insights.push_back("Very low AI indicates streaming access pattern (memory-bound)");
    } else if (result.measured_ai < 1.0) {
        insights.push_back("Low AI suggests memory bandwidth is the primary bottleneck");
    } else if (result.measured_ai < 4.0) {
        insights.push_back("Moderate AI - kernel is in transition zone between memory and compute bound");
    } else if (result.measured_ai < 16.0) {
        insights.push_back("High AI indicates kernel is approaching compute-bound regime");
    } else {
        insights.push_back("Very high AI - kernel is compute-bound, focus on instruction throughput");
    }

    // Cache amplification insights
    if (result.theoretical_ai > 0) {
        double amplification = result.cache_amplification;

        if (amplification > 2.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1);
            oss << "Cache amplification of " << amplification << "x indicates significant cache thrashing";
            insights.push_back(oss.str());
            insights.push_back("Consider loop tiling or blocking to improve cache reuse");
        } else if (amplification > 1.5) {
            insights.push_back("Moderate cache amplification - some cache inefficiency detected");
            insights.push_back("Prefetching or data layout changes may help");
        } else if (amplification < 0.8) {
            insights.push_back("Excellent cache reuse - measured AI exceeds theoretical minimum");
        }
    }

    // Read/write ratio insights
    if (result.bytes_read > 0 && result.bytes_written > 0) {
        double read_ratio = static_cast<double>(result.bytes_read) /
                            static_cast<double>(result.bytes_read + result.bytes_written);

        if (read_ratio > 0.9) {
            insights.push_back("Workload is read-dominated - good candidate for read-ahead prefetching");
        } else if (read_ratio < 0.3) {
            insights.push_back("Workload is write-dominated - consider non-temporal stores");
        }
    }

    return insights;
}

// Enhanced recommendations with dynamic AI
std::vector<RooflineRecommendation> generate_enhanced_recommendations(
    const EnhancedRooflinePoint& point,
    const RooflineModel& model
) {
    // Start with base recommendations
    std::vector<RooflineRecommendation> recommendations =
        generate_recommendations(point.base, model);

    const auto& ai = point.dynamic_ai;

    // Add dynamic AI-specific recommendations
    if (ai.cache_amplification > 2.0) {
        recommendations.push_back({
            "cache",
            "Cache amplification " + std::to_string(static_cast<int>(ai.cache_amplification)) +
            "x detected - implement loop tiling to improve cache reuse",
            ai.cache_amplification
        });
    }

    if (ai.measured_ai < 0.5 && ai.theoretical_ai > 1.0) {
        recommendations.push_back({
            "cache",
            "Measured AI is much lower than theoretical - data layout may cause cache conflicts",
            ai.theoretical_ai / ai.measured_ai
        });
    }

    if (point.cache_behavior == "thrashing") {
        recommendations.push_back({
            "cache",
            "Cache thrashing detected - consider reducing working set or using cache-oblivious algorithms",
            2.0
        });
    }

    // Add insights as info recommendations
    for (const auto& insight : ai.insights) {
        recommendations.push_back({
            "info",
            insight,
            0.0
        });
    }

    return recommendations;
}

}  // namespace simd_bench
