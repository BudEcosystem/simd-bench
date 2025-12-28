#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/roofline.h"
#include <cmath>

namespace simd_bench {
namespace testing {

class RooflineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a typical roofline model
        model_.set_peak_gflops_sp(100.0);
        model_.set_peak_gflops_dp(50.0);
        model_.add_ceiling("L1", 200.0);    // 200 GB/s L1
        model_.add_ceiling("L2", 80.0);     // 80 GB/s L2
        model_.add_ceiling("L3", 40.0);     // 40 GB/s L3
        model_.add_ceiling("DRAM", 20.0);   // 20 GB/s DRAM
    }

    RooflineModel model_;
};

// Test peak GFLOPS setting
TEST_F(RooflineTest, SetPeakGflopsStoresValue) {
    RooflineModel model;
    model.set_peak_gflops(120.0);

    // At high AI, should be compute bound at peak
    double max = model.get_theoretical_max(1000.0);
    EXPECT_NEAR(max, 120.0, 0.01);
}

TEST_F(RooflineTest, SetPeakGflopsSPAndDP) {
    RooflineModel model;
    model.set_peak_gflops_sp(200.0);
    model.set_peak_gflops_dp(100.0);

    double max_sp = model.get_theoretical_max(1000.0, false);
    double max_dp = model.get_theoretical_max(1000.0, true);

    EXPECT_NEAR(max_sp, 200.0, 0.01);
    EXPECT_NEAR(max_dp, 100.0, 0.01);
}

// Test ceiling addition
TEST_F(RooflineTest, AddCeilingIncreasesCount) {
    RooflineModel model;
    model.set_peak_gflops(100.0);

    EXPECT_EQ(model.get_ceilings().size(), 1u);  // Peak ceiling

    model.add_ceiling("L1", 200.0);
    EXPECT_EQ(model.get_ceilings().size(), 2u);

    model.add_ceiling("DRAM", 20.0);
    EXPECT_EQ(model.get_ceilings().size(), 3u);
}

TEST_F(RooflineTest, ClearCeilingsRemovesAll) {
    model_.clear_ceilings();
    model_.set_peak_gflops(100.0);

    // Should only have peak ceiling after clear + set
    EXPECT_EQ(model_.get_ceilings().size(), 1u);
}

// Test theoretical max calculation
TEST_F(RooflineTest, TheoreticalMaxIsMemoryBoundAtLowAI) {
    // At AI=0.1 FLOP/byte with 20 GB/s DRAM
    // Max = 0.1 * 20 = 2 GFLOPS
    double max = model_.get_theoretical_max(0.1);
    EXPECT_NEAR(max, 2.0, 0.1);
}

TEST_F(RooflineTest, TheoreticalMaxIsComputeBoundAtHighAI) {
    // At AI=100 FLOP/byte, should be compute bound
    double max = model_.get_theoretical_max(100.0);
    EXPECT_NEAR(max, 100.0, 0.1);
}

TEST_F(RooflineTest, TheoreticalMaxTransitionsAtRidgePoint) {
    // Ridge point = peak / bandwidth = 100 / 20 = 5 FLOP/byte
    double ridge = model_.get_ridge_point("DRAM");
    EXPECT_NEAR(ridge, 5.0, 0.1);

    // Just below ridge: memory bound
    double below_ridge = model_.get_theoretical_max(ridge * 0.5);
    EXPECT_LT(below_ridge, 100.0);

    // At ridge: should be close to peak
    double at_ridge = model_.get_theoretical_max(ridge);
    EXPECT_NEAR(at_ridge, 100.0, 5.0);
}

// Test limiting ceiling identification
TEST_F(RooflineTest, GetLimitingCeilingIdentifiesCorrectly) {
    // At very low AI, DRAM should be limiting
    std::string limiting_low = model_.get_limiting_ceiling(0.1);
    EXPECT_EQ(limiting_low, "DRAM");

    // At very high AI, Peak should be limiting
    std::string limiting_high = model_.get_limiting_ceiling(100.0);
    EXPECT_EQ(limiting_high, "Peak");
}

// Test ridge point calculation
TEST_F(RooflineTest, RidgePointCalculation) {
    double dram_ridge = model_.get_ridge_point("DRAM");
    double l1_ridge = model_.get_ridge_point("L1");

    // L1 ridge should be lower (faster memory)
    EXPECT_LT(l1_ridge, dram_ridge);

    // DRAM ridge = 100 / 20 = 5
    EXPECT_NEAR(dram_ridge, 5.0, 0.1);

    // L1 ridge = 100 / 200 = 0.5
    EXPECT_NEAR(l1_ridge, 0.5, 0.1);
}

// Test analyze function
TEST_F(RooflineTest, AnalyzeReturnsValidPoint) {
    RooflinePoint point = model_.analyze(1.0, 15.0);

    EXPECT_DOUBLE_EQ(point.arithmetic_intensity, 1.0);
    EXPECT_DOUBLE_EQ(point.achieved_gflops, 15.0);
    EXPECT_FALSE(point.bound.empty());
    EXPECT_GE(point.efficiency, 0.0);
    EXPECT_LE(point.efficiency, 1.0);
}

TEST_F(RooflineTest, AnalyzeIdentifiesMemoryBound) {
    // At AI=0.5, max = min(100, 0.5*20) = 10 GFLOPS (DRAM bound)
    RooflinePoint point = model_.analyze(0.5, 8.0);

    EXPECT_EQ(point.bound, "DRAM");
    EXPECT_NEAR(point.efficiency, 0.8, 0.1);  // 8/10 = 80%
}

TEST_F(RooflineTest, AnalyzeIdentifiesComputeBound) {
    // At AI=10, max = min(100, 10*20) = 100 GFLOPS (compute bound)
    RooflinePoint point = model_.analyze(10.0, 80.0);

    EXPECT_EQ(point.bound, "compute");
    EXPECT_NEAR(point.efficiency, 0.8, 0.1);  // 80/100 = 80%
}

// Test configure from hardware
TEST_F(RooflineTest, ConfigureFromHardwareCreatesValidModel) {
    HardwareInfo hw = HardwareInfo::detect();
    RooflineModel model;
    model.configure_from_hardware(hw);

    // Should have at least peak ceiling
    EXPECT_FALSE(model.get_ceilings().empty());

    // Should have reasonable theoretical max
    double max = model.get_theoretical_max(100.0);
    EXPECT_GT(max, 0.0);
}

// Test SVG generation
TEST_F(RooflineTest, GenerateSVGReturnsValidSVG) {
    std::vector<RooflinePoint> points = {
        {0.5, 8.0, "DRAM", 0.8},
        {2.0, 35.0, "L3", 0.9},
        {10.0, 90.0, "compute", 0.9}
    };

    std::string svg = model_.generate_svg(points);

    EXPECT_FALSE(svg.empty());
    EXPECT_NE(svg.find("<svg"), std::string::npos);
    EXPECT_NE(svg.find("</svg>"), std::string::npos);
}

TEST_F(RooflineTest, GenerateSVGWithCustomDimensions) {
    std::string svg = model_.generate_svg({}, 1024, 768);

    EXPECT_NE(svg.find("width=\"1024\""), std::string::npos);
    EXPECT_NE(svg.find("height=\"768\""), std::string::npos);
}

// Test plot data generation
TEST_F(RooflineTest, GetPlotDataReturnsValidData) {
    auto data = model_.get_plot_data(0.1, 100.0, 50);

    EXPECT_EQ(data.x_values.size(), 50u);
    EXPECT_FALSE(data.ceiling_lines.empty());
    EXPECT_FALSE(data.ceiling_names.empty());

    // X values should be in range
    EXPECT_GE(data.x_values.front(), 0.1);
    EXPECT_LE(data.x_values.back(), 100.0);
}

// Test EmpiricalRoofline
TEST_F(RooflineTest, EmpiricalRooflineMeasuresValues) {
    EmpiricalRoofline empirical;
    empirical.measure_bandwidths();

    EXPECT_GT(empirical.get_l1_bandwidth_gbps(), 0.0);
    EXPECT_GT(empirical.get_dram_bandwidth_gbps(), 0.0);
    EXPECT_GT(empirical.get_peak_gflops(), 0.0);
}

TEST_F(RooflineTest, EmpiricalRooflineCreateModel) {
    EmpiricalRoofline empirical;
    empirical.measure_bandwidths();

    RooflineModel model = empirical.create_model();

    EXPECT_FALSE(model.get_ceilings().empty());
    EXPECT_GT(model.get_theoretical_max(100.0), 0.0);
}

// Test recommendation generation
TEST_F(RooflineTest, GenerateRecommendationsForMemoryBound) {
    RooflinePoint point = {0.5, 8.0, "DRAM", 0.8};

    auto recommendations = generate_recommendations(point, model_);

    EXPECT_FALSE(recommendations.empty());
    // Should suggest improving AI or data locality
}

TEST_F(RooflineTest, GenerateRecommendationsForComputeBound) {
    RooflinePoint point = {10.0, 70.0, "compute", 0.7};

    auto recommendations = generate_recommendations(point, model_);

    EXPECT_FALSE(recommendations.empty());
    // Should suggest vectorization or algorithm improvements
}

TEST_F(RooflineTest, RecommendationsIncludePotentialSpeedup) {
    RooflinePoint point = {1.0, 10.0, "DRAM", 0.5};

    auto recommendations = generate_recommendations(point, model_);

    for (const auto& rec : recommendations) {
        // Each recommendation should have a valid category
        EXPECT_FALSE(rec.category.empty());
        EXPECT_FALSE(rec.message.empty());
        EXPECT_GE(rec.potential_speedup, 0.0);
    }
}

// Test double precision
TEST_F(RooflineTest, DoublePrecisionHalfPeak) {
    double sp_max = model_.get_theoretical_max(100.0, false);
    double dp_max = model_.get_theoretical_max(100.0, true);

    EXPECT_NEAR(sp_max / dp_max, 2.0, 0.1);
}

}  // namespace testing
}  // namespace simd_bench
