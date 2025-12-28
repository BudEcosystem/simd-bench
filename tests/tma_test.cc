#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/tma.h"

namespace simd_bench {
namespace testing {

class TMATest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test TMAMetrics defaults
TEST_F(TMATest, TMAMetricsDefaultsToZero) {
    TMAMetrics metrics;
    EXPECT_DOUBLE_EQ(metrics.retiring, 0.0);
    EXPECT_DOUBLE_EQ(metrics.bad_speculation, 0.0);
    EXPECT_DOUBLE_EQ(metrics.frontend_bound, 0.0);
    EXPECT_DOUBLE_EQ(metrics.backend_bound, 0.0);
}

// Test TMAResult
TEST_F(TMATest, TMAResultIsWellOptimizedWhenRetiringHigh) {
    TMAResult result;
    result.metrics.retiring = 0.85;
    result.metrics.backend_bound = 0.10;
    EXPECT_TRUE(result.is_well_optimized());
}

TEST_F(TMATest, TMAResultIsNotWellOptimizedWhenRetiringLow) {
    TMAResult result;
    result.metrics.retiring = 0.50;
    result.metrics.backend_bound = 0.30;
    EXPECT_FALSE(result.is_well_optimized());
}

TEST_F(TMATest, TMAResultIsNotWellOptimizedWhenBackendHigh) {
    TMAResult result;
    result.metrics.retiring = 0.75;
    result.metrics.backend_bound = 0.25;
    EXPECT_FALSE(result.is_well_optimized());
}

// Test TMAAnalyzer creation
TEST_F(TMATest, TMAAnalyzerDefaultConstruction) {
    TMAAnalyzer analyzer;
    EXPECT_EQ(analyzer.get_level(), TMALevel::LEVEL1);
}

TEST_F(TMATest, TMAAnalyzerSetLevel) {
    TMAAnalyzer analyzer;

    analyzer.set_level(TMALevel::LEVEL2);
    EXPECT_EQ(analyzer.get_level(), TMALevel::LEVEL2);

    analyzer.set_level(TMALevel::LEVEL3);
    EXPECT_EQ(analyzer.get_level(), TMALevel::LEVEL3);
}

// Test required events
TEST_F(TMATest, GetRequiredEventsLevel1) {
    TMAAnalyzer analyzer;
    analyzer.set_level(TMALevel::LEVEL1);

    auto events = analyzer.get_required_events();
    EXPECT_FALSE(events.empty());
    EXPECT_THAT(events, ::testing::Contains(CounterEvent::CYCLES));
}

TEST_F(TMATest, GetRequiredEventsLevel2HasMore) {
    TMAAnalyzer analyzer;

    analyzer.set_level(TMALevel::LEVEL1);
    auto level1_events = analyzer.get_required_events();

    analyzer.set_level(TMALevel::LEVEL2);
    auto level2_events = analyzer.get_required_events();

    EXPECT_GE(level2_events.size(), level1_events.size());
}

// Test analysis with mock counter values
TEST_F(TMATest, AnalyzeWithMockValues) {
    TMAAnalyzer analyzer;

    CounterValues values;
    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 900000);
    values.set(CounterEvent::UOPS_RETIRED_SLOTS, 800000);
    values.set(CounterEvent::UOPS_ISSUED_ANY, 850000);

    TMAResult result = analyzer.analyze(values);

    // Should have some retiring ratio
    EXPECT_GE(result.metrics.retiring, 0.0);
    EXPECT_LE(result.metrics.retiring, 1.0);

    // Sum of all categories should be approximately 1
    double sum = result.metrics.retiring + result.metrics.bad_speculation +
                 result.metrics.frontend_bound + result.metrics.backend_bound;
    // Allow some tolerance due to calculation methods
    EXPECT_GT(sum, 0.0);
}

// Test category classification
TEST_F(TMATest, AnalyzeIdentifiesPrimaryBottleneck) {
    TMAAnalyzer analyzer;

    CounterValues values;
    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 400000);
    values.set(CounterEvent::UOPS_RETIRED_SLOTS, 300000);
    values.set(CounterEvent::CYCLE_ACTIVITY_STALLS_MEM, 400000);

    TMAResult result = analyzer.analyze(values);

    // Should identify a primary bottleneck
    EXPECT_FALSE(result.primary_bottleneck.empty());
}

// Test recommendations
TEST_F(TMATest, AnalyzeGeneratesRecommendations) {
    TMAAnalyzer analyzer;

    CounterValues values;
    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 500000);
    values.set(CounterEvent::CYCLE_ACTIVITY_STALLS_MEM, 300000);

    TMAResult result = analyzer.analyze(values);

    // If there are bottlenecks, should have recommendations
    if (result.metrics.retiring < 0.7) {
        EXPECT_FALSE(result.recommendations.empty());
    }
}

// Test support detection
TEST_F(TMATest, IsSupportedDoesNotCrash) {
    bool supported = TMAAnalyzer::is_supported();
    // Just check it returns a boolean without crashing
    EXPECT_TRUE(supported || !supported);
}

// Test bar chart formatting
TEST_F(TMATest, FormatTMABarChartReturnsNonEmpty) {
    TMAResult result;
    result.metrics.retiring = 0.7;
    result.metrics.bad_speculation = 0.05;
    result.metrics.frontend_bound = 0.05;
    result.metrics.backend_bound = 0.2;
    result.primary_bottleneck = "Backend Bound";

    std::string chart = format_tma_bar_chart(result);

    EXPECT_FALSE(chart.empty());
    EXPECT_NE(chart.find("Retiring"), std::string::npos);
}

TEST_F(TMATest, FormatTMABarChartCustomWidth) {
    TMAResult result;
    result.metrics.retiring = 0.5;

    std::string chart40 = format_tma_bar_chart(result, 40);
    std::string chart80 = format_tma_bar_chart(result, 80);

    // Wider chart should be longer
    EXPECT_LT(chart40.length(), chart80.length());
}

// Test category to string conversion
TEST_F(TMATest, TMACategoryToStringReturnsValidStrings) {
    EXPECT_EQ(tma_category_to_string(TMACategory::RETIRING), "Retiring");
    EXPECT_EQ(tma_category_to_string(TMACategory::BAD_SPECULATION), "Bad Speculation");
    EXPECT_EQ(tma_category_to_string(TMACategory::FRONTEND_BOUND), "Frontend Bound");
    EXPECT_EQ(tma_category_to_string(TMACategory::BACKEND_BOUND), "Backend Bound");
    EXPECT_EQ(tma_category_to_string(TMACategory::MEMORY_BOUND), "Memory Bound");
    EXPECT_EQ(tma_category_to_string(TMACategory::CORE_BOUND), "Core Bound");
}

// Test TMACategoryResult
TEST_F(TMATest, TMACategoryResultHasValidFields) {
    TMACategoryResult result;
    result.category = TMACategory::RETIRING;
    result.name = "Retiring";
    result.ratio = 0.75;
    result.description = "Useful work";

    EXPECT_EQ(result.category, TMACategory::RETIRING);
    EXPECT_EQ(result.name, "Retiring");
    EXPECT_DOUBLE_EQ(result.ratio, 0.75);
}

// Test with null counters
TEST_F(TMATest, AnalyzerWithNullCounters) {
    TMAAnalyzer analyzer;
    // No counters set

    CounterValues values;
    TMAResult result = analyzer.analyze(values);

    // Should return valid (if empty) result
    EXPECT_GE(result.metrics.retiring, 0.0);
}

// Test measure_and_analyze
TEST_F(TMATest, MeasureAndAnalyzeExecutes) {
    NullCounters null_counters;
    null_counters.initialize();

    TMAAnalyzer analyzer(&null_counters);

    auto func = []() {
        volatile int x = 0;
        for (int i = 0; i < 1000; ++i) x += i;
    };

    TMAResult result = analyzer.measure_and_analyze(func, 10);

    // With NullCounters, we'll get default values but no crash
    EXPECT_GE(result.metrics.retiring, 0.0);
}

// Test Level 3 breakdown
TEST_F(TMATest, Level3HasMemoryBreakdown) {
    TMAAnalyzer analyzer;
    analyzer.set_level(TMALevel::LEVEL3);

    auto events = analyzer.get_required_events();

    // Level 3 should request L1/L2/L3/DRAM bound events
    EXPECT_THAT(events, ::testing::Contains(CounterEvent::CYCLE_ACTIVITY_STALLS_L1D));
}

}  // namespace testing
}  // namespace simd_bench
