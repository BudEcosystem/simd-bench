#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/metrics_analyzer.h"
#include "simd_bench/performance_counters.h"
#include "simd_bench/types.h"
#include <cmath>

using namespace simd_bench;
using ::testing::Gt;
using ::testing::Ge;
using ::testing::Le;
using ::testing::AllOf;

class MetricsAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        analyzer_ = std::make_unique<MetricsAnalyzer>();
    }

    std::unique_ptr<MetricsAnalyzer> analyzer_;
};

// Test MPKI calculation
TEST_F(MetricsAnalyzerTest, MPKICalculation) {
    CounterValues values;

    // Set up test values: 10M instructions
    values.set(CounterEvent::INSTRUCTIONS, 10000000);

    // L1: 40000 misses = 4 MPKI (acceptable)
    values.set(CounterEvent::L1D_READ_MISS, 40000);

    // L2: 15000 misses = 1.5 MPKI (acceptable)
    values.set(CounterEvent::L2_READ_MISS, 15000);

    // L3: 5000 misses = 0.5 MPKI (acceptable)
    values.set(CounterEvent::L3_READ_MISS, 5000);

    auto mpki = analyzer_->calculate_mpki(values);

    EXPECT_NEAR(mpki.l1_mpki, 4.0, 0.001);
    EXPECT_NEAR(mpki.l2_mpki, 1.5, 0.001);
    EXPECT_NEAR(mpki.l3_mpki, 0.5, 0.001);
    EXPECT_TRUE(mpki.l1_acceptable);
    EXPECT_TRUE(mpki.l2_acceptable);
    EXPECT_TRUE(mpki.l3_acceptable);
    EXPECT_FALSE(mpki.has_cache_issues);
}

TEST_F(MetricsAnalyzerTest, MPKIThresholdsExceeded) {
    CounterValues values;

    // 1M instructions
    values.set(CounterEvent::INSTRUCTIONS, 1000000);

    // L1: 10000 misses = 10 MPKI (exceeds threshold of 5)
    values.set(CounterEvent::L1D_READ_MISS, 10000);

    // L2: 5000 misses = 5 MPKI (exceeds threshold of 2)
    values.set(CounterEvent::L2_READ_MISS, 5000);

    // L3: 3000 misses = 3 MPKI (exceeds threshold of 1)
    values.set(CounterEvent::L3_READ_MISS, 3000);

    auto mpki = analyzer_->calculate_mpki(values);

    EXPECT_FALSE(mpki.l1_acceptable);
    EXPECT_FALSE(mpki.l2_acceptable);
    EXPECT_FALSE(mpki.l3_acceptable);
    EXPECT_TRUE(mpki.has_cache_issues);
}

// Test IPC calculation
TEST_F(MetricsAnalyzerTest, IPCCalculation) {
    CounterValues values;

    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 2500000);

    double ipc = analyzer_->calculate_ipc(values);

    EXPECT_NEAR(ipc, 2.5, 0.001);
    EXPECT_TRUE(is_ipc_healthy(ipc));
}

TEST_F(MetricsAnalyzerTest, LowIPCDetection) {
    CounterValues values;

    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 1000000);

    double ipc = analyzer_->calculate_ipc(values);

    EXPECT_NEAR(ipc, 1.0, 0.001);
    EXPECT_FALSE(is_ipc_healthy(ipc));
}

// Test vectorization ratio calculation
TEST_F(MetricsAnalyzerTest, VectorizationRatioCalculation) {
    CounterValues values;
    analyzer_->set_cpu_vendor(CPUVendor::INTEL);

    // 1000 scalar ops, 9000 vector ops = 90% vectorization
    values.set(CounterEvent::FP_ARITH_SCALAR_SINGLE, 500);
    values.set(CounterEvent::FP_ARITH_SCALAR_DOUBLE, 500);

    values.set(CounterEvent::FP_ARITH_128B_PACKED_SINGLE, 1000);
    values.set(CounterEvent::FP_ARITH_256B_PACKED_SINGLE, 3000);
    values.set(CounterEvent::FP_ARITH_512B_PACKED_SINGLE, 5000);

    double ratio = analyzer_->calculate_vectorization_ratio(values);

    EXPECT_NEAR(ratio, 0.9, 0.001);
    EXPECT_TRUE(is_vectorization_acceptable(ratio));
}

// Test DSB coverage calculation
TEST_F(MetricsAnalyzerTest, DSBCoverageCalculation) {
    CounterValues values;

    // 80% DSB, 15% MITE, 5% MS
    values.set(CounterEvent::IDQ_DSB_UOPS, 80000);
    values.set(CounterEvent::IDQ_MITE_UOPS, 15000);
    values.set(CounterEvent::IDQ_MS_UOPS, 5000);

    auto dsb = analyzer_->calculate_dsb_coverage(values);

    EXPECT_NEAR(dsb.dsb_coverage, 0.8, 0.001);
    EXPECT_NEAR(dsb.mite_coverage, 0.15, 0.001);
    EXPECT_NEAR(dsb.ms_coverage, 0.05, 0.001);
    EXPECT_TRUE(dsb.is_dsb_efficient);
}

TEST_F(MetricsAnalyzerTest, LowDSBCoverageDetection) {
    CounterValues values;

    // 50% DSB, 40% MITE, 10% MS - inefficient
    values.set(CounterEvent::IDQ_DSB_UOPS, 50000);
    values.set(CounterEvent::IDQ_MITE_UOPS, 40000);
    values.set(CounterEvent::IDQ_MS_UOPS, 10000);

    auto dsb = analyzer_->calculate_dsb_coverage(values);

    EXPECT_NEAR(dsb.dsb_coverage, 0.5, 0.001);
    EXPECT_FALSE(dsb.is_dsb_efficient);
    EXPECT_FALSE(dsb.recommendation.empty());
}

// Test port saturation detection
TEST_F(MetricsAnalyzerTest, PortSaturationDetection) {
    CounterValues values;

    values.set(CounterEvent::CYCLES, 1000000);

    // High Port 5 usage (shuffle bottleneck)
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_0, 500000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_1, 500000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_5, 900000);  // 90% utilization

    auto ports = analyzer_->calculate_port_utilization(values);

    EXPECT_NEAR(ports.port5_utilization, 0.9, 0.001);
    EXPECT_TRUE(ports.port5_saturated);
}

TEST_F(MetricsAnalyzerTest, BalancedPortUsage) {
    CounterValues values;

    values.set(CounterEvent::CYCLES, 1000000);

    // Balanced port usage
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_0, 600000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_1, 580000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_5, 400000);

    auto ports = analyzer_->calculate_port_utilization(values);

    EXPECT_FALSE(ports.port5_saturated);
    EXPECT_TRUE(ports.fma_ports_balanced);
}

// Test cache line split detection
TEST_F(MetricsAnalyzerTest, CacheLineSplitDetection) {
    CounterValues values;

    values.set(CounterEvent::MEM_LOAD_RETIRED, 1000000);
    values.set(CounterEvent::MEM_STORE_RETIRED, 500000);

    // 2% split loads, 1% split stores - has alignment issues
    values.set(CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS, 20000);
    values.set(CounterEvent::MEM_INST_RETIRED_SPLIT_STORES, 5000);

    auto splits = analyzer_->calculate_cache_line_splits(values);

    EXPECT_NEAR(splits.split_load_ratio, 0.02, 0.001);
    EXPECT_NEAR(splits.split_store_ratio, 0.01, 0.001);
    EXPECT_TRUE(splits.has_alignment_issues);
}

TEST_F(MetricsAnalyzerTest, GoodAlignment) {
    CounterValues values;

    values.set(CounterEvent::MEM_LOAD_RETIRED, 1000000);
    values.set(CounterEvent::MEM_STORE_RETIRED, 500000);

    // Very few splits - good alignment
    values.set(CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS, 100);
    values.set(CounterEvent::MEM_INST_RETIRED_SPLIT_STORES, 50);

    auto splits = analyzer_->calculate_cache_line_splits(values);

    EXPECT_LT(splits.split_load_ratio, 0.01);
    EXPECT_FALSE(splits.has_alignment_issues);
}

// Test AVX-512 frequency metrics
TEST_F(MetricsAnalyzerTest, AVX512FrequencyMetrics) {
    CounterValues values;

    // Heavy AVX-512 usage (mostly L1 and L2 licenses)
    values.set(CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE, 100000);
    values.set(CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE, 300000);
    values.set(CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE, 600000);

    auto freq = analyzer_->calculate_avx512_frequency(values);

    // Average level should be around 1.5 (weighted average)
    // (100000*0 + 300000*1 + 600000*2) / 1000000 = 1.5
    EXPECT_NEAR(freq.avg_license_level, 1.5, 0.001);
    EXPECT_TRUE(freq.has_frequency_penalty);
}

TEST_F(MetricsAnalyzerTest, LightAVX512Usage) {
    CounterValues values;

    // Light AVX-512 (mostly L0 license)
    values.set(CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE, 900000);
    values.set(CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE, 80000);
    values.set(CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE, 20000);

    auto freq = analyzer_->calculate_avx512_frequency(values);

    EXPECT_LT(freq.avg_license_level, 0.3);
    EXPECT_FALSE(freq.has_frequency_penalty);
}

// Test quality scoring
TEST_F(MetricsAnalyzerTest, QualityScoring) {
    CounterValues values;
    analyzer_->set_cpu_vendor(CPUVendor::INTEL);

    // Set up good metrics
    values.set(CounterEvent::CYCLES, 1000000);
    values.set(CounterEvent::INSTRUCTIONS, 3000000);  // IPC = 3
    values.set(CounterEvent::L1D_READ_MISS, 1000);    // Low MPKI
    values.set(CounterEvent::L2_READ_MISS, 500);
    values.set(CounterEvent::L3_READ_MISS, 100);

    // Good vectorization
    values.set(CounterEvent::FP_ARITH_SCALAR_SINGLE, 100);
    values.set(CounterEvent::FP_ARITH_256B_PACKED_SINGLE, 5000);

    // Good DSB coverage
    values.set(CounterEvent::IDQ_DSB_UOPS, 900000);
    values.set(CounterEvent::IDQ_MITE_UOPS, 100000);
    values.set(CounterEvent::IDQ_MS_UOPS, 10000);

    // Balanced ports
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_0, 400000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_1, 420000);
    values.set(CounterEvent::UOPS_DISPATCHED_PORT_5, 300000);

    auto extended = analyzer_->calculate_extended_simd(values, 0.001, 100000);

    EXPECT_GT(extended.quality_score, 70.0);  // Should be good quality
    EXPECT_THAT(extended.quality_rating, ::testing::AnyOf("Good", "Excellent"));
}

// Test quality rating strings
TEST(QualityRatingTest, RatingThresholds) {
    EXPECT_EQ(get_quality_rating(95.0), "Excellent");
    EXPECT_EQ(get_quality_rating(80.0), "Good");
    EXPECT_EQ(get_quality_rating(60.0), "Acceptable");
    EXPECT_EQ(get_quality_rating(30.0), "Poor");
    EXPECT_EQ(get_quality_rating(10.0), "Critical");
}

// Test threshold helper functions
TEST(ThresholdHelpersTest, IPCThresholds) {
    EXPECT_TRUE(is_ipc_healthy(2.5));
    EXPECT_TRUE(is_ipc_healthy(3.5));
    EXPECT_FALSE(is_ipc_healthy(1.5));
    EXPECT_FALSE(is_ipc_healthy(0.5));
}

TEST(ThresholdHelpersTest, VectorizationThresholds) {
    EXPECT_TRUE(is_vectorization_acceptable(0.95));
    EXPECT_TRUE(is_vectorization_acceptable(0.90));
    EXPECT_FALSE(is_vectorization_acceptable(0.85));
    EXPECT_FALSE(is_vectorization_acceptable(0.50));
}

TEST(ThresholdHelpersTest, RetiringThresholds) {
    EXPECT_TRUE(is_retiring_healthy(0.5));
    EXPECT_TRUE(is_retiring_healthy(0.7));
    EXPECT_FALSE(is_retiring_healthy(0.3));
    EXPECT_FALSE(is_retiring_healthy(0.1));
}

TEST(ThresholdHelpersTest, BackendBoundThresholds) {
    EXPECT_TRUE(is_backend_bound_acceptable(0.2));
    EXPECT_TRUE(is_backend_bound_acceptable(0.1));
    EXPECT_FALSE(is_backend_bound_acceptable(0.4));
    EXPECT_FALSE(is_backend_bound_acceptable(0.6));
}

TEST(ThresholdHelpersTest, DSBCoverageThresholds) {
    EXPECT_TRUE(is_dsb_coverage_acceptable(0.8));
    EXPECT_TRUE(is_dsb_coverage_acceptable(0.95));
    EXPECT_FALSE(is_dsb_coverage_acceptable(0.5));
    EXPECT_FALSE(is_dsb_coverage_acceptable(0.3));
}

TEST(ThresholdHelpersTest, MPKIThresholds) {
    EXPECT_TRUE(is_mpki_acceptable(4.0, 1.5, 0.5));  // All below thresholds
    EXPECT_FALSE(is_mpki_acceptable(6.0, 1.5, 0.5)); // L1 exceeds
    EXPECT_FALSE(is_mpki_acceptable(4.0, 3.0, 0.5)); // L2 exceeds
    EXPECT_FALSE(is_mpki_acceptable(4.0, 1.5, 2.0)); // L3 exceeds
}

// Test FLOPS calculation
TEST(FLOPSCalculationTest, IntelFLOPSCalculation) {
    CounterValues values;

    // Scalar: 1000 single + 500 double = 1500 FLOPs
    values.set(CounterEvent::FP_ARITH_SCALAR_SINGLE, 1000);
    values.set(CounterEvent::FP_ARITH_SCALAR_DOUBLE, 500);

    // 128-bit: 100 * 4 + 50 * 2 = 500 FLOPs
    values.set(CounterEvent::FP_ARITH_128B_PACKED_SINGLE, 100);
    values.set(CounterEvent::FP_ARITH_128B_PACKED_DOUBLE, 50);

    // 256-bit: 200 * 8 + 100 * 4 = 2000 FLOPs
    values.set(CounterEvent::FP_ARITH_256B_PACKED_SINGLE, 200);
    values.set(CounterEvent::FP_ARITH_256B_PACKED_DOUBLE, 100);

    // 512-bit: 300 * 16 + 150 * 8 = 6000 FLOPs
    values.set(CounterEvent::FP_ARITH_512B_PACKED_SINGLE, 300);
    values.set(CounterEvent::FP_ARITH_512B_PACKED_DOUBLE, 150);

    uint64_t total = calculate_total_flops(values, CPUVendor::INTEL);

    // Total = 1500 + 500 + 2000 + 6000 = 10000
    EXPECT_EQ(total, 10000);
}

// Test empty/zero counter handling
TEST_F(MetricsAnalyzerTest, ZeroInstructionsHandling) {
    CounterValues values;
    values.set(CounterEvent::INSTRUCTIONS, 0);
    values.set(CounterEvent::CYCLES, 1000);

    auto mpki = analyzer_->calculate_mpki(values);

    // Should handle gracefully without division by zero
    EXPECT_EQ(mpki.l1_mpki, 0.0);
    EXPECT_EQ(mpki.l2_mpki, 0.0);
    EXPECT_EQ(mpki.l3_mpki, 0.0);
}

TEST_F(MetricsAnalyzerTest, ZeroCyclesHandling) {
    CounterValues values;
    values.set(CounterEvent::CYCLES, 0);
    values.set(CounterEvent::INSTRUCTIONS, 1000);

    double ipc = analyzer_->calculate_ipc(values);

    // Should handle gracefully
    EXPECT_EQ(ipc, 0.0);
}

// Test IMC bandwidth analyzer availability check
TEST(IMCBandwidthTest, AvailabilityCheck) {
    // Just verify the static method doesn't crash
    bool available = IMCBandwidthAnalyzer::is_available();
    // Don't assert true/false since it depends on system
    (void)available;
}

// Test CPU vendor detection
class VendorDetectionTest : public ::testing::Test {
protected:
    MetricsAnalyzer analyzer_;
};

TEST_F(VendorDetectionTest, EventListsNotEmpty) {
    // Verify event lists are populated
    EXPECT_FALSE(analyzer_.get_intel_events().empty());
    EXPECT_FALSE(analyzer_.get_amd_events().empty());
    EXPECT_FALSE(analyzer_.get_arm_events().empty());
}

TEST_F(VendorDetectionTest, RequiredEventsIncludeBasics) {
    auto events = analyzer_.get_required_events();

    // Check that basic events are always included
    bool has_cycles = std::find(events.begin(), events.end(), CounterEvent::CYCLES) != events.end();
    bool has_instructions = std::find(events.begin(), events.end(), CounterEvent::INSTRUCTIONS) != events.end();

    EXPECT_TRUE(has_cycles);
    EXPECT_TRUE(has_instructions);
}
