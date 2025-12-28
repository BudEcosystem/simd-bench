#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/energy.h"
#include <thread>
#include <chrono>

namespace simd_bench {
namespace testing {

class EnergyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test EnergySample defaults
TEST_F(EnergyTest, EnergySampleDefaultsToZero) {
    EnergySample sample;
    EXPECT_DOUBLE_EQ(sample.package_joules, 0.0);
    EXPECT_DOUBLE_EQ(sample.cores_joules, 0.0);
    EXPECT_DOUBLE_EQ(sample.dram_joules, 0.0);
}

// Test EnergyMetrics defaults
TEST_F(EnergyTest, EnergyMetricsDefaultsToZero) {
    EnergyMetrics metrics;
    EXPECT_DOUBLE_EQ(metrics.package_power_watts, 0.0);
    EXPECT_DOUBLE_EQ(metrics.energy_joules, 0.0);
    EXPECT_DOUBLE_EQ(metrics.energy_per_op_nj, 0.0);
}

// Test NullEnergyMonitor
TEST_F(EnergyTest, NullEnergyMonitorInitializeSucceeds) {
    NullEnergyMonitor monitor;
    EXPECT_TRUE(monitor.initialize());
}

TEST_F(EnergyTest, NullEnergyMonitorOperationsSucceed) {
    NullEnergyMonitor monitor;
    monitor.initialize();

    EXPECT_TRUE(monitor.start());
    EXPECT_TRUE(monitor.stop());

    monitor.shutdown();
}

TEST_F(EnergyTest, NullEnergyMonitorReturnsEmptySample) {
    NullEnergyMonitor monitor;
    EnergySample sample = monitor.sample();

    EXPECT_DOUBLE_EQ(sample.package_joules, 0.0);
}

TEST_F(EnergyTest, NullEnergyMonitorBackendIsNone) {
    NullEnergyMonitor monitor;
    EXPECT_EQ(monitor.get_backend(), EnergyBackend::NONE);
    EXPECT_EQ(monitor.get_backend_name(), "none");
}

TEST_F(EnergyTest, NullEnergyMonitorDomainNotAvailable) {
    NullEnergyMonitor monitor;
    EXPECT_FALSE(monitor.is_domain_available(RAPLDomain::PACKAGE));
    EXPECT_FALSE(monitor.is_domain_available(RAPLDomain::CORES));
}

// Test factory
TEST_F(EnergyTest, FactoryCreateNoneReturnsNullMonitor) {
    auto monitor = EnergyMonitorFactory::create(EnergyBackend::NONE);
    ASSERT_NE(monitor, nullptr);
    EXPECT_EQ(monitor->get_backend(), EnergyBackend::NONE);
}

TEST_F(EnergyTest, FactoryCreateBestAvailableReturnsNonNull) {
    auto monitor = EnergyMonitorFactory::create_best_available();
    ASSERT_NE(monitor, nullptr);
}

TEST_F(EnergyTest, FactoryGetAvailableBackendsIncludesNone) {
    auto backends = EnergyMonitorFactory::get_available_backends();
    EXPECT_THAT(backends, ::testing::Contains(EnergyBackend::NONE));
}

TEST_F(EnergyTest, FactoryNoneBackendIsAlwaysAvailable) {
    EXPECT_TRUE(EnergyMonitorFactory::is_backend_available(EnergyBackend::NONE));
}

// Test RAPL availability check
TEST_F(EnergyTest, IsRaplAvailableDoesNotCrash) {
    bool available = is_rapl_available();
    EXPECT_TRUE(available || !available);
}

// Test energy efficiency analyzer
TEST_F(EnergyTest, CalculateEnergyPerFlopNJ) {
    double energy = 1.0;  // 1 Joule
    uint64_t flops = 1000000000;  // 1 GFLOP

    double nj_per_flop = EnergyEfficiencyAnalyzer::calculate_energy_per_flop_nj(energy, flops);

    // 1 J / 1e9 FLOP = 1 nJ/FLOP
    EXPECT_NEAR(nj_per_flop, 1.0, 0.01);
}

TEST_F(EnergyTest, CalculateEDP) {
    double energy = 10.0;  // 10 Joules
    double time = 2.0;     // 2 seconds

    double edp = EnergyEfficiencyAnalyzer::calculate_edp(energy, time);

    EXPECT_DOUBLE_EQ(edp, 20.0);  // 10 * 2 = 20 J*s
}

TEST_F(EnergyTest, CalculateED2P) {
    double energy = 10.0;  // 10 Joules
    double time = 2.0;     // 2 seconds

    double ed2p = EnergyEfficiencyAnalyzer::calculate_ed2p(energy, time);

    EXPECT_DOUBLE_EQ(ed2p, 40.0);  // 10 * 2^2 = 40 J*s^2
}

TEST_F(EnergyTest, CalculatePowerWatts) {
    double energy = 100.0;  // 100 Joules
    double time = 10.0;     // 10 seconds

    double power = EnergyEfficiencyAnalyzer::calculate_power_watts(energy, time);

    EXPECT_DOUBLE_EQ(power, 10.0);  // 100 / 10 = 10 Watts
}

TEST_F(EnergyTest, CompareEfficiency) {
    EnergyMetrics scalar_metrics, simd_metrics;
    scalar_metrics.energy_joules = 10.0;
    simd_metrics.energy_joules = 5.0;

    auto comparison = EnergyEfficiencyAnalyzer::compare_efficiency(
        scalar_metrics, 1000000, 1.0,
        simd_metrics, 1000000, 0.25  // SIMD 4x faster
    );

    // SIMD should be more efficient
    EXPECT_LT(simd_metrics.energy_joules, scalar_metrics.energy_joules);
    EXPECT_GT(comparison.energy_savings_percent, 0.0);
}

// Test scoped energy measurement
TEST_F(EnergyTest, ScopedEnergyMeasurementWorks) {
    NullEnergyMonitor monitor;
    monitor.initialize();
    EnergyMetrics result;

    {
        ScopedEnergyMeasurement scoped(monitor, result);
        // Do some work
        volatile int x = 0;
        for (int i = 0; i < 1000; ++i) x += i;
    }

    // With NullMonitor, result will be zero but no crash
    EXPECT_DOUBLE_EQ(result.energy_joules, 0.0);
}

// Test measure_energy function
TEST_F(EnergyTest, MeasureEnergyExecutes) {
    auto func = []() {
        volatile int x = 0;
        for (int i = 0; i < 10000; ++i) x += i;
    };

    EnergyMetrics metrics = measure_energy(func, 10000);

    // May return zeros if RAPL not available, but shouldn't crash
    EXPECT_GE(metrics.energy_joules, 0.0);
}

// Test RAPL monitor if available
#ifdef SIMD_BENCH_HAS_RAPL
TEST_F(EnergyTest, RAPLMonitorInitializeAndShutdown) {
    RAPLMonitor monitor;
    bool init_success = monitor.initialize();

    if (init_success) {
        EXPECT_EQ(monitor.get_backend(), EnergyBackend::RAPL);
        EXPECT_EQ(monitor.get_backend_name(), "RAPL");

        // Check if at least package domain is available
        EXPECT_TRUE(monitor.is_domain_available(RAPLDomain::PACKAGE));

        monitor.shutdown();
    }
}

TEST_F(EnergyTest, RAPLMonitorSampleReturnsValues) {
    RAPLMonitor monitor;
    if (monitor.initialize()) {
        EnergySample sample1 = monitor.sample();

        // Do some work
        volatile double x = 0;
        for (int i = 0; i < 1000000; ++i) x += i * 0.001;

        EnergySample sample2 = monitor.sample();

        // Energy should increase
        if (monitor.is_domain_available(RAPLDomain::PACKAGE)) {
            EXPECT_GT(sample2.package_joules, sample1.package_joules);
        }

        monitor.shutdown();
    }
}

TEST_F(EnergyTest, RAPLMonitorStartStopMeasuresEnergy) {
    RAPLMonitor monitor;
    if (monitor.initialize()) {
        monitor.start();

        // Do significant work
        volatile double x = 0;
        for (int i = 0; i < 10000000; ++i) x += i * 0.001;

        monitor.stop();

        EnergyMetrics metrics = monitor.get_metrics();

        // Should have consumed some energy
        EXPECT_GT(metrics.energy_joules, 0.0);
        EXPECT_GT(metrics.package_power_watts, 0.0);

        monitor.shutdown();
    }
}

TEST_F(EnergyTest, RAPLMonitorGetEnergyUnit) {
    RAPLMonitor monitor;
    if (monitor.initialize()) {
        double unit = monitor.get_energy_unit();

        // Energy unit should be a small positive value
        EXPECT_GT(unit, 0.0);
        EXPECT_LT(unit, 1.0);  // Less than 1 Joule per increment

        monitor.shutdown();
    }
}

TEST_F(EnergyTest, RAPLMonitorGetMaxEnergy) {
    RAPLMonitor monitor;
    if (monitor.initialize()) {
        double max_pkg = monitor.get_max_energy_joules(RAPLDomain::PACKAGE);

        // Max energy should be positive
        EXPECT_GT(max_pkg, 0.0);

        monitor.shutdown();
    }
}
#endif

// Test read_rapl_energy_joules
TEST_F(EnergyTest, ReadRaplEnergyDirectlyDoesNotCrash) {
    // This may fail if RAPL not available, but should not crash
    double energy = read_rapl_energy_joules(RAPLDomain::PACKAGE);
    EXPECT_GE(energy, 0.0);
}

}  // namespace testing
}  // namespace simd_bench
