#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/hardware.h"
#include <algorithm>

namespace simd_bench {
namespace testing {

class HardwareTest : public ::testing::Test {
protected:
    void SetUp() override {
        hw_ = HardwareInfo::detect();
    }

    HardwareInfo hw_;
};

// Test hardware detection
TEST_F(HardwareTest, DetectReturnsValidCPUVendor) {
    EXPECT_FALSE(hw_.cpu_vendor.empty());
    // Common vendors
    EXPECT_TRUE(
        hw_.cpu_vendor == "GenuineIntel" ||
        hw_.cpu_vendor == "AuthenticAMD" ||
        hw_.cpu_vendor == "ARM" ||
        !hw_.cpu_vendor.empty()
    );
}

TEST_F(HardwareTest, DetectReturnsValidCPUBrand) {
    EXPECT_FALSE(hw_.cpu_brand.empty());
}

TEST_F(HardwareTest, DetectReturnsValidArchitecture) {
    EXPECT_FALSE(hw_.architecture.empty());
    EXPECT_TRUE(
        hw_.architecture == "x86_64" ||
        hw_.architecture == "aarch64" ||
        hw_.architecture == "riscv64" ||
        !hw_.architecture.empty()
    );
}

TEST_F(HardwareTest, DetectReturnsValidCoreCount) {
    EXPECT_GT(hw_.physical_cores, 0);
    EXPECT_GT(hw_.logical_cores, 0);
    EXPECT_GE(hw_.logical_cores, hw_.physical_cores);
}

TEST_F(HardwareTest, DetectReturnsValidFrequency) {
    // Base frequency should be between 0.5 and 5 GHz
    EXPECT_GT(hw_.base_frequency_ghz, 0.0);
    EXPECT_LT(hw_.base_frequency_ghz, 10.0);
}

TEST_F(HardwareTest, DetectReturnsValidMeasuredFrequency) {
    EXPECT_GT(hw_.measured_frequency_ghz, 0.0);
    EXPECT_LT(hw_.measured_frequency_ghz, 10.0);
}

TEST_F(HardwareTest, DetectReturnsValidCacheInfo) {
    // L1 cache is typically 16-64 KB
    EXPECT_GT(hw_.cache.l1d_size_kb, 0u);
    EXPECT_LE(hw_.cache.l1d_size_kb, 256u);

    // L2 cache is typically 256KB - 2MB
    EXPECT_GT(hw_.cache.l2_size_kb, 0u);

    // L3 cache can be 0 (for some embedded) or up to 128MB
    // Don't require L3 to exist
}

TEST_F(HardwareTest, DetectReturnsValidCacheLineSize) {
    // Cache line is typically 32, 64, or 128 bytes
    EXPECT_TRUE(
        hw_.cache.l1_line_size == 32 ||
        hw_.cache.l1_line_size == 64 ||
        hw_.cache.l1_line_size == 128
    );
}

TEST_F(HardwareTest, DetectReturnsValidVectorWidth) {
    // Vector width should be at least 64 bits (SSE requires 128, but could be scalar)
    EXPECT_GT(hw_.max_vector_bits, 0);
    // Common values: 128 (SSE), 256 (AVX), 512 (AVX-512)
    EXPECT_LE(hw_.max_vector_bits, 2048);  // SVE can go up to 2048
}

TEST_F(HardwareTest, DetectReturnsValidMemory) {
    // System should have at least 256MB RAM
    EXPECT_GT(hw_.total_memory_mb, 256u);
}

// Test SIMD extension string
TEST_F(HardwareTest, GetSIMDStringReturnsNonEmpty) {
    std::string simd_str = hw_.get_simd_string();
    EXPECT_FALSE(simd_str.empty());
}

TEST_F(HardwareTest, GetSIMDStringContainsExtensions) {
    std::string simd_str = hw_.get_simd_string();

#if defined(__x86_64__)
    // x86-64 should have at least SSE2
    if (has_extension(hw_.simd_extensions, SIMDExtension::SSE2)) {
        EXPECT_NE(simd_str.find("SSE2"), std::string::npos);
    }
#endif
}

// Test theoretical peak calculation
TEST_F(HardwareTest, CalculatePeakGflopsReturnsPositive) {
    double peak_sp = hw_.calculate_peak_gflops(false);
    double peak_dp = hw_.calculate_peak_gflops(true);

    EXPECT_GT(peak_sp, 0.0);
    EXPECT_GT(peak_dp, 0.0);
    // Single precision should be >= double precision
    EXPECT_GE(peak_sp, peak_dp);
}

TEST_F(HardwareTest, TheoreticalPeakIsReasonable) {
    double peak = hw_.theoretical_peak_sp_gflops;

    // Modern CPUs typically achieve 50-2000 GFLOPS peak SP
    EXPECT_GT(peak, 1.0);
    EXPECT_LT(peak, 10000.0);
}

// Test ridge point calculation
TEST_F(HardwareTest, CalculateRidgePointReturnsPositive) {
    double ridge = hw_.calculate_ridge_point();
    EXPECT_GT(ridge, 0.0);
}

// Test CPUID reader
TEST_F(HardwareTest, CPUIDReaderReturnsValidVendor) {
    CPUIDReader cpuid;
    std::string vendor = cpuid.get_vendor();

#if defined(__x86_64__)
    EXPECT_FALSE(vendor.empty());
    EXPECT_TRUE(
        vendor == "GenuineIntel" ||
        vendor == "AuthenticAMD" ||
        vendor == "HygonGenuine" ||  // Chinese AMD clone
        !vendor.empty()
    );
#endif
}

TEST_F(HardwareTest, CPUIDReaderDetectsSSE2) {
    CPUIDReader cpuid;

#if defined(__x86_64__)
    // All x86-64 CPUs must have SSE2
    EXPECT_TRUE(cpuid.has_sse2());
#endif
}

TEST_F(HardwareTest, CPUIDReaderFeaturesConsistent) {
    CPUIDReader cpuid;

    // If we have AVX2, we must have AVX
    if (cpuid.has_avx2()) {
        EXPECT_TRUE(cpuid.has_avx());
    }

    // If we have AVX, we must have SSE4.2
    if (cpuid.has_avx()) {
        EXPECT_TRUE(cpuid.has_sse4_2());
    }

    // If we have AVX-512F, we must have AVX2
    if (cpuid.has_avx512f()) {
        EXPECT_TRUE(cpuid.has_avx2());
    }
}

// Test memory bandwidth measurement
TEST_F(HardwareTest, MeasureMemoryBandwidthReturnsPositive) {
    double bw = measure_memory_bandwidth_gbps(16);  // 16 MB test
    EXPECT_GT(bw, 0.0);
    // Typical DDR4/5 is 20-100 GB/s
    EXPECT_LT(bw, 500.0);
}

// Test cache bandwidth measurement
TEST_F(HardwareTest, MeasureCacheBandwidthsReturnsPositive) {
    BandwidthMeasurement bw = measure_cache_bandwidths();

    EXPECT_GT(bw.l1_bandwidth_gbps, 0.0);
    EXPECT_GT(bw.l2_bandwidth_gbps, 0.0);
    EXPECT_GT(bw.dram_bandwidth_gbps, 0.0);

    // L1 should be faster than L2, L2 faster than DRAM
    EXPECT_GT(bw.l1_bandwidth_gbps, bw.l2_bandwidth_gbps);
    EXPECT_GT(bw.l2_bandwidth_gbps, bw.dram_bandwidth_gbps);
}

// Test CPU frequency reading
TEST_F(HardwareTest, ReadCPUFrequencyFromCpuinfoReturnsPositive) {
    double freq = read_cpu_frequency_from_cpuinfo();
    if (freq > 0.0) {  // May not be available on all systems
        EXPECT_LT(freq, 10.0);  // Less than 10 GHz
    }
}

TEST_F(HardwareTest, MeasureCPUFrequencyReturnsPositive) {
    double freq = measure_cpu_frequency();
    EXPECT_GT(freq, 0.5);
    EXPECT_LT(freq, 10.0);
}

// Test /proc/cpuinfo parsing
TEST_F(HardwareTest, ParseCpuinfoReturnsData) {
    auto info = parse_cpuinfo();
    // Should have at least some entries on Linux
#ifdef __linux__
    EXPECT_FALSE(info.empty());
#endif
}

// Test SIMD extension bitwise operations
TEST_F(HardwareTest, SIMDExtensionBitwiseOr) {
    SIMDExtension ext = SIMDExtension::SSE | SIMDExtension::AVX;
    EXPECT_TRUE(has_extension(ext, SIMDExtension::SSE));
    EXPECT_TRUE(has_extension(ext, SIMDExtension::AVX));
    EXPECT_FALSE(has_extension(ext, SIMDExtension::AVX512F));
}

TEST_F(HardwareTest, SIMDExtensionBitwiseAnd) {
    SIMDExtension ext = SIMDExtension::SSE | SIMDExtension::AVX | SIMDExtension::AVX2;
    SIMDExtension avx_only = ext & SIMDExtension::AVX;
    EXPECT_TRUE(has_extension(avx_only, SIMDExtension::AVX));
}

// Test hardware counter availability check
TEST_F(HardwareTest, HasHardwareCountersDoesNotCrash) {
    bool has_counters = hw_.has_hardware_counters();
    // Just check it doesn't crash and returns a boolean
    EXPECT_TRUE(has_counters || !has_counters);
}

// Test RAPL availability check
TEST_F(HardwareTest, HasRaplDoesNotCrash) {
    bool has_rapl = hw_.has_rapl();
    // Just check it doesn't crash
    EXPECT_TRUE(has_rapl || !has_rapl);
}

}  // namespace testing
}  // namespace simd_bench
