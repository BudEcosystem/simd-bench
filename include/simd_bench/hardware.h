#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

namespace simd_bench {

// SIMD extension flags
enum class SIMDExtension {
    NONE = 0,
    SSE = 1 << 0,
    SSE2 = 1 << 1,
    SSE3 = 1 << 2,
    SSSE3 = 1 << 3,
    SSE4_1 = 1 << 4,
    SSE4_2 = 1 << 5,
    AVX = 1 << 6,
    AVX2 = 1 << 7,
    FMA = 1 << 8,
    AVX512F = 1 << 9,
    AVX512DQ = 1 << 10,
    AVX512BW = 1 << 11,
    AVX512VL = 1 << 12,
    NEON = 1 << 13,
    SVE = 1 << 14,
    SVE2 = 1 << 15,
    RVV = 1 << 16
};

inline SIMDExtension operator|(SIMDExtension a, SIMDExtension b) {
    return static_cast<SIMDExtension>(static_cast<int>(a) | static_cast<int>(b));
}

inline SIMDExtension operator&(SIMDExtension a, SIMDExtension b) {
    return static_cast<SIMDExtension>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool has_extension(SIMDExtension flags, SIMDExtension ext) {
    return (static_cast<int>(flags) & static_cast<int>(ext)) != 0;
}

// Cache information
struct CacheInfo {
    size_t l1d_size_kb = 0;      // L1 data cache size
    size_t l1i_size_kb = 0;      // L1 instruction cache size
    size_t l2_size_kb = 0;       // L2 cache size
    size_t l3_size_kb = 0;       // L3 cache size

    size_t l1_line_size = 64;    // Cache line size
    size_t l1_associativity = 8;
    size_t l2_associativity = 8;
    size_t l3_associativity = 16;
};

// Hardware information
struct HardwareInfo {
    // CPU identification
    std::string cpu_vendor;        // "GenuineIntel", "AuthenticAMD", etc.
    std::string cpu_brand;         // Full CPU name
    std::string architecture;      // "x86_64", "aarch64", "riscv64"

    // Core topology
    int physical_cores = 0;
    int logical_cores = 0;
    int numa_nodes = 1;

    // Frequency
    double base_frequency_ghz = 0.0;
    double max_frequency_ghz = 0.0;
    double measured_frequency_ghz = 0.0;

    // SIMD capabilities
    SIMDExtension simd_extensions = SIMDExtension::NONE;
    int max_vector_bits = 0;       // Maximum SIMD register width
    int fma_units = 0;             // Number of FMA units per core

    // Cache hierarchy
    CacheInfo cache;

    // Memory
    size_t total_memory_mb = 0;
    double measured_memory_bw_gbps = 0.0;

    // Theoretical peak performance
    double theoretical_peak_sp_gflops = 0.0;  // Single precision
    double theoretical_peak_dp_gflops = 0.0;  // Double precision

    // Detection methods
    static HardwareInfo detect();

    // Helper methods
    std::string get_simd_string() const;

    // Calculate peak GFLOPS with optional AVX-512 throttling adjustment
    double calculate_peak_gflops(bool double_precision = false) const;
    double calculate_peak_gflops_throttled(bool double_precision,
                                            AVX512License license) const;

    // Calculate realistic sustained peak (accounts for thermal throttling)
    double calculate_sustained_peak_gflops(bool double_precision = false) const;

    double calculate_ridge_point() const;  // FLOP/byte where compute-bound
    double calculate_ridge_point_throttled(AVX512License license) const;

    // Validate hardware counters are available
    bool has_hardware_counters() const;
    bool has_rapl() const;
};

// CPU feature detection using CPUID (x86)
class CPUIDReader {
public:
    CPUIDReader();

    bool has_sse() const { return sse_; }
    bool has_sse2() const { return sse2_; }
    bool has_sse3() const { return sse3_; }
    bool has_ssse3() const { return ssse3_; }
    bool has_sse4_1() const { return sse4_1_; }
    bool has_sse4_2() const { return sse4_2_; }
    bool has_avx() const { return avx_; }
    bool has_avx2() const { return avx2_; }
    bool has_fma() const { return fma_; }
    bool has_avx512f() const { return avx512f_; }
    bool has_avx512dq() const { return avx512dq_; }
    bool has_avx512bw() const { return avx512bw_; }
    bool has_avx512vl() const { return avx512vl_; }

    std::string get_vendor() const { return vendor_; }
    std::string get_brand() const { return brand_; }

    int get_family() const { return family_; }
    int get_model() const { return model_; }
    int get_stepping() const { return stepping_; }

private:
    void query_cpuid();
    void detect_features();

    std::string vendor_;
    std::string brand_;
    int family_ = 0;
    int model_ = 0;
    int stepping_ = 0;

    bool sse_ = false;
    bool sse2_ = false;
    bool sse3_ = false;
    bool ssse3_ = false;
    bool sse4_1_ = false;
    bool sse4_2_ = false;
    bool avx_ = false;
    bool avx2_ = false;
    bool fma_ = false;
    bool avx512f_ = false;
    bool avx512dq_ = false;
    bool avx512bw_ = false;
    bool avx512vl_ = false;
};

// Memory bandwidth measurement
double measure_memory_bandwidth_gbps(size_t size_mb = 64);

// Streaming bandwidth measurement using non-temporal stores (true peak DRAM bandwidth)
double measure_streaming_bandwidth_gbps(size_t size_mb = 64);

// Read-only bandwidth measurement
double measure_read_bandwidth_gbps(size_t size_mb = 64);

// Write-only bandwidth measurement (non-temporal)
double measure_write_bandwidth_gbps(size_t size_mb = 64);

// L1/L2/L3/DRAM bandwidth measurement
struct BandwidthMeasurement {
    double l1_bandwidth_gbps = 0.0;
    double l2_bandwidth_gbps = 0.0;
    double l3_bandwidth_gbps = 0.0;
    double dram_bandwidth_gbps = 0.0;
    double streaming_bandwidth_gbps = 0.0;  // Non-temporal stores
};

BandwidthMeasurement measure_cache_bandwidths();

// Read CPU frequency from various sources
double read_cpu_frequency_from_cpuinfo();
double read_cpu_frequency_from_scaling();
double measure_cpu_frequency();

// Parse /proc/cpuinfo
std::vector<std::pair<std::string, std::string>> parse_cpuinfo();

}  // namespace simd_bench
