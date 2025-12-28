#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <cstdint>

namespace simd_bench {

// Memory traffic metrics from IMC (Integrated Memory Controller) or uncore counters
struct MemoryTrafficMetrics {
    // Raw byte counts
    uint64_t bytes_read_dram = 0;
    uint64_t bytes_written_dram = 0;
    uint64_t total_bytes_dram = 0;

    // Derived metrics
    double measured_arithmetic_intensity = 0.0;
    double theoretical_arithmetic_intensity = 0.0;
    double cache_amplification = 1.0;  // > 1.0 means cache thrashing
    double bandwidth_utilization = 0.0; // Fraction of peak DRAM bandwidth used
    double read_write_ratio = 0.0;      // Fraction of reads vs total

    // Performance context
    uint64_t total_flops = 0;
    double elapsed_seconds = 0.0;
    double achieved_bandwidth_gbps = 0.0;
    double peak_bandwidth_gbps = 0.0;

    // Quality assessment
    std::string bandwidth_efficiency;  // "excellent", "good", "moderate", "poor"
    std::vector<std::string> recommendations;
};

// IMC (Integrated Memory Controller) counter interface
// Abstracts uncore counter access for Intel/AMD platforms
class MemoryTrafficAnalyzer {
public:
    MemoryTrafficAnalyzer();
    ~MemoryTrafficAnalyzer();

    // Check if IMC counters are available on this platform
    static bool is_available();

    // Initialize counter collection
    bool initialize();

    // Start/stop measurement
    void start();
    void stop();

    // Reset counters
    void reset();

    // Get metrics (call after stop)
    MemoryTrafficMetrics get_metrics(uint64_t total_flops, double elapsed_seconds) const;

    // Configure peak bandwidth for utilization calculation
    void set_peak_bandwidth_gbps(double peak_bw);

    // Set theoretical AI for cache amplification calculation
    void set_theoretical_ai(double theoretical_ai);

    // Check if currently measuring
    bool is_measuring() const { return measuring_.load(); }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    double peak_bandwidth_gbps_ = 0.0;
    double theoretical_ai_ = 0.0;
    std::atomic<bool> measuring_{false};

    uint64_t start_reads_ = 0;
    uint64_t start_writes_ = 0;
    uint64_t end_reads_ = 0;
    uint64_t end_writes_ = 0;

    // Platform-specific initialization
    bool init_intel_imc();
    bool init_amd_df();  // AMD Data Fabric counters
    bool init_fallback();  // Use L3 miss counters as proxy

    // Read current counter values
    bool read_counters(uint64_t& reads, uint64_t& writes) const;
};

// DRAM channel information for multi-channel systems
struct DRAMChannelInfo {
    int channel_id;
    uint64_t reads;
    uint64_t writes;
    double bandwidth_gbps;
    double utilization;
};

// Extended memory traffic analyzer with per-channel granularity
class DetailedMemoryTrafficAnalyzer {
public:
    DetailedMemoryTrafficAnalyzer();
    ~DetailedMemoryTrafficAnalyzer();

    static bool is_available();

    bool initialize();
    void start();
    void stop();
    void reset();

    // Get per-channel metrics
    std::vector<DRAMChannelInfo> get_channel_metrics() const;

    // Get aggregated metrics
    MemoryTrafficMetrics get_aggregated_metrics(uint64_t total_flops,
                                                  double elapsed_seconds) const;

    // Detect memory bandwidth bottlenecks
    struct BottleneckAnalysis {
        bool has_bottleneck = false;
        double severity = 0.0;  // 0-1
        std::string bottleneck_type;  // "channel_imbalance", "bandwidth_limit", etc.
        std::vector<std::string> recommendations;
    };

    BottleneckAnalysis analyze_bottlenecks() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> measuring_{false};
};

// Memory access pattern analyzer
struct AccessPatternMetrics {
    double sequential_ratio = 0.0;   // Fraction of sequential accesses
    double random_ratio = 0.0;       // Fraction of random accesses
    double stride_ratio = 0.0;       // Fraction of strided accesses
    size_t average_stride = 0;       // Average access stride in bytes
    double prefetch_friendly = 0.0;  // 0-1, how prefetch-friendly the pattern is
    std::string pattern_type;        // "sequential", "strided", "random", "mixed"
    std::vector<std::string> recommendations;
};

// Analyze memory access patterns from L1/L2 hit/miss data
AccessPatternMetrics analyze_access_pattern(
    uint64_t l1_hits,
    uint64_t l1_misses,
    uint64_t l2_hits,
    uint64_t l2_misses,
    uint64_t l3_hits,
    uint64_t l3_misses,
    size_t working_set_bytes,
    size_t element_stride = sizeof(float)
);

// Memory bandwidth prediction based on access pattern
struct BandwidthPrediction {
    double expected_bandwidth_gbps = 0.0;
    double bandwidth_efficiency = 0.0;  // Fraction of peak expected
    double latency_bound_factor = 0.0;  // How much latency limits bandwidth
    std::string limiting_factor;  // "bandwidth", "latency", "cache"
};

BandwidthPrediction predict_bandwidth(
    const AccessPatternMetrics& pattern,
    const HardwareInfo& hw,
    size_t working_set_bytes
);

// Helper: Calculate bytes per DRAM transaction (varies by DDR generation)
constexpr size_t get_dram_transaction_bytes(int ddr_generation) {
    // DDR4/DDR5: 64 bytes per burst (8 beats * 8 bytes)
    // But prefetch buffer is 64 bytes for DDR4, 32 bytes for DDR5
    switch (ddr_generation) {
        case 3: return 64;   // DDR3: 64 bytes
        case 4: return 64;   // DDR4: 64 bytes
        case 5: return 64;   // DDR5: Still 64 bytes per transaction
        default: return 64;
    }
}

}  // namespace simd_bench
