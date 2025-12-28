#include "simd_bench/memory_traffic.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <dirent.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace simd_bench {

// ============================================================================
// MemoryTrafficAnalyzer::Impl - Platform-specific implementation
// ============================================================================

class MemoryTrafficAnalyzer::Impl {
public:
    enum class Backend {
        NONE,
        INTEL_IMC,    // Intel Uncore IMC counters via perf/MSR
        AMD_DF,       // AMD Data Fabric counters
        PERF_UNCORE,  // Generic perf uncore
        L3_FALLBACK   // L3 miss counters as proxy
    };

    Backend backend = Backend::NONE;
    bool initialized = false;

    // Intel IMC file descriptors (one per channel)
    std::vector<int> imc_fds;

    // Cached sysfs paths
    std::vector<std::string> imc_read_paths;
    std::vector<std::string> imc_write_paths;

    // AMD DF counters
    int amd_df_fd = -1;

    ~Impl() {
        for (int fd : imc_fds) {
            if (fd >= 0) close(fd);
        }
        if (amd_df_fd >= 0) close(amd_df_fd);
    }
};

// ============================================================================
// MemoryTrafficAnalyzer implementation
// ============================================================================

MemoryTrafficAnalyzer::MemoryTrafficAnalyzer()
    : impl_(std::make_unique<Impl>()) {
}

MemoryTrafficAnalyzer::~MemoryTrafficAnalyzer() = default;

bool MemoryTrafficAnalyzer::is_available() {
#ifdef __linux__
    // Check for Intel IMC uncore counters
    DIR* dir = opendir("/sys/devices/uncore_imc_0");
    if (dir) {
        closedir(dir);
        return true;
    }

    // Check for AMD Data Fabric counters
    dir = opendir("/sys/devices/amd_df");
    if (dir) {
        closedir(dir);
        return true;
    }

    // Check for generic perf uncore support
    std::ifstream perf_paranoid("/proc/sys/kernel/perf_event_paranoid");
    if (perf_paranoid.is_open()) {
        int level;
        perf_paranoid >> level;
        // Level -1 or 0 allows uncore access
        if (level <= 0) {
            return true;
        }
    }

    // L3 miss counters are always available as fallback
    return true;
#else
    return false;
#endif
}

bool MemoryTrafficAnalyzer::initialize() {
    if (impl_->initialized) return true;

    // Try Intel IMC first
    if (init_intel_imc()) {
        impl_->backend = Impl::Backend::INTEL_IMC;
        impl_->initialized = true;
        return true;
    }

    // Try AMD Data Fabric
    if (init_amd_df()) {
        impl_->backend = Impl::Backend::AMD_DF;
        impl_->initialized = true;
        return true;
    }

    // Fallback to L3 miss counters
    if (init_fallback()) {
        impl_->backend = Impl::Backend::L3_FALLBACK;
        impl_->initialized = true;
        return true;
    }

    return false;
}

bool MemoryTrafficAnalyzer::init_intel_imc() {
#ifdef __linux__
    // Look for Intel IMC PMU devices
    // Typical path: /sys/devices/uncore_imc_0/events/

    for (int imc = 0; imc < 8; ++imc) {  // Up to 8 IMC channels
        std::string base = "/sys/devices/uncore_imc_" + std::to_string(imc);
        struct stat st;
        if (stat(base.c_str(), &st) != 0) break;

        // Check for CAS count events
        std::string cas_rd = base + "/events/cas_count_read";
        std::string cas_wr = base + "/events/cas_count_write";

        if (stat(cas_rd.c_str(), &st) == 0 && stat(cas_wr.c_str(), &st) == 0) {
            impl_->imc_read_paths.push_back(cas_rd);
            impl_->imc_write_paths.push_back(cas_wr);
        }
    }

    return !impl_->imc_read_paths.empty();
#else
    return false;
#endif
}

bool MemoryTrafficAnalyzer::init_amd_df() {
#ifdef __linux__
    // AMD Data Fabric uses different PMU events
    // Check for amd_df device
    struct stat st;
    if (stat("/sys/devices/amd_df", &st) != 0) {
        return false;
    }

    // AMD uses different event encoding
    // We'll use UMC (Unified Memory Controller) events when available
    return true;  // Mark as available, actual counting done via perf
#else
    return false;
#endif
}

bool MemoryTrafficAnalyzer::init_fallback() {
    // L3 miss counters are always available via perf_event
    // This is a proxy for DRAM traffic
    return true;
}

void MemoryTrafficAnalyzer::start() {
    if (!impl_->initialized) {
        initialize();
    }

    measuring_.store(true);
    read_counters(start_reads_, start_writes_);
}

void MemoryTrafficAnalyzer::stop() {
    read_counters(end_reads_, end_writes_);
    measuring_.store(false);
}

void MemoryTrafficAnalyzer::reset() {
    start_reads_ = 0;
    start_writes_ = 0;
    end_reads_ = 0;
    end_writes_ = 0;
}

bool MemoryTrafficAnalyzer::read_counters(uint64_t& reads, uint64_t& writes) const {
    reads = 0;
    writes = 0;

#ifdef __linux__
    switch (impl_->backend) {
        case Impl::Backend::INTEL_IMC: {
            // Read from sysfs or perf
            // For now, use a simplified approach with /proc
            // Real implementation would use perf_event_open

            // Aggregate across all IMC channels
            for (size_t i = 0; i < impl_->imc_read_paths.size(); ++i) {
                std::ifstream rd_file(impl_->imc_read_paths[i]);
                std::ifstream wr_file(impl_->imc_write_paths[i]);

                if (rd_file.is_open() && wr_file.is_open()) {
                    uint64_t rd_val = 0, wr_val = 0;
                    rd_file >> rd_val;
                    wr_file >> wr_val;
                    reads += rd_val;
                    writes += wr_val;
                }
            }
            break;
        }

        case Impl::Backend::AMD_DF:
            // AMD implementation would read from DF counters
            // Simplified: return 0 for now (real impl needs MSR/perf)
            break;

        case Impl::Backend::L3_FALLBACK:
            // Use L3 miss counters from perf
            // This is approximate - actual impl would use perf_event
            break;

        case Impl::Backend::PERF_UNCORE:
            // Generic perf uncore implementation
            break;

        case Impl::Backend::NONE:
        default:
            return false;
    }
#endif

    return true;
}

void MemoryTrafficAnalyzer::set_peak_bandwidth_gbps(double peak_bw) {
    peak_bandwidth_gbps_ = peak_bw;
}

void MemoryTrafficAnalyzer::set_theoretical_ai(double theoretical_ai) {
    theoretical_ai_ = theoretical_ai;
}

MemoryTrafficMetrics MemoryTrafficAnalyzer::get_metrics(
    uint64_t total_flops,
    double elapsed_seconds
) const {
    MemoryTrafficMetrics metrics;
    metrics.total_flops = total_flops;
    metrics.elapsed_seconds = elapsed_seconds;
    metrics.peak_bandwidth_gbps = peak_bandwidth_gbps_;
    metrics.theoretical_arithmetic_intensity = theoretical_ai_;

    // Calculate delta with underflow protection
    // Counter wraparound or reset could cause end < start
    uint64_t read_delta = (end_reads_ >= start_reads_) ?
        (end_reads_ - start_reads_) : end_reads_;  // Assume counter wrapped
    uint64_t write_delta = (end_writes_ >= start_writes_) ?
        (end_writes_ - start_writes_) : end_writes_;  // Assume counter wrapped

    // Convert to bytes (each CAS transaction is 64 bytes)
    constexpr size_t BYTES_PER_TRANSACTION = 64;
    metrics.bytes_read_dram = read_delta * BYTES_PER_TRANSACTION;
    metrics.bytes_written_dram = write_delta * BYTES_PER_TRANSACTION;
    metrics.total_bytes_dram = metrics.bytes_read_dram + metrics.bytes_written_dram;

    // Calculate derived metrics
    if (metrics.total_bytes_dram > 0) {
        metrics.measured_arithmetic_intensity =
            static_cast<double>(total_flops) / static_cast<double>(metrics.total_bytes_dram);

        metrics.read_write_ratio =
            static_cast<double>(metrics.bytes_read_dram) /
            static_cast<double>(metrics.total_bytes_dram);
    }

    if (elapsed_seconds > 0) {
        metrics.achieved_bandwidth_gbps =
            static_cast<double>(metrics.total_bytes_dram) / (elapsed_seconds * 1e9);

        if (peak_bandwidth_gbps_ > 0) {
            metrics.bandwidth_utilization =
                metrics.achieved_bandwidth_gbps / peak_bandwidth_gbps_;
        }
    }

    // Calculate cache amplification
    if (theoretical_ai_ > 0 && metrics.measured_arithmetic_intensity > 0) {
        metrics.cache_amplification =
            theoretical_ai_ / metrics.measured_arithmetic_intensity;
    }

    // Generate quality assessment
    if (metrics.bandwidth_utilization >= 0.8) {
        metrics.bandwidth_efficiency = "excellent";
    } else if (metrics.bandwidth_utilization >= 0.6) {
        metrics.bandwidth_efficiency = "good";
    } else if (metrics.bandwidth_utilization >= 0.4) {
        metrics.bandwidth_efficiency = "moderate";
    } else {
        metrics.bandwidth_efficiency = "poor";
    }

    // Generate recommendations
    if (metrics.cache_amplification > 2.0) {
        metrics.recommendations.push_back(
            "High cache amplification (" +
            std::to_string(static_cast<int>(metrics.cache_amplification)) +
            "x) - consider loop tiling to improve cache reuse"
        );
    }

    if (metrics.bandwidth_utilization < 0.5 && metrics.measured_arithmetic_intensity < 1.0) {
        metrics.recommendations.push_back(
            "Low bandwidth utilization - kernel may be latency-bound, try prefetching"
        );
    }

    if (metrics.read_write_ratio > 0.9) {
        metrics.recommendations.push_back(
            "Read-dominated workload - consider software prefetching"
        );
    } else if (metrics.read_write_ratio < 0.3) {
        metrics.recommendations.push_back(
            "Write-dominated workload - consider non-temporal stores"
        );
    }

    return metrics;
}

// ============================================================================
// DetailedMemoryTrafficAnalyzer implementation
// ============================================================================

class DetailedMemoryTrafficAnalyzer::Impl {
public:
    std::vector<DRAMChannelInfo> channels;
    bool initialized = false;
};

DetailedMemoryTrafficAnalyzer::DetailedMemoryTrafficAnalyzer()
    : impl_(std::make_unique<Impl>()) {
}

DetailedMemoryTrafficAnalyzer::~DetailedMemoryTrafficAnalyzer() = default;

bool DetailedMemoryTrafficAnalyzer::is_available() {
    return MemoryTrafficAnalyzer::is_available();
}

bool DetailedMemoryTrafficAnalyzer::initialize() {
#ifdef __linux__
    // Count available IMC channels
    for (int imc = 0; imc < 8; ++imc) {
        std::string path = "/sys/devices/uncore_imc_" + std::to_string(imc);
        struct stat st;
        if (stat(path.c_str(), &st) != 0) break;

        DRAMChannelInfo channel;
        channel.channel_id = imc;
        channel.reads = 0;
        channel.writes = 0;
        channel.bandwidth_gbps = 0.0;
        channel.utilization = 0.0;
        impl_->channels.push_back(channel);
    }

    impl_->initialized = !impl_->channels.empty();
    return impl_->initialized;
#else
    return false;
#endif
}

void DetailedMemoryTrafficAnalyzer::start() {
    measuring_.store(true);
    // Reset channel counters
    for (auto& channel : impl_->channels) {
        channel.reads = 0;
        channel.writes = 0;
    }
}

void DetailedMemoryTrafficAnalyzer::stop() {
    measuring_.store(false);
    // Read final counter values
    // Implementation would read from perf/MSR
}

void DetailedMemoryTrafficAnalyzer::reset() {
    for (auto& channel : impl_->channels) {
        channel.reads = 0;
        channel.writes = 0;
        channel.bandwidth_gbps = 0.0;
        channel.utilization = 0.0;
    }
}

std::vector<DRAMChannelInfo> DetailedMemoryTrafficAnalyzer::get_channel_metrics() const {
    return impl_->channels;
}

MemoryTrafficMetrics DetailedMemoryTrafficAnalyzer::get_aggregated_metrics(
    uint64_t total_flops,
    double elapsed_seconds
) const {
    MemoryTrafficMetrics metrics;
    metrics.total_flops = total_flops;
    metrics.elapsed_seconds = elapsed_seconds;

    // Aggregate across all channels
    for (const auto& channel : impl_->channels) {
        constexpr size_t BYTES_PER_TRANSACTION = 64;
        metrics.bytes_read_dram += channel.reads * BYTES_PER_TRANSACTION;
        metrics.bytes_written_dram += channel.writes * BYTES_PER_TRANSACTION;
    }
    metrics.total_bytes_dram = metrics.bytes_read_dram + metrics.bytes_written_dram;

    // Calculate derived metrics
    if (metrics.total_bytes_dram > 0) {
        metrics.measured_arithmetic_intensity =
            static_cast<double>(total_flops) / static_cast<double>(metrics.total_bytes_dram);
    }

    if (elapsed_seconds > 0) {
        metrics.achieved_bandwidth_gbps =
            static_cast<double>(metrics.total_bytes_dram) / (elapsed_seconds * 1e9);
    }

    return metrics;
}

DetailedMemoryTrafficAnalyzer::BottleneckAnalysis
DetailedMemoryTrafficAnalyzer::analyze_bottlenecks() const {
    BottleneckAnalysis analysis;

    if (impl_->channels.size() < 2) {
        analysis.has_bottleneck = false;
        return analysis;
    }

    // Check for channel imbalance
    double total_traffic = 0.0;
    double max_traffic = 0.0;
    double min_traffic = std::numeric_limits<double>::max();

    for (const auto& channel : impl_->channels) {
        double traffic = static_cast<double>(channel.reads + channel.writes);
        total_traffic += traffic;
        max_traffic = std::max(max_traffic, traffic);
        min_traffic = std::min(min_traffic, traffic);
    }

    if (total_traffic > 0 && min_traffic > 0) {
        double imbalance = max_traffic / min_traffic;
        if (imbalance > 1.5) {
            analysis.has_bottleneck = true;
            analysis.severity = std::min(1.0, (imbalance - 1.0) / 2.0);
            analysis.bottleneck_type = "channel_imbalance";
            analysis.recommendations.push_back(
                "Memory access pattern causes uneven channel loading"
            );
            analysis.recommendations.push_back(
                "Consider interleaving data across NUMA nodes"
            );
        }
    }

    // Check for bandwidth saturation
    double avg_utilization = 0.0;
    for (const auto& channel : impl_->channels) {
        avg_utilization += channel.utilization;
    }
    avg_utilization /= impl_->channels.size();

    if (avg_utilization > 0.85) {
        analysis.has_bottleneck = true;
        analysis.severity = std::max(analysis.severity, avg_utilization);
        analysis.bottleneck_type = "bandwidth_limit";
        analysis.recommendations.push_back(
            "Memory bandwidth is saturated - optimize for cache reuse"
        );
    }

    return analysis;
}

// ============================================================================
// Access pattern analysis
// ============================================================================

AccessPatternMetrics analyze_access_pattern(
    uint64_t l1_hits,
    uint64_t l1_misses,
    uint64_t l2_hits,
    uint64_t l2_misses,
    uint64_t l3_hits,
    uint64_t l3_misses,
    size_t working_set_bytes,
    size_t element_stride
) {
    AccessPatternMetrics metrics;

    uint64_t total_accesses = l1_hits + l1_misses;
    if (total_accesses == 0) {
        metrics.pattern_type = "unknown";
        return metrics;
    }

    // Calculate hit rates
    double l1_hit_rate = static_cast<double>(l1_hits) / total_accesses;
    double l2_hit_rate = (l1_misses > 0) ?
        static_cast<double>(l2_hits) / l1_misses : 0.0;
    double l3_hit_rate = (l2_misses > 0) ?
        static_cast<double>(l3_hits) / l2_misses : 0.0;

    // Infer access pattern from cache behavior

    // Sequential access: high L1 hit rate, low DRAM traffic
    // Random access: poor hit rates across all levels
    // Strided access: moderate hit rates, depends on stride

    if (l1_hit_rate > 0.9) {
        // Very high L1 hit rate suggests temporal locality
        metrics.sequential_ratio = l1_hit_rate;
        metrics.pattern_type = "sequential";
        metrics.prefetch_friendly = 0.95;
    } else if (l1_hit_rate > 0.7 && l2_hit_rate > 0.8) {
        // Good L2 capture suggests moderate stride
        metrics.stride_ratio = 0.8;
        metrics.sequential_ratio = 0.2;
        metrics.pattern_type = "strided";
        metrics.prefetch_friendly = 0.7;

        // Estimate stride from L1 miss rate
        // Miss rate ~ stride / cache_line_size for unit stride
        double miss_rate = 1.0 - l1_hit_rate;
        metrics.average_stride = static_cast<size_t>(miss_rate * 64);  // 64 byte lines
    } else if (l3_hit_rate < 0.5) {
        // Poor hit rates everywhere suggests random access
        metrics.random_ratio = 0.8;
        metrics.pattern_type = "random";
        metrics.prefetch_friendly = 0.2;
    } else {
        // Mixed pattern
        metrics.sequential_ratio = l1_hit_rate * 0.5;
        metrics.stride_ratio = l2_hit_rate * 0.3;
        metrics.random_ratio = 1.0 - metrics.sequential_ratio - metrics.stride_ratio;
        metrics.pattern_type = "mixed";
        metrics.prefetch_friendly = 0.5;
    }

    // Generate recommendations
    if (metrics.pattern_type == "random") {
        metrics.recommendations.push_back(
            "Random access pattern detected - consider data layout restructuring"
        );
        metrics.recommendations.push_back(
            "Prefetching will not help random access patterns"
        );
    } else if (metrics.pattern_type == "strided" && metrics.average_stride > 64) {
        metrics.recommendations.push_back(
            "Large stride detected (" + std::to_string(metrics.average_stride) +
            " bytes) - consider data transposition or blocking"
        );
    } else if (metrics.pattern_type == "sequential") {
        metrics.recommendations.push_back(
            "Sequential access pattern - hardware prefetcher should be effective"
        );
    }

    return metrics;
}

BandwidthPrediction predict_bandwidth(
    const AccessPatternMetrics& pattern,
    const HardwareInfo& hw,
    size_t working_set_bytes
) {
    BandwidthPrediction prediction;

    double peak_bw = hw.measured_memory_bw_gbps;
    if (peak_bw <= 0) peak_bw = 20.0;  // Default estimate

    // Base efficiency from access pattern
    double pattern_efficiency = pattern.prefetch_friendly * 0.5 +
                                 pattern.sequential_ratio * 0.3 +
                                 (1.0 - pattern.random_ratio) * 0.2;

    // Latency factor: larger working sets have more latency hiding opportunity
    size_t l3_size = hw.cache.l3_size_kb * 1024;
    double latency_factor;
    if (working_set_bytes <= l3_size) {
        latency_factor = 0.9;  // Mostly in L3, latency hidden
    } else {
        // Larger working sets are more latency sensitive
        double ratio = static_cast<double>(working_set_bytes) / l3_size;
        latency_factor = 0.9 / std::sqrt(ratio);
    }

    prediction.bandwidth_efficiency = pattern_efficiency * latency_factor;
    prediction.expected_bandwidth_gbps = peak_bw * prediction.bandwidth_efficiency;
    prediction.latency_bound_factor = 1.0 - latency_factor;

    // Determine limiting factor
    if (pattern.random_ratio > 0.5) {
        prediction.limiting_factor = "latency";
    } else if (working_set_bytes > l3_size * 2) {
        prediction.limiting_factor = "bandwidth";
    } else {
        prediction.limiting_factor = "cache";
    }

    return prediction;
}

}  // namespace simd_bench
