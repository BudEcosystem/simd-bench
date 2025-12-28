#include "simd_bench/hardware.h"
#include "simd_bench/timing.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace hn = hwy::HWY_NAMESPACE;

namespace simd_bench {

// CPUIDReader implementation
CPUIDReader::CPUIDReader() {
    query_cpuid();
    detect_features();
}

void CPUIDReader::query_cpuid() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

    // Get vendor string
    __cpuid(0, eax, ebx, ecx, edx);
    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    vendor_ = vendor;

    // Get processor info
    __cpuid(1, eax, ebx, ecx, edx);
    family_ = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
    model_ = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);
    stepping_ = eax & 0xF;

    // Get brand string
    char brand[49];
    memset(brand, 0, sizeof(brand));

    __cpuid(0x80000000, eax, ebx, ecx, edx);
    if (eax >= 0x80000004) {
        __cpuid(0x80000002, eax, ebx, ecx, edx);
        memcpy(brand, &eax, 4);
        memcpy(brand + 4, &ebx, 4);
        memcpy(brand + 8, &ecx, 4);
        memcpy(brand + 12, &edx, 4);

        __cpuid(0x80000003, eax, ebx, ecx, edx);
        memcpy(brand + 16, &eax, 4);
        memcpy(brand + 20, &ebx, 4);
        memcpy(brand + 24, &ecx, 4);
        memcpy(brand + 28, &edx, 4);

        __cpuid(0x80000004, eax, ebx, ecx, edx);
        memcpy(brand + 32, &eax, 4);
        memcpy(brand + 36, &ebx, 4);
        memcpy(brand + 40, &ecx, 4);
        memcpy(brand + 44, &edx, 4);
    }
    brand_ = brand;
    // Trim leading spaces
    size_t start = brand_.find_first_not_of(' ');
    if (start != std::string::npos) {
        brand_ = brand_.substr(start);
    }
#endif
}

void CPUIDReader::detect_features() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

    __cpuid(1, eax, ebx, ecx, edx);

    sse_ = (edx & (1 << 25)) != 0;
    sse2_ = (edx & (1 << 26)) != 0;
    sse3_ = (ecx & (1 << 0)) != 0;
    ssse3_ = (ecx & (1 << 9)) != 0;
    sse4_1_ = (ecx & (1 << 19)) != 0;
    sse4_2_ = (ecx & (1 << 20)) != 0;
    fma_ = (ecx & (1 << 12)) != 0;
    avx_ = (ecx & (1 << 28)) != 0;

    // Check for AVX2 and AVX-512
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    avx2_ = (ebx & (1 << 5)) != 0;
    avx512f_ = (ebx & (1 << 16)) != 0;
    avx512dq_ = (ebx & (1 << 17)) != 0;
    avx512bw_ = (ebx & (1 << 30)) != 0;
    avx512vl_ = (ebx & (1 << 31)) != 0;
#endif
}

// Hardware detection
HardwareInfo HardwareInfo::detect() {
    HardwareInfo info;

#if defined(__x86_64__) || defined(_M_X64)
    info.architecture = "x86_64";

    CPUIDReader cpuid;
    info.cpu_vendor = cpuid.get_vendor();
    info.cpu_brand = cpuid.get_brand();

    // Detect SIMD extensions
    if (cpuid.has_sse()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSE;
    if (cpuid.has_sse2()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSE2;
    if (cpuid.has_sse3()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSE3;
    if (cpuid.has_ssse3()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSSE3;
    if (cpuid.has_sse4_1()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSE4_1;
    if (cpuid.has_sse4_2()) info.simd_extensions = info.simd_extensions | SIMDExtension::SSE4_2;
    if (cpuid.has_avx()) info.simd_extensions = info.simd_extensions | SIMDExtension::AVX;
    if (cpuid.has_avx2()) info.simd_extensions = info.simd_extensions | SIMDExtension::AVX2;
    if (cpuid.has_fma()) info.simd_extensions = info.simd_extensions | SIMDExtension::FMA;
    if (cpuid.has_avx512f()) info.simd_extensions = info.simd_extensions | SIMDExtension::AVX512F;

    // Determine max vector width
    if (cpuid.has_avx512f()) {
        info.max_vector_bits = 512;
        info.fma_units = 2;
    } else if (cpuid.has_avx2() || cpuid.has_avx()) {
        info.max_vector_bits = 256;
        info.fma_units = 2;
    } else if (cpuid.has_sse2()) {
        info.max_vector_bits = 128;
        info.fma_units = 1;
    }

#elif defined(__aarch64__)
    info.architecture = "aarch64";
    info.simd_extensions = SIMDExtension::NEON;
    info.max_vector_bits = 128;
    info.fma_units = 2;
#endif

    // Core counts
    info.logical_cores = std::thread::hardware_concurrency();
    info.physical_cores = info.logical_cores / 2;  // Estimate for SMT
    if (info.physical_cores == 0) info.physical_cores = 1;

    // Read /proc/cpuinfo for more details
    auto cpuinfo = parse_cpuinfo();
    for (const auto& [key, value] : cpuinfo) {
        if (key == "cpu MHz") {
            try {
                info.base_frequency_ghz = std::stod(value) / 1000.0;
            } catch (...) {}
        }
        if (key == "cache size") {
            try {
                // Parse cache size (e.g., "16384 KB")
                size_t kb = std::stoull(value);
                info.cache.l3_size_kb = kb;
            } catch (...) {}
        }
    }

    // Measure actual frequency
    info.measured_frequency_ghz = measure_cpu_frequency();
    if (info.base_frequency_ghz == 0) {
        info.base_frequency_ghz = info.measured_frequency_ghz;
    }
    info.max_frequency_ghz = info.measured_frequency_ghz * 1.2;  // Estimate turbo

    // Cache sizes (defaults, may be overridden by cpuinfo)
    if (info.cache.l1d_size_kb == 0) info.cache.l1d_size_kb = 32;
    if (info.cache.l1i_size_kb == 0) info.cache.l1i_size_kb = 32;
    if (info.cache.l2_size_kb == 0) info.cache.l2_size_kb = 256;
    if (info.cache.l3_size_kb == 0) info.cache.l3_size_kb = 8192;

    // Memory info
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                try {
                    size_t kb = std::stoull(line.substr(10));
                    info.total_memory_mb = kb / 1024;
                } catch (...) {}
                break;
            }
        }
    }

    // Measure memory bandwidth
    info.measured_memory_bw_gbps = measure_memory_bandwidth_gbps(64);

    // Calculate theoretical peak
    info.theoretical_peak_sp_gflops = info.calculate_peak_gflops(false);
    info.theoretical_peak_dp_gflops = info.calculate_peak_gflops(true);

    return info;
}

std::string HardwareInfo::get_simd_string() const {
    std::string result;

    if (has_extension(simd_extensions, SIMDExtension::SSE)) result += "SSE ";
    if (has_extension(simd_extensions, SIMDExtension::SSE2)) result += "SSE2 ";
    if (has_extension(simd_extensions, SIMDExtension::SSE3)) result += "SSE3 ";
    if (has_extension(simd_extensions, SIMDExtension::SSSE3)) result += "SSSE3 ";
    if (has_extension(simd_extensions, SIMDExtension::SSE4_1)) result += "SSE4.1 ";
    if (has_extension(simd_extensions, SIMDExtension::SSE4_2)) result += "SSE4.2 ";
    if (has_extension(simd_extensions, SIMDExtension::AVX)) result += "AVX ";
    if (has_extension(simd_extensions, SIMDExtension::AVX2)) result += "AVX2 ";
    if (has_extension(simd_extensions, SIMDExtension::FMA)) result += "FMA ";
    if (has_extension(simd_extensions, SIMDExtension::AVX512F)) result += "AVX-512F ";
    if (has_extension(simd_extensions, SIMDExtension::NEON)) result += "NEON ";
    if (has_extension(simd_extensions, SIMDExtension::SVE)) result += "SVE ";

    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

double HardwareInfo::calculate_peak_gflops(bool double_precision) const {
    // Peak GFLOPS = freq * vector_lanes * FMA_units * 2 (for FMA = mul + add)
    int lanes = max_vector_bits / (double_precision ? 64 : 32);
    double freq = measured_frequency_ghz > 0 ? measured_frequency_ghz : base_frequency_ghz;
    return freq * lanes * fma_units * 2;
}

double HardwareInfo::calculate_ridge_point() const {
    // Ridge point = peak GFLOPS / memory bandwidth (GB/s)
    // Result is in FLOP/byte
    if (measured_memory_bw_gbps <= 0) return 10.0;  // Default
    return theoretical_peak_sp_gflops / measured_memory_bw_gbps;
}

bool HardwareInfo::has_hardware_counters() const {
#ifdef __linux__
    std::ifstream file("/proc/sys/kernel/perf_event_paranoid");
    if (file.is_open()) {
        int level;
        file >> level;
        return level <= 2;  // Level 2 or below allows user-space access
    }
#endif
    return false;
}

bool HardwareInfo::has_rapl() const {
#ifdef __linux__
    std::ifstream file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    return file.is_open();
#endif
    return false;
}

// Memory bandwidth measurement
double measure_memory_bandwidth_gbps(size_t size_mb) {
    // Minimum 1 MB
    if (size_mb < 1) size_mb = 1;
    const size_t size = size_mb * 1024 * 1024;
    const size_t count = size / sizeof(float);

    auto src = hwy::AllocateAligned<float>(count);
    auto dst = hwy::AllocateAligned<float>(count);

    // Initialize
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<float>(i);
    }

    // Warmup
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t i = 0; i + N <= count; i += N) {
        hn::Store(hn::Load(d, src.get() + i), d, dst.get() + i);
    }

    // Measure
    Timer timer;
    timer.start();

    const int iterations = 10;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + N <= count; i += N) {
            hn::Store(hn::Load(d, src.get() + i), d, dst.get() + i);
        }
    }

    timer.stop();

    // Bytes transferred = read + write = 2 * size * iterations
    double bytes = 2.0 * size * iterations;
    double seconds = timer.elapsed_seconds();

    return bytes / seconds / 1e9;  // GB/s
}

BandwidthMeasurement measure_cache_bandwidths() {
    BandwidthMeasurement bw;

    // All measurements use minimum 1 MB due to allocation constraints
    // The actual cache level tested depends on working set vs cache size

    // L1/L2 estimate: 1 MB (repeated access warms caches)
    bw.l1_bandwidth_gbps = measure_memory_bandwidth_gbps(1);

    // L2/L3 estimate: 2 MB
    bw.l2_bandwidth_gbps = measure_memory_bandwidth_gbps(2);

    // L3: 4 MB
    bw.l3_bandwidth_gbps = measure_memory_bandwidth_gbps(4);

    // DRAM: 64 MB
    bw.dram_bandwidth_gbps = measure_memory_bandwidth_gbps(64);

    return bw;
}

double read_cpu_frequency_from_cpuinfo() {
    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open()) return 0.0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                try {
                    return std::stod(line.substr(pos + 1)) / 1000.0;
                } catch (...) {}
            }
        }
    }
    return 0.0;
}

double read_cpu_frequency_from_scaling() {
    std::ifstream file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (file.is_open()) {
        uint64_t freq_khz;
        file >> freq_khz;
        return static_cast<double>(freq_khz) / 1e6;  // Convert to GHz
    }
    return 0.0;
}

double measure_cpu_frequency() {
    return Timer::measure_frequency_ghz();
}

std::vector<std::pair<std::string, std::string>> parse_cpuinfo() {
    std::vector<std::pair<std::string, std::string>> result;

    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open()) return result;

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Trim whitespace
            auto trim = [](std::string& s) {
                size_t start = s.find_first_not_of(" \t");
                size_t end = s.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos) {
                    s = s.substr(start, end - start + 1);
                }
            };

            trim(key);
            trim(value);

            result.emplace_back(key, value);
        }
    }

    return result;
}

}  // namespace simd_bench
