#include "simd_bench/metrics_analyzer.h"
#include "simd_bench/timing.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace simd_bench {

MetricsAnalyzer::MetricsAnalyzer() {
    cpu_vendor_ = detect_vendor();
}

MetricsAnalyzer::MetricsAnalyzer(IPerformanceCounters* counters)
    : counters_(counters) {
    cpu_vendor_ = detect_vendor();
}

void MetricsAnalyzer::set_counters(IPerformanceCounters* counters) {
    counters_ = counters;
}

void MetricsAnalyzer::set_cpu_vendor(CPUVendor vendor) {
    cpu_vendor_ = vendor;
}

CPUVendor MetricsAnalyzer::detect_vendor() const {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0) return CPUVendor::INTEL;
    if (strcmp(vendor, "AuthenticAMD") == 0) return CPUVendor::AMD;
#elif defined(__aarch64__)
    return CPUVendor::ARM;
#elif defined(__riscv)
    return CPUVendor::RISCV;
#elif defined(__powerpc__) || defined(__powerpc64__)
    return CPUVendor::IBM_POWER;
#endif
    return CPUVendor::UNKNOWN;
}

MPKIMetrics MetricsAnalyzer::calculate_mpki(const CounterValues& values) const {
    MPKIMetrics mpki;

    uint64_t instructions = values.get(CounterEvent::INSTRUCTIONS);
    if (instructions == 0) {
        return mpki;
    }

    double kilo_instructions = static_cast<double>(instructions) / 1000.0;

    // L1 MPKI
    uint64_t l1_misses = values.get(CounterEvent::L1D_READ_MISS);
    mpki.l1_mpki = static_cast<double>(l1_misses) / kilo_instructions;
    mpki.l1_acceptable = mpki.l1_mpki < MPKIThresholds::L1_ACCEPTABLE;

    // L2 MPKI
    uint64_t l2_misses = values.get(CounterEvent::L2_READ_MISS);
    mpki.l2_mpki = static_cast<double>(l2_misses) / kilo_instructions;
    mpki.l2_acceptable = mpki.l2_mpki < MPKIThresholds::L2_ACCEPTABLE;

    // L3 MPKI
    uint64_t l3_misses = values.get(CounterEvent::L3_READ_MISS);
    mpki.l3_mpki = static_cast<double>(l3_misses) / kilo_instructions;
    mpki.l3_acceptable = mpki.l3_mpki < MPKIThresholds::L3_ACCEPTABLE;

    mpki.has_cache_issues = !mpki.l1_acceptable || !mpki.l2_acceptable || !mpki.l3_acceptable;

    // Generate evaluation string
    std::ostringstream oss;
    if (!mpki.has_cache_issues) {
        oss << "Cache hierarchy performing well";
    } else {
        oss << "Cache issues detected: ";
        if (!mpki.l1_acceptable) {
            oss << "L1 MPKI=" << mpki.l1_mpki << " (threshold=" << MPKIThresholds::L1_ACCEPTABLE << "); ";
        }
        if (!mpki.l2_acceptable) {
            oss << "L2 MPKI=" << mpki.l2_mpki << " (threshold=" << MPKIThresholds::L2_ACCEPTABLE << "); ";
        }
        if (!mpki.l3_acceptable) {
            oss << "L3 MPKI=" << mpki.l3_mpki << " (threshold=" << MPKIThresholds::L3_ACCEPTABLE << ")";
        }
    }
    mpki.evaluation = oss.str();

    return mpki;
}

DSBMetrics MetricsAnalyzer::calculate_dsb_coverage(const CounterValues& values) const {
    DSBMetrics dsb;

    dsb.dsb_uops = values.get(CounterEvent::IDQ_DSB_UOPS);
    dsb.mite_uops = values.get(CounterEvent::IDQ_MITE_UOPS);
    dsb.ms_uops = values.get(CounterEvent::IDQ_MS_UOPS);

    uint64_t total_uops = dsb.dsb_uops + dsb.mite_uops + dsb.ms_uops;
    if (total_uops == 0) {
        dsb.recommendation = "DSB metrics not available (Intel-specific)";
        return dsb;
    }

    dsb.dsb_coverage = static_cast<double>(dsb.dsb_uops) / total_uops;
    dsb.mite_coverage = static_cast<double>(dsb.mite_uops) / total_uops;
    dsb.ms_coverage = static_cast<double>(dsb.ms_uops) / total_uops;

    dsb.is_dsb_efficient = dsb.dsb_coverage >= QualityThresholds::DSB_COVERAGE_ACCEPTABLE;

    if (dsb.is_dsb_efficient) {
        dsb.recommendation = "Good DSB coverage - instruction cache is efficient";
    } else if (dsb.mite_coverage > 0.3) {
        dsb.recommendation = "High MITE usage - consider code layout optimization or inlining hot functions";
    } else if (dsb.ms_coverage > 0.1) {
        dsb.recommendation = "High microcode sequencer usage - reduce complex operations (divisions, transcendentals)";
    } else {
        dsb.recommendation = "Consider optimizing instruction cache usage";
    }

    return dsb;
}

PortMetrics MetricsAnalyzer::calculate_port_utilization(const CounterValues& values) const {
    PortMetrics ports;

    uint64_t cycles = values.get(CounterEvent::CYCLES);
    if (cycles == 0) {
        return ports;
    }

    // Get port dispatch counts
    uint64_t port0 = values.get(CounterEvent::UOPS_DISPATCHED_PORT_0);
    uint64_t port1 = values.get(CounterEvent::UOPS_DISPATCHED_PORT_1);
    uint64_t port5 = values.get(CounterEvent::UOPS_DISPATCHED_PORT_5);
    uint64_t port6 = values.get(CounterEvent::UOPS_DISPATCHED_PORT_6);

    // Calculate utilization (uops/cycle for each port)
    ports.port0_utilization = static_cast<double>(port0) / cycles;
    ports.port1_utilization = static_cast<double>(port1) / cycles;
    ports.port5_utilization = static_cast<double>(port5) / cycles;
    ports.port6_utilization = static_cast<double>(port6) / cycles;

    // Check for Port 5 saturation (shuffle bottleneck)
    ports.port5_saturated = ports.port5_utilization > QualityThresholds::PORT_SATURATION_WARNING;

    // Check FMA port balance
    double fma_imbalance = std::abs(ports.port0_utilization - ports.port1_utilization);
    ports.fma_ports_balanced = fma_imbalance < 0.2;  // < 20% imbalance

    // Identify bottleneck
    if (ports.port5_saturated) {
        ports.bottleneck = "Port 5 (shuffle/permute) saturated - reduce shuffle operations or restructure algorithm";
    } else if (!ports.fma_ports_balanced) {
        ports.bottleneck = "FMA port imbalance - consider interleaving different operations";
    } else if (ports.port0_utilization > QualityThresholds::PORT_SATURATION_CRITICAL ||
               ports.port1_utilization > QualityThresholds::PORT_SATURATION_CRITICAL) {
        ports.bottleneck = "FMA ports saturated - increase instruction-level parallelism";
    } else {
        ports.bottleneck = "No significant port bottleneck detected";
    }

    return ports;
}

CacheLineSplitMetrics MetricsAnalyzer::calculate_cache_line_splits(const CounterValues& values) const {
    CacheLineSplitMetrics splits;

    splits.split_loads = values.get(CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS);
    splits.split_stores = values.get(CounterEvent::MEM_INST_RETIRED_SPLIT_STORES);
    splits.total_loads = values.get(CounterEvent::MEM_LOAD_RETIRED);
    splits.total_stores = values.get(CounterEvent::MEM_STORE_RETIRED);

    if (splits.total_loads > 0) {
        splits.split_load_ratio = static_cast<double>(splits.split_loads) / splits.total_loads;
    }
    if (splits.total_stores > 0) {
        splits.split_store_ratio = static_cast<double>(splits.split_stores) / splits.total_stores;
    }

    // > 1% split accesses indicates alignment issues
    splits.has_alignment_issues = (splits.split_load_ratio > 0.01) || (splits.split_store_ratio > 0.01);

    if (splits.has_alignment_issues) {
        splits.recommendation = "Alignment issues detected - ensure data is aligned to cache line boundaries (64 bytes)";
    } else {
        splits.recommendation = "Memory alignment is good";
    }

    return splits;
}

AVX512FrequencyMetrics MetricsAnalyzer::calculate_avx512_frequency(const CounterValues& values) const {
    AVX512FrequencyMetrics freq;

    freq.l0_cycles = values.get(CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE);
    freq.l1_cycles = values.get(CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE);
    freq.l2_cycles = values.get(CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE);

    uint64_t total_license_cycles = freq.l0_cycles + freq.l1_cycles + freq.l2_cycles;

    if (total_license_cycles == 0) {
        freq.recommendation = "AVX-512 license metrics not available or no AVX-512 usage";
        return freq;
    }

    // Calculate weighted average license level (0=L0, 1=L1, 2=L2)
    freq.avg_license_level = (freq.l0_cycles * 0.0 + freq.l1_cycles * 1.0 + freq.l2_cycles * 2.0) /
                             total_license_cycles;

    // Estimate frequency penalty (roughly 5-15% per level on many chips)
    freq.frequency_penalty_estimate = freq.avg_license_level * 0.08;  // ~8% per level
    freq.has_frequency_penalty = freq.avg_license_level > 0.5;

    if (freq.avg_license_level < 0.3) {
        freq.recommendation = "Light AVX-512 usage - minimal frequency impact";
    } else if (freq.avg_license_level < 1.0) {
        freq.recommendation = "Moderate AVX-512 usage - some frequency reduction expected";
    } else {
        freq.recommendation = "Heavy AVX-512 usage - significant frequency downclocking. Consider mixing with non-AVX-512 work or using AVX2 for light operations";
    }

    return freq;
}

double MetricsAnalyzer::calculate_vectorization_ratio(const CounterValues& values) const {
    uint64_t scalar_ops = 0;
    uint64_t vector_ops = 0;

    if (cpu_vendor_ == CPUVendor::INTEL) {
        scalar_ops = values.get(CounterEvent::FP_ARITH_SCALAR_SINGLE) +
                     values.get(CounterEvent::FP_ARITH_SCALAR_DOUBLE);
        vector_ops = values.get(CounterEvent::FP_ARITH_128B_PACKED_SINGLE) +
                     values.get(CounterEvent::FP_ARITH_128B_PACKED_DOUBLE) +
                     values.get(CounterEvent::FP_ARITH_256B_PACKED_SINGLE) +
                     values.get(CounterEvent::FP_ARITH_256B_PACKED_DOUBLE) +
                     values.get(CounterEvent::FP_ARITH_512B_PACKED_SINGLE) +
                     values.get(CounterEvent::FP_ARITH_512B_PACKED_DOUBLE);
    } else if (cpu_vendor_ == CPUVendor::AMD) {
        // AMD uses combined event, need to parse by umask
        // For now, use the combined event
        uint64_t total = values.get(CounterEvent::AMD_FP_RET_SSE_AVX_OPS);
        scalar_ops = values.get(CounterEvent::FP_ARITH_SCALAR_SINGLE) +
                     values.get(CounterEvent::FP_ARITH_SCALAR_DOUBLE);
        vector_ops = total > scalar_ops ? total - scalar_ops : 0;
    } else if (cpu_vendor_ == CPUVendor::ARM) {
        scalar_ops = values.get(CounterEvent::ARM_FP_SPEC);
        vector_ops = values.get(CounterEvent::ARM_ASE_SPEC) +
                     values.get(CounterEvent::ARM_SVE_INST_SPEC);
    }

    uint64_t total_ops = scalar_ops + vector_ops;
    if (total_ops == 0) {
        return 0.0;
    }

    return static_cast<double>(vector_ops) / total_ops;
}

double MetricsAnalyzer::calculate_vector_width_utilization(const CounterValues& values) const {
    if (cpu_vendor_ != CPUVendor::INTEL && cpu_vendor_ != CPUVendor::AMD) {
        return 0.0;  // Width utilization is x86-specific in this implementation
    }

    // Calculate weighted average vector width
    uint64_t ops_128 = values.get(CounterEvent::FP_ARITH_128B_PACKED_SINGLE) +
                       values.get(CounterEvent::FP_ARITH_128B_PACKED_DOUBLE);
    uint64_t ops_256 = values.get(CounterEvent::FP_ARITH_256B_PACKED_SINGLE) +
                       values.get(CounterEvent::FP_ARITH_256B_PACKED_DOUBLE);
    uint64_t ops_512 = values.get(CounterEvent::FP_ARITH_512B_PACKED_SINGLE) +
                       values.get(CounterEvent::FP_ARITH_512B_PACKED_DOUBLE);

    uint64_t total_vector_ops = ops_128 + ops_256 + ops_512;
    if (total_vector_ops == 0) {
        return 0.0;
    }

    // Weight by bits
    double avg_bits = (ops_128 * 128.0 + ops_256 * 256.0 + ops_512 * 512.0) / total_vector_ops;

    // Determine max available width (assume 512 if 512-bit ops exist, else 256)
    double max_bits = ops_512 > 0 ? 512.0 : (ops_256 > 0 ? 256.0 : 128.0);

    return avg_bits / max_bits;
}

double MetricsAnalyzer::calculate_ipc(const CounterValues& values) const {
    uint64_t cycles = values.get(CounterEvent::CYCLES);
    uint64_t instructions = values.get(CounterEvent::INSTRUCTIONS);

    if (cycles == 0) {
        return 0.0;
    }

    return static_cast<double>(instructions) / cycles;
}

ExtendedSIMDMetrics MetricsAnalyzer::calculate_extended_simd(
    const CounterValues& values,
    double elapsed_seconds,
    size_t total_flops
) const {
    ExtendedSIMDMetrics metrics;

    metrics.vectorization_ratio = calculate_vectorization_ratio(values);
    metrics.vector_width_utilization = calculate_vector_width_utilization(values);

    // FMA utilization (estimate from port balance)
    auto ports = calculate_port_utilization(values);
    double fma_port_avg = (ports.port0_utilization + ports.port1_utilization) / 2.0;
    metrics.fma_utilization = std::min(1.0, fma_port_avg);

    // Architecture-specific metrics
    if (cpu_vendor_ == CPUVendor::INTEL) {
        auto freq = calculate_avx512_frequency(values);
        metrics.avx512_license_level = freq.avg_license_level;
    } else if (cpu_vendor_ == CPUVendor::AMD) {
        // AMD dual-pump ratio for 512-bit operations
        uint64_t ops_256 = values.get(CounterEvent::FP_ARITH_256B_PACKED_SINGLE) +
                           values.get(CounterEvent::FP_ARITH_256B_PACKED_DOUBLE);
        uint64_t ops_512 = values.get(CounterEvent::FP_ARITH_512B_PACKED_SINGLE) +
                           values.get(CounterEvent::FP_ARITH_512B_PACKED_DOUBLE);
        if (ops_256 + ops_512 > 0) {
            metrics.amd_dual_pump_ratio = static_cast<double>(ops_512) / (ops_256 + ops_512);
        }
    } else if (cpu_vendor_ == CPUVendor::ARM) {
        // SVE predicate efficiency
        uint64_t partial = values.get(CounterEvent::ARM_SVE_PRED_PARTIAL);
        uint64_t full = values.get(CounterEvent::ARM_SVE_PRED_FULL);
        uint64_t empty = values.get(CounterEvent::ARM_SVE_PRED_EMPTY);
        uint64_t total_pred = partial + full + empty;
        if (total_pred > 0) {
            metrics.sve_predicate_efficiency = static_cast<double>(full) / total_pred;
        }
    }

    // Calculate quality score (0-100)
    double ipc = calculate_ipc(values);
    double ipc_score = score_ipc(ipc);
    double vec_score = score_vectorization(metrics.vectorization_ratio);
    auto mpki = calculate_mpki(values);
    double cache_score = score_cache_efficiency(mpki);
    auto dsb = calculate_dsb_coverage(values);
    double dsb_score = score_dsb_efficiency(dsb);
    double port_score = score_port_balance(ports);

    // Weighted average
    metrics.quality_score = (ipc_score * 0.25 + vec_score * 0.25 + cache_score * 0.2 +
                             dsb_score * 0.15 + port_score * 0.15);

    metrics.quality_rating = get_quality_rating(metrics.quality_score);

    // Generate issues and suggestions
    if (ipc < QualityThresholds::IPC_HEALTHY) {
        metrics.issues.push_back("Low IPC (" + std::to_string(ipc) + ")");
        metrics.suggestions.push_back("Increase instruction-level parallelism");
    }
    if (metrics.vectorization_ratio < QualityThresholds::VECTORIZATION_ACCEPTABLE) {
        metrics.issues.push_back("Low vectorization ratio (" +
                                  std::to_string(metrics.vectorization_ratio * 100) + "%)");
        metrics.suggestions.push_back("Increase SIMD usage - avoid scalar operations");
    }
    if (mpki.has_cache_issues) {
        metrics.issues.push_back("Cache efficiency issues");
        metrics.suggestions.push_back("Improve data locality and access patterns");
    }
    if (ports.port5_saturated) {
        metrics.issues.push_back("Shuffle port (Port 5) saturated");
        metrics.suggestions.push_back("Reduce shuffle/permute operations");
    }

    return metrics;
}

std::vector<std::string> MetricsAnalyzer::evaluate_quality(const CounterValues& values) const {
    std::vector<std::string> recommendations;

    double ipc = calculate_ipc(values);
    if (ipc < QualityThresholds::IPC_HEALTHY) {
        recommendations.push_back("Low IPC (" + std::to_string(ipc) +
                                   ") - increase instruction-level parallelism");
    }

    auto mpki = calculate_mpki(values);
    if (mpki.has_cache_issues) {
        recommendations.push_back(mpki.evaluation);
    }

    auto dsb = calculate_dsb_coverage(values);
    if (!dsb.is_dsb_efficient) {
        recommendations.push_back(dsb.recommendation);
    }

    auto ports = calculate_port_utilization(values);
    if (ports.port5_saturated || !ports.fma_ports_balanced) {
        recommendations.push_back(ports.bottleneck);
    }

    auto splits = calculate_cache_line_splits(values);
    if (splits.has_alignment_issues) {
        recommendations.push_back(splits.recommendation);
    }

    if (cpu_vendor_ == CPUVendor::INTEL) {
        auto freq = calculate_avx512_frequency(values);
        if (freq.has_frequency_penalty) {
            recommendations.push_back(freq.recommendation);
        }
    }

    if (recommendations.empty()) {
        recommendations.push_back("Kernel is well-optimized - no significant issues detected");
    }

    return recommendations;
}

MetricsAnalyzer::AnalysisResult MetricsAnalyzer::measure_and_analyze(
    const std::function<void()>& func,
    size_t iterations,
    size_t flops_per_iteration
) {
    AnalysisResult result;

    if (!counters_) {
        return result;
    }

    // Configure events
    counters_->clear_events();
    for (auto event : get_required_events()) {
        counters_->add_event(event);
    }

    Timer timer;
    timer.start();
    counters_->start();

    for (size_t i = 0; i < iterations; ++i) {
        func();
    }

    counters_->stop();
    timer.stop();

    CounterValues values = counters_->read();
    double elapsed = timer.elapsed_seconds();
    size_t total_flops = iterations * flops_per_iteration;

    result.mpki = calculate_mpki(values);
    result.dsb = calculate_dsb_coverage(values);
    result.ports = calculate_port_utilization(values);
    result.cache_splits = calculate_cache_line_splits(values);
    result.avx512_freq = calculate_avx512_frequency(values);
    result.extended_simd = calculate_extended_simd(values, elapsed, total_flops);
    result.ipc = calculate_ipc(values);
    result.vectorization_ratio = calculate_vectorization_ratio(values);
    result.recommendations = evaluate_quality(values);
    result.overall_quality_score = result.extended_simd.quality_score;

    return result;
}

std::vector<CounterEvent> MetricsAnalyzer::get_required_events() const {
    std::vector<CounterEvent> events = {
        CounterEvent::CYCLES,
        CounterEvent::INSTRUCTIONS,
        CounterEvent::L1D_READ_MISS,
        CounterEvent::L2_READ_MISS,
        CounterEvent::L3_READ_MISS,
        CounterEvent::MEM_LOAD_RETIRED,
        CounterEvent::MEM_STORE_RETIRED,
    };

    // Add architecture-specific events
    switch (cpu_vendor_) {
        case CPUVendor::INTEL:
            for (auto e : get_intel_events()) {
                events.push_back(e);
            }
            break;
        case CPUVendor::AMD:
            for (auto e : get_amd_events()) {
                events.push_back(e);
            }
            break;
        case CPUVendor::ARM:
        case CPUVendor::APPLE_SILICON:
            for (auto e : get_arm_events()) {
                events.push_back(e);
            }
            break;
        default:
            break;
    }

    return events;
}

std::vector<CounterEvent> MetricsAnalyzer::get_intel_events() const {
    return {
        CounterEvent::FP_ARITH_SCALAR_SINGLE,
        CounterEvent::FP_ARITH_SCALAR_DOUBLE,
        CounterEvent::FP_ARITH_128B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_128B_PACKED_DOUBLE,
        CounterEvent::FP_ARITH_256B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_256B_PACKED_DOUBLE,
        CounterEvent::FP_ARITH_512B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_512B_PACKED_DOUBLE,
        CounterEvent::IDQ_DSB_UOPS,
        CounterEvent::IDQ_MITE_UOPS,
        CounterEvent::IDQ_MS_UOPS,
        CounterEvent::UOPS_DISPATCHED_PORT_0,
        CounterEvent::UOPS_DISPATCHED_PORT_1,
        CounterEvent::UOPS_DISPATCHED_PORT_5,
        CounterEvent::MEM_INST_RETIRED_SPLIT_LOADS,
        CounterEvent::MEM_INST_RETIRED_SPLIT_STORES,
        CounterEvent::CORE_POWER_LVL0_TURBO_LICENSE,
        CounterEvent::CORE_POWER_LVL1_TURBO_LICENSE,
        CounterEvent::CORE_POWER_LVL2_TURBO_LICENSE,
    };
}

std::vector<CounterEvent> MetricsAnalyzer::get_amd_events() const {
    return {
        CounterEvent::AMD_FP_RET_SSE_AVX_OPS,
        CounterEvent::FP_ARITH_SCALAR_SINGLE,
        CounterEvent::FP_ARITH_SCALAR_DOUBLE,
        CounterEvent::FP_ARITH_128B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_128B_PACKED_DOUBLE,
        CounterEvent::FP_ARITH_256B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_256B_PACKED_DOUBLE,
        CounterEvent::FP_ARITH_512B_PACKED_SINGLE,
        CounterEvent::FP_ARITH_512B_PACKED_DOUBLE,
    };
}

std::vector<CounterEvent> MetricsAnalyzer::get_arm_events() const {
    return {
        CounterEvent::ARM_FP_SPEC,
        CounterEvent::ARM_ASE_SPEC,
        CounterEvent::ARM_SVE_INST_SPEC,
        CounterEvent::ARM_SVE_PRED_PARTIAL,
        CounterEvent::ARM_SVE_PRED_FULL,
        CounterEvent::ARM_SVE_PRED_EMPTY,
    };
}

double MetricsAnalyzer::score_ipc(double ipc) const {
    if (ipc >= QualityThresholds::IPC_GOOD) return 100.0;
    if (ipc >= QualityThresholds::IPC_HEALTHY) return 70.0 + 30.0 * (ipc - QualityThresholds::IPC_HEALTHY) /
                                                            (QualityThresholds::IPC_GOOD - QualityThresholds::IPC_HEALTHY);
    return std::max(0.0, 70.0 * ipc / QualityThresholds::IPC_HEALTHY);
}

double MetricsAnalyzer::score_vectorization(double ratio) const {
    if (ratio >= QualityThresholds::VECTORIZATION_GOOD) return 100.0;
    if (ratio >= QualityThresholds::VECTORIZATION_ACCEPTABLE) return 70.0 + 30.0 * (ratio - QualityThresholds::VECTORIZATION_ACCEPTABLE) /
                                                                           (QualityThresholds::VECTORIZATION_GOOD - QualityThresholds::VECTORIZATION_ACCEPTABLE);
    return std::max(0.0, 70.0 * ratio / QualityThresholds::VECTORIZATION_ACCEPTABLE);
}

double MetricsAnalyzer::score_cache_efficiency(const MPKIMetrics& mpki) const {
    double score = 100.0;

    if (!mpki.l1_acceptable) {
        score -= 20.0 * std::min(1.0, mpki.l1_mpki / MPKIThresholds::L1_CRITICAL);
    }
    if (!mpki.l2_acceptable) {
        score -= 20.0 * std::min(1.0, mpki.l2_mpki / MPKIThresholds::L2_CRITICAL);
    }
    if (!mpki.l3_acceptable) {
        score -= 20.0 * std::min(1.0, mpki.l3_mpki / MPKIThresholds::L3_CRITICAL);
    }

    return std::max(0.0, score);
}

double MetricsAnalyzer::score_dsb_efficiency(const DSBMetrics& dsb) const {
    if (dsb.dsb_coverage >= QualityThresholds::DSB_COVERAGE_GOOD) return 100.0;
    if (dsb.dsb_coverage >= QualityThresholds::DSB_COVERAGE_ACCEPTABLE) return 70.0;
    return std::max(0.0, 70.0 * dsb.dsb_coverage / QualityThresholds::DSB_COVERAGE_ACCEPTABLE);
}

double MetricsAnalyzer::score_port_balance(const PortMetrics& ports) const {
    double score = 100.0;

    if (ports.port5_saturated) {
        score -= 30.0;
    }
    if (!ports.fma_ports_balanced) {
        score -= 20.0;
    }

    return std::max(0.0, score);
}

// IMC Bandwidth Analyzer implementation
struct IMCBandwidthAnalyzer::Impl {
    bool initialized = false;
    uint64_t start_reads = 0;
    uint64_t start_writes = 0;
    std::chrono::steady_clock::time_point start_time;
    double theoretical_max_gbps = 0.0;
};

IMCBandwidthAnalyzer::IMCBandwidthAnalyzer() : impl_(std::make_unique<Impl>()) {}

bool IMCBandwidthAnalyzer::is_available() {
    // Check for Intel uncore PMU
    std::ifstream check("/sys/bus/event_source/devices/uncore_imc_0/type");
    return check.is_open();
}

bool IMCBandwidthAnalyzer::initialize() {
    if (!is_available()) {
        return false;
    }
    impl_->initialized = true;

    // Estimate theoretical max bandwidth (DDR4 ~25GB/s per channel, DDR5 ~50GB/s)
    impl_->theoretical_max_gbps = 50.0;  // Conservative estimate

    return true;
}

void IMCBandwidthAnalyzer::shutdown() {
    impl_->initialized = false;
}

bool IMCBandwidthAnalyzer::start() {
    if (!impl_->initialized) return false;
    impl_->start_time = std::chrono::steady_clock::now();
    // In a full implementation, we would read uncore PMU here
    return true;
}

bool IMCBandwidthAnalyzer::stop() {
    return impl_->initialized;
}

IMCBandwidthMetrics IMCBandwidthAnalyzer::get_metrics() const {
    IMCBandwidthMetrics metrics;

    if (!impl_->initialized) {
        return metrics;
    }

    auto elapsed = std::chrono::steady_clock::now() - impl_->start_time;
    double seconds = std::chrono::duration<double>(elapsed).count();

    // In a full implementation, we would read uncore PMU counters here
    // and calculate actual bandwidth

    metrics.bandwidth_utilization = 0.0;  // Would be calculated from actual data

    return metrics;
}

// Utility functions
uint64_t calculate_total_flops(const CounterValues& values, CPUVendor vendor) {
    uint64_t total = 0;

    if (vendor == CPUVendor::INTEL || vendor == CPUVendor::AMD) {
        // Scalar ops = 1 FLOP each
        total += values.get(CounterEvent::FP_ARITH_SCALAR_SINGLE);
        total += values.get(CounterEvent::FP_ARITH_SCALAR_DOUBLE);

        // 128-bit = 4 SP or 2 DP FLOPs
        total += values.get(CounterEvent::FP_ARITH_128B_PACKED_SINGLE) * 4;
        total += values.get(CounterEvent::FP_ARITH_128B_PACKED_DOUBLE) * 2;

        // 256-bit = 8 SP or 4 DP FLOPs
        total += values.get(CounterEvent::FP_ARITH_256B_PACKED_SINGLE) * 8;
        total += values.get(CounterEvent::FP_ARITH_256B_PACKED_DOUBLE) * 4;

        // 512-bit = 16 SP or 8 DP FLOPs
        total += values.get(CounterEvent::FP_ARITH_512B_PACKED_SINGLE) * 16;
        total += values.get(CounterEvent::FP_ARITH_512B_PACKED_DOUBLE) * 8;
    }

    return total;
}

std::string get_quality_rating(double score) {
    if (score >= 90.0) return "Excellent";
    if (score >= 75.0) return "Good";
    if (score >= 50.0) return "Acceptable";
    if (score >= 25.0) return "Poor";
    return "Critical";
}

}  // namespace simd_bench
