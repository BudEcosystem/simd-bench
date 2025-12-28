#include "simd_bench/tma.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace simd_bench {

// AMD-specific counter events for Pipeline Utilization
// These are based on AMD Zen 4+ PMU documentation
enum class AMDEvent {
    // AMD Pipeline Utilization events (approximate equivalents)
    EX_RET_INSTR,           // Retired Instructions
    EX_RET_UOPS,            // Retired micro-ops
    DE_DIS_UOPS_FROM_DECODER,// uops from decoder
    DE_NO_DISPATCH_CYCLES,  // Cycles with no dispatch
    EX_RET_BRN_MISP,        // Mispredicted branches
    LS_DC_ACCESSES,         // Data cache accesses
    LS_L2_MISS_DC,          // L2 misses from DC
    IC_FETCH_STALL,         // Instruction fetch stalls
};

TMAAnalyzer::TMAAnalyzer() {}

TMAAnalyzer::TMAAnalyzer(IPerformanceCounters* counters)
    : counters_(counters) {}

void TMAAnalyzer::set_counters(IPerformanceCounters* counters) {
    counters_ = counters;
}

std::vector<CounterEvent> TMAAnalyzer::get_required_events() const {
    std::vector<CounterEvent> events;

    // Level 1 events (basic)
    events.push_back(CounterEvent::CYCLES);
    events.push_back(CounterEvent::INSTRUCTIONS);
    events.push_back(CounterEvent::UOPS_RETIRED_SLOTS);
    events.push_back(CounterEvent::UOPS_ISSUED_ANY);

    if (level_ >= TMALevel::LEVEL2) {
        events.push_back(CounterEvent::INT_MISC_RECOVERY_CYCLES);
        events.push_back(CounterEvent::CYCLE_ACTIVITY_STALLS_MEM);
    }

    if (level_ >= TMALevel::LEVEL3) {
        events.push_back(CounterEvent::CYCLE_ACTIVITY_STALLS_L1D);
        events.push_back(CounterEvent::CYCLE_ACTIVITY_STALLS_L2);
        events.push_back(CounterEvent::CYCLE_ACTIVITY_STALLS_L3);
    }

    return events;
}

TMAResult TMAAnalyzer::analyze(const CounterValues& values) const {
    TMAResult result;

    // Select appropriate metrics calculation based on vendor
    TMAVendor effective_vendor = vendor_;
    if (effective_vendor == TMAVendor::AUTO) {
        effective_vendor = detect_vendor();
    }

    if (effective_vendor == TMAVendor::AMD) {
        result.metrics = calculate_amd_metrics(values);
    } else {
        result.metrics = calculate_metrics(values);
    }

    // Classify each category
    result.categories.push_back(classify_category(TMACategory::RETIRING, result.metrics));
    result.categories.push_back(classify_category(TMACategory::BAD_SPECULATION, result.metrics));
    result.categories.push_back(classify_category(TMACategory::FRONTEND_BOUND, result.metrics));
    result.categories.push_back(classify_category(TMACategory::BACKEND_BOUND, result.metrics));

    if (level_ >= TMALevel::LEVEL2) {
        result.categories.push_back(classify_category(TMACategory::MEMORY_BOUND, result.metrics));
        result.categories.push_back(classify_category(TMACategory::CORE_BOUND, result.metrics));
    }

    // Identify bottlenecks
    double max_ratio = 0;
    for (const auto& cat : result.categories) {
        if (cat.ratio > max_ratio && cat.category != TMACategory::RETIRING) {
            max_ratio = cat.ratio;
            result.primary_bottleneck = cat.name;
            result.bottleneck_ratio = cat.ratio;
        }
    }

    // Generate recommendations
    result.recommendations = generate_recommendations(result.metrics);

    return result;
}

TMAResult TMAAnalyzer::measure_and_analyze(
    const std::function<void()>& func,
    size_t iterations
) {
    if (!counters_) {
        return TMAResult{};
    }

    // Configure events
    counters_->clear_events();
    for (auto event : get_required_events()) {
        counters_->add_event(event);
    }

    counters_->start();

    for (size_t i = 0; i < iterations; ++i) {
        func();
    }

    counters_->stop();

    CounterValues values = counters_->read();
    return analyze(values);
}

TMAMetrics TMAAnalyzer::calculate_metrics(const CounterValues& values) const {
    TMAMetrics metrics;

    uint64_t cycles = values.get(CounterEvent::CYCLES);
    uint64_t uops_retired = values.get(CounterEvent::UOPS_RETIRED_SLOTS);
    uint64_t uops_issued = values.get(CounterEvent::UOPS_ISSUED_ANY);
    uint64_t recovery_cycles = values.get(CounterEvent::INT_MISC_RECOVERY_CYCLES);
    uint64_t mem_stalls = values.get(CounterEvent::CYCLE_ACTIVITY_STALLS_MEM);

    if (cycles == 0) {
        return metrics;
    }

    // Pipeline width assumption (4 for modern CPUs)
    const double pipeline_width = 4.0;
    double total_slots = cycles * pipeline_width;

    // Level 1 metrics
    metrics.retiring = static_cast<double>(uops_retired) / total_slots;
    metrics.bad_speculation = static_cast<double>(uops_issued - uops_retired + recovery_cycles * pipeline_width) / total_slots;

    // Clamp values to [0, 1]
    metrics.retiring = std::clamp(metrics.retiring, 0.0, 1.0);
    metrics.bad_speculation = std::clamp(metrics.bad_speculation, 0.0, 1.0);

    // Remaining is split between frontend and backend
    double remaining = 1.0 - metrics.retiring - metrics.bad_speculation;
    remaining = std::max(0.0, remaining);

    // Estimate memory bound from stall cycles
    double mem_bound_estimate = static_cast<double>(mem_stalls) / cycles;
    mem_bound_estimate = std::clamp(mem_bound_estimate, 0.0, remaining);

    metrics.backend_bound = remaining * 0.7;  // Estimate
    metrics.frontend_bound = remaining * 0.3;

    // Level 2 breakdown
    if (level_ >= TMALevel::LEVEL2) {
        metrics.memory_bound = metrics.backend_bound * 0.7;
        metrics.core_bound = metrics.backend_bound * 0.3;
    }

    // Level 3 breakdown
    if (level_ >= TMALevel::LEVEL3) {
        uint64_t l1_stalls = values.get(CounterEvent::CYCLE_ACTIVITY_STALLS_L1D);
        uint64_t l2_stalls = values.get(CounterEvent::CYCLE_ACTIVITY_STALLS_L2);
        uint64_t l3_stalls = values.get(CounterEvent::CYCLE_ACTIVITY_STALLS_L3);

        double total_mem_stalls = l1_stalls + l2_stalls + l3_stalls;
        if (total_mem_stalls > 0) {
            metrics.l1_bound = metrics.memory_bound * (l1_stalls / total_mem_stalls);
            metrics.l2_bound = metrics.memory_bound * (l2_stalls / total_mem_stalls);
            metrics.l3_bound = metrics.memory_bound * (l3_stalls / total_mem_stalls);
            metrics.dram_bound = metrics.memory_bound - metrics.l1_bound - metrics.l2_bound - metrics.l3_bound;
        }
    }

    return metrics;
}

TMACategoryResult TMAAnalyzer::classify_category(TMACategory cat, const TMAMetrics& metrics) const {
    TMACategoryResult result;
    result.category = cat;
    result.name = tma_category_to_string(cat);

    switch (cat) {
        case TMACategory::RETIRING:
            result.ratio = metrics.retiring;
            result.description = "Fraction of slots used for useful work";
            if (result.ratio > 0.7) {
                result.recommendations.push_back("Good utilization of pipeline slots");
            }
            break;

        case TMACategory::BAD_SPECULATION:
            result.ratio = metrics.bad_speculation;
            result.description = "Slots wasted due to incorrect speculation";
            if (result.ratio > 0.1) {
                result.recommendations.push_back("Consider improving branch prediction");
                result.recommendations.push_back("Use profile-guided optimization");
            }
            break;

        case TMACategory::FRONTEND_BOUND:
            result.ratio = metrics.frontend_bound;
            result.description = "Stalls due to instruction fetch/decode";
            if (result.ratio > 0.15) {
                result.recommendations.push_back("Code may benefit from better layout");
                result.recommendations.push_back("Consider inlining hot functions");
            }
            break;

        case TMACategory::BACKEND_BOUND:
            result.ratio = metrics.backend_bound;
            result.description = "Stalls due to execution resources";
            break;

        case TMACategory::MEMORY_BOUND:
            result.ratio = metrics.memory_bound;
            result.description = "Stalls waiting for memory";
            if (result.ratio > 0.2) {
                result.recommendations.push_back("Consider prefetching");
                result.recommendations.push_back("Improve data locality");
            }
            break;

        case TMACategory::CORE_BOUND:
            result.ratio = metrics.core_bound;
            result.description = "Stalls due to execution port contention";
            if (result.ratio > 0.15) {
                result.recommendations.push_back("Increase instruction-level parallelism");
            }
            break;

        default:
            result.ratio = 0.0;
            break;
    }

    return result;
}

std::vector<std::string> TMAAnalyzer::generate_recommendations(const TMAMetrics& metrics) const {
    std::vector<std::string> recommendations;

    if (metrics.retiring < 0.5) {
        recommendations.push_back("Low retiring ratio indicates significant pipeline inefficiency");
    }

    if (metrics.backend_bound > 0.3) {
        recommendations.push_back("Backend-bound: focus on execution optimizations");

        if (metrics.memory_bound > 0.2) {
            recommendations.push_back("Memory-bound: improve data access patterns");

            if (metrics.l1_bound > 0.1) {
                recommendations.push_back("L1-bound: increase cache line utilization");
            }
            if (metrics.dram_bound > 0.1) {
                recommendations.push_back("DRAM-bound: consider streaming stores or prefetching");
            }
        }

        if (metrics.core_bound > 0.15) {
            recommendations.push_back("Core-bound: increase instruction-level parallelism");
        }
    }

    if (metrics.bad_speculation > 0.1) {
        recommendations.push_back("High speculation overhead: improve branch prediction");
    }

    if (metrics.frontend_bound > 0.2) {
        recommendations.push_back("Frontend-bound: consider code layout optimization");
    }

    return recommendations;
}

bool TMAAnalyzer::is_supported() {
    // TMA is primarily supported on Intel CPUs
#if defined(__x86_64__)
    return true;  // May not work on all CPUs, but worth trying
#else
    return false;
#endif
}

bool TMAAnalyzer::is_amd_supported() {
    // AMD Pipeline Utilization is supported on Zen 4+
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "AuthenticAMD") == 0) {
        // Check for Zen 4+ (family 0x19 or higher)
        __cpuid(1, eax, ebx, ecx, edx);
        uint32_t family = ((eax >> 8) & 0xF);
        if (family == 0xF) {
            family += ((eax >> 20) & 0xFF);
        }
        return family >= 0x19;  // Zen 3+ family, Zen 4 is better supported
    }
#endif
    return false;
}

TMAVendor TMAAnalyzer::detect_vendor() const {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0) return TMAVendor::INTEL;
    if (strcmp(vendor, "AuthenticAMD") == 0) return TMAVendor::AMD;
#endif
    return TMAVendor::INTEL;  // Default to Intel methodology
}

std::vector<CounterEvent> TMAAnalyzer::get_amd_required_events() const {
    std::vector<CounterEvent> events;

    // Basic events for AMD Pipeline Utilization
    events.push_back(CounterEvent::CYCLES);
    events.push_back(CounterEvent::INSTRUCTIONS);
    events.push_back(CounterEvent::UOPS_RETIRED_SLOTS);  // Maps to EX_RET_UOPS
    events.push_back(CounterEvent::BRANCH_MISSES);       // Maps to EX_RET_BRN_MISP

    if (level_ >= TMALevel::LEVEL2) {
        events.push_back(CounterEvent::L1D_READ_MISS);
        events.push_back(CounterEvent::L2_READ_MISS);
    }

    if (level_ >= TMALevel::LEVEL3) {
        events.push_back(CounterEvent::L3_READ_MISS);
        events.push_back(CounterEvent::AMD_FP_RET_SSE_AVX_OPS);
    }

    return events;
}

TMAMetrics TMAAnalyzer::calculate_amd_metrics(const CounterValues& values) const {
    TMAMetrics metrics;

    uint64_t cycles = values.get(CounterEvent::CYCLES);
    uint64_t instructions = values.get(CounterEvent::INSTRUCTIONS);
    uint64_t uops_retired = values.get(CounterEvent::UOPS_RETIRED_SLOTS);
    uint64_t branch_misses = values.get(CounterEvent::BRANCH_MISSES);

    if (cycles == 0) {
        return metrics;
    }

    // AMD Zen has a 4-wide pipeline (dispatch width)
    const double pipeline_width = 4.0;
    double total_slots = cycles * pipeline_width;

    // Retiring: fraction of slots used by retired uops
    metrics.retiring = static_cast<double>(uops_retired) / total_slots;
    metrics.retiring = std::clamp(metrics.retiring, 0.0, 1.0);

    // Bad Speculation: estimated from branch mispredictions
    // Each mispredict causes ~15-20 cycle penalty on Zen
    const double avg_mispredict_penalty = 18.0;
    double mispredict_slots = branch_misses * avg_mispredict_penalty * pipeline_width;
    metrics.bad_speculation = mispredict_slots / total_slots;
    metrics.bad_speculation = std::clamp(metrics.bad_speculation, 0.0, 0.5);

    // Remaining is split between frontend and backend
    double remaining = 1.0 - metrics.retiring - metrics.bad_speculation;
    remaining = std::max(0.0, remaining);

    // AMD doesn't expose exact frontend/backend split like Intel
    // Estimate based on IPC (low IPC often indicates backend bound)
    double ipc = static_cast<double>(instructions) / cycles;
    double backend_factor = ipc < 2.0 ? 0.75 : (ipc < 3.0 ? 0.5 : 0.3);

    metrics.backend_bound = remaining * backend_factor;
    metrics.frontend_bound = remaining * (1.0 - backend_factor);

    // Level 2 breakdown
    if (level_ >= TMALevel::LEVEL2) {
        uint64_t l1_misses = values.get(CounterEvent::L1D_READ_MISS);
        uint64_t l2_misses = values.get(CounterEvent::L2_READ_MISS);

        // Memory bound estimate from cache misses
        double miss_penalty_cycles = l1_misses * 4 + l2_misses * 12;  // Approximate
        double mem_bound_ratio = miss_penalty_cycles / cycles;
        mem_bound_ratio = std::clamp(mem_bound_ratio, 0.0, metrics.backend_bound);

        metrics.memory_bound = mem_bound_ratio;
        metrics.core_bound = metrics.backend_bound - metrics.memory_bound;
        metrics.core_bound = std::max(0.0, metrics.core_bound);
    }

    // Level 3 breakdown
    if (level_ >= TMALevel::LEVEL3) {
        uint64_t l3_misses = values.get(CounterEvent::L3_READ_MISS);
        uint64_t l1_misses = values.get(CounterEvent::L1D_READ_MISS);
        uint64_t l2_misses = values.get(CounterEvent::L2_READ_MISS);

        // Estimate cache level contributions
        double total_weighted_misses = l1_misses * 4.0 + l2_misses * 12.0 + l3_misses * 50.0;
        if (total_weighted_misses > 0 && metrics.memory_bound > 0) {
            metrics.l1_bound = metrics.memory_bound * (l1_misses * 4.0) / total_weighted_misses;
            metrics.l2_bound = metrics.memory_bound * (l2_misses * 12.0) / total_weighted_misses;
            metrics.l3_bound = metrics.memory_bound * (l3_misses * 50.0) / total_weighted_misses;
            metrics.dram_bound = metrics.memory_bound - metrics.l1_bound - metrics.l2_bound - metrics.l3_bound;
            metrics.dram_bound = std::max(0.0, metrics.dram_bound);
        }
    }

    return metrics;
}

// String formatting
std::string tma_category_to_string(TMACategory category) {
    switch (category) {
        case TMACategory::RETIRING: return "Retiring";
        case TMACategory::BAD_SPECULATION: return "Bad Speculation";
        case TMACategory::FRONTEND_BOUND: return "Frontend Bound";
        case TMACategory::BACKEND_BOUND: return "Backend Bound";
        case TMACategory::RETIRING_BASE: return "Base";
        case TMACategory::RETIRING_VECTORIZED: return "Vectorized";
        case TMACategory::BRANCH_MISPREDICTS: return "Branch Mispredicts";
        case TMACategory::MACHINE_CLEARS: return "Machine Clears";
        case TMACategory::FETCH_LATENCY: return "Fetch Latency";
        case TMACategory::FETCH_BANDWIDTH: return "Fetch Bandwidth";
        case TMACategory::CORE_BOUND: return "Core Bound";
        case TMACategory::MEMORY_BOUND: return "Memory Bound";
        case TMACategory::DIVIDER: return "Divider";
        case TMACategory::PORTS_UTILIZATION: return "Ports Utilization";
        case TMACategory::L1_BOUND: return "L1 Bound";
        case TMACategory::L2_BOUND: return "L2 Bound";
        case TMACategory::L3_BOUND: return "L3 Bound";
        case TMACategory::DRAM_BOUND: return "DRAM Bound";
        default: return "Unknown";
    }
}

std::string format_tma_bar_chart(const TMAResult& result, int width) {
    std::ostringstream oss;

    // Helper to repeat a string n times
    auto repeat = [](const std::string& s, int n) {
        std::string result;
        result.reserve(s.size() * n);
        for (int i = 0; i < n; ++i) {
            result += s;
        }
        return result;
    };

    oss << "+" << repeat("-", width + 20) << "+\n";
    oss << "|" << std::setw(width + 20) << std::left << "  Pipeline Slot Breakdown" << "|\n";
    oss << "+" << repeat("-", width + 20) << "+\n";

    auto format_bar = [width, &repeat](const std::string& name, double ratio) {
        std::ostringstream bar;
        int filled = static_cast<int>(ratio * width);
        filled = std::clamp(filled, 0, width);

        bar << "| " << std::setw(18) << std::left << name << " ";
        bar << repeat("#", filled);
        bar << repeat(".", width - filled);
        bar << " " << std::fixed << std::setprecision(1) << (ratio * 100) << "% |\n";

        return bar.str();
    };

    oss << format_bar("Retiring", result.metrics.retiring);
    oss << format_bar("Bad Speculation", result.metrics.bad_speculation);
    oss << format_bar("Frontend Bound", result.metrics.frontend_bound);
    oss << format_bar("Backend Bound", result.metrics.backend_bound);

    if (result.metrics.memory_bound > 0 || result.metrics.core_bound > 0) {
        oss << "|" << repeat("-", width + 20) << "|\n";
        oss << format_bar("  -> Memory Bound", result.metrics.memory_bound);
        oss << format_bar("  -> Core Bound", result.metrics.core_bound);
    }

    oss << "+" << repeat("-", width + 20) << "+\n";

    if (!result.primary_bottleneck.empty()) {
        oss << "\nPrimary bottleneck: " << result.primary_bottleneck
            << " (" << std::fixed << std::setprecision(1)
            << (result.bottleneck_ratio * 100) << "%)\n";
    }

    return oss.str();
}

}  // namespace simd_bench
