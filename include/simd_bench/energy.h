#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>

namespace simd_bench {

// Energy monitoring backend types
enum class EnergyBackend {
    NONE,
    RAPL,       // Intel/AMD RAPL (Running Average Power Limit)
    POWERCAP,   // Linux powercap interface
    PERF_EVENT  // perf event interface for RAPL
};

// RAPL domain types
enum class RAPLDomain {
    PACKAGE,    // Entire CPU package
    CORES,      // CPU cores only
    UNCORE,     // Uncore (L3 cache, memory controller)
    DRAM,       // DRAM (if available)
    PSYS        // Platform (if available)
};

// Energy sample at a point in time
struct EnergySample {
    std::chrono::steady_clock::time_point timestamp;
    double package_joules = 0.0;
    double cores_joules = 0.0;
    double dram_joules = 0.0;
    double psys_joules = 0.0;
};

// Abstract energy monitoring interface
class IEnergyMonitor {
public:
    virtual ~IEnergyMonitor() = default;

    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    virtual EnergySample sample() = 0;

    virtual bool start() = 0;
    virtual bool stop() = 0;

    virtual EnergyMetrics get_metrics() = 0;

    virtual EnergyBackend get_backend() const = 0;
    virtual std::string get_backend_name() const = 0;

    // Check if specific domain is available
    virtual bool is_domain_available(RAPLDomain domain) const = 0;
};

// Factory for energy monitors
class EnergyMonitorFactory {
public:
    static std::unique_ptr<IEnergyMonitor> create(EnergyBackend backend = EnergyBackend::NONE);
    static std::unique_ptr<IEnergyMonitor> create_best_available();
    static std::vector<EnergyBackend> get_available_backends();
    static bool is_backend_available(EnergyBackend backend);
};

// RAPL implementation using powercap sysfs interface
#ifdef SIMD_BENCH_HAS_RAPL
class RAPLMonitor : public IEnergyMonitor {
public:
    RAPLMonitor();
    ~RAPLMonitor() override;

    bool initialize() override;
    void shutdown() override;

    EnergySample sample() override;

    bool start() override;
    bool stop() override;

    EnergyMetrics get_metrics() override;

    EnergyBackend get_backend() const override { return EnergyBackend::RAPL; }
    std::string get_backend_name() const override { return "RAPL"; }

    bool is_domain_available(RAPLDomain domain) const override;

    // Get max energy range before wraparound
    double get_max_energy_joules(RAPLDomain domain) const;

    // Get energy unit (joules per increment)
    double get_energy_unit() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

// Null implementation
class NullEnergyMonitor : public IEnergyMonitor {
public:
    bool initialize() override { return true; }
    void shutdown() override {}
    EnergySample sample() override { return EnergySample{}; }
    bool start() override { return true; }
    bool stop() override { return true; }
    EnergyMetrics get_metrics() override { return EnergyMetrics{}; }
    EnergyBackend get_backend() const override { return EnergyBackend::NONE; }
    std::string get_backend_name() const override { return "none"; }
    bool is_domain_available(RAPLDomain) const override { return false; }
};

// Scoped energy measurement
class ScopedEnergyMeasurement {
public:
    ScopedEnergyMeasurement(IEnergyMonitor& monitor, EnergyMetrics& result);
    ~ScopedEnergyMeasurement();

    ScopedEnergyMeasurement(const ScopedEnergyMeasurement&) = delete;
    ScopedEnergyMeasurement& operator=(const ScopedEnergyMeasurement&) = delete;

private:
    IEnergyMonitor& monitor_;
    EnergyMetrics& result_;
    EnergySample start_sample_;
    std::chrono::steady_clock::time_point start_time_;
};

// Energy efficiency metrics calculator
class EnergyEfficiencyAnalyzer {
public:
    // Calculate energy per operation
    static double calculate_energy_per_flop_nj(
        double energy_joules,
        uint64_t flops
    );

    // Calculate energy-delay product
    static double calculate_edp(
        double energy_joules,
        double elapsed_seconds
    );

    // Calculate energy-delay^2 product (for power-constrained scenarios)
    static double calculate_ed2p(
        double energy_joules,
        double elapsed_seconds
    );

    // Calculate average power
    static double calculate_power_watts(
        double energy_joules,
        double elapsed_seconds
    );

    // Compare scalar vs SIMD energy efficiency
    struct EfficiencyComparison {
        double scalar_energy_per_op;
        double simd_energy_per_op;
        double energy_savings_percent;
        double scalar_edp;
        double simd_edp;
        double edp_improvement_percent;
    };

    static EfficiencyComparison compare_efficiency(
        const EnergyMetrics& scalar_metrics,
        uint64_t scalar_ops,
        double scalar_time,
        const EnergyMetrics& simd_metrics,
        uint64_t simd_ops,
        double simd_time
    );
};

// Measure energy consumption of a function
EnergyMetrics measure_energy(
    const std::function<void()>& func,
    uint64_t total_flops
);

// Check if RAPL is available on this system
bool is_rapl_available();

// Read RAPL energy directly (for debugging)
double read_rapl_energy_joules(RAPLDomain domain);

}  // namespace simd_bench
