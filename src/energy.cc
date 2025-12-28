#include "simd_bench/energy.h"
#include <fstream>
#include <sstream>
#include <cstring>

namespace simd_bench {

// Factory implementation
std::unique_ptr<IEnergyMonitor> EnergyMonitorFactory::create(EnergyBackend backend) {
    switch (backend) {
#ifdef SIMD_BENCH_HAS_RAPL
        case EnergyBackend::RAPL:
        case EnergyBackend::POWERCAP:
            return std::make_unique<RAPLMonitor>();
#endif
        case EnergyBackend::NONE:
        default:
            return std::make_unique<NullEnergyMonitor>();
    }
}

std::unique_ptr<IEnergyMonitor> EnergyMonitorFactory::create_best_available() {
#ifdef SIMD_BENCH_HAS_RAPL
    if (is_rapl_available()) {
        auto monitor = std::make_unique<RAPLMonitor>();
        if (monitor->initialize()) {
            return monitor;
        }
    }
#endif
    return std::make_unique<NullEnergyMonitor>();
}

std::vector<EnergyBackend> EnergyMonitorFactory::get_available_backends() {
    std::vector<EnergyBackend> backends;
    backends.push_back(EnergyBackend::NONE);

#ifdef SIMD_BENCH_HAS_RAPL
    if (is_rapl_available()) {
        backends.push_back(EnergyBackend::RAPL);
        backends.push_back(EnergyBackend::POWERCAP);
    }
#endif

    return backends;
}

bool EnergyMonitorFactory::is_backend_available(EnergyBackend backend) {
    switch (backend) {
        case EnergyBackend::NONE:
            return true;
#ifdef SIMD_BENCH_HAS_RAPL
        case EnergyBackend::RAPL:
        case EnergyBackend::POWERCAP:
            return is_rapl_available();
#endif
        default:
            return false;
    }
}

// RAPL implementation with Intel and AMD support
#ifdef SIMD_BENCH_HAS_RAPL

enum class RAPLVendor {
    UNKNOWN,
    INTEL,
    AMD
};

static RAPLVendor detect_rapl_vendor() {
    // Check Intel RAPL first
    std::ifstream intel_check("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    if (intel_check.is_open()) {
        return RAPLVendor::INTEL;
    }

    // Check AMD RAPL (different path on AMD systems)
    std::ifstream amd_check("/sys/class/powercap/amd-rapl/amd-rapl:0/energy_uj");
    if (amd_check.is_open()) {
        return RAPLVendor::AMD;
    }

    // Some AMD systems use intel-rapl naming
    std::ifstream amd_alt("/sys/class/hwmon");
    if (amd_alt.is_open()) {
        // Check hwmon interface for AMD
        for (int i = 0; i < 10; ++i) {
            std::string path = "/sys/class/hwmon/hwmon" + std::to_string(i) + "/name";
            std::ifstream name_file(path);
            if (name_file.is_open()) {
                std::string name;
                name_file >> name;
                if (name.find("amd") != std::string::npos || name.find("k10temp") != std::string::npos) {
                    return RAPLVendor::AMD;
                }
            }
        }
    }

    return RAPLVendor::UNKNOWN;
}

struct RAPLMonitor::Impl {
    bool initialized = false;
    EnergySample start_sample;
    std::chrono::steady_clock::time_point start_time;
    bool running = false;
    RAPLVendor vendor = RAPLVendor::UNKNOWN;

    // Intel RAPL paths
    std::string intel_package_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
    std::string intel_cores_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
    std::string intel_dram_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:2/energy_uj";

    // AMD RAPL paths (varies by kernel version)
    std::string amd_package_path = "/sys/class/powercap/amd-rapl/amd-rapl:0/energy_uj";
    std::string amd_cores_path = "/sys/class/powercap/amd-rapl/amd-rapl:0/amd-rapl:0:0/energy_uj";

    // Active paths (set during initialization)
    std::string package_energy_path;
    std::string cores_energy_path;
    std::string dram_energy_path;

    bool package_available = false;
    bool cores_available = false;
    bool dram_available = false;

    double energy_unit = 1e-6;  // microjoules to joules
    double max_energy_uj = 0;
};

RAPLMonitor::RAPLMonitor() : impl_(std::make_unique<Impl>()) {}
RAPLMonitor::~RAPLMonitor() { shutdown(); }

bool RAPLMonitor::initialize() {
    // Detect RAPL vendor
    impl_->vendor = detect_rapl_vendor();

    // Set paths based on vendor
    if (impl_->vendor == RAPLVendor::INTEL) {
        impl_->package_energy_path = impl_->intel_package_path;
        impl_->cores_energy_path = impl_->intel_cores_path;
        impl_->dram_energy_path = impl_->intel_dram_path;
    } else if (impl_->vendor == RAPLVendor::AMD) {
        impl_->package_energy_path = impl_->amd_package_path;
        impl_->cores_energy_path = impl_->amd_cores_path;
        impl_->dram_energy_path = "";  // AMD doesn't expose DRAM via RAPL typically
    } else {
        // Try Intel paths as fallback (some AMD systems use them)
        impl_->package_energy_path = impl_->intel_package_path;
        impl_->cores_energy_path = impl_->intel_cores_path;
        impl_->dram_energy_path = impl_->intel_dram_path;
    }

    // Check which domains are available
    std::ifstream pkg_file(impl_->package_energy_path);
    impl_->package_available = pkg_file.is_open();

    std::ifstream cores_file(impl_->cores_energy_path);
    impl_->cores_available = cores_file.is_open();

    if (!impl_->dram_energy_path.empty()) {
        std::ifstream dram_file(impl_->dram_energy_path);
        impl_->dram_available = dram_file.is_open();
    }

    if (!impl_->package_available) {
        return false;
    }

    // Read max energy range
    std::string max_range_path;
    if (impl_->vendor == RAPLVendor::INTEL) {
        max_range_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj";
    } else if (impl_->vendor == RAPLVendor::AMD) {
        max_range_path = "/sys/class/powercap/amd-rapl/amd-rapl:0/max_energy_range_uj";
    } else {
        max_range_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj";
    }

    std::ifstream max_file(max_range_path);
    if (max_file.is_open()) {
        max_file >> impl_->max_energy_uj;
    }

    impl_->initialized = true;
    return true;
}

void RAPLMonitor::shutdown() {
    impl_->initialized = false;
    impl_->running = false;
}

static double read_energy_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0.0;
    }

    uint64_t energy_uj;
    file >> energy_uj;
    return static_cast<double>(energy_uj) * 1e-6;  // Convert to joules
}

EnergySample RAPLMonitor::sample() {
    EnergySample sample;
    sample.timestamp = std::chrono::steady_clock::now();

    if (impl_->package_available) {
        sample.package_joules = read_energy_file(impl_->package_energy_path);
    }
    if (impl_->cores_available) {
        sample.cores_joules = read_energy_file(impl_->cores_energy_path);
    }
    if (impl_->dram_available) {
        sample.dram_joules = read_energy_file(impl_->dram_energy_path);
    }

    return sample;
}

bool RAPLMonitor::start() {
    impl_->start_sample = sample();
    impl_->start_time = std::chrono::steady_clock::now();
    impl_->running = true;
    return true;
}

bool RAPLMonitor::stop() {
    impl_->running = false;
    return true;
}

EnergyMetrics RAPLMonitor::get_metrics() {
    EnergyMetrics metrics;

    EnergySample end_sample = sample();
    auto end_time = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(end_time - impl_->start_time).count();

    // Handle wraparound
    double pkg_energy = end_sample.package_joules - impl_->start_sample.package_joules;
    if (pkg_energy < 0 && impl_->max_energy_uj > 0) {
        pkg_energy += impl_->max_energy_uj * 1e-6;
    }

    double cores_energy = end_sample.cores_joules - impl_->start_sample.cores_joules;
    if (cores_energy < 0 && impl_->max_energy_uj > 0) {
        cores_energy += impl_->max_energy_uj * 1e-6;
    }

    double dram_energy = end_sample.dram_joules - impl_->start_sample.dram_joules;
    if (dram_energy < 0 && impl_->max_energy_uj > 0) {
        dram_energy += impl_->max_energy_uj * 1e-6;
    }

    metrics.energy_joules = pkg_energy;

    if (elapsed > 0) {
        metrics.package_power_watts = pkg_energy / elapsed;
        metrics.core_power_watts = cores_energy / elapsed;
        metrics.dram_power_watts = dram_energy / elapsed;
    }

    return metrics;
}

bool RAPLMonitor::is_domain_available(RAPLDomain domain) const {
    switch (domain) {
        case RAPLDomain::PACKAGE:
            return impl_->package_available;
        case RAPLDomain::CORES:
            return impl_->cores_available;
        case RAPLDomain::DRAM:
            return impl_->dram_available;
        default:
            return false;
    }
}

double RAPLMonitor::get_max_energy_joules(RAPLDomain) const {
    return impl_->max_energy_uj * 1e-6;
}

double RAPLMonitor::get_energy_unit() const {
    return impl_->energy_unit;
}
#endif

// ScopedEnergyMeasurement implementation
ScopedEnergyMeasurement::ScopedEnergyMeasurement(IEnergyMonitor& monitor, EnergyMetrics& result)
    : monitor_(monitor), result_(result) {
    start_sample_ = monitor_.sample();
    start_time_ = std::chrono::steady_clock::now();
    monitor_.start();
}

ScopedEnergyMeasurement::~ScopedEnergyMeasurement() {
    monitor_.stop();
    result_ = monitor_.get_metrics();
}

// EnergyEfficiencyAnalyzer implementation
double EnergyEfficiencyAnalyzer::calculate_energy_per_flop_nj(
    double energy_joules,
    uint64_t flops
) {
    if (flops == 0) return 0.0;
    return (energy_joules * 1e9) / static_cast<double>(flops);
}

double EnergyEfficiencyAnalyzer::calculate_edp(
    double energy_joules,
    double elapsed_seconds
) {
    return energy_joules * elapsed_seconds;
}

double EnergyEfficiencyAnalyzer::calculate_ed2p(
    double energy_joules,
    double elapsed_seconds
) {
    return energy_joules * elapsed_seconds * elapsed_seconds;
}

double EnergyEfficiencyAnalyzer::calculate_power_watts(
    double energy_joules,
    double elapsed_seconds
) {
    if (elapsed_seconds <= 0) return 0.0;
    return energy_joules / elapsed_seconds;
}

EnergyEfficiencyAnalyzer::EfficiencyComparison
EnergyEfficiencyAnalyzer::compare_efficiency(
    const EnergyMetrics& scalar_metrics,
    uint64_t scalar_ops,
    double scalar_time,
    const EnergyMetrics& simd_metrics,
    uint64_t simd_ops,
    double simd_time
) {
    EfficiencyComparison comparison;

    comparison.scalar_energy_per_op = calculate_energy_per_flop_nj(
        scalar_metrics.energy_joules, scalar_ops);
    comparison.simd_energy_per_op = calculate_energy_per_flop_nj(
        simd_metrics.energy_joules, simd_ops);

    if (comparison.scalar_energy_per_op > 0) {
        comparison.energy_savings_percent =
            (comparison.scalar_energy_per_op - comparison.simd_energy_per_op) /
            comparison.scalar_energy_per_op * 100.0;
    }

    comparison.scalar_edp = calculate_edp(scalar_metrics.energy_joules, scalar_time);
    comparison.simd_edp = calculate_edp(simd_metrics.energy_joules, simd_time);

    if (comparison.scalar_edp > 0) {
        comparison.edp_improvement_percent =
            (comparison.scalar_edp - comparison.simd_edp) /
            comparison.scalar_edp * 100.0;
    }

    return comparison;
}

// Utility functions
EnergyMetrics measure_energy(
    const std::function<void()>& func,
    uint64_t total_flops
) {
    auto monitor = EnergyMonitorFactory::create_best_available();
    if (!monitor->initialize()) {
        return EnergyMetrics{};
    }

    monitor->start();
    func();
    monitor->stop();

    EnergyMetrics metrics = monitor->get_metrics();
    metrics.energy_per_op_nj = EnergyEfficiencyAnalyzer::calculate_energy_per_flop_nj(
        metrics.energy_joules, total_flops);

    return metrics;
}

bool is_rapl_available() {
#ifdef __linux__
    std::ifstream file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    return file.is_open();
#else
    return false;
#endif
}

double read_rapl_energy_joules(RAPLDomain domain) {
#ifdef __linux__
    std::string path;
    switch (domain) {
        case RAPLDomain::PACKAGE:
            path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
            break;
        case RAPLDomain::CORES:
            path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";
            break;
        case RAPLDomain::DRAM:
            path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:2/energy_uj";
            break;
        default:
            return 0.0;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return 0.0;
    }

    uint64_t energy_uj;
    file >> energy_uj;
    return static_cast<double>(energy_uj) * 1e-6;
#else
    (void)domain;
    return 0.0;
#endif
}

}  // namespace simd_bench
