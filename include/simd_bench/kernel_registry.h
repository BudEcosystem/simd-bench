#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>
#include <shared_mutex>

namespace simd_bench {

// Kernel registration and management

class KernelRegistry {
public:
    // Singleton access (thread-safe via C++11 static initialization)
    static KernelRegistry& instance();

    // Register a kernel (thread-safe)
    void register_kernel(const KernelConfig& config);

    // Get a kernel by name (thread-safe)
    const KernelConfig* get_kernel(const std::string& name) const;

    // Get all registered kernels (thread-safe)
    std::vector<std::string> get_kernel_names() const;

    // Get kernels by category (thread-safe)
    std::vector<const KernelConfig*> get_kernels_by_category(const std::string& category) const;

    // Check if kernel exists (thread-safe)
    bool has_kernel(const std::string& name) const;

    // Get number of registered kernels (thread-safe)
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return kernels_.size();
    }

    // Clear all registered kernels (for testing, thread-safe)
    void clear();

private:
    KernelRegistry() = default;
    mutable std::shared_mutex mutex_;  // Reader-writer lock for thread safety
    std::unordered_map<std::string, KernelConfig> kernels_;
};

// Macro for registering kernels
#define SIMD_BENCH_REGISTER_KERNEL(name, ...) \
    namespace { \
        struct KernelRegistrar_##name { \
            KernelRegistrar_##name() { \
                simd_bench::KernelConfig config = __VA_ARGS__; \
                config.name = #name; \
                simd_bench::KernelRegistry::instance().register_kernel(config); \
            } \
        } kernel_registrar_##name; \
    }

// Builder pattern for kernel configuration
class KernelBuilder {
public:
    KernelBuilder(const std::string& name);

    KernelBuilder& description(const std::string& desc);
    KernelBuilder& category(const std::string& cat);
    KernelBuilder& arithmetic_intensity(double ai);
    KernelBuilder& flops_per_element(size_t flops);
    KernelBuilder& bytes_per_element(size_t bytes);

    KernelBuilder& add_variant(const std::string& name,
                               KernelFunction func,
                               const std::string& isa = "scalar",
                               bool is_reference = false);

    KernelBuilder& sizes(std::vector<size_t> sizes);
    KernelBuilder& default_iterations(size_t iters);

    KernelBuilder& setup(SetupFunction func);
    KernelBuilder& teardown(TeardownFunction func);
    KernelBuilder& verify(VerifyFunction func);

    KernelConfig build() const;
    void register_kernel();

private:
    KernelConfig config_;
};

// Common kernel data structures
struct VectorData {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t size = 0;
};

struct MatrixData {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    size_t M = 0;
    size_t N = 0;
    size_t K = 0;
};

// Standard setup/teardown functions
void* setup_vector_add(size_t size);
void teardown_vector_data(void* data);
bool verify_vector_add(const void* result, const void* reference, size_t size);

void* setup_dot_product(size_t size);
bool verify_dot_product(const void* result, const void* reference, size_t size);

void* setup_matrix_multiply(size_t size);
void teardown_matrix_data(void* data);
bool verify_matrix_multiply(const void* result, const void* reference, size_t size);

}  // namespace simd_bench
