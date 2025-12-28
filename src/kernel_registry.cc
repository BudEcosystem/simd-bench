#include "simd_bench/kernel_registry.h"
#include "hwy/aligned_allocator.h"
#include <algorithm>

namespace simd_bench {

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry registry;
    return registry;
}

void KernelRegistry::register_kernel(const KernelConfig& config) {
    kernels_[config.name] = config;
}

const KernelConfig* KernelRegistry::get_kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    if (it != kernels_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> KernelRegistry::get_kernel_names() const {
    std::vector<std::string> names;
    names.reserve(kernels_.size());
    for (const auto& [name, _] : kernels_) {
        names.push_back(name);
    }
    return names;
}

std::vector<const KernelConfig*> KernelRegistry::get_kernels_by_category(
    const std::string& category
) const {
    std::vector<const KernelConfig*> result;
    for (const auto& [_, config] : kernels_) {
        if (config.category == category) {
            result.push_back(&config);
        }
    }
    return result;
}

bool KernelRegistry::has_kernel(const std::string& name) const {
    return kernels_.find(name) != kernels_.end();
}

void KernelRegistry::clear() {
    kernels_.clear();
}

// KernelBuilder implementation
KernelBuilder::KernelBuilder(const std::string& name) {
    config_.name = name;
}

KernelBuilder& KernelBuilder::description(const std::string& desc) {
    config_.description = desc;
    return *this;
}

KernelBuilder& KernelBuilder::category(const std::string& cat) {
    config_.category = cat;
    return *this;
}

KernelBuilder& KernelBuilder::arithmetic_intensity(double ai) {
    config_.arithmetic_intensity = ai;
    return *this;
}

KernelBuilder& KernelBuilder::flops_per_element(size_t flops) {
    config_.flops_per_element = flops;
    return *this;
}

KernelBuilder& KernelBuilder::bytes_per_element(size_t bytes) {
    config_.bytes_per_element = bytes;
    return *this;
}

KernelBuilder& KernelBuilder::add_variant(
    const std::string& name,
    KernelFunction func,
    const std::string& isa,
    bool is_reference
) {
    KernelVariant variant;
    variant.name = name;
    variant.func = func;
    variant.isa = isa;
    variant.is_reference = is_reference;
    config_.variants.push_back(variant);
    return *this;
}

KernelBuilder& KernelBuilder::sizes(std::vector<size_t> sizes) {
    config_.sizes = std::move(sizes);
    return *this;
}

KernelBuilder& KernelBuilder::default_iterations(size_t iters) {
    config_.default_iterations = iters;
    return *this;
}

KernelBuilder& KernelBuilder::setup(SetupFunction func) {
    config_.setup = func;
    return *this;
}

KernelBuilder& KernelBuilder::teardown(TeardownFunction func) {
    config_.teardown = func;
    return *this;
}

KernelBuilder& KernelBuilder::verify(VerifyFunction func) {
    config_.verify = func;
    return *this;
}

KernelConfig KernelBuilder::build() const {
    return config_;
}

void KernelBuilder::register_kernel() {
    KernelRegistry::instance().register_kernel(config_);
}

// Standard setup/teardown functions
void* setup_vector_add(size_t size) {
    auto* data = new VectorData();
    data->size = size;
    data->a = static_cast<float*>(hwy::AllocateAlignedBytes(size * sizeof(float),
                                                             nullptr, 0));
    data->b = static_cast<float*>(hwy::AllocateAlignedBytes(size * sizeof(float),
                                                             nullptr, 0));
    data->c = static_cast<float*>(hwy::AllocateAlignedBytes(size * sizeof(float),
                                                             nullptr, 0));

    for (size_t i = 0; i < size; ++i) {
        data->a[i] = static_cast<float>(i % 100) / 100.0f;
        data->b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }

    return data;
}

void teardown_vector_data(void* ptr) {
    auto* data = static_cast<VectorData*>(ptr);
    hwy::FreeAlignedBytes(data->a, nullptr, 0);
    hwy::FreeAlignedBytes(data->b, nullptr, 0);
    hwy::FreeAlignedBytes(data->c, nullptr, 0);
    delete data;
}

bool verify_vector_add(const void* result, const void* reference, size_t size) {
    auto* res = static_cast<const float*>(result);
    auto* ref = static_cast<const float*>(reference);

    for (size_t i = 0; i < size; ++i) {
        if (std::abs(res[i] - ref[i]) > 1e-5f) {
            return false;
        }
    }
    return true;
}

void* setup_dot_product(size_t size) {
    return setup_vector_add(size);  // Same setup
}

bool verify_dot_product(const void* result, const void* reference, size_t) {
    auto res = *static_cast<const float*>(result);
    auto ref = *static_cast<const float*>(reference);

    return std::abs(res - ref) / (std::abs(ref) + 1e-10f) < 1e-4f;
}

void* setup_matrix_multiply(size_t size) {
    // size is the matrix dimension (N for NxN)
    size_t n = size;
    auto* data = new MatrixData();
    data->M = n;
    data->N = n;
    data->K = n;

    size_t elements = n * n;
    data->A = static_cast<float*>(hwy::AllocateAlignedBytes(elements * sizeof(float),
                                                             nullptr, 0));
    data->B = static_cast<float*>(hwy::AllocateAlignedBytes(elements * sizeof(float),
                                                             nullptr, 0));
    data->C = static_cast<float*>(hwy::AllocateAlignedBytes(elements * sizeof(float),
                                                             nullptr, 0));

    for (size_t i = 0; i < elements; ++i) {
        data->A[i] = static_cast<float>(i % 100) / 100.0f;
        data->B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
        data->C[i] = 0.0f;
    }

    return data;
}

void teardown_matrix_data(void* ptr) {
    auto* data = static_cast<MatrixData*>(ptr);
    hwy::FreeAlignedBytes(data->A, nullptr, 0);
    hwy::FreeAlignedBytes(data->B, nullptr, 0);
    hwy::FreeAlignedBytes(data->C, nullptr, 0);
    delete data;
}

bool verify_matrix_multiply(const void* result, const void* reference, size_t size) {
    auto* res = static_cast<const float*>(result);
    auto* ref = static_cast<const float*>(reference);

    for (size_t i = 0; i < size * size; ++i) {
        float rel_err = std::abs(res[i] - ref[i]) / (std::abs(ref[i]) + 1e-10f);
        if (rel_err > 1e-3f) {
            return false;
        }
    }
    return true;
}

}  // namespace simd_bench
