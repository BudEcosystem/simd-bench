#include "simd_bench/correctness.h"
#include "hwy/aligned_allocator.h"
#include <cstring>
#include <algorithm>
#include <numeric>

namespace simd_bench {

// ULP distance implementation
template<>
int64_t ulp_distance<float>(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<int64_t>::max();
    }
    if (a == b) return 0;

    int32_t ai, bi;
    std::memcpy(&ai, &a, sizeof(float));
    std::memcpy(&bi, &b, sizeof(float));

    // Handle negative zero
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;

    return std::abs(static_cast<int64_t>(ai) - static_cast<int64_t>(bi));
}

template<>
int64_t ulp_distance<double>(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<int64_t>::max();
    }
    if (a == b) return 0;

    int64_t ai, bi;
    std::memcpy(&ai, &a, sizeof(double));
    std::memcpy(&bi, &b, sizeof(double));

    // Handle negative zero
    if (ai < 0) ai = 0x8000000000000000LL - ai;
    if (bi < 0) bi = 0x8000000000000000LL - bi;

    return std::abs(ai - bi);
}

// CorrectnessVerifier implementation
CorrectnessVerifier::CorrectnessVerifier(const VerificationConfig& config)
    : config_(config) {}

template<typename T>
ElementComparisonResult CorrectnessVerifier::compare_elements(T expected, T actual, size_t index) {
    ElementComparisonResult result;
    result.index = index;
    result.expected = static_cast<double>(expected);
    result.actual = static_cast<double>(actual);
    result.is_nan = std::isnan(actual);
    result.is_inf = std::isinf(actual);

    if (result.is_nan) {
        result.passed = config_.allow_nan;
        return result;
    }

    if (result.is_inf) {
        result.passed = config_.allow_inf;
        return result;
    }

    result.absolute_error = std::abs(static_cast<double>(expected) - static_cast<double>(actual));
    result.relative_error = result.absolute_error / (std::abs(static_cast<double>(expected)) + 1e-10);
    result.ulp_error = ulp_distance(expected, actual);

    result.passed = (result.absolute_error <= config_.max_absolute_error ||
                    result.relative_error <= config_.max_relative_error) &&
                   result.ulp_error <= config_.max_ulp_error;

    return result;
}

template<typename T>
VerificationResult CorrectnessVerifier::verify_arrays(
    const T* expected,
    const T* actual,
    size_t count
) {
    VerificationResult result;
    result.total_elements = count;

    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;

    for (size_t i = 0; i < count; ++i) {
        auto comparison = compare_elements(expected[i], actual[i], i);

        if (comparison.is_nan) result.metrics.nan_count++;
        if (comparison.is_inf) result.metrics.inf_count++;

        sum_abs_error += comparison.absolute_error;
        sum_rel_error += comparison.relative_error;

        result.metrics.max_absolute_error = std::max(
            result.metrics.max_absolute_error, comparison.absolute_error);
        result.metrics.max_relative_error = std::max(
            result.metrics.max_relative_error, comparison.relative_error);
        result.metrics.max_ulp_error = std::max(
            result.metrics.max_ulp_error, static_cast<double>(comparison.ulp_error));

        if (!comparison.passed) {
            result.total_failures++;
            if (result.failures.size() < 10) {  // Store first 10 failures
                result.failures.push_back(comparison);
            }
        }
    }

    result.metrics.mean_absolute_error = sum_abs_error / count;
    result.metrics.mean_relative_error = sum_rel_error / count;

    result.metrics.passed = (result.total_failures == 0);
    if (!result.metrics.passed) {
        result.metrics.failure_reason = std::to_string(result.total_failures) +
                                        " elements failed verification";
    }

    return result;
}

// Explicit template instantiations
template VerificationResult CorrectnessVerifier::verify_arrays<float>(
    const float*, const float*, size_t);
template VerificationResult CorrectnessVerifier::verify_arrays<double>(
    const double*, const double*, size_t);

VerificationResult CorrectnessVerifier::verify_kernel(
    const std::function<void(float*, size_t)>& simd_kernel,
    const std::function<void(float*, size_t)>& reference_kernel,
    size_t size
) {
    auto simd_data = hwy::AllocateAligned<float>(size);
    auto ref_data = hwy::AllocateAligned<float>(size);

    // Initialize with same data
    for (size_t i = 0; i < size; ++i) {
        simd_data[i] = ref_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Run kernels
    simd_kernel(simd_data.get(), size);
    reference_kernel(ref_data.get(), size);

    return verify_arrays(ref_data.get(), simd_data.get(), size);
}

VerificationResult CorrectnessVerifier::verify_with_random_inputs(
    const std::function<void(float*, const float*, size_t)>& simd_kernel,
    const std::function<void(float*, const float*, size_t)>& reference_kernel,
    size_t size,
    float min_val,
    float max_val
) {
    uint64_t seed = config_.random_seed.value_or(std::random_device{}());
    RandomInputGenerator gen(seed);

    auto input = hwy::AllocateAligned<float>(size);
    auto simd_output = hwy::AllocateAligned<float>(size);
    auto ref_output = hwy::AllocateAligned<float>(size);

    gen.generate_uniform(input.get(), size, min_val, max_val);

    simd_kernel(simd_output.get(), input.get(), size);
    reference_kernel(ref_output.get(), input.get(), size);

    return verify_arrays(ref_output.get(), simd_output.get(), size);
}

VerificationResult CorrectnessVerifier::verify_reduction(
    const std::function<float(const float*, size_t)>& simd_kernel,
    const std::function<float(const float*, size_t)>& reference_kernel,
    size_t size
) {
    uint64_t seed = config_.random_seed.value_or(std::random_device{}());
    RandomInputGenerator gen(seed);

    auto input = hwy::AllocateAligned<float>(size);
    gen.generate_uniform(input.get(), size, -100.0f, 100.0f);

    float simd_result = simd_kernel(input.get(), size);
    float ref_result = reference_kernel(input.get(), size);

    return verify_arrays(&ref_result, &simd_result, 1);
}

// RandomInputGenerator implementation
RandomInputGenerator::RandomInputGenerator(uint64_t seed) : rng_(seed) {}

void RandomInputGenerator::generate_uniform(float* data, size_t count, float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng_);
    }
}

void RandomInputGenerator::generate_normal(float* data, size_t count, float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng_);
    }
}

void RandomInputGenerator::generate_integers(int32_t* data, size_t count, int32_t min_val, int32_t max_val) {
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng_);
    }
}

void RandomInputGenerator::generate_edge_cases(float* data, size_t count) {
    std::uniform_real_distribution<float> uniform(-1e10f, 1e10f);
    std::uniform_int_distribution<int> type_dist(0, 10);

    for (size_t i = 0; i < count; ++i) {
        int type = type_dist(rng_);
        switch (type) {
            case 0: data[i] = 0.0f; break;
            case 1: data[i] = -0.0f; break;
            case 2: data[i] = std::numeric_limits<float>::infinity(); break;
            case 3: data[i] = -std::numeric_limits<float>::infinity(); break;
            case 4: data[i] = std::nanf(""); break;
            case 5: data[i] = std::numeric_limits<float>::min(); break;
            case 6: data[i] = std::numeric_limits<float>::max(); break;
            case 7: data[i] = std::numeric_limits<float>::denorm_min(); break;
            case 8: data[i] = 1.0f; break;
            case 9: data[i] = -1.0f; break;
            default: data[i] = uniform(rng_); break;
        }
    }
}

void RandomInputGenerator::generate_positive(float* data, size_t count, float max_val) {
    generate_uniform(data, count, std::numeric_limits<float>::min(), max_val);
}

void RandomInputGenerator::generate_unit_range(float* data, size_t count) {
    generate_uniform(data, count, 0.0f, 1.0f);
}

void RandomInputGenerator::generate_centered(float* data, size_t count, float range) {
    generate_uniform(data, count, -range, range);
}

// Reference implementations
namespace reference {

void vector_add(float* result, const float* a, const float* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul(float* result, const float* a, const float* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void vector_fma(float* result, const float* a, const float* b, const float* c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

float sum(const float* data, size_t count) {
    double sum = 0.0;  // Use double for accuracy
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return static_cast<float>(sum);
}

float dot_product(const float* a, const float* b, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

float min(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    float m = data[0];
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < m) m = data[i];
    }
    return m;
}

float max(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    float m = data[0];
    for (size_t i = 1; i < count; ++i) {
        if (data[i] > m) m = data[i];
    }
    return m;
}

void matrix_multiply(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

void exp(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::exp(input[i]);
    }
}

void log(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::log(input[i]);
    }
}

void sin(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::sin(input[i]);
    }
}

void cos(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::cos(input[i]);
    }
}

void softmax(float* result, const float* input, size_t count) {
    // Find max for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < count; ++i) {
        result[i] /= static_cast<float>(sum);
    }
}

void relu(float* result, const float* input, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = input[i] > 0 ? input[i] : 0;
    }
}

}  // namespace reference

}  // namespace simd_bench
