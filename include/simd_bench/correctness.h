#pragma once

#include "types.h"
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <limits>
#include <string>

namespace simd_bench {

// ULP (Units in Last Place) calculation
template<typename T>
int64_t ulp_distance(T a, T b);

// Specializations
template<>
int64_t ulp_distance<float>(float a, float b);

template<>
int64_t ulp_distance<double>(double a, double b);

// Correctness verification configuration
struct VerificationConfig {
    double max_absolute_error = 1e-6;
    double max_relative_error = 1e-5;
    int64_t max_ulp_error = 4;
    bool allow_nan = false;
    bool allow_inf = false;
    size_t num_random_tests = 1000;
    std::optional<uint64_t> random_seed;
};

// Individual element comparison result
struct ElementComparisonResult {
    size_t index;
    double expected;
    double actual;
    double absolute_error;
    double relative_error;
    int64_t ulp_error;
    bool is_nan;
    bool is_inf;
    bool passed;
};

// Detailed verification result
struct VerificationResult {
    CorrectnessMetrics metrics;
    std::vector<ElementComparisonResult> failures;  // First N failures
    size_t total_failures = 0;
    size_t total_elements = 0;

    bool passed() const { return metrics.passed; }
};

// Correctness verifier class
class CorrectnessVerifier {
public:
    explicit CorrectnessVerifier(const VerificationConfig& config = VerificationConfig{});

    // Set configuration
    void set_config(const VerificationConfig& config) { config_ = config; }
    const VerificationConfig& get_config() const { return config_; }

    // Verify two arrays are approximately equal
    template<typename T>
    VerificationResult verify_arrays(
        const T* expected,
        const T* actual,
        size_t count
    );

    // Verify a SIMD kernel against a reference scalar implementation
    VerificationResult verify_kernel(
        const std::function<void(float*, size_t)>& simd_kernel,
        const std::function<void(float*, size_t)>& reference_kernel,
        size_t size
    );

    // Verify with random inputs
    VerificationResult verify_with_random_inputs(
        const std::function<void(float*, const float*, size_t)>& simd_kernel,
        const std::function<void(float*, const float*, size_t)>& reference_kernel,
        size_t size,
        float min_val = -1000.0f,
        float max_val = 1000.0f
    );

    // Verify reduction operations (e.g., sum, dot product)
    VerificationResult verify_reduction(
        const std::function<float(const float*, size_t)>& simd_kernel,
        const std::function<float(const float*, size_t)>& reference_kernel,
        size_t size
    );

private:
    VerificationConfig config_;

    template<typename T>
    ElementComparisonResult compare_elements(T expected, T actual, size_t index);
};

// Random input generators
class RandomInputGenerator {
public:
    explicit RandomInputGenerator(uint64_t seed = std::random_device{}());

    // Generate uniform random floats
    void generate_uniform(float* data, size_t count, float min_val = 0.0f, float max_val = 1.0f);

    // Generate normal distribution
    void generate_normal(float* data, size_t count, float mean = 0.0f, float stddev = 1.0f);

    // Generate integers in range
    void generate_integers(int32_t* data, size_t count, int32_t min_val, int32_t max_val);

    // Generate edge cases (NaN, Inf, denormals, etc.)
    void generate_edge_cases(float* data, size_t count);

    // Generate data with specific properties
    void generate_positive(float* data, size_t count, float max_val = 1000.0f);
    void generate_unit_range(float* data, size_t count);  // [0, 1]
    void generate_centered(float* data, size_t count, float range = 1.0f);  // [-range, range]

private:
    std::mt19937_64 rng_;
};

// Reference implementations for common operations
namespace reference {

// Element-wise operations
void vector_add(float* result, const float* a, const float* b, size_t count);
void vector_mul(float* result, const float* a, const float* b, size_t count);
void vector_fma(float* result, const float* a, const float* b, const float* c, size_t count);

// Reductions
float sum(const float* data, size_t count);
float dot_product(const float* a, const float* b, size_t count);
float min(const float* data, size_t count);
float max(const float* data, size_t count);

// Matrix operations
void matrix_multiply(
    float* C,
    const float* A,
    const float* B,
    size_t M, size_t N, size_t K
);

// Transcendental functions
void exp(float* result, const float* input, size_t count);
void log(float* result, const float* input, size_t count);
void sin(float* result, const float* input, size_t count);
void cos(float* result, const float* input, size_t count);

// Neural network operations
void softmax(float* result, const float* input, size_t count);
void relu(float* result, const float* input, size_t count);

}  // namespace reference

// Helper macros for testing
#define SIMD_BENCH_EXPECT_NEAR(expected, actual, tolerance) \
    do { \
        double _e = (expected); \
        double _a = (actual); \
        if (std::abs(_e - _a) > (tolerance)) { \
            return false; \
        } \
    } while(0)

#define SIMD_BENCH_EXPECT_ULP(expected, actual, max_ulp) \
    do { \
        if (simd_bench::ulp_distance((expected), (actual)) > (max_ulp)) { \
            return false; \
        } \
    } while(0)

}  // namespace simd_bench
