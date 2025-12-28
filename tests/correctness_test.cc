#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/correctness.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace simd_bench {
namespace testing {

class CorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test ULP distance calculation
TEST_F(CorrectnessTest, ULPDistanceZeroForIdentical) {
    float a = 1.0f;
    EXPECT_EQ(ulp_distance(a, a), 0);
}

TEST_F(CorrectnessTest, ULPDistanceOneForAdjacent) {
    float a = 1.0f;
    float b = std::nextafter(a, 2.0f);
    EXPECT_EQ(ulp_distance(a, b), 1);
}

TEST_F(CorrectnessTest, ULPDistanceSymmetric) {
    float a = 1.5f;
    float b = 1.500001f;
    EXPECT_EQ(ulp_distance(a, b), ulp_distance(b, a));
}

TEST_F(CorrectnessTest, ULPDistanceDouble) {
    double a = 1.0;
    double b = std::nextafter(a, 2.0);
    EXPECT_EQ(ulp_distance(a, b), 1);
}

// Test VerificationConfig defaults
TEST_F(CorrectnessTest, VerificationConfigDefaults) {
    VerificationConfig config;
    EXPECT_DOUBLE_EQ(config.max_absolute_error, 1e-6);
    EXPECT_DOUBLE_EQ(config.max_relative_error, 1e-5);
    EXPECT_EQ(config.max_ulp_error, 4);
    EXPECT_FALSE(config.allow_nan);
    EXPECT_FALSE(config.allow_inf);
}

// Test CorrectnessVerifier
TEST_F(CorrectnessTest, VerifyArraysIdenticalPasses) {
    CorrectnessVerifier verifier;

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual = expected;

    auto result = verifier.verify_arrays(expected.data(), actual.data(), expected.size());

    EXPECT_TRUE(result.passed());
    EXPECT_DOUBLE_EQ(result.metrics.max_absolute_error, 0.0);
    EXPECT_EQ(result.total_failures, 0u);
}

TEST_F(CorrectnessTest, VerifyArraysWithSmallDifferencePasses) {
    CorrectnessVerifier verifier;

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual = {1.0f + 1e-7f, 2.0f, 3.0f, 4.0f};

    auto result = verifier.verify_arrays(expected.data(), actual.data(), expected.size());

    EXPECT_TRUE(result.passed());
}

TEST_F(CorrectnessTest, VerifyArraysWithLargeDifferenceFails) {
    VerificationConfig config;
    config.max_absolute_error = 1e-6;
    CorrectnessVerifier verifier(config);

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual = {1.0f + 0.01f, 2.0f, 3.0f, 4.0f};

    auto result = verifier.verify_arrays(expected.data(), actual.data(), expected.size());

    EXPECT_FALSE(result.passed());
    EXPECT_GT(result.total_failures, 0u);
}

TEST_F(CorrectnessTest, VerifyArraysDetectsNaN) {
    VerificationConfig config;
    config.allow_nan = false;
    CorrectnessVerifier verifier(config);

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual = {1.0f, std::nanf(""), 3.0f, 4.0f};

    auto result = verifier.verify_arrays(expected.data(), actual.data(), expected.size());

    EXPECT_FALSE(result.passed());
    EXPECT_GT(result.metrics.nan_count, 0);
}

TEST_F(CorrectnessTest, VerifyArraysDetectsInf) {
    VerificationConfig config;
    config.allow_inf = false;
    CorrectnessVerifier verifier(config);

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual = {1.0f, std::numeric_limits<float>::infinity(), 3.0f, 4.0f};

    auto result = verifier.verify_arrays(expected.data(), actual.data(), expected.size());

    EXPECT_FALSE(result.passed());
    EXPECT_GT(result.metrics.inf_count, 0);
}

// Test verify_kernel
TEST_F(CorrectnessTest, VerifyKernelWithIdenticalImplementations) {
    CorrectnessVerifier verifier;

    auto kernel = [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) data[i] *= 2.0f;
    };

    auto reference = [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) data[i] *= 2.0f;
    };

    auto result = verifier.verify_kernel(kernel, reference, 1024);

    EXPECT_TRUE(result.passed());
}

// Test RandomInputGenerator
TEST_F(CorrectnessTest, RandomGeneratorUniformInRange) {
    RandomInputGenerator gen(42);

    std::vector<float> data(1000);
    gen.generate_uniform(data.data(), data.size(), -10.0f, 10.0f);

    for (float v : data) {
        EXPECT_GE(v, -10.0f);
        EXPECT_LE(v, 10.0f);
    }
}

TEST_F(CorrectnessTest, RandomGeneratorNormalDistribution) {
    RandomInputGenerator gen(42);

    std::vector<float> data(10000);
    gen.generate_normal(data.data(), data.size(), 0.0f, 1.0f);

    // Calculate mean
    double sum = 0;
    for (float v : data) sum += v;
    double mean = sum / data.size();

    // Mean should be close to 0
    EXPECT_NEAR(mean, 0.0, 0.1);
}

TEST_F(CorrectnessTest, RandomGeneratorEdgeCases) {
    RandomInputGenerator gen(42);

    std::vector<float> data(100);
    gen.generate_edge_cases(data.data(), data.size());

    bool has_nan = false, has_inf = false, has_denormal = false;
    for (float v : data) {
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
        if (std::fpclassify(v) == FP_SUBNORMAL) has_denormal = true;
    }

    // Should generate some edge cases
    EXPECT_TRUE(has_nan || has_inf || has_denormal);
}

TEST_F(CorrectnessTest, RandomGeneratorPositive) {
    RandomInputGenerator gen(42);

    std::vector<float> data(1000);
    gen.generate_positive(data.data(), data.size());

    for (float v : data) {
        EXPECT_GT(v, 0.0f);
    }
}

TEST_F(CorrectnessTest, RandomGeneratorUnitRange) {
    RandomInputGenerator gen(42);

    std::vector<float> data(1000);
    gen.generate_unit_range(data.data(), data.size());

    for (float v : data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST_F(CorrectnessTest, RandomGeneratorDeterministicWithSeed) {
    std::vector<float> data1(100), data2(100);

    RandomInputGenerator gen1(12345);
    gen1.generate_uniform(data1.data(), data1.size());

    RandomInputGenerator gen2(12345);
    gen2.generate_uniform(data2.data(), data2.size());

    EXPECT_EQ(data1, data2);
}

// Test reference implementations
TEST_F(CorrectnessTest, ReferenceVectorAdd) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> result(4);

    reference::vector_add(result.data(), a.data(), b.data(), 4);

    EXPECT_FLOAT_EQ(result[0], 6.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 10.0f);
    EXPECT_FLOAT_EQ(result[3], 12.0f);
}

TEST_F(CorrectnessTest, ReferenceVectorMul) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> result(4);

    reference::vector_mul(result.data(), a.data(), b.data(), 4);

    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 6.0f);
    EXPECT_FLOAT_EQ(result[2], 12.0f);
    EXPECT_FLOAT_EQ(result[3], 20.0f);
}

TEST_F(CorrectnessTest, ReferenceSum) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    float sum = reference::sum(data.data(), data.size());
    EXPECT_FLOAT_EQ(sum, 10.0f);
}

TEST_F(CorrectnessTest, ReferenceDotProduct) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f};
    float dot = reference::dot_product(a.data(), b.data(), 4);
    EXPECT_FLOAT_EQ(dot, 10.0f);
}

TEST_F(CorrectnessTest, ReferenceMinMax) {
    std::vector<float> data = {3.0f, 1.0f, 4.0f, 1.5f, 9.0f, 2.6f};
    EXPECT_FLOAT_EQ(reference::min(data.data(), data.size()), 1.0f);
    EXPECT_FLOAT_EQ(reference::max(data.data(), data.size()), 9.0f);
}

TEST_F(CorrectnessTest, ReferenceMatrixMultiply) {
    // 2x2 * 2x2
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};  // Row-major: [[1,2],[3,4]]
    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};  // Row-major: [[5,6],[7,8]]
    std::vector<float> C(4);

    reference::matrix_multiply(C.data(), A.data(), B.data(), 2, 2, 2);

    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //   = [[19, 22], [43, 50]]
    EXPECT_FLOAT_EQ(C[0], 19.0f);
    EXPECT_FLOAT_EQ(C[1], 22.0f);
    EXPECT_FLOAT_EQ(C[2], 43.0f);
    EXPECT_FLOAT_EQ(C[3], 50.0f);
}

TEST_F(CorrectnessTest, ReferenceExp) {
    std::vector<float> input = {0.0f, 1.0f, 2.0f};
    std::vector<float> result(3);

    reference::exp(result.data(), input.data(), 3);

    EXPECT_NEAR(result[0], 1.0f, 1e-5);
    EXPECT_NEAR(result[1], std::exp(1.0f), 1e-5);
    EXPECT_NEAR(result[2], std::exp(2.0f), 1e-4);
}

TEST_F(CorrectnessTest, ReferenceSoftmax) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> result(3);

    reference::softmax(result.data(), input.data(), 3);

    // Sum should be 1
    float sum = result[0] + result[1] + result[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);

    // All values should be positive
    for (float v : result) {
        EXPECT_GT(v, 0.0f);
    }

    // Larger input should have larger output
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
}

TEST_F(CorrectnessTest, ReferenceRelu) {
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> result(5);

    reference::relu(result.data(), input.data(), 5);

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 1.0f);
    EXPECT_FLOAT_EQ(result[4], 2.0f);
}

// Test verification with random inputs
TEST_F(CorrectnessTest, VerifyWithRandomInputs) {
    CorrectnessVerifier verifier;

    auto simd_add = [](float* result, const float* input, size_t count) {
        for (size_t i = 0; i < count; ++i) result[i] = input[i] + 1.0f;
    };

    auto ref_add = [](float* result, const float* input, size_t count) {
        for (size_t i = 0; i < count; ++i) result[i] = input[i] + 1.0f;
    };

    auto result = verifier.verify_with_random_inputs(simd_add, ref_add, 1024);
    EXPECT_TRUE(result.passed());
}

// Test verify reduction
TEST_F(CorrectnessTest, VerifyReduction) {
    // Use higher tolerance because reference::sum uses double accumulator
    // while the test uses float, leading to different rounding
    VerificationConfig config;
    config.max_absolute_error = 1e-3;  // Looser tolerance for float accumulation
    config.max_relative_error = 1e-3;
    config.max_ulp_error = 1000;  // Float sum has different precision than double
    CorrectnessVerifier verifier(config);

    auto simd_sum = [](const float* data, size_t count) -> float {
        float sum = 0;
        for (size_t i = 0; i < count; ++i) sum += data[i];
        return sum;
    };

    auto result = verifier.verify_reduction(simd_sum, reference::sum, 1024);
    EXPECT_TRUE(result.passed());
}

}  // namespace testing
}  // namespace simd_bench
