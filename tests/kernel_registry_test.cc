#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/kernel_registry.h"

namespace simd_bench {
namespace testing {

class KernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        KernelRegistry::instance().clear();
    }

    void TearDown() override {
        KernelRegistry::instance().clear();
    }
};

// Test singleton pattern
TEST_F(KernelRegistryTest, SingletonReturnsConsistentInstance) {
    KernelRegistry& instance1 = KernelRegistry::instance();
    KernelRegistry& instance2 = KernelRegistry::instance();
    EXPECT_EQ(&instance1, &instance2);
}

// Test kernel registration
TEST_F(KernelRegistryTest, RegisterKernelAddsToRegistry) {
    KernelConfig config;
    config.name = "test_kernel";
    config.description = "Test kernel";
    config.category = "test";

    KernelRegistry::instance().register_kernel(config);

    EXPECT_TRUE(KernelRegistry::instance().has_kernel("test_kernel"));
    EXPECT_EQ(KernelRegistry::instance().size(), 1u);
}

TEST_F(KernelRegistryTest, RegisterMultipleKernels) {
    for (int i = 0; i < 5; ++i) {
        KernelConfig config;
        config.name = "kernel_" + std::to_string(i);
        KernelRegistry::instance().register_kernel(config);
    }

    EXPECT_EQ(KernelRegistry::instance().size(), 5u);
}

// Test kernel retrieval
TEST_F(KernelRegistryTest, GetKernelReturnsRegisteredKernel) {
    KernelConfig config;
    config.name = "my_kernel";
    config.description = "My test kernel";
    config.category = "math";
    config.arithmetic_intensity = 0.5;

    KernelRegistry::instance().register_kernel(config);

    const KernelConfig* retrieved = KernelRegistry::instance().get_kernel("my_kernel");

    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->name, "my_kernel");
    EXPECT_EQ(retrieved->description, "My test kernel");
    EXPECT_EQ(retrieved->category, "math");
    EXPECT_DOUBLE_EQ(retrieved->arithmetic_intensity, 0.5);
}

TEST_F(KernelRegistryTest, GetKernelReturnsNullForUnknown) {
    const KernelConfig* result = KernelRegistry::instance().get_kernel("unknown_kernel");
    EXPECT_EQ(result, nullptr);
}

// Test has_kernel
TEST_F(KernelRegistryTest, HasKernelReturnsTrueForRegistered) {
    KernelConfig config;
    config.name = "existing_kernel";
    KernelRegistry::instance().register_kernel(config);

    EXPECT_TRUE(KernelRegistry::instance().has_kernel("existing_kernel"));
}

TEST_F(KernelRegistryTest, HasKernelReturnsFalseForUnregistered) {
    EXPECT_FALSE(KernelRegistry::instance().has_kernel("nonexistent"));
}

// Test get_kernel_names
TEST_F(KernelRegistryTest, GetKernelNamesReturnsAllNames) {
    for (const auto& name : {"alpha", "beta", "gamma"}) {
        KernelConfig config;
        config.name = name;
        KernelRegistry::instance().register_kernel(config);
    }

    std::vector<std::string> names = KernelRegistry::instance().get_kernel_names();

    EXPECT_EQ(names.size(), 3u);
    EXPECT_THAT(names, ::testing::UnorderedElementsAre("alpha", "beta", "gamma"));
}

// Test get_kernels_by_category
TEST_F(KernelRegistryTest, GetKernelsByCategoryFiltersCorrectly) {
    std::vector<std::pair<std::string, std::string>> kernels = {
        {"kernel1", "blas"},
        {"kernel2", "blas"},
        {"kernel3", "image"},
        {"kernel4", "blas"},
        {"kernel5", "neural"}
    };

    for (const auto& [name, category] : kernels) {
        KernelConfig config;
        config.name = name;
        config.category = category;
        KernelRegistry::instance().register_kernel(config);
    }

    auto blas_kernels = KernelRegistry::instance().get_kernels_by_category("blas");
    EXPECT_EQ(blas_kernels.size(), 3u);

    auto image_kernels = KernelRegistry::instance().get_kernels_by_category("image");
    EXPECT_EQ(image_kernels.size(), 1u);

    auto unknown_kernels = KernelRegistry::instance().get_kernels_by_category("unknown");
    EXPECT_EQ(unknown_kernels.size(), 0u);
}

// Test clear
TEST_F(KernelRegistryTest, ClearRemovesAllKernels) {
    for (int i = 0; i < 3; ++i) {
        KernelConfig config;
        config.name = "kernel_" + std::to_string(i);
        KernelRegistry::instance().register_kernel(config);
    }

    EXPECT_EQ(KernelRegistry::instance().size(), 3u);

    KernelRegistry::instance().clear();

    EXPECT_EQ(KernelRegistry::instance().size(), 0u);
}

// Test KernelBuilder
TEST_F(KernelRegistryTest, KernelBuilderCreatesValidConfig) {
    auto dummy_func = [](void*, size_t, size_t) {};

    KernelConfig config = KernelBuilder("dot_product")
        .description("Vector dot product")
        .category("BLAS Level 1")
        .arithmetic_intensity(0.25)
        .flops_per_element(2)
        .bytes_per_element(8)
        .add_variant("scalar", dummy_func, "scalar", true)
        .add_variant("avx2", dummy_func, "avx2", false)
        .sizes({1024, 4096, 16384})
        .default_iterations(1000)
        .build();

    EXPECT_EQ(config.name, "dot_product");
    EXPECT_EQ(config.description, "Vector dot product");
    EXPECT_EQ(config.category, "BLAS Level 1");
    EXPECT_DOUBLE_EQ(config.arithmetic_intensity, 0.25);
    EXPECT_EQ(config.flops_per_element, 2u);
    EXPECT_EQ(config.bytes_per_element, 8u);
    EXPECT_EQ(config.variants.size(), 2u);
    EXPECT_EQ(config.sizes.size(), 3u);
    EXPECT_EQ(config.default_iterations, 1000u);
}

TEST_F(KernelRegistryTest, KernelBuilderRegisterKernel) {
    auto dummy_func = [](void*, size_t, size_t) {};

    KernelBuilder("builder_test")
        .description("Builder test")
        .add_variant("test", dummy_func)
        .register_kernel();

    EXPECT_TRUE(KernelRegistry::instance().has_kernel("builder_test"));
}

TEST_F(KernelRegistryTest, KernelBuilderVariantProperties) {
    auto dummy_func = [](void*, size_t, size_t) {};

    KernelConfig config = KernelBuilder("variant_test")
        .add_variant("reference", dummy_func, "scalar", true)
        .add_variant("optimized", dummy_func, "avx512", false)
        .build();

    ASSERT_EQ(config.variants.size(), 2u);

    EXPECT_EQ(config.variants[0].name, "reference");
    EXPECT_EQ(config.variants[0].isa, "scalar");
    EXPECT_TRUE(config.variants[0].is_reference);

    EXPECT_EQ(config.variants[1].name, "optimized");
    EXPECT_EQ(config.variants[1].isa, "avx512");
    EXPECT_FALSE(config.variants[1].is_reference);
}

// Test setup and teardown functions
TEST_F(KernelRegistryTest, KernelBuilderWithSetupTeardown) {
    bool setup_called = false;
    bool teardown_called = false;

    auto setup = [&setup_called](size_t) -> void* {
        setup_called = true;
        return nullptr;
    };

    auto teardown = [&teardown_called](void*) {
        teardown_called = true;
    };

    KernelConfig config = KernelBuilder("lifecycle_test")
        .setup(setup)
        .teardown(teardown)
        .build();

    ASSERT_NE(config.setup, nullptr);
    ASSERT_NE(config.teardown, nullptr);

    void* data = config.setup(100);
    config.teardown(data);

    EXPECT_TRUE(setup_called);
    EXPECT_TRUE(teardown_called);
}

// Test verify function
TEST_F(KernelRegistryTest, KernelBuilderWithVerify) {
    bool verify_called = false;

    auto verify = [&verify_called](const void*, const void*, size_t) -> bool {
        verify_called = true;
        return true;
    };

    KernelConfig config = KernelBuilder("verify_test")
        .verify(verify)
        .build();

    ASSERT_NE(config.verify, nullptr);

    bool result = config.verify(nullptr, nullptr, 100);

    EXPECT_TRUE(verify_called);
    EXPECT_TRUE(result);
}

// Test VectorData structure
TEST_F(KernelRegistryTest, VectorDataDefaultsToNull) {
    VectorData data;
    EXPECT_EQ(data.a, nullptr);
    EXPECT_EQ(data.b, nullptr);
    EXPECT_EQ(data.c, nullptr);
    EXPECT_EQ(data.size, 0u);
}

// Test MatrixData structure
TEST_F(KernelRegistryTest, MatrixDataDefaultsToNull) {
    MatrixData data;
    EXPECT_EQ(data.A, nullptr);
    EXPECT_EQ(data.B, nullptr);
    EXPECT_EQ(data.C, nullptr);
    EXPECT_EQ(data.M, 0u);
    EXPECT_EQ(data.N, 0u);
    EXPECT_EQ(data.K, 0u);
}

}  // namespace testing
}  // namespace simd_bench
