#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "simd_bench/performance_counters.h"

namespace simd_bench {
namespace testing {

class PerformanceCountersTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test NullCounters
TEST_F(PerformanceCountersTest, NullCountersInitializeSucceeds) {
    NullCounters counters;
    EXPECT_TRUE(counters.initialize());
}

TEST_F(PerformanceCountersTest, NullCountersOperationsSucceed) {
    NullCounters counters;
    counters.initialize();

    EXPECT_TRUE(counters.add_event(CounterEvent::CYCLES));
    EXPECT_TRUE(counters.start());
    EXPECT_TRUE(counters.stop());
    EXPECT_TRUE(counters.reset());

    counters.clear_events();
    counters.shutdown();
}

TEST_F(PerformanceCountersTest, NullCountersReturnsEmptyValues) {
    NullCounters counters;
    CounterValues values = counters.read();
    EXPECT_TRUE(values.values.empty());
}

TEST_F(PerformanceCountersTest, NullCountersBackendIsNone) {
    NullCounters counters;
    EXPECT_EQ(counters.get_backend(), CounterBackend::NONE);
    EXPECT_EQ(counters.get_backend_name(), "none");
}

TEST_F(PerformanceCountersTest, NullCountersEventNotSupported) {
    NullCounters counters;
    EXPECT_FALSE(counters.is_event_supported(CounterEvent::CYCLES));
}

// Test CounterValues
TEST_F(PerformanceCountersTest, CounterValuesGetReturnsZeroForMissing) {
    CounterValues values;
    EXPECT_EQ(values.get(CounterEvent::CYCLES), 0u);
}

TEST_F(PerformanceCountersTest, CounterValuesSetAndGet) {
    CounterValues values;
    values.set(CounterEvent::CYCLES, 12345);
    values.set(CounterEvent::INSTRUCTIONS, 67890);

    EXPECT_EQ(values.get(CounterEvent::CYCLES), 12345u);
    EXPECT_EQ(values.get(CounterEvent::INSTRUCTIONS), 67890u);
}

// Test factory
TEST_F(PerformanceCountersTest, FactoryCreateNoneReturnsNullCounters) {
    auto counters = PerformanceCounterFactory::create(CounterBackend::NONE);
    ASSERT_NE(counters, nullptr);
    EXPECT_EQ(counters->get_backend(), CounterBackend::NONE);
}

TEST_F(PerformanceCountersTest, FactoryCreateBestAvailableReturnsNonNull) {
    auto counters = PerformanceCounterFactory::create_best_available();
    ASSERT_NE(counters, nullptr);
    // Should return at least NullCounters
}

TEST_F(PerformanceCountersTest, FactoryGetAvailableBackendsIncludesNone) {
    auto backends = PerformanceCounterFactory::get_available_backends();
    EXPECT_THAT(backends, ::testing::Contains(CounterBackend::NONE));
}

TEST_F(PerformanceCountersTest, FactoryNoneBackendIsAlwaysAvailable) {
    EXPECT_TRUE(PerformanceCounterFactory::is_backend_available(CounterBackend::NONE));
}

// Test event string conversion
TEST_F(PerformanceCountersTest, CounterEventToStringReturnsValidString) {
    EXPECT_EQ(counter_event_to_string(CounterEvent::CYCLES), "CYCLES");
    EXPECT_EQ(counter_event_to_string(CounterEvent::INSTRUCTIONS), "INSTRUCTIONS");
    EXPECT_EQ(counter_event_to_string(CounterEvent::FP_ARITH_256B_PACKED_SINGLE),
              "FP_ARITH_256B_PACKED_SINGLE");
}

TEST_F(PerformanceCountersTest, StringToCounterEventReturnsCorrectEvent) {
    EXPECT_EQ(string_to_counter_event("CYCLES"), CounterEvent::CYCLES);
    EXPECT_EQ(string_to_counter_event("INSTRUCTIONS"), CounterEvent::INSTRUCTIONS);
}

// Test standard event sets
TEST_F(PerformanceCountersTest, GetFlopsEventsReturnsNonEmpty) {
    auto events = get_flops_events();
    EXPECT_FALSE(events.empty());
    EXPECT_THAT(events, ::testing::Contains(CounterEvent::FP_ARITH_256B_PACKED_SINGLE));
}

TEST_F(PerformanceCountersTest, GetCacheEventsReturnsNonEmpty) {
    auto events = get_cache_events();
    EXPECT_FALSE(events.empty());
    EXPECT_THAT(events, ::testing::Contains(CounterEvent::L1D_READ_MISS));
}

TEST_F(PerformanceCountersTest, GetMemoryEventsReturnsNonEmpty) {
    auto events = get_memory_events();
    EXPECT_FALSE(events.empty());
}

TEST_F(PerformanceCountersTest, GetTmaEventsReturnsNonEmpty) {
    auto events = get_tma_events();
    EXPECT_FALSE(events.empty());
}

// Test ScopedCounters (with NullCounters)
TEST_F(PerformanceCountersTest, ScopedCountersStartsAndStops) {
    NullCounters counters;
    counters.initialize();
    CounterValues result;

    {
        ScopedCounters scoped(counters, result);
        // Do some work
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) x += i;
    }

    // With NullCounters, result will be empty
    // But the test verifies no crash occurs
}

// Test PAPI if available
#ifdef SIMD_BENCH_HAS_PAPI
TEST_F(PerformanceCountersTest, PAPICountersInitializeAndShutdown) {
    PAPICounters counters;
    bool init_success = counters.initialize();
    // May fail if PAPI not properly set up, that's OK
    if (init_success) {
        EXPECT_EQ(counters.get_backend(), CounterBackend::PAPI);
        EXPECT_EQ(counters.get_backend_name(), "PAPI");
        counters.shutdown();
    }
}

TEST_F(PerformanceCountersTest, PAPIFactoryAvailable) {
    bool available = PerformanceCounterFactory::is_backend_available(CounterBackend::PAPI);
    if (available) {
        auto counters = PerformanceCounterFactory::create(CounterBackend::PAPI);
        ASSERT_NE(counters, nullptr);
        EXPECT_EQ(counters->get_backend(), CounterBackend::PAPI);
    }
}
#endif

// Test LIKWID if available
#ifdef SIMD_BENCH_HAS_LIKWID
TEST_F(PerformanceCountersTest, LIKWIDCountersInitializeAndShutdown) {
    LIKWIDCounters counters;
    bool init_success = counters.initialize();
    // May fail if LIKWID not properly set up
    if (init_success) {
        EXPECT_EQ(counters.get_backend(), CounterBackend::LIKWID);
        EXPECT_EQ(counters.get_backend_name(), "LIKWID");
        counters.shutdown();
    }
}

TEST_F(PerformanceCountersTest, LIKWIDGetAvailableGroups) {
    LIKWIDCounters counters;
    if (counters.initialize()) {
        auto groups = counters.get_available_groups();
        // If LIKWID works, should have at least some groups
        counters.shutdown();
    }
}
#endif

// Test perf_event if on Linux
#ifdef __linux__
TEST_F(PerformanceCountersTest, PerfEventCountersBasicOperations) {
    auto counters = PerformanceCounterFactory::create(CounterBackend::PERF_EVENT);
    if (counters && counters->initialize()) {
        EXPECT_EQ(counters->get_backend(), CounterBackend::PERF_EVENT);
        EXPECT_EQ(counters->get_backend_name(), "perf_event");

        // Try adding basic events
        bool cycles_added = counters->add_event(CounterEvent::CYCLES);
        bool instr_added = counters->add_event(CounterEvent::INSTRUCTIONS);

        if (cycles_added && instr_added) {
            EXPECT_TRUE(counters->start());

            volatile int x = 0;
            for (int i = 0; i < 1000000; ++i) x += i;

            EXPECT_TRUE(counters->stop());

            CounterValues values = counters->read();
            // Should have some cycles
            EXPECT_GT(values.get(CounterEvent::CYCLES), 0u);
        }

        counters->shutdown();
    }
}
#endif

}  // namespace testing
}  // namespace simd_bench
