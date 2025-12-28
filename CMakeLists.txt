cmake_minimum_required(VERSION 3.16)
project(simd_bench VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(SIMD_BENCH_BUILD_TESTS "Build tests" ON)
option(SIMD_BENCH_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SIMD_BENCH_ENABLE_PAPI "Enable PAPI support" ON)
option(SIMD_BENCH_ENABLE_LIKWID "Enable LIKWID support" ON)
option(SIMD_BENCH_ENABLE_RAPL "Enable RAPL energy profiling" ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O3 -march=native)
endif()

# Find required packages
find_package(Threads REQUIRED)

# Find optional packages
find_package(GTest QUIET)
find_package(benchmark QUIET)

# Find PAPI
if(SIMD_BENCH_ENABLE_PAPI)
    find_path(PAPI_INCLUDE_DIR papi.h PATHS /usr/include /usr/local/include)
    find_library(PAPI_LIBRARY papi PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    if(PAPI_INCLUDE_DIR AND PAPI_LIBRARY)
        set(PAPI_FOUND TRUE)
        message(STATUS "PAPI found: ${PAPI_LIBRARY}")
    else()
        set(PAPI_FOUND FALSE)
        message(STATUS "PAPI not found - disabling PAPI support")
    endif()
endif()

# Find LIKWID
if(SIMD_BENCH_ENABLE_LIKWID)
    find_path(LIKWID_INCLUDE_DIR likwid.h PATHS /usr/include /usr/local/include)
    find_library(LIKWID_LIBRARY likwid PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    if(LIKWID_INCLUDE_DIR AND LIKWID_LIBRARY)
        set(LIKWID_FOUND TRUE)
        message(STATUS "LIKWID found: ${LIKWID_LIBRARY}")
    else()
        set(LIKWID_FOUND FALSE)
        message(STATUS "LIKWID not found - disabling LIKWID support")
    endif()
endif()

# Find Highway - try system first, then FetchContent as fallback
find_package(hwy QUIET)
if(NOT hwy_FOUND)
    # Try to find from parent project (for local development)
    if(EXISTS "${CMAKE_SOURCE_DIR}/../highway/hwy/highway.h")
        message(STATUS "Using Highway from parent directory")
        set(HWY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../highway")
        set(HWY_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../build")
        set(HWY_FOUND_LOCAL TRUE)
    else()
        # Fetch Highway from GitHub
        message(STATUS "Highway not found - fetching from GitHub")
        include(FetchContent)
        FetchContent_Declare(
            highway
            GIT_REPOSITORY https://github.com/google/highway.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(highway)
        set(HWY_FETCHED TRUE)
    endif()
else()
    message(STATUS "Found system Highway: ${hwy_DIR}")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
if(HWY_FOUND_LOCAL)
    include_directories(${HWY_INCLUDE_DIR})
    link_directories(${HWY_LIBRARY_DIR})
endif()

# nlohmann_json
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(json)
endif()

# Core library
set(SIMD_BENCH_SOURCES
    src/timing.cc
    src/hardware.cc
    src/kernel_registry.cc
    src/performance_counters.cc
    src/roofline.cc
    src/tma.cc
    src/energy.cc
    src/correctness.cc
    src/report_generator.cc
    src/runner.cc
    src/insights.cc
    src/metrics_analyzer.cc
    # New feature modules
    src/memory_traffic.cc
    src/prefetch.cc
    src/branch_analysis.cc
    src/scaling.cc
    src/register_pressure.cc
    src/loop_tiling.cc
    src/config.cc
    src/regression.cc
    src/autovec.cc
    src/backend_registry.cc
)

add_library(simd_bench_core ${SIMD_BENCH_SOURCES})
target_include_directories(simd_bench_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simd_bench_core PUBLIC Threads::Threads)

# Link dl library for dynamic loading (Linux)
if(UNIX AND NOT APPLE)
    target_link_libraries(simd_bench_core PUBLIC dl)
endif()

# Link optional libraries
if(PAPI_FOUND)
    target_compile_definitions(simd_bench_core PUBLIC SIMD_BENCH_HAS_PAPI)
    target_include_directories(simd_bench_core PUBLIC ${PAPI_INCLUDE_DIR})
    target_link_libraries(simd_bench_core PUBLIC ${PAPI_LIBRARY})
endif()

if(LIKWID_FOUND)
    target_compile_definitions(simd_bench_core PUBLIC SIMD_BENCH_HAS_LIKWID)
    target_include_directories(simd_bench_core PUBLIC ${LIKWID_INCLUDE_DIR})
    target_link_libraries(simd_bench_core PUBLIC ${LIKWID_LIBRARY})
endif()

if(SIMD_BENCH_ENABLE_RAPL)
    target_compile_definitions(simd_bench_core PUBLIC SIMD_BENCH_HAS_RAPL)
endif()

target_link_libraries(simd_bench_core PUBLIC nlohmann_json::nlohmann_json)

# Highway library
target_link_libraries(simd_bench_core PUBLIC hwy)

# Main executable
add_executable(simd-bench src/main.cc)
target_link_libraries(simd-bench simd_bench_core)

# Tests
if(SIMD_BENCH_BUILD_TESTS AND GTest_FOUND)
    enable_testing()

    add_executable(simd_bench_tests
        tests/timing_test.cc
        tests/hardware_test.cc
        tests/kernel_registry_test.cc
        tests/performance_counters_test.cc
        tests/roofline_test.cc
        tests/tma_test.cc
        tests/energy_test.cc
        tests/correctness_test.cc
        tests/report_generator_test.cc
        tests/runner_test.cc
        tests/integration_test.cc
        tests/metrics_analyzer_test.cc
    )
    target_link_libraries(simd_bench_tests simd_bench_core GTest::gtest GTest::gtest_main GTest::gmock)

    include(GoogleTest)
    gtest_discover_tests(simd_bench_tests)
endif()

# Benchmarks
if(SIMD_BENCH_BUILD_BENCHMARKS AND benchmark_FOUND)
    add_executable(timing_benchmarks benchmarks/timing_bench.cc)
    target_link_libraries(timing_benchmarks simd_bench_core benchmark::benchmark)

    add_executable(kernel_benchmarks benchmarks/kernel_bench.cc)
    target_link_libraries(kernel_benchmarks simd_bench_core benchmark::benchmark)
endif()

# Examples
add_executable(example_dot_product examples/dot_product.cc)
target_link_libraries(example_dot_product simd_bench_core)

add_executable(example_matmul examples/matmul.cc)
target_link_libraries(example_matmul simd_bench_core)

add_executable(full_benchmark examples/full_benchmark.cc)
target_link_libraries(full_benchmark simd_bench_core)

add_executable(insights_demo examples/insights_demo.cc)
target_link_libraries(insights_demo simd_bench_core)

add_executable(insights_validator examples/insights_validator.cc)
target_link_libraries(insights_validator simd_bench_core)

add_executable(insights_validation_loop examples/insights_validation_loop.cc)
target_link_libraries(insights_validation_loop simd_bench_core)

# Installation
install(TARGETS simd_bench_core simd-bench
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/simd_bench DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "SIMD-Bench Configuration:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  PAPI:         ${PAPI_FOUND}")
message(STATUS "  LIKWID:       ${LIKWID_FOUND}")
message(STATUS "  RAPL:         ${SIMD_BENCH_ENABLE_RAPL}")
message(STATUS "  Tests:        ${SIMD_BENCH_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${SIMD_BENCH_BUILD_BENCHMARKS}")
message(STATUS "")
