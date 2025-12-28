#pragma once

#include "types.h"
#include "hardware.h"
#include <vector>
#include <string>
#include <cstddef>

namespace simd_bench {

// Tiling recommendation for a single cache level
struct TileSizeRecommendation {
    size_t tile_size;                    // Elements per tile
    size_t tile_bytes;                   // Bytes per tile
    std::string cache_level;             // "L1", "L2", "L3"
    double expected_hit_rate;            // Expected cache hit rate
    double expected_speedup;             // Speedup vs untiled
    std::string rationale;
};

// Complete tiling recommendation
struct TilingRecommendation {
    // Recommended tile sizes for each cache level
    size_t l1_tile_size = 0;
    size_t l2_tile_size = 0;
    size_t l3_tile_size = 0;

    // Multi-dimensional tiling (for 2D/3D problems)
    std::vector<size_t> l1_tile_dims;    // e.g., {64, 64} for 2D
    std::vector<size_t> l2_tile_dims;
    std::vector<size_t> l3_tile_dims;

    // Expected performance improvement
    double estimated_speedup = 1.0;
    double estimated_cache_efficiency = 0.0;

    // Code generation
    std::string code_example;
    std::vector<std::string> implementation_notes;

    // Validation
    bool is_valid = false;
    std::string validation_message;
};

// Loop tiling advisor
class LoopTilingAdvisor {
public:
    explicit LoopTilingAdvisor(const HardwareInfo& hw);

    // Get tiling recommendation for 1D problem
    TilingRecommendation recommend_1d(
        size_t total_elements,
        size_t element_bytes,
        size_t reuse_distance = 1          // How far apart are reuses?
    );

    // Get tiling recommendation for 2D problem (e.g., matrix operations)
    TilingRecommendation recommend_2d(
        size_t rows,
        size_t cols,
        size_t element_bytes,
        size_t num_arrays = 3              // Typical: A, B, C for matmul
    );

    // Get tiling recommendation for 3D problem (e.g., stencils)
    TilingRecommendation recommend_3d(
        size_t dim_x,
        size_t dim_y,
        size_t dim_z,
        size_t element_bytes,
        size_t stencil_radius = 1
    );

    // Get tiling recommendation from measured cache behavior
    TilingRecommendation recommend_from_metrics(
        size_t working_set_bytes,
        double l1_miss_rate,
        double l2_miss_rate,
        double l3_miss_rate,
        size_t element_bytes
    );

    // Calculate optimal tile size for a given cache level
    size_t calculate_optimal_tile(
        size_t cache_size_kb,
        size_t element_bytes,
        size_t num_arrays = 1,
        double cache_utilization_target = 0.75
    );

    // Generate tiled loop code
    std::string generate_tiled_loop_code(
        const TilingRecommendation& rec,
        const std::string& loop_var = "i",
        const std::string& array_name = "data"
    );

    // Validate tiling parameters
    bool validate_tiling(
        const TilingRecommendation& rec,
        size_t total_elements
    );

private:
    HardwareInfo hw_;

    // Calculate cache-optimal tile size
    size_t tile_for_cache(
        size_t cache_kb,
        size_t element_bytes,
        size_t num_simultaneous_arrays,
        double fill_ratio = 0.75
    );

    // Estimate speedup from tiling
    double estimate_speedup(
        size_t working_set,
        size_t tile_size,
        size_t cache_size,
        double current_miss_rate
    );
};

// Cache blocking analysis
struct CacheBlockingAnalysis {
    bool needs_blocking = false;
    size_t optimal_block_size = 0;
    double current_efficiency = 0.0;
    double expected_efficiency = 0.0;
    std::vector<std::string> recommendations;
};

CacheBlockingAnalysis analyze_cache_blocking(
    size_t working_set_bytes,
    const CacheInfo& cache,
    double l1_miss_rate,
    double l2_miss_rate,
    double l3_miss_rate
);

// Matrix multiplication specific tiling
struct MatmulTiling {
    size_t m_tile;                        // Tile size in M dimension
    size_t n_tile;                        // Tile size in N dimension
    size_t k_tile;                        // Tile size in K dimension
    size_t register_block_m = 4;          // Register blocking in M
    size_t register_block_n = 4;          // Register blocking in N
    double expected_efficiency = 0.0;
    std::string code_example;
};

MatmulTiling recommend_matmul_tiling(
    size_t M,
    size_t N,
    size_t K,
    size_t element_bytes,
    const CacheInfo& cache,
    int simd_width_bytes = 32            // AVX2 default
);

// Stencil-specific tiling
struct StencilTiling {
    std::vector<size_t> tile_dims;        // Tile size per dimension
    size_t time_steps_per_tile = 1;       // For temporal blocking
    bool use_diamond_tiling = false;      // For wavefront parallelism
    double expected_efficiency = 0.0;
    std::string code_example;
};

StencilTiling recommend_stencil_tiling(
    const std::vector<size_t>& grid_dims,
    size_t stencil_radius,
    size_t element_bytes,
    const CacheInfo& cache
);

// Tiling parameter constraints
struct TilingConstraints {
    // Minimum tile size (for SIMD efficiency)
    size_t min_tile_elements = 64;

    // Maximum tile size (to avoid TLB pressure)
    size_t max_tile_bytes = 4 * 1024 * 1024;  // 4MB

    // Alignment requirements
    size_t alignment_bytes = 64;              // Cache line

    // SIMD width
    size_t simd_elements = 8;                 // AVX2 floats
};

}  // namespace simd_bench
