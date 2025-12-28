#include "simd_bench/loop_tiling.h"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace simd_bench {

// ============================================================================
// LoopTilingAdvisor implementation
// ============================================================================

LoopTilingAdvisor::LoopTilingAdvisor(const HardwareInfo& hw) : hw_(hw) {}

size_t LoopTilingAdvisor::tile_for_cache(
    size_t cache_kb,
    size_t element_bytes,
    size_t num_simultaneous_arrays,
    double fill_ratio
) {
    // Target: fit all arrays in cache with some headroom
    size_t cache_bytes = cache_kb * 1024;
    size_t usable_bytes = static_cast<size_t>(cache_bytes * fill_ratio);
    size_t bytes_per_array = usable_bytes / num_simultaneous_arrays;
    size_t elements_per_array = bytes_per_array / element_bytes;

    // Round down to SIMD-friendly size
    size_t simd_elements = 64 / element_bytes;  // Cache line
    elements_per_array = (elements_per_array / simd_elements) * simd_elements;

    return std::max(simd_elements, elements_per_array);
}

TilingRecommendation LoopTilingAdvisor::recommend_1d(
    size_t total_elements,
    size_t element_bytes,
    size_t reuse_distance
) {
    TilingRecommendation rec;

    // Calculate tile sizes for each cache level
    // For 1D with reuse, we need reuse_distance elements to fit in cache

    size_t working_elements = total_elements / reuse_distance;

    // L1 tile: fit in L1 with room for other data
    rec.l1_tile_size = tile_for_cache(hw_.cache.l1d_size_kb, element_bytes, 2, 0.5);

    // L2 tile: fit in L2
    rec.l2_tile_size = tile_for_cache(hw_.cache.l2_size_kb, element_bytes, 2, 0.75);

    // L3 tile: fit in L3
    rec.l3_tile_size = tile_for_cache(hw_.cache.l3_size_kb, element_bytes, 2, 0.75);

    // Estimate speedup
    size_t working_set = total_elements * element_bytes;
    size_t l3_bytes = hw_.cache.l3_size_kb * 1024;

    if (working_set > l3_bytes) {
        rec.estimated_speedup = 2.0 + std::log2(static_cast<double>(working_set) / l3_bytes);
    } else if (working_set > hw_.cache.l2_size_kb * 1024) {
        rec.estimated_speedup = 1.5;
    } else {
        rec.estimated_speedup = 1.1;
    }

    rec.is_valid = (rec.l1_tile_size > 0);
    rec.estimated_cache_efficiency = 0.85;

    // Generate code example
    std::ostringstream code;
    code << "// 1D tiled loop with L2 blocking\n";
    code << "const size_t TILE_SIZE = " << rec.l2_tile_size << ";\n";
    code << "for (size_t tile = 0; tile < n; tile += TILE_SIZE) {\n";
    code << "    size_t tile_end = std::min(tile + TILE_SIZE, n);\n";
    code << "    for (size_t i = tile; i < tile_end; ++i) {\n";
    code << "        // Process element i\n";
    code << "    }\n";
    code << "}\n";
    rec.code_example = code.str();

    rec.implementation_notes.push_back(
        "Tile size chosen to fit in L2 cache (" +
        std::to_string(hw_.cache.l2_size_kb) + " KB)");

    return rec;
}

TilingRecommendation LoopTilingAdvisor::recommend_2d(
    size_t rows,
    size_t cols,
    size_t element_bytes,
    size_t num_arrays
) {
    TilingRecommendation rec;

    // For 2D, we need tile_rows * tile_cols * num_arrays to fit in cache

    // L1: Small tiles for register-level blocking
    size_t l1_elements = tile_for_cache(hw_.cache.l1d_size_kb, element_bytes, num_arrays, 0.5);
    size_t l1_tile = static_cast<size_t>(std::sqrt(static_cast<double>(l1_elements)));
    l1_tile = std::max(size_t(8), (l1_tile / 8) * 8);  // Round to SIMD-friendly, min 8
    rec.l1_tile_size = l1_tile * l1_tile;
    rec.l1_tile_dims = {l1_tile, l1_tile};

    // L2: Medium tiles for cache blocking
    size_t l2_elements = tile_for_cache(hw_.cache.l2_size_kb, element_bytes, num_arrays, 0.75);
    size_t l2_tile = static_cast<size_t>(std::sqrt(static_cast<double>(l2_elements)));
    l2_tile = std::max(size_t(16), (l2_tile / 16) * 16);  // Min 16
    rec.l2_tile_size = l2_tile * l2_tile;
    rec.l2_tile_dims = {l2_tile, l2_tile};

    // L3: Large tiles
    size_t l3_elements = tile_for_cache(hw_.cache.l3_size_kb, element_bytes, num_arrays, 0.75);
    size_t l3_tile = static_cast<size_t>(std::sqrt(static_cast<double>(l3_elements)));
    l3_tile = std::max(size_t(32), (l3_tile / 32) * 32);  // Min 32
    rec.l3_tile_size = l3_tile * l3_tile;
    rec.l3_tile_dims = {l3_tile, l3_tile};

    // Estimate speedup
    size_t total_bytes = rows * cols * element_bytes * num_arrays;
    size_t l2_bytes = hw_.cache.l2_size_kb * 1024;

    if (total_bytes > l2_bytes) {
        double ratio = static_cast<double>(total_bytes) / l2_bytes;
        rec.estimated_speedup = 1.0 + std::log2(ratio) * 0.5;
    } else {
        rec.estimated_speedup = 1.1;
    }

    rec.estimated_cache_efficiency = 0.80;
    rec.is_valid = (l1_tile > 0 && l2_tile > 0);

    // Generate code example
    std::ostringstream code;
    code << "// 2D tiled loop with cache blocking\n";
    code << "const size_t TILE_I = " << rec.l2_tile_dims[0] << ";\n";
    code << "const size_t TILE_J = " << rec.l2_tile_dims[1] << ";\n\n";
    code << "for (size_t ii = 0; ii < rows; ii += TILE_I) {\n";
    code << "    for (size_t jj = 0; jj < cols; jj += TILE_J) {\n";
    code << "        size_t i_end = std::min(ii + TILE_I, rows);\n";
    code << "        size_t j_end = std::min(jj + TILE_J, cols);\n";
    code << "        for (size_t i = ii; i < i_end; ++i) {\n";
    code << "            for (size_t j = jj; j < j_end; ++j) {\n";
    code << "                // Process element (i, j)\n";
    code << "            }\n";
    code << "        }\n";
    code << "    }\n";
    code << "}\n";
    rec.code_example = code.str();

    rec.implementation_notes.push_back(
        "Tile dimensions chosen for L2 cache (" +
        std::to_string(hw_.cache.l2_size_kb) + " KB)");
    rec.implementation_notes.push_back(
        "Consider adding L1 micro-tiling for better register utilization");

    return rec;
}

TilingRecommendation LoopTilingAdvisor::recommend_3d(
    size_t dim_x,
    size_t dim_y,
    size_t dim_z,
    size_t element_bytes,
    size_t stencil_radius
) {
    TilingRecommendation rec;

    // 3D requires careful tile sizing due to stencil halos

    size_t halo = 2 * stencil_radius;

    // L2 tile (primary focus for 3D)
    size_t l2_elements = tile_for_cache(hw_.cache.l2_size_kb, element_bytes, 2, 0.6);
    size_t cube_side = static_cast<size_t>(std::cbrt(static_cast<double>(l2_elements)));

    // Account for halo overhead
    cube_side = std::max(size_t(16), cube_side - halo);
    cube_side = (cube_side / 8) * 8;

    rec.l2_tile_size = cube_side * cube_side * cube_side;
    rec.l2_tile_dims = {cube_side, cube_side, cube_side};

    // L3 tile
    size_t l3_elements = tile_for_cache(hw_.cache.l3_size_kb, element_bytes, 2, 0.6);
    size_t l3_side = static_cast<size_t>(std::cbrt(static_cast<double>(l3_elements)));
    l3_side = std::max(size_t(32), l3_side - halo);
    l3_side = (l3_side / 16) * 16;

    rec.l3_tile_size = l3_side * l3_side * l3_side;
    rec.l3_tile_dims = {l3_side, l3_side, l3_side};

    rec.estimated_speedup = 1.5;
    rec.estimated_cache_efficiency = 0.70;
    rec.is_valid = (cube_side >= 8);

    // Generate code example
    std::ostringstream code;
    code << "// 3D tiled stencil with cache blocking\n";
    code << "const size_t TILE = " << cube_side << ";\n";
    code << "const size_t HALO = " << stencil_radius << ";\n\n";
    code << "for (size_t bz = HALO; bz < dim_z - HALO; bz += TILE) {\n";
    code << "    for (size_t by = HALO; by < dim_y - HALO; by += TILE) {\n";
    code << "        for (size_t bx = HALO; bx < dim_x - HALO; bx += TILE) {\n";
    code << "            // Process tile [bx:bx+TILE, by:by+TILE, bz:bz+TILE]\n";
    code << "            for (size_t z = bz; z < std::min(bz + TILE, dim_z - HALO); ++z) {\n";
    code << "                for (size_t y = by; y < std::min(by + TILE, dim_y - HALO); ++y) {\n";
    code << "                    for (size_t x = bx; x < std::min(bx + TILE, dim_x - HALO); ++x) {\n";
    code << "                        // Stencil computation at (x, y, z)\n";
    code << "                    }\n";
    code << "                }\n";
    code << "            }\n";
    code << "        }\n";
    code << "    }\n";
    code << "}\n";
    rec.code_example = code.str();

    rec.implementation_notes.push_back(
        "Halo of " + std::to_string(stencil_radius) + " elements per dimension");
    rec.implementation_notes.push_back(
        "Consider temporal blocking for iterative stencils");

    return rec;
}

TilingRecommendation LoopTilingAdvisor::recommend_from_metrics(
    size_t working_set_bytes,
    double l1_miss_rate,
    double l2_miss_rate,
    double l3_miss_rate,
    size_t element_bytes
) {
    TilingRecommendation rec;

    // Determine which cache level is the bottleneck
    std::string bottleneck;
    if (l1_miss_rate > 0.1) {
        bottleneck = "L1";
    } else if (l2_miss_rate > 0.1) {
        bottleneck = "L2";
    } else if (l3_miss_rate > 0.05) {
        bottleneck = "L3";
    } else {
        rec.is_valid = false;
        rec.validation_message = "Cache behavior is already efficient";
        return rec;
    }

    // Calculate tile sizes based on bottleneck
    size_t l1_kb = hw_.cache.l1d_size_kb;
    size_t l2_kb = hw_.cache.l2_size_kb;
    size_t l3_kb = hw_.cache.l3_size_kb;

    rec.l1_tile_size = (l1_kb * 1024 * 3 / 4) / element_bytes;
    rec.l2_tile_size = (l2_kb * 1024 * 3 / 4) / element_bytes;
    rec.l3_tile_size = (l3_kb * 1024 * 3 / 4) / element_bytes;

    // Estimate speedup based on miss rate reduction
    if (bottleneck == "L1") {
        rec.estimated_speedup = 1.0 + l1_miss_rate * 5.0;  // L1 misses are expensive
    } else if (bottleneck == "L2") {
        rec.estimated_speedup = 1.0 + l2_miss_rate * 3.0;
    } else {
        rec.estimated_speedup = 1.0 + l3_miss_rate * 10.0;  // DRAM access very expensive
    }

    rec.estimated_cache_efficiency = 1.0 - (l1_miss_rate * 0.1 + l2_miss_rate * 0.3 + l3_miss_rate * 0.6);
    rec.is_valid = true;

    rec.implementation_notes.push_back("Bottleneck detected at " + bottleneck + " cache");
    rec.implementation_notes.push_back(
        "Expected " + std::to_string(static_cast<int>((rec.estimated_speedup - 1.0) * 100)) +
        "% improvement from tiling");

    return rec;
}

size_t LoopTilingAdvisor::calculate_optimal_tile(
    size_t cache_size_kb,
    size_t element_bytes,
    size_t num_arrays,
    double cache_utilization_target
) {
    return tile_for_cache(cache_size_kb, element_bytes, num_arrays, cache_utilization_target);
}

std::string LoopTilingAdvisor::generate_tiled_loop_code(
    const TilingRecommendation& rec,
    const std::string& loop_var,
    const std::string& array_name
) {
    std::ostringstream code;

    code << "const size_t TILE_SIZE = " << rec.l2_tile_size << ";\n\n";
    code << "for (size_t tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {\n";
    code << "    size_t tile_end = std::min(tile_start + TILE_SIZE, n);\n";
    code << "    for (size_t " << loop_var << " = tile_start; "
         << loop_var << " < tile_end; ++" << loop_var << ") {\n";
    code << "        // Process " << array_name << "[" << loop_var << "]\n";
    code << "    }\n";
    code << "}\n";

    return code.str();
}

bool LoopTilingAdvisor::validate_tiling(
    const TilingRecommendation& rec,
    size_t total_elements
) {
    if (rec.l2_tile_size == 0) return false;
    if (rec.l2_tile_size > total_elements) return false;
    return true;
}

// ============================================================================
// Cache blocking analysis
// ============================================================================

CacheBlockingAnalysis analyze_cache_blocking(
    size_t working_set_bytes,
    const CacheInfo& cache,
    double l1_miss_rate,
    double l2_miss_rate,
    double l3_miss_rate
) {
    CacheBlockingAnalysis analysis;

    size_t l1_bytes = cache.l1d_size_kb * 1024;
    size_t l2_bytes = cache.l2_size_kb * 1024;
    size_t l3_bytes = cache.l3_size_kb * 1024;

    // Determine if blocking would help
    if (working_set_bytes <= l1_bytes && l1_miss_rate < 0.05) {
        analysis.needs_blocking = false;
        analysis.current_efficiency = 0.95;
        analysis.recommendations.push_back("Working set fits in L1 - blocking not needed");
        return analysis;
    }

    if (working_set_bytes <= l2_bytes && l2_miss_rate < 0.05) {
        analysis.needs_blocking = false;
        analysis.current_efficiency = 0.85;
        analysis.recommendations.push_back("Working set fits in L2 - blocking optional");
        return analysis;
    }

    // Blocking would help
    analysis.needs_blocking = true;

    // Calculate current efficiency
    analysis.current_efficiency =
        (1.0 - l1_miss_rate) * 0.4 +
        (1.0 - l2_miss_rate) * 0.35 +
        (1.0 - l3_miss_rate) * 0.25;

    // Determine optimal block size
    if (working_set_bytes > l3_bytes) {
        analysis.optimal_block_size = l3_bytes * 3 / 4;
        analysis.recommendations.push_back("Block for L3 cache (" +
            std::to_string(cache.l3_size_kb) + " KB)");
    } else if (working_set_bytes > l2_bytes) {
        analysis.optimal_block_size = l2_bytes * 3 / 4;
        analysis.recommendations.push_back("Block for L2 cache (" +
            std::to_string(cache.l2_size_kb) + " KB)");
    } else {
        analysis.optimal_block_size = l1_bytes * 3 / 4;
        analysis.recommendations.push_back("Block for L1 cache (" +
            std::to_string(cache.l1d_size_kb) + " KB)");
    }

    analysis.expected_efficiency = 0.85;

    return analysis;
}

// ============================================================================
// Matrix multiplication tiling
// ============================================================================

MatmulTiling recommend_matmul_tiling(
    size_t M,
    size_t N,
    size_t K,
    size_t element_bytes,
    const CacheInfo& cache,
    int simd_width_bytes
) {
    MatmulTiling tiling;

    size_t simd_elements = simd_width_bytes / element_bytes;

    // Register blocking: fit in SIMD registers
    // For FMA: need A panel, B panel, C block
    tiling.register_block_m = 4;
    tiling.register_block_n = simd_elements;  // One SIMD width

    // L1 blocking: fit A panel + B panel + C block in L1
    size_t l1_bytes = cache.l1d_size_kb * 1024;
    // A: m_tile * k_tile, B: k_tile * n_tile, C: m_tile * n_tile
    // Approximate: 3 * tile^2 * element_bytes < L1
    size_t l1_tile = static_cast<size_t>(std::sqrt(l1_bytes / (3.0 * element_bytes)));
    l1_tile = (l1_tile / simd_elements) * simd_elements;

    // L2 blocking
    size_t l2_bytes = cache.l2_size_kb * 1024;
    size_t l2_tile = static_cast<size_t>(std::sqrt(l2_bytes / (3.0 * element_bytes)));
    l2_tile = (l2_tile / 16) * 16;

    tiling.m_tile = std::min({l2_tile, M});
    tiling.n_tile = std::min({l2_tile, N});
    tiling.k_tile = std::min({l1_tile, K});

    // Estimate efficiency
    double arithmetic_intensity = 2.0 * M * N * K / (M * K + K * N + M * N) / element_bytes;
    tiling.expected_efficiency = std::min(0.9, arithmetic_intensity / 10.0);

    // Generate code example
    std::ostringstream code;
    code << "// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]\n";
    code << "const size_t M_TILE = " << tiling.m_tile << ";\n";
    code << "const size_t N_TILE = " << tiling.n_tile << ";\n";
    code << "const size_t K_TILE = " << tiling.k_tile << ";\n\n";
    code << "for (size_t jj = 0; jj < N; jj += N_TILE) {\n";
    code << "    for (size_t kk = 0; kk < K; kk += K_TILE) {\n";
    code << "        for (size_t ii = 0; ii < M; ii += M_TILE) {\n";
    code << "            // Micro-kernel: C[ii:ii+M_TILE, jj:jj+N_TILE] += \n";
    code << "            //              A[ii:ii+M_TILE, kk:kk+K_TILE] * B[kk:kk+K_TILE, jj:jj+N_TILE]\n";
    code << "        }\n";
    code << "    }\n";
    code << "}\n";
    tiling.code_example = code.str();

    return tiling;
}

// ============================================================================
// Stencil tiling
// ============================================================================

StencilTiling recommend_stencil_tiling(
    const std::vector<size_t>& grid_dims,
    size_t stencil_radius,
    size_t element_bytes,
    const CacheInfo& cache
) {
    StencilTiling tiling;

    size_t ndim = grid_dims.size();
    size_t halo = 2 * stencil_radius;

    // L2-focused tiling for stencils
    size_t l2_bytes = cache.l2_size_kb * 1024;
    size_t elements_per_tile = l2_bytes / (2 * element_bytes);  // Input + output

    // Calculate tile size per dimension
    double elements_per_dim = std::pow(static_cast<double>(elements_per_tile), 1.0 / ndim);
    size_t tile_side = static_cast<size_t>(elements_per_dim);
    tile_side = std::max(size_t(16), tile_side - halo);
    tile_side = (tile_side / 8) * 8;

    tiling.tile_dims.resize(ndim, tile_side);

    // For large stencils, consider temporal blocking
    if (stencil_radius >= 2 && ndim >= 2) {
        tiling.time_steps_per_tile = 2;
        tiling.use_diamond_tiling = true;
    }

    tiling.expected_efficiency = 0.75;

    // Generate code example for 2D
    std::ostringstream code;
    if (ndim == 2) {
        code << "// 2D stencil with cache blocking\n";
        code << "const size_t TILE = " << tile_side << ";\n";
        code << "const size_t R = " << stencil_radius << ";  // Stencil radius\n\n";
        code << "for (size_t by = R; by < ny - R; by += TILE) {\n";
        code << "    for (size_t bx = R; bx < nx - R; bx += TILE) {\n";
        code << "        for (size_t y = by; y < std::min(by + TILE, ny - R); ++y) {\n";
        code << "            for (size_t x = bx; x < std::min(bx + TILE, nx - R); ++x) {\n";
        code << "                out[y][x] = stencil_op(in, x, y, R);\n";
        code << "            }\n";
        code << "        }\n";
        code << "    }\n";
        code << "}\n";
    }
    tiling.code_example = code.str();

    return tiling;
}

}  // namespace simd_bench
