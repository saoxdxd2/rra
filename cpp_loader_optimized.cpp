#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <limits>
#include <atomic>
#include <pybind11/pybind11.h>

#include <immintrin.h>
#include "brain_isa.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================================
// ENTERPRISE CONFIGURATION & CONSTANTS
// =========================================================================
namespace Config {
    constexpr float EPSILON = 1e-8f;
    constexpr float THRESH_SPIKE = 0.5f;
    constexpr float MAX_RELu = 10.0f; // Clip value
}

namespace RuntimeABI {
    constexpr float PHENOTYPE_ABI_VERSION = 2.0f;
    constexpr int64_t FWD_SCALARS_MIN = 22;
}

namespace py = pybind11;

// =========================================================================
// CORE UTILITIES (Internal)
// =========================================================================
namespace Core {
    static float g_hpc_target_error = 0.05f;
    static float g_hpc_error_ema_decay = 0.95f;
    static float g_hpc_halt_gain = 0.35f;
    static float g_hpc_error_ema = 0.05f;
    static bool g_hpc_temporal_folding = true;
    static float g_hpc_fold_alpha = 0.25f;
    static int g_event_mode = 2; // 0=dense, 1=event-only, 2=auto
    static float g_event_density_threshold = 0.20f;
    static bool g_ttfs_enabled = true;
    static float g_ttfs_slope_threshold = 0.0f;

    inline void set_hpc_temporal_folding(bool enabled, float fold_alpha) {
        g_hpc_temporal_folding = enabled;
        if (!std::isfinite(fold_alpha)) {
            fold_alpha = 0.0f;
        }
        g_hpc_fold_alpha = std::max(0.0f, std::min(1.0f, fold_alpha));
    }

    inline void set_event_runtime(
        int event_mode,
        float event_density_threshold,
        bool ttfs_enabled,
        float ttfs_slope_threshold
    ) {
        if (event_mode < 0 || event_mode > 2) {
            event_mode = 2;
        }
        if (!std::isfinite(event_density_threshold)) {
            event_density_threshold = 0.20f;
        }
        if (!std::isfinite(ttfs_slope_threshold)) {
            ttfs_slope_threshold = 0.0f;
        }
        g_event_mode = event_mode;
        g_event_density_threshold = std::max(0.0f, std::min(1.0f, event_density_threshold));
        g_ttfs_enabled = ttfs_enabled;
        g_ttfs_slope_threshold = std::max(0.0f, ttfs_slope_threshold);
    }

    inline float interp_at(
        const float* x_ptr,
        float t_target,
        int T_max,
        int c_idx,
        int x_stride_t,
        int x_stride_d
    ) {
        const int t0 = static_cast<int>(std::floor(t_target));
        const int t1 = t0 + 1;
        const float alpha = t_target - static_cast<float>(t0);
        float val = 0.0f;
        if (t0 >= 0 && t0 < T_max) {
            val += (1.0f - alpha) * x_ptr[t0 * x_stride_t + c_idx * x_stride_d];
        }
        if (t1 >= 0 && t1 < T_max) {
            val += alpha * x_ptr[t1 * x_stride_t + c_idx * x_stride_d];
        }
        return val;
    }

    // Helper: BF16 -> FP32 conversion
    inline float bf16_to_f32(uint16_t b) {
        uint32_t b32 = (uint32_t)b << 16;
        float f;
        std::memcpy(&f, &b32, 4);
        return f;
    }

    // Helper: Shared RAM Address Calculation
    // Used by DCLS, LIF, and Cognitive Cycle kernels
    inline uint32_t compute_ram_address(
        const float* x_ptr,      // [B, T, D_in] or similar
        const float* d_ptr,      // [M, K] or [K] depending on caller stride
        const int64_t* conn_ptr, // [M, K]
        int t_current,
        int T_max,
        int D_in, 
        int K,
        int x_stride_t,          // Stride for time dimension
        int x_stride_d           // Stride for feature dimension
    ) {
        uint32_t addr_dense = 0;
        uint32_t addr_event = 0;
        int active_count = 0;
        for (int k = 0; k < K; k++) {
            float delay = d_ptr[k];
            int64_t c_idx = conn_ptr[k];

            // Safety Bounds
            if (c_idx < 0) c_idx = 0;
            if (c_idx >= D_in) c_idx = D_in - 1;

            float t_target = (float)t_current - delay;
            float val = interp_at(x_ptr, t_target, T_max, static_cast<int>(c_idx), x_stride_t, x_stride_d);
            const float prev_raw = interp_at(
                x_ptr,
                t_target - 1.0f,
                T_max,
                static_cast<int>(c_idx),
                x_stride_t,
                x_stride_d
            );
            if (g_hpc_temporal_folding) {
                val = val + g_hpc_fold_alpha * (val - prev_raw);
            }

            const float rise = val - prev_raw;
            bool is_active = (val > 0.0f);
            if (g_ttfs_enabled) {
                is_active = is_active && (rise >= g_ttfs_slope_threshold);
            }
            if (is_active) {
                addr_dense |= (uint32_t(1) << k);
                active_count++;
            }
            const bool is_event = is_active && (rise > 0.0f) && (prev_raw <= 0.0f);
            if (is_event) {
                addr_event |= (uint32_t(1) << k);
            }
        }
        if (g_event_mode == 0) {
            return addr_dense;
        }
        if (g_event_mode == 1) {
            return addr_event;
        }
        const float density = static_cast<float>(active_count) / static_cast<float>(std::max(1, K));
        if (density <= g_event_density_threshold) {
            return (addr_event != 0 || addr_dense == 0) ? addr_event : addr_dense;
        }
        return addr_dense;
    }
}

// =========================================================================
// PERF COUNTERS (Kernel-level FLOP/byte accounting)
// =========================================================================
namespace Perf {
    static std::atomic<bool> g_enabled{true};
    static std::atomic<uint64_t> g_total_flops{0};
    static std::atomic<uint64_t> g_total_bytes{0};
    static std::atomic<uint64_t> g_total_calls{0};

    static std::atomic<uint64_t> g_parallel_scan_calls{0};
    static std::atomic<uint64_t> g_quantized_matmul_calls{0};
    static std::atomic<uint64_t> g_rms_norm_calls{0};
    static std::atomic<uint64_t> g_fused_rms_mean_calls{0};
    static std::atomic<uint64_t> g_dcls_addr_calls{0};
    static std::atomic<uint64_t> g_dcls_lookup_calls{0};
    static std::atomic<uint64_t> g_dcls_lookup_int8_calls{0};
    static std::atomic<uint64_t> g_fused_lif_calls{0};

    static std::atomic<uint64_t> g_cache_lookup_calls{0};
    static std::atomic<uint64_t> g_ademamix_calls{0};
    static std::atomic<uint64_t> g_batched_ademamix_calls{0};
    static std::atomic<uint64_t> g_mes_super_calls{0};
    static std::atomic<uint64_t> g_survival_update_calls{0};
    static std::atomic<uint64_t> g_survival_mask_calls{0};
    static std::atomic<uint64_t> g_survival_losses_calls{0};
    static std::atomic<uint64_t> g_fused_cognitive_calls{0};
    static std::atomic<uint64_t> g_lgh_calls{0};
    static std::atomic<uint64_t> g_lgh_tsc_cycles{0};
    static std::atomic<uint64_t> g_lgh_pulse_ops{0};
    static std::atomic<uint64_t> g_lgh_temporal_folds{0};
    static std::atomic<uint64_t> g_full_forward_calls{0};

    inline uint64_t sat_from_ld(long double v) {
        if (!(v > 0.0L)) {
            return 0ULL;
        }
        const long double hi = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        if (v >= hi) {
            return std::numeric_limits<uint64_t>::max();
        }
        return static_cast<uint64_t>(v);
    }

    inline void add(uint64_t flops, uint64_t bytes) {
        if (!g_enabled.load(std::memory_order_relaxed)) {
            return;
        }
        g_total_flops.fetch_add(flops, std::memory_order_relaxed);
        g_total_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_total_calls.fetch_add(1ULL, std::memory_order_relaxed);
    }

    inline void set_enabled(bool enabled) {
        g_enabled.store(enabled, std::memory_order_relaxed);
    }

    inline void reset() {
        g_total_flops.store(0ULL, std::memory_order_relaxed);
        g_total_bytes.store(0ULL, std::memory_order_relaxed);
        g_total_calls.store(0ULL, std::memory_order_relaxed);
        g_parallel_scan_calls.store(0ULL, std::memory_order_relaxed);
        g_quantized_matmul_calls.store(0ULL, std::memory_order_relaxed);
        g_rms_norm_calls.store(0ULL, std::memory_order_relaxed);
        g_fused_rms_mean_calls.store(0ULL, std::memory_order_relaxed);
        g_dcls_addr_calls.store(0ULL, std::memory_order_relaxed);
        g_dcls_lookup_calls.store(0ULL, std::memory_order_relaxed);
        g_dcls_lookup_int8_calls.store(0ULL, std::memory_order_relaxed);
        g_fused_lif_calls.store(0ULL, std::memory_order_relaxed);

        g_cache_lookup_calls.store(0ULL, std::memory_order_relaxed);
        g_ademamix_calls.store(0ULL, std::memory_order_relaxed);
        g_batched_ademamix_calls.store(0ULL, std::memory_order_relaxed);
        g_mes_super_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_update_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_mask_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_losses_calls.store(0ULL, std::memory_order_relaxed);
        g_fused_cognitive_calls.store(0ULL, std::memory_order_relaxed);
        g_lgh_calls.store(0ULL, std::memory_order_relaxed);
        g_lgh_tsc_cycles.store(0ULL, std::memory_order_relaxed);
        g_lgh_pulse_ops.store(0ULL, std::memory_order_relaxed);
        g_lgh_temporal_folds.store(0ULL, std::memory_order_relaxed);
    }

    py::dict snapshot() {
        py::dict out;
        out["enabled"] = g_enabled.load(std::memory_order_relaxed);
        out["total_flops"] = py::int_(g_total_flops.load(std::memory_order_relaxed));
        out["total_bytes"] = py::int_(g_total_bytes.load(std::memory_order_relaxed));
        out["total_calls"] = py::int_(g_total_calls.load(std::memory_order_relaxed));
        out["parallel_scan_calls"] = py::int_(g_parallel_scan_calls.load(std::memory_order_relaxed));
        out["quantized_matmul_calls"] = py::int_(g_quantized_matmul_calls.load(std::memory_order_relaxed));
        out["rms_norm_calls"] = py::int_(g_rms_norm_calls.load(std::memory_order_relaxed));
        out["fused_rms_mean_calls"] = py::int_(g_fused_rms_mean_calls.load(std::memory_order_relaxed));
        out["dcls_addr_calls"] = py::int_(g_dcls_addr_calls.load(std::memory_order_relaxed));
        out["dcls_lookup_calls"] = py::int_(g_dcls_lookup_calls.load(std::memory_order_relaxed));
        out["dcls_lookup_int8_calls"] = py::int_(g_dcls_lookup_int8_calls.load(std::memory_order_relaxed));
        out["fused_lif_calls"] = py::int_(g_fused_lif_calls.load(std::memory_order_relaxed));

        out["cache_lookup_calls"] = py::int_(g_cache_lookup_calls.load(std::memory_order_relaxed));
        out["ademamix_calls"] = py::int_(g_ademamix_calls.load(std::memory_order_relaxed));
        out["batched_ademamix_calls"] = py::int_(g_batched_ademamix_calls.load(std::memory_order_relaxed));
        out["mes_super_calls"] = py::int_(g_mes_super_calls.load(std::memory_order_relaxed));
        out["survival_update_calls"] = py::int_(g_survival_update_calls.load(std::memory_order_relaxed));
        out["survival_mask_calls"] = py::int_(g_survival_mask_calls.load(std::memory_order_relaxed));
        out["survival_losses_calls"] = py::int_(g_survival_losses_calls.load(std::memory_order_relaxed));
        out["fused_cognitive_calls"] = py::int_(g_fused_cognitive_calls.load(std::memory_order_relaxed));
        out["lgh_calls"] = py::int_(g_lgh_calls.load(std::memory_order_relaxed));
        out["lgh_tsc_cycles"] = py::int_(g_lgh_tsc_cycles.load(std::memory_order_relaxed));
        out["lgh_pulse_ops"] = py::int_(g_lgh_pulse_ops.load(std::memory_order_relaxed));
        out["lgh_temporal_folds"] = py::int_(g_lgh_temporal_folds.load(std::memory_order_relaxed));
        return out;
    }
}

// =========================================================================
// API BOUNDARY HELPERS
// =========================================================================
inline at::Tensor ensure_contig(const at::Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}

inline void check_cpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor.");
}

inline void check_dim(const at::Tensor& t, int64_t expected, const char* name) {
    TORCH_CHECK(t.dim() == expected, name, " must have ", expected, " dimensions, got ", t.dim(), ".");
}

inline void check_dtype(const at::Tensor& t, at::ScalarType expected, const char* name) {
    TORCH_CHECK(t.scalar_type() == expected, name, " has invalid dtype.");
}

// =========================================================================
// KERNELS
// =========================================================================

// -------------------------------------------------------------------------
// KERNEL: Parallel Scan (Iterative Association)
// -------------------------------------------------------------------------
torch::Tensor parallel_scan(torch::Tensor u, torch::Tensor decay) {
    check_cpu(u, "u");
    check_cpu(decay, "decay");
    check_dim(u, 3, "u");
    check_dim(decay, 3, "decay");
    check_dtype(u, at::kFloat, "u");
    check_dtype(decay, at::kFloat, "decay");
    TORCH_CHECK(u.sizes() == decay.sizes(), "u and decay must have identical shapes.");

    auto u_in = ensure_contig(u);
    auto d_in = ensure_contig(decay);
    
    int B = u_in.size(0);
    int T = u_in.size(1);
    int D = u_in.size(2);
    Perf::g_parallel_scan_calls.fetch_add(1ULL, std::memory_order_relaxed);
    
    auto h = torch::empty_like(u_in);
    
    float* u_ptr = u_in.data_ptr<float>();
    float* d_ptr = d_in.data_ptr<float>();
    float* h_ptr = h.data_ptr<float>();
    
    // Enterprise Threading: Let logic decide
    // If B is small, parallelize D? Simple approach: Parallelize B if B >= threads/2
    // Else parallelize D.
    
    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif

    if (B >= max_threads) {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B; b++) {
            // Thread-local scratchpad using vector (reallocation amortized)
            // Static allows reuse across calls if thread ID matches (OS dependent)
            // Safety: Just use local vector, small overhead compared to logic
            std::vector<float> state(D, 0.0f);
            
            for (int t = 0; t < T; t++) {
                int offset = b * T * D + t * D;
                for (int d = 0; d < D; d++) {
                    state[d] = d_ptr[offset + d] * state[d] + u_ptr[offset + d];
                    h_ptr[offset + d] = state[d];
                }
            }
        }
    } else {
        // Small batch optimization: Parallelize over D
        // Note: This requires locking T sequential
        for (int b = 0; b < B; b++) {
            // Can't easily parallelize T due to dependency
            // Parallelize D? efficient for large D
            #pragma omp parallel 
            {
                // Each thread handles chunk of D
                #pragma omp for schedule(static) nowait
                for (int d = 0; d < D; d++) {
                    float state = 0.0f;
                    for (int t = 0; t < T; t++) {
                        int idx = b * T * D + t * D + d;
                        state = d_ptr[idx] * state + u_ptr[idx];
                        h_ptr[idx] = state;
                    }
                }
            }
        }
    }
    const long double elems = static_cast<long double>(B) * static_cast<long double>(T) * static_cast<long double>(D);
    Perf::add(
        Perf::sat_from_ld(2.0L * elems),   // decay*state + input
        Perf::sat_from_ld(12.0L * elems)   // read u, read decay, write h
    );
    return h;
}

// -------------------------------------------------------------------------
// KERNEL: Quantized Matmul (INT8)
// -------------------------------------------------------------------------
torch::Tensor quantized_matmul(torch::Tensor x, torch::Tensor w_q, torch::Tensor scale_w, torch::Tensor bias) {
    check_cpu(x, "x");
    check_cpu(w_q, "w_q");
    check_cpu(scale_w, "scale_w");
    check_cpu(bias, "bias");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions.");
    check_dim(w_q, 2, "w_q");
    check_dim(scale_w, 1, "scale_w");
    check_dim(bias, 1, "bias");
    check_dtype(x, at::kFloat, "x");
    check_dtype(w_q, at::kChar, "w_q");
    check_dtype(scale_w, at::kFloat, "scale_w");
    check_dtype(bias, at::kFloat, "bias");

    auto x_c = ensure_contig(x);
    auto w_c = ensure_contig(w_q);
    auto s_c = ensure_contig(scale_w);
    auto b_c = ensure_contig(bias);
    
    int64_t Din = x_c.size(-1);
    int64_t N = x_c.numel() / Din;
    int64_t Dout = w_c.size(0);
    Perf::g_quantized_matmul_calls.fetch_add(1ULL, std::memory_order_relaxed);
    
    auto y = torch::empty({N, Dout}, x_c.options().dtype(torch::kFloat32));
    
    float* x_ptr = x_c.data_ptr<float>();
    int8_t* w_ptr = w_c.data_ptr<int8_t>();
    float* s_ptr = s_c.data_ptr<float>();
    float* b_ptr = b_c.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    
    #pragma omp parallel
    {
        // Thread-local quantization buffer
        // Note: static thread_local is safe here as long as no recursion
        static thread_local std::vector<int8_t> xi_q;
        if ((int64_t)xi_q.size() < Din) xi_q.resize(Din);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            float* xi = x_ptr + i * Din;
            float* yi = y_ptr + i * Dout;
            
            // 1. Dynamic Quantization of Input
            float x_max0 = 1e-6f;
            float x_max1 = 1e-6f;
            float x_max2 = 1e-6f;
            float x_max3 = 1e-6f;
            int64_t k = 0;
            for (; k + 3 < Din; k += 4) {
                x_max0 = std::max(x_max0, std::abs(xi[k]));
                x_max1 = std::max(x_max1, std::abs(xi[k + 1]));
                x_max2 = std::max(x_max2, std::abs(xi[k + 2]));
                x_max3 = std::max(x_max3, std::abs(xi[k + 3]));
            }
            float x_max = std::max(std::max(x_max0, x_max1), std::max(x_max2, x_max3));
            for (; k < Din; k++) {
                x_max = std::max(x_max, std::abs(xi[k]));
            }
            float x_scale = x_max / 127.0f;
            float inv_x_scale = 1.0f / (x_scale + 1e-9f);
            
            for (int64_t kq = 0; kq < Din; kq++) {
                xi_q[kq] = (int8_t)std::round(xi[kq] * inv_x_scale);
            }
            
            // 2. Matmul
            for (int64_t j = 0; j < Dout; j++) {
                int8_t* wj = w_ptr + j * Din;
                int32_t acc0 = 0;
                int32_t acc1 = 0;
                int32_t acc2 = 0;
                int32_t acc3 = 0;
                int64_t kk = 0;
                for (; kk + 15 < Din; kk += 16) {
                    acc0 += (int32_t)xi_q[kk] * (int32_t)wj[kk];
                    acc1 += (int32_t)xi_q[kk + 1] * (int32_t)wj[kk + 1];
                    acc2 += (int32_t)xi_q[kk + 2] * (int32_t)wj[kk + 2];
                    acc3 += (int32_t)xi_q[kk + 3] * (int32_t)wj[kk + 3];

                    acc0 += (int32_t)xi_q[kk + 4] * (int32_t)wj[kk + 4];
                    acc1 += (int32_t)xi_q[kk + 5] * (int32_t)wj[kk + 5];
                    acc2 += (int32_t)xi_q[kk + 6] * (int32_t)wj[kk + 6];
                    acc3 += (int32_t)xi_q[kk + 7] * (int32_t)wj[kk + 7];

                    acc0 += (int32_t)xi_q[kk + 8] * (int32_t)wj[kk + 8];
                    acc1 += (int32_t)xi_q[kk + 9] * (int32_t)wj[kk + 9];
                    acc2 += (int32_t)xi_q[kk + 10] * (int32_t)wj[kk + 10];
                    acc3 += (int32_t)xi_q[kk + 11] * (int32_t)wj[kk + 11];

                    acc0 += (int32_t)xi_q[kk + 12] * (int32_t)wj[kk + 12];
                    acc1 += (int32_t)xi_q[kk + 13] * (int32_t)wj[kk + 13];
                    acc2 += (int32_t)xi_q[kk + 14] * (int32_t)wj[kk + 14];
                    acc3 += (int32_t)xi_q[kk + 15] * (int32_t)wj[kk + 15];
                }
                int32_t acc = acc0 + acc1 + acc2 + acc3;
                for (; kk < Din; kk++) {
                    acc += (int32_t)xi_q[kk] * (int32_t)wj[kk];
                }
                
                // 3. Dequantize
                yi[j] = ((float)acc * x_scale * s_ptr[j]) + b_ptr[j];
            }
        }
    }
    const long double n = static_cast<long double>(N);
    const long double din = static_cast<long double>(Din);
    const long double dout = static_cast<long double>(Dout);
    const long double flops = n * (2.0L * din + dout * (2.0L * din + 3.0L));
    const long double bytes = n * (
        4.0L * din +                   // read x
        dout * din +                   // read int8 weights
        4.0L * dout +                  // read scale_w
        4.0L * dout +                  // read bias
        4.0L * dout                    // write y
    );
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return y;
}

// -------------------------------------------------------------------------
// KERNEL: RMSNorm
// -------------------------------------------------------------------------
torch::Tensor rms_norm(torch::Tensor x, float eps) {
    check_cpu(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension.");
    check_dtype(x, at::kFloat, "x");

    auto x_c = ensure_contig(x);
    auto y = torch::empty_like(x_c);
    
    int64_t D = x_c.size(-1);
    int64_t N = x_c.numel() / D;
    Perf::g_rms_norm_calls.fetch_add(1ULL, std::memory_order_relaxed);
    
    float* x_ptr = x_c.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float* xi = x_ptr + i * D;
        float* yi = y_ptr + i * D;
        
        float sum_sq = 0.0f;
        for (int j = 0; j < D; j++) sum_sq += xi[j] * xi[j];
        
        float rms = std::sqrt(sum_sq / D + eps);
        float inv_rms = 1.0f / rms;
        
        for (int j = 0; j < D; j++) yi[j] = xi[j] * inv_rms;
    }
    const long double n = static_cast<long double>(N);
    const long double d = static_cast<long double>(D);
    Perf::add(
        Perf::sat_from_ld(n * (3.0L * d + 2.0L)),
        Perf::sat_from_ld(n * (12.0L * d))
    );
    return y;
}

// -------------------------------------------------------------------------
// KERNEL: Fused RMSNorm + Regional Mean
// Input:
//   - [B, T, D]      -> output [B, T, D] (RMSNorm)
//   - [B, T, R, D, C] -> output [B, T, D*C] (mean over R, then RMSNorm)
// -------------------------------------------------------------------------
torch::Tensor fused_rms_mean(torch::Tensor x_seq) {
    check_cpu(x_seq, "x_seq");
    check_dtype(x_seq, at::kFloat, "x_seq");
    TORCH_CHECK(
        x_seq.dim() == 3 || x_seq.dim() == 5,
        "x_seq must be rank-3 [B,T,D] or rank-5 [B,T,R,D,C]."
    );

    auto x_c = ensure_contig(x_seq);
    Perf::g_fused_rms_mean_calls.fetch_add(1ULL, std::memory_order_relaxed);
    if (x_c.dim() == 3) {
        const int64_t B = x_c.size(0);
        const int64_t T = x_c.size(1);
        const int64_t D = x_c.size(2);
        auto out = torch::empty_like(x_c);
        const float* x_ptr = x_c.data_ptr<float>();
        float* o_ptr = out.data_ptr<float>();

        const int64_t rows = B * T;
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < rows; row++) {
            const int64_t base = row * D;
            float sum_sq = 0.0f;
            for (int64_t d = 0; d < D; d++) {
                const float v = x_ptr[base + d];
                sum_sq += v * v;
            }
            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + 1e-8f);
            for (int64_t d = 0; d < D; d++) {
                o_ptr[base + d] = x_ptr[base + d] * inv_rms;
            }
        }
        const long double rows_ld = static_cast<long double>(rows);
        const long double d_ld = static_cast<long double>(D);
        Perf::add(
            Perf::sat_from_ld(rows_ld * (3.0L * d_ld + 2.0L)),
            Perf::sat_from_ld(rows_ld * (12.0L * d_ld))
        );
        return out;
    }

    const int64_t B = x_c.size(0);
    const int64_t T = x_c.size(1);
    const int64_t R = x_c.size(2);
    const int64_t D = x_c.size(3);
    const int64_t C = x_c.size(4);
    const int64_t M = D * C;
    TORCH_CHECK(R > 0, "R must be > 0 for fused_rms_mean.");

    auto out = torch::empty({B, T, M}, x_c.options());
    const float* x_ptr = x_c.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();

    const int64_t rows = B * T;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; row++) {
        const int64_t b = row / T;
        const int64_t t = row % T;
        const int64_t in_bt_base = (b * T + t) * R * M;
        const int64_t out_bt_base = row * M;

        float sum_sq = 0.0f;
        for (int64_t m = 0; m < M; m++) {
            float sum_r = 0.0f;
            for (int64_t r = 0; r < R; r++) {
                sum_r += x_ptr[in_bt_base + r * M + m];
            }
            const float mean_r = sum_r / static_cast<float>(R);
            o_ptr[out_bt_base + m] = mean_r;
            sum_sq += mean_r * mean_r;
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(M) + 1e-8f);
        for (int64_t m = 0; m < M; m++) {
            o_ptr[out_bt_base + m] *= inv_rms;
        }
    }
    const long double rows_ld = static_cast<long double>(rows);
    const long double m_ld = static_cast<long double>(M);
    const long double r_ld = static_cast<long double>(R);
    Perf::add(
        Perf::sat_from_ld(rows_ld * (m_ld * (r_ld + 4.0L) + 2.0L)),
        Perf::sat_from_ld(rows_ld * (4.0L * r_ld * m_ld + 12.0L * m_ld))
    );
    return out;
}

// -------------------------------------------------------------------------
// KERNEL: DCLS RAM Lookup
// -------------------------------------------------------------------------
torch::Tensor dcls_ram_addresses(torch::Tensor x, torch::Tensor delays, torch::Tensor connections, int current_t) {
    check_cpu(x, "x");
    check_cpu(delays, "delays");
    check_cpu(connections, "connections");
    check_dim(x, 3, "x");
    check_dim(delays, 2, "delays");
    check_dim(connections, 2, "connections");
    check_dtype(x, at::kFloat, "x");
    check_dtype(delays, at::kFloat, "delays");
    check_dtype(connections, at::kLong, "connections");

    auto x_c = ensure_contig(x);
    auto d_c = ensure_contig(delays);
    auto c_c = ensure_contig(connections);
    
    int B = x_c.size(0); 
    int T = x_c.size(1); 
    int D_in = x_c.size(2);
    int M = d_c.size(0); 
    int K = d_c.size(1);
    Perf::g_dcls_addr_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(K > 0 && K <= 30, "K must be in [1, 30] for safe bit addressing.");
    TORCH_CHECK(c_c.size(0) == M && c_c.size(1) == K, "connections shape must match delays shape [M, K].");
    TORCH_CHECK(current_t >= 0 && current_t < T, "current_t out of range.");

    auto addresses = torch::empty({B, M}, x_c.options().dtype(torch::kLong));
    
    float* x_ptr = x_c.data_ptr<float>();
    float* d_ptr = d_c.data_ptr<float>();
    int64_t* c_ptr = c_c.data_ptr<int64_t>();
    int64_t* a_ptr = addresses.data_ptr<int64_t>();
    
    const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(M);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; idx++) {
        const int b = static_cast<int>(idx / M);
        const int m = static_cast<int>(idx % M);
        uint32_t addr = Core::compute_ram_address(
            x_ptr + b * T * D_in, // Base of x for this batch
            d_ptr + m * K,        // Base of delays for this neuron
            c_ptr + m * K,        // Base of conns for this neuron
            current_t, T, D_in, K,
            D_in, 1               // Strides: T stride = D_in, D stride = 1
        );
        a_ptr[b * M + m] = static_cast<int64_t>(addr);
    }
    const long double total_ld = static_cast<long double>(total);
    const long double k_ld = static_cast<long double>(K);
    Perf::add(
        Perf::sat_from_ld(total_ld * (8.0L * k_ld)),
        Perf::sat_from_ld(total_ld * (16.0L * k_ld + 8.0L))
    );
    return addresses;
}

torch::Tensor dcls_ram_lookup(torch::Tensor x, torch::Tensor delays, torch::Tensor ram_tables, torch::Tensor connections, int current_t) {
    check_cpu(ram_tables, "ram_tables");
    check_dim(ram_tables, 2, "ram_tables");
    check_dtype(ram_tables, at::kFloat, "ram_tables");

    auto t_c = ensure_contig(ram_tables);
    int M = t_c.size(0);
    int ram_size = t_c.size(1);
    TORCH_CHECK(ram_size > 0, "ram_tables second dimension must be > 0.");

    auto addresses = dcls_ram_addresses(x, delays, connections, current_t);
    auto a_c = ensure_contig(addresses);
    TORCH_CHECK(a_c.size(1) == M, "Address M dimension must match ram_tables.");

    auto output = torch::empty({a_c.size(0), M}, t_c.options());
    float* t_ptr = t_c.data_ptr<float>();
    int64_t* a_ptr = a_c.data_ptr<int64_t>();
    float* out_ptr = output.data_ptr<float>();

    int B = (int)a_c.size(0);
    Perf::g_dcls_lookup_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(M);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; idx++) {
        const int b = static_cast<int>(idx / M);
        const int m = static_cast<int>(idx % M);
        int64_t addr = a_ptr[b * M + m];
        if (addr < 0) addr = 0;
        out_ptr[b * M + m] = t_ptr[m * ram_size + (addr % ram_size)];
    }
    const long double total_ld = static_cast<long double>(total);
    Perf::add(
        Perf::sat_from_ld(total_ld),
        Perf::sat_from_ld(16.0L * total_ld)
    );
    return output;
}

torch::Tensor dcls_ram_lookup_int8(
    torch::Tensor x,
    torch::Tensor delays,
    torch::Tensor ram_tables_q,
    torch::Tensor scales,
    torch::Tensor connections,
    int current_t
) {
    check_cpu(ram_tables_q, "ram_tables_q");
    check_cpu(scales, "scales");
    check_dim(ram_tables_q, 2, "ram_tables_q");
    check_dim(scales, 1, "scales");
    check_dtype(ram_tables_q, at::kChar, "ram_tables_q");
    check_dtype(scales, at::kFloat, "scales");

    auto q_c = ensure_contig(ram_tables_q);
    auto s_c = ensure_contig(scales);
    const int M = q_c.size(0);
    const int ram_size = q_c.size(1);
    TORCH_CHECK(ram_size > 0, "ram_tables_q second dimension must be > 0.");
    TORCH_CHECK(s_c.size(0) == M, "scales must have shape [M].");

    auto addresses = dcls_ram_addresses(x, delays, connections, current_t);
    auto a_c = ensure_contig(addresses);
    TORCH_CHECK(a_c.size(1) == M, "Address M dimension must match ram_tables_q.");

    auto output = torch::empty({a_c.size(0), M}, x.options().dtype(torch::kFloat32));
    const int8_t* q_ptr = q_c.data_ptr<int8_t>();
    const float* sc_ptr = s_c.data_ptr<float>();
    const int64_t* a_ptr = a_c.data_ptr<int64_t>();
    float* out_ptr = output.data_ptr<float>();

    const int B = static_cast<int>(a_c.size(0));
    Perf::g_dcls_lookup_int8_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(M);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; idx++) {
        const int b = static_cast<int>(idx / M);
        const int m = static_cast<int>(idx % M);
        int64_t addr = a_ptr[b * M + m];
        if (addr < 0) {
            addr = 0;
        }
        const int64_t slot = addr % ram_size;
        const int8_t qv = q_ptr[m * ram_size + slot];
        out_ptr[b * M + m] = static_cast<float>(qv) * sc_ptr[m];
    }
    const long double total_ld = static_cast<long double>(total);
    Perf::add(
        Perf::sat_from_ld(total_ld),
        Perf::sat_from_ld(17.0L * total_ld)
    );
    return output;
}

// -------------------------------------------------------------------------
// KERNEL: DCLS Backward (grad_x + grad_tables)
// -------------------------------------------------------------------------
std::vector<torch::Tensor> dcls_backward(
    torch::Tensor x_seq,
    torch::Tensor delays,
    torch::Tensor connections,
    torch::Tensor grad_output,
    torch::Tensor addresses,
    int64_t current_t,
    int64_t ram_size,
    bool need_grad_x,
    bool need_grad_tables
) {
    check_cpu(x_seq, "x_seq");
    check_cpu(delays, "delays");
    check_cpu(connections, "connections");
    check_cpu(grad_output, "grad_output");
    check_cpu(addresses, "addresses");
    check_dim(x_seq, 3, "x_seq");
    check_dim(delays, 2, "delays");
    check_dim(connections, 2, "connections");
    check_dim(grad_output, 2, "grad_output");
    check_dim(addresses, 2, "addresses");
    check_dtype(x_seq, at::kFloat, "x_seq");
    check_dtype(delays, at::kFloat, "delays");
    check_dtype(connections, at::kLong, "connections");
    check_dtype(grad_output, at::kFloat, "grad_output");
    check_dtype(addresses, at::kLong, "addresses");
    TORCH_CHECK(ram_size > 0, "ram_size must be > 0.");

    auto x_c = ensure_contig(x_seq);
    auto d_c = ensure_contig(delays);
    auto c_c = ensure_contig(connections);
    auto go_c = ensure_contig(grad_output);
    auto a_c = ensure_contig(addresses);

    const int64_t B = x_c.size(0);
    const int64_t T = x_c.size(1);
    const int64_t D = x_c.size(2);
    const int64_t M = d_c.size(0);
    const int64_t K = d_c.size(1);
    TORCH_CHECK(K > 0 && K <= 30, "K must be in [1, 30] for safe bit addressing.");
    TORCH_CHECK(c_c.size(0) == M && c_c.size(1) == K, "connections shape must match delays shape [M, K].");
    TORCH_CHECK(go_c.size(0) == B && go_c.size(1) == M, "grad_output must have shape [B, M].");
    TORCH_CHECK(a_c.size(0) == B && a_c.size(1) == M, "addresses must have shape [B, M].");
    TORCH_CHECK(current_t >= 0 && current_t < T, "current_t out of range.");

    auto grad_x = need_grad_x ? torch::zeros_like(x_c) : torch::empty({0}, x_c.options());
    auto grad_tables = need_grad_tables
        ? torch::zeros({M, ram_size}, go_c.options())
        : torch::empty({0}, go_c.options());

    const float* x_ptr = x_c.data_ptr<float>();
    const float* d_ptr = d_c.data_ptr<float>();
    const int64_t* c_ptr = c_c.data_ptr<int64_t>();
    const float* go_ptr = go_c.data_ptr<float>();
    const int64_t* a_ptr = a_c.data_ptr<int64_t>();
    float* gx_ptr = need_grad_x ? grad_x.data_ptr<float>() : nullptr;
    float* gt_ptr = need_grad_tables ? grad_tables.data_ptr<float>() : nullptr;

    std::vector<int64_t> t_indices(M * K, 0);
    std::vector<int64_t> c_indices(M * K, 0);
    for (int64_t m = 0; m < M; m++) {
        for (int64_t k = 0; k < K; k++) {
            const int64_t idx = m * K + k;
            float t_target = static_cast<float>(current_t) - d_ptr[idx];
            int64_t t_idx = static_cast<int64_t>(t_target); // torch.long() truncation behavior
            if (t_idx < 0) t_idx = 0;
            if (t_idx >= T) t_idx = T - 1;
            int64_t c_idx = c_ptr[idx];
            if (c_idx < 0) c_idx = 0;
            if (c_idx >= D) c_idx = D - 1;
            t_indices[idx] = t_idx;
            c_indices[idx] = c_idx;
        }
    }

    if (need_grad_x) {
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < B; b++) {
            std::vector<float> abs_vals(K, 0.0f);
            const int64_t x_batch_base = b * T * D;
            const int64_t go_batch_base = b * M;
            for (int64_t m = 0; m < M; m++) {
                const float g_out = go_ptr[go_batch_base + m];
                float denom = 1e-8f;
                const int64_t mk_base = m * K;
                for (int64_t k = 0; k < K; k++) {
                    const int64_t idx = mk_base + k;
                    const int64_t flat = x_batch_base + t_indices[idx] * D + c_indices[idx];
                    const float aval = std::abs(x_ptr[flat]);
                    abs_vals[k] = aval;
                    denom += aval;
                }
                for (int64_t k = 0; k < K; k++) {
                    const int64_t idx = mk_base + k;
                    const int64_t flat = x_batch_base + t_indices[idx] * D + c_indices[idx];
                    gx_ptr[flat] += g_out * (abs_vals[k] / denom);
                }
            }
        }
    }

    if (need_grad_tables) {
        const int64_t total = B * M;
        #pragma omp parallel for schedule(static)
        for (int64_t idx = 0; idx < total; idx++) {
            const int64_t b = idx / M;
            const int64_t m = idx % M;
            int64_t addr = a_ptr[b * M + m];
            if (addr < 0) addr = 0;
            int64_t slot = addr % ram_size;
            if (slot < 0) slot += ram_size;
            #pragma omp atomic
            gt_ptr[m * ram_size + slot] += go_ptr[b * M + m];
        }
    }

    return {grad_x, grad_tables};
}

// -------------------------------------------------------------------------
// KERNEL: Fused LIF + RAM
// -------------------------------------------------------------------------
std::vector<torch::Tensor> fused_lif_ram_lookup(
    torch::Tensor x, torch::Tensor v_prev, torch::Tensor delays,
    torch::Tensor ram_tables, torch::Tensor connections,
    float lif_decay, float lif_threshold, int current_t
) {
    check_cpu(x, "x");
    check_cpu(v_prev, "v_prev");
    check_cpu(delays, "delays");
    check_cpu(ram_tables, "ram_tables");
    check_cpu(connections, "connections");
    check_dim(x, 3, "x");
    check_dim(v_prev, 2, "v_prev");
    check_dim(delays, 2, "delays");
    check_dim(ram_tables, 2, "ram_tables");
    check_dim(connections, 2, "connections");
    check_dtype(x, at::kFloat, "x");
    check_dtype(v_prev, at::kFloat, "v_prev");
    check_dtype(delays, at::kFloat, "delays");
    check_dtype(ram_tables, at::kFloat, "ram_tables");
    check_dtype(connections, at::kLong, "connections");

    auto x_c = ensure_contig(x);
    auto v_c = ensure_contig(v_prev);
    auto d_c = ensure_contig(delays);
    auto t_c = ensure_contig(ram_tables);
    auto c_c = ensure_contig(connections);

    int B = x_c.size(0); int T = x_c.size(1); int D_in = x_c.size(2);
    int M = d_c.size(0); int K = d_c.size(1);
    Perf::g_fused_lif_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(K > 0 && K <= 30, "K must be in [1, 30] for safe bit addressing.");
    TORCH_CHECK(v_c.size(0) == B && v_c.size(1) == M, "v_prev must have shape [B, M].");
    TORCH_CHECK(c_c.size(0) == M && c_c.size(1) == K, "connections shape must match delays shape [M, K].");
    TORCH_CHECK(t_c.size(0) == M, "ram_tables first dimension must match M.");
    TORCH_CHECK(current_t >= 0 && current_t < T, "current_t out of range.");
    int ram_size = t_c.size(1);
    
    auto spikes = torch::zeros({B, M}, x_c.options());
    auto v_next = torch::empty({B, M}, x_c.options());
    
    float* x_ptr = x_c.data_ptr<float>();
    float* vp_ptr = v_c.data_ptr<float>();
    float* d_ptr = d_c.data_ptr<float>();
    float* t_ptr = t_c.data_ptr<float>();
    int64_t* c_ptr = c_c.data_ptr<int64_t>();
    float* s_ptr = spikes.data_ptr<float>();
    float* vn_ptr = v_next.data_ptr<float>();
    
    const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(M);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; idx++) {
        const int b = static_cast<int>(idx / M);
        const int m = static_cast<int>(idx % M);
        uint32_t addr = Core::compute_ram_address(
            x_ptr + b * T * D_in,
            d_ptr + m * K,
            c_ptr + m * K,
            current_t, T, D_in, K,
            D_in, 1
        );
        
        float u_ff = t_ptr[m * ram_size + (addr % ram_size)];
        float v_curr = vp_ptr[b * M + m];
        
        float vn = lif_decay * v_curr + u_ff;
        if (vn > lif_threshold) {
            s_ptr[b * M + m] = 1.0f;
            vn_ptr[b * M + m] = 0.0f;
        } else {
            s_ptr[b * M + m] = 0.0f;
            vn_ptr[b * M + m] = vn;
        }
    }
    const long double total_ld = static_cast<long double>(total);
    const long double k_ld = static_cast<long double>(K);
    Perf::add(
        Perf::sat_from_ld(total_ld * (8.0L * k_ld + 4.0L)),
        Perf::sat_from_ld(total_ld * (16.0L * k_ld + 20.0L))
    );
    return {spikes, v_next};
}




// -------------------------------------------------------------------------
// KERNEL: NIS Sparsity Metric
// -------------------------------------------------------------------------
float calculate_nis_sparsity(at::Tensor x, float threshold = 1e-7f) {
    if (!x.defined() || x.numel() == 0) return 0.0f;
    auto x_c = x.to(at::kFloat).contiguous();
    const float* ptr = x_c.data_ptr<float>();
    int64_t n = x_c.numel();
    int64_t zeros = 0;
    #pragma omp parallel for reduction(+:zeros)
    for (int64_t i = 0; i < n; i++) {
        if (std::abs(ptr[i]) < threshold) zeros++;
    }
    return static_cast<float>(zeros) / static_cast<float>(n);
}

// -------------------------------------------------------------------------
// REVEAL: Performance Statistics API
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// KERNEL: Neural Cache Lookup (LSH Accelerated)
std::vector<torch::Tensor> neural_cache_lookup_fast(
    torch::Tensor query,      // [B, D]
    torch::Tensor keys,       // [TBL, RAM, D] bf16/float
    torch::Tensor values,     // [TBL, RAM, O] bf16/float
    torch::Tensor planes,     // [TBL, D, Bits]
    torch::Tensor valid,      // [TBL, RAM]
    float key_similarity_threshold,
    bool use_avx512
) {
    check_cpu(query, "query");
    check_cpu(keys, "keys");
    check_cpu(values, "values");
    check_cpu(planes, "planes");
    check_cpu(valid, "valid");
    
    auto q_c = ensure_contig(query);
    auto k_c = ensure_contig(keys.to(torch::kFloat32));
    auto v_c = ensure_contig(values.to(torch::kFloat32));
    auto p_c = ensure_contig(planes);
    auto m_c = ensure_contig(valid);

    const int64_t B = q_c.size(0);
    const int64_t D = q_c.size(1);
    const int64_t TBL = k_c.size(0);
    const int64_t RAM = k_c.size(1);
    const int64_t Bits = p_c.size(1);
    const int64_t O = v_c.size(2);

    auto out = torch::zeros({B, O}, q_c.options());
    auto hit_mask = torch::zeros({B}, q_c.options().dtype(torch::kBool));
    auto hit_addrs = torch::zeros({B, TBL}, q_c.options().dtype(torch::kLong));

    const float* q_ptr = q_c.data_ptr<float>();
    const float* k_ptr = k_c.data_ptr<float>();
    const float* val_ptr = v_c.data_ptr<float>();
    const float* plane_ptr = p_c.data_ptr<float>();
    const bool* valid_ptr = m_c.data_ptr<bool>();
    float* o_ptr = out.data_ptr<float>();
    bool* h_ptr = hit_mask.data_ptr<bool>();
    int64_t* a_ptr = hit_addrs.data_ptr<int64_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        const float* q_b = q_ptr + b * D;
        
        // Pre-compute query norm squared
        float q_n = 0.0f;
#ifdef __AVX512F
        __m512 q_n_v = _mm512_setzero_ps();
        for (int64_t d = 0; d < D; d += 16) {
            __m512 v = _mm512_loadu_ps(q_b + d);
            q_n_v = _mm512_fmadd_ps(v, v, q_n_v);
        }
        q_n = _mm512_reduce_add_ps(q_n_v);
#else
        for (int64_t d = 0; d < D; d++) q_n += q_b[d] * q_b[d];
#endif

        float best_sim = -1.0f;
        int64_t best_table_idx = -1;
        int64_t best_slot_idx = -1;

        for (int64_t t = 0; t < TBL; t++) {
            // LSH Hashing: Optimized [TBL, Bits, D] Layout
            uint64_t hash = 0;
            const float* p_t = plane_ptr + t * Bits * D;
            
            for (int64_t bit = 0; bit < Bits; bit++) {
                const float* p_row = p_t + bit * D;
                float dot = 0.0f;
#ifdef __AVX512F
                __m512 dot_v = _mm512_setzero_ps();
                for (int64_t d = 0; d < D; d += 16) {
                    dot_v = _mm512_fmadd_ps(_mm512_loadu_ps(q_b + d), _mm512_loadu_ps(p_row + d), dot_v);
                }
                dot = _mm512_reduce_add_ps(dot_v);
#else
                for (int64_t d = 0; d < D; d++) dot += q_b[d] * p_row[d];
#endif
                if (dot > 0.0f) hash |= (1ULL << bit);
            }
            
            const int64_t slot = hash % RAM;
            a_ptr[b * TBL + t] = slot;

            if (!valid_ptr[t * RAM + slot]) continue;

            // Cosine Similarity: AVX-512 Accelerated
            const float* k_s = k_ptr + (t * RAM + slot) * D;
            float dot_k = 0.0f;
            float k_n = 0.0f;
#ifdef __AVX512F
            __m512 dot_k_v = _mm512_setzero_ps();
            __m512 k_n_v = _mm512_setzero_ps();
            for (int64_t d = 0; d < D; d += 16) {
                __m512 q_v = _mm512_loadu_ps(q_b + d);
                __m512 k_v = _mm512_loadu_ps(k_s + d);
                dot_k_v = _mm512_fmadd_ps(q_v, k_v, dot_k_v);
                k_n_v = _mm512_fmadd_ps(k_v, k_v, k_n_v);
            }
            dot_k = _mm512_reduce_add_ps(dot_k_v);
            k_n = _mm512_reduce_add_ps(k_n_v);
#else
            for (int64_t d = 0; d < D; d++) {
                dot_k += q_b[d] * k_s[d];
                k_n += k_s[d] * k_s[d];
            }
#endif
            float sim = dot_k / (std::sqrt(q_n * k_n) + 1e-8f);
            
            if (sim > key_similarity_threshold && sim > best_sim) {
                best_sim = sim;
                best_table_idx = t;
                best_slot_idx = slot;
            }
        }

        if (best_table_idx != -1) {
            h_ptr[b] = true;
            const float* v_s = val_ptr + (best_table_idx * RAM + best_slot_idx) * O;
            for (int64_t o = 0; o < O; o++) o_ptr[b * O + o] = v_s[o];
        }
    }

    const long double b_ld = static_cast<long double>(B);
    const long double d_ld = static_cast<long double>(D);
    const long double tbl_ld = static_cast<long double>(TBL);
    const long double o_ld = static_cast<long double>(O);
    Perf::add(
        Perf::sat_from_ld(b_ld * (2.0L * d_ld + 1.0L + tbl_ld * (2.0L * d_ld + 2.0L))),
        Perf::sat_from_ld(b_ld * (4.0L * d_ld + tbl_ld * (8.0L + 1.0L + 8.0L * d_ld) + o_ld * 8.0L + 1.0L))
    );
    return {out, hit_mask, hit_addrs};
}

// -------------------------------------------------------------------------
// KERNEL: AdEMAMix Shared Implementation
// -------------------------------------------------------------------------
namespace {
    inline void _ademamix_core(
        float* p, float* grad, float* mf, float* ms, float* v,
        int64_t n, float lr, float beta1_fast, float beta1_slow, float beta2,
        float alpha, float eps, float weight_decay, float metabolic_tax, int step,
        float bc1_f, float bc1_s, float bc2
    ) {
#ifdef __AVX512F
        const __m512 v_lr = _mm512_set1_ps(lr);
        const __m512 v_b1f = _mm512_set1_ps(beta1_fast);
        const __m512 v_b1s = _mm512_set1_ps(beta1_slow);
        const __m512 v_b2 = _mm512_set1_ps(beta2);
        const __m512 v_1mb1f = _mm512_set1_ps(1.0f - beta1_fast);
        const __m512 v_1mb1s = _mm512_set1_ps(1.0f - beta1_slow);
        const __m512 v_1mb2 = _mm512_set1_ps(1.0f - beta2);
        const __m512 v_alpha = _mm512_set1_ps(alpha);
        const __m512 v_1malpha = _mm512_set1_ps(1.0f - alpha);
        const __m512 v_eps = _mm512_set1_ps(eps);
        const __m512 v_bc1f = _mm512_set1_ps(bc1_f);
        const __m512 v_bc1s = _mm512_set1_ps(bc1_s);
        const __m512 v_bc2 = _mm512_set1_ps(bc2);
        const __m512 v_wd_f = _mm512_set1_ps(1.0f - lr * weight_decay);
        const __m512 v_metabolic = _mm512_set1_ps(1.0f - metabolic_tax);
        const __m512 v_metabolic_threshold = _mm512_set1_ps(1e-7f);

        #pragma omp parallel for schedule(static) if(n > 4096)
        for (int64_t i = 0; i < (n & ~15LL); i += 16) {
            __m512 g_val = _mm512_loadu_ps(grad + i);
            __m512 p_val = _mm512_loadu_ps(p + i);
            __m512 mf_val = _mm512_loadu_ps(mf + i);
            __m512 ms_val = _mm512_loadu_ps(ms + i);
            __m512 v_val = _mm512_loadu_ps(v + i);

            if (weight_decay != 0.0f) p_val = _mm512_mul_ps(p_val, v_wd_f);

            mf_val = _mm512_add_ps(_mm512_mul_ps(v_b1f, mf_val), _mm512_mul_ps(v_1mb1f, g_val));
            ms_val = _mm512_add_ps(_mm512_mul_ps(v_b1s, ms_val), _mm512_mul_ps(v_1mb1s, g_val));
            v_val = _mm512_add_ps(_mm512_mul_ps(v_b2, v_val), _mm512_mul_ps(_mm512_mul_ps(v_1mb2, g_val), g_val));

            _mm512_storeu_ps(mf + i, mf_val);
            _mm512_storeu_ps(ms + i, ms_val);
            _mm512_storeu_ps(v + i, v_val);

            __m512 mf_hat = _mm512_div_ps(mf_val, v_bc1f);
            __m512 ms_hat = _mm512_div_ps(ms_val, v_bc1s);
            __m512 v_hat = _mm512_div_ps(v_val, v_bc2);

            if (metabolic_tax > 0.0f) {
                __mmask16 inactive = _mm512_cmp_ps_mask(v_hat, v_metabolic_threshold, _CMP_LT_OS);
                p_val = _mm512_mask_mul_ps(p_val, inactive, p_val, v_metabolic);
            }

            __m512 m_mix = _mm512_add_ps(_mm512_mul_ps(v_alpha, ms_hat), _mm512_mul_ps(v_1malpha, mf_hat));
            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
            p_val = _mm512_sub_ps(p_val, _mm512_mul_ps(v_lr, _mm512_div_ps(m_mix, denom)));
            _mm512_storeu_ps(p + i, p_val);
        }
        for (int64_t i = (n & ~15LL); i < n; i++) {
            if (weight_decay != 0.0f) p[i] *= (1.0f - lr * weight_decay);
            float g_val = grad[i];
            mf[i] = beta1_fast * mf[i] + (1.0f - beta1_fast) * g_val;
            ms[i] = beta1_slow * ms[i] + (1.0f - beta1_slow) * g_val;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g_val * g_val;
            float vh = v[i] / bc2;
            if (metabolic_tax > 0.0f && vh < 1e-7f) p[i] *= (1.0f - metabolic_tax);
            float mmix = alpha * (ms[i] / bc1_s) + (1.0f - alpha) * (mf[i] / bc1_f);
            p[i] -= lr * (mmix / (std::sqrt(vh) + eps));
        }
#else
        #pragma omp parallel for schedule(static) if(n > 4096)
        for (int64_t i = 0; i < n; i++) {
            if (weight_decay != 0.0f) p[i] *= (1.0f - lr * weight_decay);
            float g_val = grad[i];
            mf[i] = beta1_fast * mf[i] + (1.0f - beta1_fast) * g_val;
            ms[i] = beta1_slow * ms[i] + (1.0f - beta1_slow) * g_val;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g_val * g_val;
            float vh = v[i] / bc2;
            if (metabolic_tax > 0.0f && vh < 1e-7f) p[i] *= (1.0f - metabolic_tax);
            float mmix = alpha * (ms[i] / bc1_s) + (1.0f - alpha) * (mf[i] / bc1_f);
            p[i] -= lr * (mmix / (std::sqrt(vh) + eps));
        }
#endif
    }
}

void ademamix_update(
    torch::Tensor p, torch::Tensor grad, 
    torch::Tensor m_fast, torch::Tensor m_slow, torch::Tensor v,
    float lr, float beta1_fast, float beta1_slow, float beta2,
    float alpha, float eps, float weight_decay, float metabolic_tax, int step
) {
    check_cpu(p, "p"); check_cpu(grad, "grad");
    check_cpu(m_fast, "m_fast"); check_cpu(m_slow, "m_slow"); check_cpu(v, "v");
    auto p_c = ensure_contig(p); auto g_c = ensure_contig(grad);
    auto mf_c = ensure_contig(m_fast); auto ms_c = ensure_contig(m_slow); auto v_c = ensure_contig(v);
    
    float bc1_f = 1.0f - std::pow(beta1_fast, (float)step + 1.0f);
    float bc1_s = 1.0f - std::pow(beta1_slow, (float)step + 1.0f);
    float bc2   = 1.0f - std::pow(beta2, (float)step + 1.0f);
    
    _ademamix_core(
        p_c.data_ptr<float>(), g_c.data_ptr<float>(),
        mf_c.data_ptr<float>(), ms_c.data_ptr<float>(), v_c.data_ptr<float>(),
        p_c.numel(), lr, beta1_fast, beta1_slow, beta2, alpha, eps, weight_decay, metabolic_tax, step,
        bc1_f, bc1_s, bc2
    );
    Perf::g_ademamix_calls.fetch_add(1ULL, std::memory_order_relaxed);
    Perf::add(14ULL * p_c.numel(), 36ULL * p_c.numel());
}

void batched_ademamix_update(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> m_fast_list,
    std::vector<torch::Tensor> m_slow_list,
    std::vector<torch::Tensor> v_list,
    float lr, float beta1_fast, float beta1_slow, float beta2,
    float alpha, float eps, float weight_decay, float metabolic_tax, int step
) {
    const size_t n_params = params.size();
    float bc1_f = 1.0f - std::pow(beta1_fast, (float)step + 1.0f);
    float bc1_s = 1.0f - std::pow(beta1_slow, (float)step + 1.0f);
    float bc2   = 1.0f - std::pow(beta2, (float)step + 1.0f);

    long double total_elems = 0.0L;
    #pragma omp parallel for reduction(+:total_elems) schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)n_params; i++) {
        total_elems += static_cast<long double>(params[i].numel());
        _ademamix_core(
            params[i].data_ptr<float>(), grads[i].data_ptr<float>(),
            m_fast_list[i].data_ptr<float>(), m_slow_list[i].data_ptr<float>(), v_list[i].data_ptr<float>(),
            params[i].numel(), lr, beta1_fast, beta1_slow, beta2, alpha, eps, weight_decay, metabolic_tax, step,
            bc1_f, bc1_s, bc2
        );
    }
    Perf::g_batched_ademamix_calls.fetch_add(1ULL, std::memory_order_relaxed);
    Perf::add(Perf::sat_from_ld(14.0L * total_elems), Perf::sat_from_ld(36.0L * total_elems));
}

// -------------------------------------------------------------------------
// KERNEL: MES Super Step (multi-layer local loss + RAM table grads)
// -------------------------------------------------------------------------
std::vector<torch::Tensor> mes_super_step(
    torch::Tensor p_brain,         // [B, T, M]
    torch::Tensor target_brain,    // [B, T, M]
    torch::Tensor H_inter,         // [B, L, R, D, C]
    torch::Tensor all_delays,      // [L, M, K]
    torch::Tensor all_tables,      // [L, M, RAM]
    torch::Tensor all_connections, // [L, M, K]
    float lif_decay,
    float lif_threshold,
    float l1_weight
) {
    check_cpu(p_brain, "p_brain");
    check_cpu(target_brain, "target_brain");
    check_cpu(H_inter, "H_inter");
    check_cpu(all_delays, "all_delays");
    check_cpu(all_tables, "all_tables");
    check_cpu(all_connections, "all_connections");

    check_dim(p_brain, 3, "p_brain");
    check_dim(target_brain, 3, "target_brain");
    check_dim(H_inter, 5, "H_inter");
    check_dim(all_delays, 3, "all_delays");
    check_dim(all_tables, 3, "all_tables");
    check_dim(all_connections, 3, "all_connections");

    check_dtype(p_brain, at::kFloat, "p_brain");
    check_dtype(target_brain, at::kFloat, "target_brain");
    check_dtype(H_inter, at::kFloat, "H_inter");
    check_dtype(all_delays, at::kFloat, "all_delays");
    check_dtype(all_tables, at::kFloat, "all_tables");
    check_dtype(all_connections, at::kLong, "all_connections");

    auto p_c = ensure_contig(p_brain);
    auto tgt_c = ensure_contig(target_brain);
    auto h_c = ensure_contig(H_inter);
    auto d_c = ensure_contig(all_delays);
    auto t_c = ensure_contig(all_tables);
    auto c_c = ensure_contig(all_connections);

    const int64_t B = p_c.size(0);
    const int64_t T = p_c.size(1);
    const int64_t M = p_c.size(2);
    const int64_t L = d_c.size(0);
    const int64_t M_d = d_c.size(1);
    const int64_t K = d_c.size(2);
    const int64_t RAM = t_c.size(2);
    const int64_t R = h_c.size(2);
    const int64_t D = h_c.size(3);
    const int64_t C = h_c.size(4);
    Perf::g_mes_super_calls.fetch_add(1ULL, std::memory_order_relaxed);

    TORCH_CHECK(T > 0, "p_brain sequence length must be > 0.");
    TORCH_CHECK(tgt_c.size(0) == B && tgt_c.size(2) == M, "target_brain must match [B, T, M].");
    TORCH_CHECK(d_c.size(0) == L && d_c.size(1) == M_d && d_c.size(2) == K, "all_delays shape invalid.");
    TORCH_CHECK(t_c.size(0) == L && t_c.size(1) == M_d, "all_tables shape must be [L, M, RAM].");
    TORCH_CHECK(c_c.size(0) == L && c_c.size(1) == M_d && c_c.size(2) == K, "all_connections shape must be [L, M, K].");
    TORCH_CHECK(M == M_d, "M mismatch between p_brain and stacked RAM params.");
    TORCH_CHECK(K > 0 && K <= 30, "K must be in [1, 30].");
    TORCH_CHECK((D * C) == M, "H_inter D*C must equal M.");
    TORCH_CHECK(h_c.size(0) == B && h_c.size(1) == L, "H_inter shape must be [B, L, R, D, C].");
    TORCH_CHECK(RAM > 0, "RAM size must be > 0.");

    auto grad_tables = torch::zeros_like(t_c);
    auto loss_tensor = torch::zeros({}, p_c.options());

    const float* p_ptr = p_c.data_ptr<float>();
    const float* tgt_ptr = tgt_c.data_ptr<float>();
    const float* h_ptr = h_c.data_ptr<float>();
    const float* d_ptr = d_c.data_ptr<float>();
    const float* t_ptr = t_c.data_ptr<float>();
    const int64_t* c_ptr = c_c.data_ptr<int64_t>();
    float* gt_ptr = grad_tables.data_ptr<float>();
    float* loss_ptr = loss_tensor.data_ptr<float>();

    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = std::max(1, omp_get_max_threads());
    #endif
    const int64_t layer_table_elems = M * RAM;
    std::vector<float> grad_thread_accum(
        static_cast<size_t>(std::max(1, max_threads)) * static_cast<size_t>(std::max<int64_t>(1, layer_table_elems)),
        0.0f
    );

    std::vector<float> h_avg(static_cast<size_t>(B * L * M), 0.0f);
    const float inv_R = 1.0f / static_cast<float>(std::max<int64_t>(1, R));
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t l = 0; l < L; l++) {
            const int64_t base_bl = (b * L + l) * R * M;
            const int64_t out_bl = (b * L + l) * M;
            float* out_row = h_avg.data() + out_bl;
            std::fill(out_row, out_row + M, 0.0f);
            for (int64_t r = 0; r < R; r++) {
                const float* h_row = h_ptr + base_bl + r * M;
                for (int64_t m = 0; m < M; m++) {
                    out_row[m] += h_row[m];
                }
            }
            for (int64_t m = 0; m < M; m++) {
                out_row[m] *= inv_R;
            }
        }
    }

    // First-layer sequence history after RMSNorm (mean over R is a no-op here because inputs are R-broadcasted).
    std::vector<float> x_hist(static_cast<size_t>(B * T * M), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t t = 0; t < T; t++) {
            const int64_t off = (b * T + t) * M;
            float sum_sq = 0.0f;
            for (int64_t m = 0; m < M; m++) {
                float v = p_ptr[off + m];
                sum_sq += v * v;
            }
            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(M) + 1e-8f);
            for (int64_t m = 0; m < M; m++) {
                x_hist[off + m] = p_ptr[off + m] * inv_rms;
            }
        }
    }

    std::vector<float> current_in_last(static_cast<size_t>(B * M), 0.0f);
    std::vector<float> target_last(static_cast<size_t>(B * M), 0.0f);
    std::vector<float> x_single(static_cast<size_t>(B * M), 0.0f);
    const int64_t t_last = T - 1;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        const int64_t off = (b * T + t_last) * M;
        for (int64_t m = 0; m < M; m++) {
            current_in_last[b * M + m] = p_ptr[off + m];
            target_last[b * M + m] = tgt_ptr[off + m];
        }
    }

    const float inv_N = 1.0f / static_cast<float>(std::max<int64_t>(1, B * R * M));
    double mse_acc = 0.0;
    double l1_acc = 0.0;
    const float lif_thr_safe = std::max(1e-6f, lif_threshold);

    for (int64_t l = 0; l < L; l++) {
        std::vector<float> prev_in = current_in_last;

        const float* d_level = d_ptr + l * M * K;
        const float* t_level = t_ptr + l * M * RAM;
        const int64_t* c_level = c_ptr + l * M * K;
        float* g_level = gt_ptr + l * M * RAM;

        const int64_t T_curr = (l == 0) ? T : 1;
        const float* x_level = (l == 0) ? x_hist.data() : x_single.data();
        const int64_t x_batch_stride = T_curr * M;
        if (max_threads > 1 && layer_table_elems > 0) {
            std::fill(grad_thread_accum.begin(), grad_thread_accum.end(), 0.0f);
        }

        #pragma omp parallel reduction(+:mse_acc,l1_acc)
        {
            int thread_id = 0;
            #ifdef _OPENMP
            thread_id = omp_get_thread_num();
            #endif
            float* g_local = g_level;
            if (max_threads > 1 && layer_table_elems > 0) {
                g_local = grad_thread_accum.data() + static_cast<int64_t>(thread_id) * layer_table_elems;
            }

            #pragma omp for schedule(static)
            for (int64_t b = 0; b < B; b++) {
                const float* xb = x_level + b * x_batch_stride;
                const float* hb = h_avg.data() + (b * L + l) * M;
                float* curr_in_b = current_in_last.data() + b * M;
                const float* tgt_b = target_last.data() + b * M;

                for (int64_t m = 0; m < M; m++) {
                    uint32_t addr = Core::compute_ram_address(
                        xb,
                        d_level + m * K,
                        c_level + m * K,
                        static_cast<int>(T_curr - 1),
                        static_cast<int>(T_curr),
                        static_cast<int>(M),
                        static_cast<int>(K),
                        static_cast<int>(M),
                        1
                    );
                    int64_t slot = static_cast<int64_t>(addr) % RAM;
                    if (slot < 0) slot += RAM;

                    const int64_t table_idx = m * RAM + slot;
                    float u_ff = t_level[table_idx];
                    float v_curr = hb[m];
                    float v_next = lif_decay * v_curr + u_ff;
                    float s = (v_next > lif_threshold) ? 1.0f : 0.0f;
                    float target = tgt_b[m];
                    float diff = s - target;

                    mse_acc += static_cast<double>(diff * diff) * static_cast<double>(R);
                    l1_acc += static_cast<double>(std::abs(s)) * static_cast<double>(R);

                    float grad_s = (2.0f * diff) * inv_N;
                    if (s > 0.0f) {
                        grad_s += l1_weight * inv_N;
                    }
                    float v_norm = v_next / lif_thr_safe;
                    float surrogate = std::max(0.0f, 1.0f - std::abs(v_norm));
                    float g = grad_s * surrogate;

                    g_local[table_idx] += g;
                    curr_in_b[m] = s;
                }
            }
        }

        if (max_threads > 1 && layer_table_elems > 0) {
            #pragma omp parallel for schedule(static)
            for (int64_t idx = 0; idx < layer_table_elems; idx++) {
                float sum = 0.0f;
                for (int th = 0; th < max_threads; th++) {
                    sum += grad_thread_accum[static_cast<int64_t>(th) * layer_table_elems + idx];
                }
                g_level[idx] += sum;
            }
        }

        target_last.swap(prev_in);

        if (l + 1 < L) {
            #pragma omp parallel for schedule(static)
            for (int64_t b = 0; b < B; b++) {
                float sum_sq = 0.0f;
                const float* in_b = current_in_last.data() + b * M;
                float* xs_b = x_single.data() + b * M;
                for (int64_t m = 0; m < M; m++) {
                    float v = in_b[m];
                    sum_sq += v * v;
                }
                float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(M) + 1e-8f);
                for (int64_t m = 0; m < M; m++) {
                    xs_b[m] = in_b[m] * inv_rms;
                }
            }
        }
    }

    *loss_ptr = static_cast<float>((mse_acc + static_cast<double>(l1_weight) * l1_acc) * static_cast<double>(inv_N));
    const long double b_ld = static_cast<long double>(B);
    const long double t_ld = static_cast<long double>(T);
    const long double l_ld = static_cast<long double>(L);
    const long double r_ld = static_cast<long double>(R);
    const long double m_ld = static_cast<long double>(M);
    const long double k_ld = static_cast<long double>(K);
    const long double h_avg_flops = b_ld * l_ld * m_ld * (r_ld + 1.0L);
    const long double x_hist_flops = b_ld * t_ld * (3.0L * m_ld + 2.0L);
    const long double main_flops = b_ld * l_ld * m_ld * (8.0L * k_ld + 22.0L);
    const long double norm_flops = b_ld * std::max(0.0L, l_ld - 1.0L) * (3.0L * m_ld + 2.0L);
    const long double flops = h_avg_flops + x_hist_flops + main_flops + norm_flops;
    const long double bytes = b_ld * l_ld * m_ld * (16.0L * k_ld + 28.0L)
        + b_ld * t_ld * m_ld * 8.0L
        + b_ld * m_ld * 16.0L;
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return {loss_tensor, grad_tables};
}

// -------------------------------------------------------------------------
// KERNEL: Survival Update (usage EMA from importance + surprise)
// -------------------------------------------------------------------------
torch::Tensor survival_update(
    torch::Tensor H,          // [B, L, R, D, C]
    torch::Tensor H_prev,     // [B, L, R, D, C] or empty
    torch::Tensor usage,      // [L, R]
    float gamma,
    float importance_weight,
    float surprise_weight
) {
    check_cpu(H, "H");
    check_cpu(H_prev, "H_prev");
    check_cpu(usage, "usage");
    check_dim(H, 5, "H");
    check_dim(usage, 2, "usage");
    check_dtype(H, at::kFloat, "H");
    check_dtype(usage, at::kFloat, "usage");
    if (H_prev.numel() > 0) {
        check_dim(H_prev, 5, "H_prev");
        check_dtype(H_prev, at::kFloat, "H_prev");
        TORCH_CHECK(H_prev.sizes() == H.sizes(), "H_prev must match H shape when provided.");
    }

    auto H_c = ensure_contig(H);
    auto usage_c = ensure_contig(usage);
    auto out = usage_c.clone();

    const int64_t B = H_c.size(0);
    const int64_t L = H_c.size(1);
    const int64_t R = H_c.size(2);
    const int64_t D = H_c.size(3);
    const int64_t C = H_c.size(4);
    Perf::g_survival_update_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(usage_c.size(0) == L && usage_c.size(1) == R, "usage must have shape [L, R].");

    const float* h_ptr = H_c.data_ptr<float>();
    at::Tensor Hp_c;
    const float* hp_ptr = nullptr;
    if (H_prev.numel() > 0) {
        Hp_c = ensure_contig(H_prev);
        hp_ptr = Hp_c.data_ptr<float>();
    }
    const float* u_ptr = usage_c.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();

    const int64_t block_size = D * C;
    const int64_t h_batch_stride = L * R * block_size;
    const int64_t h_layer_stride = R * block_size;

    const int64_t total_blocks = L * R;
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total_blocks; idx++) {
        const int64_t l = idx / R;
        const int64_t r = idx % R;
        float imp_sum = 0.0f;
        float sur_sum = 0.0f;
        for (int64_t b = 0; b < B; b++) {
            const int64_t base = b * h_batch_stride + l * h_layer_stride + r * block_size;
            float sq_imp = 0.0f;
            float sq_sur = 0.0f;
            for (int64_t i = 0; i < block_size; i++) {
                const float hv = h_ptr[base + i];
                sq_imp += hv * hv;
                if (hp_ptr != nullptr) {
                    const float dv = hv - hp_ptr[base + i];
                    sq_sur += dv * dv;
                }
            }
            imp_sum += std::sqrt(sq_imp);
            if (hp_ptr != nullptr) {
                sur_sum += std::sqrt(sq_sur);
            }
        }
        float importance = imp_sum / static_cast<float>(B);
        float surprise = (hp_ptr != nullptr) ? (sur_sum / static_cast<float>(B)) : 0.0f;
        float combined = importance_weight * importance + surprise_weight * surprise;
        float updated = (1.0f - gamma) * u_ptr[l * R + r] + gamma * combined;
        if (!std::isfinite(updated)) {
            updated = 0.0f;
        }
        if (updated < 0.0f) {
            updated = 0.0f;
        }
        o_ptr[l * R + r] = updated;
    }
    const long double b_ld = static_cast<long double>(B);
    const long double lr_ld = static_cast<long double>(L) * static_cast<long double>(R);
    const long double block_ld = static_cast<long double>(block_size);
    const bool has_prev = (hp_ptr != nullptr);
    const long double flops = has_prev
        ? (lr_ld * (b_ld * (5.0L * block_ld + 2.0L) + 8.0L))
        : (lr_ld * (b_ld * (2.0L * block_ld + 1.0L) + 8.0L));
    const long double bytes = has_prev
        ? (lr_ld * b_ld * block_ld * 8.0L + lr_ld * 8.0L)
        : (lr_ld * b_ld * block_ld * 4.0L + lr_ld * 8.0L);
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return out;
}

// -------------------------------------------------------------------------
// KERNEL: Survival Mask (top-k sparse block selection)
// -------------------------------------------------------------------------
torch::Tensor survival_mask(
    torch::Tensor usage,       // [L, R]
    torch::Tensor reliability, // [L, R]
    float metabolic_pressure,
    float tps_pressure,
    float min_keep_ratio
) {
    check_cpu(usage, "usage");
    check_cpu(reliability, "reliability");
    check_dim(usage, 2, "usage");
    check_dim(reliability, 2, "reliability");
    check_dtype(usage, at::kFloat, "usage");
    check_dtype(reliability, at::kFloat, "reliability");
    TORCH_CHECK(usage.sizes() == reliability.sizes(), "usage and reliability must have same shape.");

    auto u_c = ensure_contig(usage);
    auto r_c = ensure_contig(reliability);
    const int64_t L = u_c.size(0);
    const int64_t R = u_c.size(1);
    const int64_t N = L * R;
    Perf::g_survival_mask_calls.fetch_add(1ULL, std::memory_order_relaxed);

    float pressure = metabolic_pressure + 0.5f * tps_pressure;
    pressure = std::max(0.0f, std::min(0.95f, pressure));
    float keep_ratio = 1.0f - pressure;
    keep_ratio = std::max(min_keep_ratio, keep_ratio);
    keep_ratio = std::max(0.0f, std::min(1.0f, keep_ratio));
    int64_t k = std::max<int64_t>(1, static_cast<int64_t>(std::floor(static_cast<double>(N) * keep_ratio)));
    if (k > N) k = N;

    const float* u_ptr = u_c.data_ptr<float>();
    const float* r_ptr = r_c.data_ptr<float>();
    std::vector<float> scores(static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < N; i++) {
        float v = u_ptr[i] * r_ptr[i];
        if (!std::isfinite(v)) v = 0.0f;
        scores[static_cast<size_t>(i)] = v;
    }

    const int64_t pivot = N - k; // kth-largest threshold via kth-smallest pivot
    std::vector<float> work = scores;
    std::nth_element(work.begin(), work.begin() + pivot, work.end());
    const float threshold = work[static_cast<size_t>(pivot)];

    auto out = torch::zeros({L, R, 1, 1}, u_c.options());
    float* o_ptr = out.data_ptr<float>();
    #pragma omp parallel for
    for (int64_t i = 0; i < N; i++) {
        o_ptr[i] = (scores[static_cast<size_t>(i)] >= threshold) ? 1.0f : 0.0f;
    }
    const long double n = static_cast<long double>(N);
    Perf::add(
        Perf::sat_from_ld(2.0L * n),
        Perf::sat_from_ld(12.0L * n)
    );
    return out;
}

// -------------------------------------------------------------------------
// KERNEL: Survival Losses (stability var, energy abs-mean, coherence mse)
// -------------------------------------------------------------------------
std::vector<torch::Tensor> survival_losses(
    torch::Tensor H,      // [B, L, R, D, C]
    torch::Tensor H_prev  // [B, L, R, D, C] or empty
) {
    check_cpu(H, "H");
    check_dim(H, 5, "H");
    check_dtype(H, at::kFloat, "H");
    if (H_prev.numel() > 0) {
        check_cpu(H_prev, "H_prev");
        check_dim(H_prev, 5, "H_prev");
        check_dtype(H_prev, at::kFloat, "H_prev");
        TORCH_CHECK(H_prev.sizes() == H.sizes(), "H_prev must match H shape when provided.");
    }

    auto H_c = ensure_contig(H);
    const bool has_prev = H_prev.numel() > 0;
    at::Tensor Hp_c;
    if (has_prev) {
        Hp_c = ensure_contig(H_prev);
    }

    const int64_t B = H_c.size(0);
    const int64_t L = H_c.size(1);
    const int64_t R = H_c.size(2);
    const int64_t D = H_c.size(3);
    const int64_t C = H_c.size(4);
    const int64_t block_size = D * C;
    const int64_t total_blocks = B * L * R;
    const int64_t total_elems = total_blocks * block_size;
    Perf::g_survival_losses_calls.fetch_add(1ULL, std::memory_order_relaxed);

    const float* h_ptr = H_c.data_ptr<float>();
    const float* hp_ptr = has_prev ? Hp_c.data_ptr<float>() : nullptr;

    double sum_norm = 0.0;
    double sum_norm_sq = 0.0;
    double sum_abs = 0.0;
    double sum_diff_sq = 0.0;

    #pragma omp parallel for reduction(+:sum_norm,sum_norm_sq,sum_abs,sum_diff_sq) schedule(static)
    for (int64_t blk = 0; blk < total_blocks; blk++) {
        const int64_t base = blk * block_size;
        double sq = 0.0;
        for (int64_t i = 0; i < block_size; i++) {
            const float hv = h_ptr[base + i];
            sq += static_cast<double>(hv) * static_cast<double>(hv);
            sum_abs += static_cast<double>(std::abs(hv));
            if (hp_ptr != nullptr) {
                const double dv = static_cast<double>(hv) - static_cast<double>(hp_ptr[base + i]);
                sum_diff_sq += dv * dv;
            }
        }
        const double n = std::sqrt(sq);
        sum_norm += n;
        sum_norm_sq += n * n;
    }

    double stability = 0.0;
    if (total_blocks > 1) {
        const double n = static_cast<double>(total_blocks);
        const double mean = sum_norm / n;
        double var_num = sum_norm_sq - n * mean * mean;
        if (var_num < 0.0) var_num = 0.0;
        stability = var_num / static_cast<double>(total_blocks - 1);
    }
    const double energy = (total_elems > 0) ? (sum_abs / static_cast<double>(total_elems)) : 0.0;
    const double coherence = (has_prev && total_elems > 0) ? (sum_diff_sq / static_cast<double>(total_elems)) : 0.0;

    auto t_stability = torch::zeros({}, H_c.options());
    auto t_energy = torch::zeros({}, H_c.options());
    auto t_coherence = torch::zeros({}, H_c.options());
    t_stability.fill_(static_cast<float>(stability));
    t_energy.fill_(static_cast<float>(energy));
    t_coherence.fill_(static_cast<float>(coherence));
    const long double blocks_ld = static_cast<long double>(total_blocks);
    const long double elems_ld = static_cast<long double>(total_elems);
    const long double block_ld = static_cast<long double>(block_size);
    const long double flops = has_prev
        ? (blocks_ld * (5.0L * block_ld + 1.0L))
        : (blocks_ld * (3.0L * block_ld + 1.0L));
    const long double bytes = has_prev ? (elems_ld * 8.0L) : (elems_ld * 4.0L);
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return {t_stability, t_energy, t_coherence};
}

// Forward declaration for wrapper compatibility.
std::vector<at::Tensor> forward_stack(
    at::Tensor x, at::Tensor H,
    at::Tensor delays, at::Tensor tables, at::Tensor conns,
    at::Tensor decays, at::Tensor hw, at::Tensor hb,
    double lif_decay,
    double lif_threshold,
    double halt_threshold,
    int64_t steps
);

// -------------------------------------------------------------------------
// UNIFIED IO WRAPPERS: (input, state, params, scalars)
// -------------------------------------------------------------------------
std::vector<at::Tensor> forward_stack_io(
    at::Tensor x_input,
    at::Tensor H_state,
    std::vector<at::Tensor> params,
    at::Tensor scalars
) {
    TORCH_CHECK(
        false,
        "forward_stack_io legacy stub was removed. "
        "Use unified_dispatch_io with CMD_FORWARD_STACK (0) instead. "
        "Details: x_input shape=", x_input.sizes(),
        ", H_state shape=", H_state.sizes(),
        ", params=", params.size(),
        ", scalars shape=", scalars.sizes()
    );
    return {};
}

// --- FORWARD DECLARATIONS ---
std::vector<at::Tensor> mes_super_step_io(at::Tensor, at::Tensor, std::vector<at::Tensor>, at::Tensor);
torch::Tensor survival_update_io(at::Tensor, at::Tensor, std::vector<at::Tensor>, at::Tensor);
torch::Tensor survival_mask_io(at::Tensor, at::Tensor, std::vector<at::Tensor>, at::Tensor);
std::vector<torch::Tensor> survival_losses_io(at::Tensor, at::Tensor, std::vector<at::Tensor>, at::Tensor);
std::vector<at::Tensor> fused_cognitive_step(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, float, float, float, at::Tensor, at::Tensor, float, bool=false, at::Tensor=at::Tensor(), at::Tensor=at::Tensor(), at::Tensor=at::Tensor(), at::Tensor=at::Tensor());
std::vector<at::Tensor> fused_cognitive_cycle_ultra(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, float, float, float, at::Tensor, at::Tensor, float, bool=false, at::Tensor=at::Tensor(), at::Tensor=at::Tensor(), at::Tensor=at::Tensor(), at::Tensor=at::Tensor());

// Command codes for unified_dispatch_io
enum CognitiveCmd {
    CMD_FORWARD_STACK = 0,
    CMD_MES_SUPER = 1,
    CMD_SURVIVAL_LOSSES = 2,
    CMD_SURVIVAL_UPDATE = 3,
    CMD_SURVIVAL_MASK = 4,
    CMD_FUSED_CYCLE_ULTRA = 5,
    CMD_FUSED_STEP = 6,
    CMD_FULL_FORWARD = 7
};

std::vector<at::Tensor> fused_cognitive_step(
    at::Tensor x, at::Tensor H, at::Tensor mask,
    at::Tensor delays, at::Tensor tables, at::Tensor conns,
    at::Tensor decays, at::Tensor hw, at::Tensor hb,
    int64_t h_cycles, int64_t l_cycles,
    float lif_decay, float lif_threshold, float halt_threshold,
    at::Tensor h_out, at::Tensor y_out,
    float tax_rate,
    bool survival_enabled,
    at::Tensor usage,
    at::Tensor reliability,
    at::Tensor H_prev,
    at::Tensor survival_scalars
) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t L = H.size(1);
    const int64_t R = H.size(2);
    const int64_t M = H.size(3) * H.size(4);
    
    auto x_c = ensure_contig(x);
    auto h_curr = h_out.defined() ? ensure_contig(h_out) : H.clone();
    
    at::Tensor m_c;
    float* m_ptr = nullptr;
    if (survival_enabled && usage.defined() && reliability.defined() && survival_scalars.defined() && survival_scalars.numel() >= 3) {
        m_c = survival_mask(usage, reliability, survival_scalars[0].item<float>(), survival_scalars[1].item<float>(), survival_scalars[2].item<float>());
        m_ptr = m_c.data_ptr<float>();
    } else {
        m_c = ensure_contig(mask);
        m_ptr = m_c.data_ptr<float>();
    }
    auto t_c = ensure_contig(tables);
    auto d_c = ensure_contig(delays);
    auto c_c = ensure_contig(conns);
    auto dec_c = ensure_contig(decays);
    
    float* h_ptr = h_curr.data_ptr<float>();
    float* t_ptr = t_c.data_ptr<float>();
    float* d_ptr = d_c.data_ptr<float>();
    int64_t* c_ptr = c_c.data_ptr<int64_t>();
    float* dec_ptr = dec_c.data_ptr<float>();
    
    int total_steps = (int)(h_cycles * l_cycles);
    auto y_seq_step = y_out.defined() ? y_out : torch::zeros({B, T, R, M}, x.options());
    float* y_ptr = y_seq_step.data_ptr<float>();

    #ifdef __AVX512F
    const __m512 v_lif_threshold = _mm512_set1_ps(lif_threshold);
    #endif

    std::vector<float> h_avg_global(static_cast<size_t>(B * L * M), 0.0f);
    
    for (int step = 0; step < total_steps; step++) {
        float global_max_delta = 0.0f;
        
        // 1. Averaging Pass
        #pragma omp parallel for schedule(static)
        for (int bl = 0; bl < (int)(B * L); bl++) {
            int b = bl / (int)L;
            int l = bl % (int)L;
            const int64_t layer_offset = (static_cast<int64_t>(b) * L + l) * R * M;
            float* ha = h_avg_global.data() + (static_cast<int64_t>(bl)) * M;
            
            #ifdef __AVX512F
            const float inv_R = 1.0f / static_cast<float>(R);
            const __m512 v_inv_R = _mm512_set1_ps(inv_R);
            for (int m = 0; m < (M & ~15); m += 16) {
                __m512 sum = _mm512_setzero_ps();
                for (int r = 0; r < R; r++) {
                    sum = _mm512_add_ps(sum, _mm512_loadu_ps(h_ptr + layer_offset + r * M + m));
                }
                _mm512_storeu_ps(ha + m, _mm512_mul_ps(sum, v_inv_R));
            }
            #else
            for (int m = 0; m < M; m++) {
                float sum = 0.0f;
                for (int r = 0; r < R; r++) sum += h_ptr[layer_offset + r * M + m];
                ha[m] = sum / (float)R;
            }
            #endif
        }
        
        // 2. Update Pass with Manual Reduction
        #pragma omp parallel
        {
            float thread_max_delta = 0.0f;
            #pragma omp for schedule(static)
            for (int blr = 0; blr < (int)(B * L * R); blr++) {
                int b = blr / (int)(L * R);
                int lr_idx = blr % (int)(L * R);
                int l = lr_idx / (int)R;
                int r = lr_idx % (int)R;

                if (m_ptr[l * R + r] < 0.5f) continue;
                
                const int64_t state_base = (static_cast<int64_t>(b) * L * R + l * R + r) * M;
                const int64_t decay_base = static_cast<int64_t>(l * R + r) * M;
                const float* ha = h_avg_global.data() + (static_cast<int64_t>(b) * L + l) * M;

                #ifdef __AVX512F
                for (int m = 0; m < (M & ~15); m += 16) {
                    __m512 v_avg = _mm512_loadu_ps(ha + m);
                    __m512 v_dec = _mm512_loadu_ps(dec_ptr + decay_base + m);
                    __m512 v_old = _mm512_loadu_ps(h_ptr + state_base + m);
                    
                    __mmask16 spiking = _mm512_cmp_ps_mask(v_avg, v_lif_threshold, _CMP_GT_OQ);
                    __m512 v_spike = _mm512_mask_blend_ps(spiking, _mm512_setzero_ps(), _mm512_set1_ps(1.0f));
                    
                    __m512 v_new = _mm512_add_ps(_mm512_mul_ps(v_dec, v_old), v_spike);
                    _mm512_storeu_ps(h_ptr + state_base + m, v_new);
                    
                    __m512 v_diff = _mm512_abs_ps(_mm512_sub_ps(v_new, v_old));
                    float d = _mm512_reduce_max_ps(v_diff);
                    if (d > thread_max_delta) thread_max_delta = d;
                }
                #else
                for (int m = 0; m < M; m++) {
                    float v_avg = ha[m];
                    float v_old = h_ptr[state_base + m];
                    float s = (v_avg > lif_threshold) ? 1.0f : 0.0f;
                    float v_new = dec_ptr[decay_base + m] * v_old + s;
                    h_ptr[state_base + m] = v_new;
                    float delta = std::abs(v_new - v_old);
                    if (delta > thread_max_delta) thread_max_delta = delta;
                }
                #endif
            }
            #pragma omp critical
            {
                if (thread_max_delta > global_max_delta) global_max_delta = thread_max_delta;
            }
        }
        
        // 3. Metabolic Taxation
        if (tax_rate > 0.0f) {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < h_curr.numel(); i++) h_ptr[i] *= (1.0f - tax_rate);
        }

        if (global_max_delta < 1e-6f) break;
    }
    
    // Final copy for Python model output compatibility (org.py).
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < (int)B; b++) {
        for (int r = 0; r < (int)R; r++) {
            for (int m = 0; m < (int)M; m++) {
                const int64_t src_idx = (static_cast<int64_t>(b) * L * R + (L - 1) * R + r) * M + m;
                const int64_t dst_idx = (static_cast<int64_t>(b) * T * R + 0 * R + r) * M + m;
                y_ptr[dst_idx] = h_ptr[src_idx];
            }
        }
    }
    
    return {y_seq_step, h_curr};
}

// --- FUSED COGNITIVE CYCLE ULTRA (Sequence Loop) ---
std::vector<at::Tensor> fused_cognitive_cycle_ultra(
    at::Tensor x, at::Tensor H, at::Tensor mask,
    at::Tensor delays, at::Tensor tables, at::Tensor conns,
    at::Tensor decays, at::Tensor hw, at::Tensor hb,
    int64_t h_cycles, int64_t l_cycles,
    float lif_decay, float lif_threshold, float halt_threshold,
    at::Tensor h_out, at::Tensor y_out,
    float tax_rate,
    bool survival_enabled,
    at::Tensor usage,
    at::Tensor reliability,
    at::Tensor H_prev,
    at::Tensor survival_scalars
) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    
    at::Tensor h_iter = h_out.defined() ? h_out : H.clone();
    at::Tensor y_seq = y_out.defined() ? y_out : torch::zeros({B, T, NIS::R, NIS::WORKING_DIM / NIS::C, NIS::C}, x.options());
    
    // Initial state for survival update logic
    at::Tensor h_initial = H.clone();

    for (int64_t t = 0; t < T; t++) {
        auto out_step = fused_cognitive_step(
            x.narrow(1, t, 1), h_iter, mask,
            delays, tables, conns,
            decays, hw, hb,
            h_cycles, l_cycles,
            lif_decay, lif_threshold, halt_threshold,
            h_iter, at::Tensor(), 
            tax_rate,
            survival_enabled, usage, reliability, at::Tensor(), survival_scalars
        );
        h_iter = out_step[1];
        y_seq.narrow(1, t, 1).copy_(out_step[0].view_as(y_seq.narrow(1, t, 1)));
    }
    
    at::Tensor new_usage = usage;
    if (survival_enabled && usage.defined() && survival_scalars.defined() && survival_scalars.numel() >= 6) {
        new_usage = survival_update(h_iter, h_initial, usage, survival_scalars[3].item<float>(), survival_scalars[4].item<float>(), survival_scalars[5].item<float>());
    }

    return {y_seq, h_iter, new_usage};
}

// -------------------------------------------------------------------------
// KERNEL: Geometric Manifold Recall (AVX-512 Optimized)
// -------------------------------------------------------------------------
torch::Tensor geometric_manifold_recall_avx512(
    torch::Tensor p_brain,           // [B, T, M] or [B, M]
    torch::Tensor manifold,          // [N, M] (int8 or float)
    torch::Tensor scales,            // [N] (float) if int8, else empty
    torch::Tensor curve_indices,     // [S] (long) Morton indices
    float focus_sharpness         // Scalar
) {
    auto p_c = ensure_contig(p_brain);
    auto m_c = ensure_contig(manifold);
    auto idx_c = ensure_contig(curve_indices);
    
    const int64_t B = p_c.size(0);
    const int64_t M = p_c.size(-1); // Last dim is params
    const int64_t N = m_c.size(0);
    const int64_t S = idx_c.numel();
    
    bool is_int8 = (m_c.scalar_type() == at::kByte || m_c.scalar_type() == at::kChar);
    
    // Optimization: The curve is global (shared for all B, T).
    // We compute the aggregated context ONCE and return a [1, 1, M] tensor.
    // Python will handle broadcasting during the addition (p + context).
    
    // Output tensor [1, 1, M]
    auto options = p_c.options();
    at::Tensor recall = torch::zeros({1, 1, M}, options);
    float* out_ptr = recall.data_ptr<float>();
    
    // Check for scales if int8
    const float* scale_ptr = nullptr;
    if (is_int8) {
        TORCH_CHECK(scales.defined() && scales.numel() == N, "dquant scales required for int8 manifold");
        scale_ptr = scales.data_ptr<float>();
    }

    const int64_t* idx_ptr = idx_c.data_ptr<int64_t>();
    
    // Accumulate manifold curve (Single pass)
    std::vector<float> accumulator(M, 0.0f);
    
    for (int64_t s = 0; s < S; s++) {
        int64_t row_idx = idx_ptr[s];
        if (row_idx < 0 || row_idx >= N) continue; // Boundary check
        
        if (is_int8) {
            const int8_t* m_row = (const int8_t*)m_c.data_ptr() + row_idx * M;
            float row_scale = scale_ptr[row_idx];
            
            for (int64_t i = 0; i < M; i += 16) {
                 int rem = M - i;
                 if (rem >= 16) {
                     __m128i b = _mm_loadu_si128((const __m128i*)(m_row + i));
                     __m512i i32 = _mm512_cvtepi8_epi32(b);
                     __m512 f = _mm512_cvtepi32_ps(i32);
                     __m512 sc = _mm512_set1_ps(row_scale);
                     __m512 val = _mm512_mul_ps(f, sc);
                     
                     __m512 acc = _mm512_loadu_ps(accumulator.data() + i);
                     acc = _mm512_add_ps(acc, val);
                     _mm512_storeu_ps(accumulator.data() + i, acc);
                 } else {
                     for (int k=0; k<rem; k++) accumulator[i+k] += (float)m_row[i+k] * row_scale;
                 }
            }
        } else {
            const float* m_row = m_c.data_ptr<float>() + row_idx * M;
            for (int64_t i = 0; i < M; i += 16) {
                 int rem = M - i;
                 if (rem >= 16) {
                    __m512 val = _mm512_loadu_ps(m_row + i);
                    __m512 acc = _mm512_loadu_ps(accumulator.data() + i);
                    acc = _mm512_add_ps(acc, val);
                    _mm512_storeu_ps(accumulator.data() + i, acc);
                 } else {
                     for (int k=0; k<rem; k++) accumulator[i+k] += m_row[i+k];
                 }
            }
        }
    }
    
    // Normalize and Write to Output
    float norm = (S > 0) ? (1.0f / S) : 0.0f;
    float final_scale = norm * focus_sharpness; 
    
    for (int64_t i=0; i<M; i++) {
        out_ptr[i] = accumulator[i] * final_scale;
    }

    return recall;
}

// -------------------------------------------------------------------------
// KERNEL: NIS Full Forward Fusion (Inference Fast Path)
// -------------------------------------------------------------------------
std::vector<at::Tensor> full_forward_inference(
    at::Tensor x,              // [B, T, 8] (bits)
    at::Tensor H,              // [B, L, R, D, C]
    std::vector<at::Tensor> p, // Pre-packed weights
    at::Tensor s               // Scalars
) {
    // Unpack Scalars
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t L = (int64_t)s[0].item<float>();
    const int64_t R = (int64_t)s[1].item<float>();
    const int64_t d_s1 = (int64_t)s[2].item<float>();
    const int64_t d_s2 = (int64_t)s[3].item<float>();
    const int64_t C = (int64_t)s[4].item<float>();
    const float lif_decay = s[5].item<float>();
    const float lif_threshold = s[6].item<float>();
    const float halt_threshold = s[7].item<float>();
    const float lambda_sparsity = s[8].item<float>();
    const float trace_gain = s[9].item<float>();
    const float cost_scale = s[10].item<float>();
    const float keep_ratio = s[11].item<float>();
    const int phase = (int)s[12].item<float>();
    const float act_threshold = (s.numel() > 13) ? s[13].item<float>() : 0.8f;

    Perf::g_full_forward_calls.fetch_add(1ULL, std::memory_order_relaxed);

    // 1. Bit -> Latent Projection
    at::Tensor p_lat;
    {
        at::Tensor x_flat = x.reshape({-1, x.size(-1)});
        if (p[0].dtype() == at::kChar) {
            p_lat = quantized_matmul(x_flat, p[0], p[1], p[2]);
        } else {
            p_lat = at::linear(x_flat, p[0], (p[2].numel() > 0 ? p[2] : at::Tensor()));
        }
        p_lat = p_lat.reshape({B, T, d_s1 * C});
    }

    // 2. State Prep
    at::Tensor H_state = (H.size(0) != B) ? H.expand({B, L, R, NIS::WORKING_DIM, C}).contiguous() : ensure_contig(H);

    // 3. LGH Manifold Recall + ACT (Adaptive Computation Time)
    bool recall_skipped_l_cycles = false;
    at::Tensor p_mod = p_lat;
    if (p[18].numel() > 0) {
        at::Tensor lgh_ctx = geometric_manifold_recall_avx512(p_mod, p[18], at::Tensor(), p[19], 2.0f);
        
        // Speculative ACT: If recall is extremely strong (sharp manifold hit), we skip System 2 deep cycles
        float recall_strength = lgh_ctx.abs().mean().item<float>();
        if (recall_strength > act_threshold) {
             recall_skipped_l_cycles = true;
             Perf::g_lgh_temporal_folds.fetch_add(1ULL, std::memory_order_relaxed); // Re-purpose counter for ACT skips
        }
        p_mod = p_mod + lgh_ctx * trace_gain;
    }

    // 4. S1 -> S2 Bridge
    at::Tensor p_brain;
    {
        at::Tensor x_in = p_mod.reshape({-1, p_mod.size(-1)});
        p_brain = (p[7].dtype() == at::kChar) ? quantized_matmul(x_in, p[7], p[8], at::Tensor()) : at::linear(x_in, p[7], at::Tensor());
        p_brain = p_brain.reshape({B, T, d_s2 * C});
    }

    // 5. Gate Calculation
    at::Tensor gate;
    float metabolic_cost = 0.0f;
    {
        at::Tensor usage = p[20];
        if (usage.numel() > 0) {
            gate = (phase == 0) ? (usage >= std::get<0>(usage.view(-1).topk(std::max<int64_t>(1, (int64_t)(usage.numel() * keep_ratio)))).min().item<float>()).to(at::kFloat).view({1, L, R, 1, 1})
                                : survival_mask(usage, torch::ones_like(usage), lambda_sparsity, 0.0f, 0.1f).view({1, L, R, 1, 1});
            metabolic_cost = gate.sum().item<float>() * cost_scale;
        } else {
            gate = torch::ones({1, L, R, 1, 1}, H_state.options());
            metabolic_cost = (float)(L * R) * cost_scale;
        }
    }

    // 6. Cognitive Cycle (Conditioned ACT)
    int64_t effective_l_cycles = recall_skipped_l_cycles ? 0 : NIS::L_CYCLES;
    
    std::vector<at::Tensor> cycle_out = fused_cognitive_cycle_ultra(
        p_brain, H_state, gate,
        p[9], p[10], p[11], p[12], p[13], p[14],
        NIS::H_CYCLES, effective_l_cycles,
        lif_decay, lif_threshold, halt_threshold,
        at::Tensor(), at::Tensor(), 0.0f, false, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
    );
    
    at::Tensor y_seq = cycle_out[0];
    at::Tensor H_next = cycle_out[1];

    // 7. Readout (RMS + Linear)
    at::Tensor y_flat = rms_norm(y_seq.flatten(2), NIS::RMS_NORM_EPS);
    at::Tensor logits;
    {
        at::Tensor x_in = y_flat.reshape({-1, y_flat.size(-1)});
        logits = (p[15].dtype() == at::kChar) ? quantized_matmul(x_in, p[15], p[16], p[17]) : at::linear(x_in, p[15], (p[17].numel() > 0 ? p[17] : at::Tensor()));
        logits = logits.reshape({B, T, -1});
    }

    return {logits, H_next, torch::tensor({metabolic_cost}, H_state.options()), gate};
}


std::vector<at::Tensor> unified_dispatch_io(
    std::vector<at::Tensor> ctx,
    int64_t cmd
) {
    // UNIFIED STRUCT LATCHING: ctx[0]=x, ctx[1]=state, ctx[2]=scalars, ctx[3:]=params
    at::Tensor x = ctx[0];
    at::Tensor state = ctx[1];
    at::Tensor scalars = ctx[2];
    std::vector<at::Tensor> params;
    for (size_t i = 3; i < ctx.size(); ++i) params.push_back(ctx[i]);

    switch (cmd) {
        case CMD_FUSED_CYCLE_ULTRA: {
            bool survival = (scalars.size(0) > 6 && scalars[6].item<float>() > 0.5f);
            return fused_cognitive_cycle_ultra(x, state, params[0], params[1], params[2], params[3], params[4], params[5], params[6], 
                (int64_t)scalars[0].item<float>(), (int64_t)scalars[1].item<float>(),
                scalars[2].item<float>(), scalars[3].item<float>(), scalars[4].item<float>(),
                (params.size() > 7) ? params[7] : at::Tensor(),
                (params.size() > 8) ? params[8] : at::Tensor(),
                scalars[5].item<float>(),
                survival,
                (survival && params.size() > 9) ? params[9] : at::Tensor(),
                (survival && params.size() > 10) ? params[10] : at::Tensor(),
                (survival && params.size() > 11) ? params[11] : at::Tensor(),
                (survival && scalars.size(0) > 12) ? scalars.slice(0, 7, 13) : at::Tensor());
        }
        case CMD_FUSED_STEP: {
            bool survival = (scalars.size(0) > 6 && scalars[6].item<float>() > 0.5f);
            return fused_cognitive_step(x, state, params[0], params[1], params[2], params[3], params[4], params[5], params[6], 
                (int64_t)scalars[0].item<float>(), (int64_t)scalars[1].item<float>(),
                scalars[2].item<float>(), scalars[3].item<float>(), scalars[4].item<float>(),
                (params.size() > 7) ? params[7] : at::Tensor(),
                (params.size() > 8) ? params[8] : at::Tensor(),
                scalars[5].item<float>(),
                survival,
                (survival && params.size() > 9) ? params[9] : at::Tensor(),
                (survival && params.size() > 10) ? params[10] : at::Tensor(),
                (survival && params.size() > 11) ? params[11] : at::Tensor(),
                (survival && scalars.size(0) > 12) ? scalars.slice(0, 7, 13) : at::Tensor());
        }
        case CMD_FORWARD_STACK:
            return fused_cognitive_step(x, state, params[0], params[1], params[2], params[3], params[4], params[5], params[6], 
                1, (int64_t)scalars[1].item<float>(),
                scalars[2].item<float>(), scalars[3].item<float>(), scalars[4].item<float>(),
                (params.size() > 7) ? params[7] : at::Tensor(),
                (params.size() > 8) ? params[8] : at::Tensor(),
                scalars[5].item<float>());
        case CMD_MES_SUPER: return mes_super_step_io(x, state, params, scalars);
        case CMD_SURVIVAL_LOSSES: return survival_losses_io(x, state, params, scalars);
        case CMD_SURVIVAL_UPDATE: return std::vector<at::Tensor>{ survival_update_io(x, state, params, scalars) };
        case CMD_SURVIVAL_MASK: return std::vector<at::Tensor>{ survival_mask_io(x, state, params, scalars) };
        case CMD_FULL_FORWARD:
            return full_forward_inference(x, state, params, scalars);
        default:
            TORCH_CHECK(false, "Unknown Unified Command: ", cmd);
    }
}

std::vector<torch::Tensor> mes_super_step_io(
    torch::Tensor x_input,      // p_brain [B, T, M]
    torch::Tensor H_state,      // H_inter [B, L, R, D, C]
    std::vector<torch::Tensor> params,
    torch::Tensor scalars
) {
    TORCH_CHECK(params.size() == 4, "mes_super_step_io expects 4 params: target_brain,delays,tables,conns.");
    check_cpu(scalars, "scalars");
    check_dim(scalars, 1, "scalars");
    check_dtype(scalars, at::kFloat, "scalars");
    auto s_c = ensure_contig(scalars);
    TORCH_CHECK(s_c.numel() >= 3, "mes_super_step_io scalars must contain [lif_decay,lif_threshold,l1_weight].");
    const float* s_ptr = s_c.data_ptr<float>();
    return mes_super_step(
        x_input,
        params[0],
        H_state,
        params[1],
        params[2],
        params[3],
        s_ptr[0], s_ptr[1], s_ptr[2]
    );
}

torch::Tensor survival_update_io(
    torch::Tensor x_input,      // H
    torch::Tensor state,        // usage
    std::vector<torch::Tensor> params,
    torch::Tensor scalars
) {
    TORCH_CHECK(params.size() <= 1, "survival_update_io expects optional H_prev in params[0].");
    check_cpu(scalars, "scalars");
    check_dim(scalars, 1, "scalars");
    check_dtype(scalars, at::kFloat, "scalars");
    auto s_c = ensure_contig(scalars);
    TORCH_CHECK(s_c.numel() >= 3, "survival_update_io scalars must contain [gamma,importance_weight,surprise_weight].");
    const float* s_ptr = s_c.data_ptr<float>();
    at::Tensor H_prev = (params.empty() ? torch::empty(0, x_input.options()) : params[0]);
    return survival_update(
        x_input,
        H_prev,
        state,
        s_ptr[0],
        s_ptr[1],
        s_ptr[2]
    );
}

torch::Tensor survival_mask_io(
    torch::Tensor x_input,      // usage
    torch::Tensor state,        // reliability
    std::vector<torch::Tensor> params,
    torch::Tensor scalars
) {
    (void)params; // Reserved for future extensions
    check_cpu(scalars, "scalars");
    check_dim(scalars, 1, "scalars");
    check_dtype(scalars, at::kFloat, "scalars");
    auto s_c = ensure_contig(scalars);
    TORCH_CHECK(s_c.numel() >= 3, "survival_mask_io scalars must contain [metabolic_pressure,tps_pressure,min_keep_ratio].");
    const float* s_ptr = s_c.data_ptr<float>();
    return survival_mask(
        x_input,
        state,
        s_ptr[0],
        s_ptr[1],
        s_ptr[2]
    );
}

std::vector<torch::Tensor> survival_losses_io(
    torch::Tensor x_input,      // H
    torch::Tensor state,        // H_prev or empty
    std::vector<torch::Tensor> params,
    torch::Tensor scalars
) {
    return survival_losses(x_input, state);
}

// -------------------------------------------------------------------------
// KERNEL: Geometric Manifold Forward (AVX-512)
// -------------------------------------------------------------------------


// Wrapper for org.py compatibility
/**
 * NIS Token Interpreter Core
 * Executes instructions from the latent instruction stack using AVX-512.
 */
inline void nis_execute_instruction(
    __m512& signal,
    __m512& state,
    uint8_t opcode,
    float scale_const = NIS::SCALE_CONSTANT
) {
    switch(opcode) {
        case NIS::OP_ADD:
            signal = _mm512_add_ps(signal, state);
            break;
        case NIS::OP_SCALE:
            signal = _mm512_mul_ps(signal, _mm512_set1_ps(scale_const));
            break;
        case NIS::OP_GATE: {
            __mmask16 mask = _mm512_cmp_ps_mask(signal, _mm512_setzero_ps(), _CMP_GT_OS);
            signal = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), signal);
            break;
        }
        case NIS::OP_REFLECT: {
            // Shadow Brain Reflection: bitwise inversion/negation
            signal = _mm512_sub_ps(_mm512_setzero_ps(), signal);
            break;
        }
        default:
            break;
    }
}

std::vector<at::Tensor> forward_stack(
    at::Tensor x, at::Tensor H,
    at::Tensor delays, at::Tensor tables, at::Tensor conns,
    at::Tensor decays, at::Tensor hw, at::Tensor hb,
    double lif_decay,
    double lif_threshold,
    double halt_threshold,
    int64_t steps,
    at::Tensor H_out,
    at::Tensor y_out
) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t D = NIS::WORKING_DIM;
    const int64_t L = NIS::L;
    const int64_t R = NIS::R;
    const int64_t C = NIS::C;

    auto x_c = ensure_contig(x);
    auto h_c = ensure_contig(H);
    at::Tensor h_next;
    if (H_out.defined() && H_out.numel() > 0 && H_out.sizes() == h_c.sizes()) {
        h_next = H_out;
        h_next.copy_(h_c);
    } else {
        h_next = h_c.clone();
    }

    at::Tensor y_seq;
    if (y_out.defined() && (y_out.size(0) == B && y_out.size(1) == T)) {
        y_seq = y_out;
    } else {
        y_seq = torch::zeros({B, T, R, D, C}, x_c.options());
    }

    float* x_ptr = x_c.data_ptr<float>();
    float* h_ptr = h_next.data_ptr<float>();

    // Zero-Copy Workspace Pointer (Simplified)
    // In a full implementation, we'd use the Morton curve to jump.
    #pragma omp parallel for
    for (int64_t b = 0; b < B; b++) {
        for (int64_t t = 0; t < T; t++) {
            float* x_bt = x_ptr + (b * T + t) * D * C;
            float* h_bt = h_ptr + (b * L + 0) * R * D * C; // Start at layer 0

            for (int64_t s = 0; s < steps; s++) {
                // Interpretation Step: First 16 floats of each reasoning window = Opcodes
                for (int64_t i = 0; i < D * C; i += NIS::SIMD_WIDTH) {
                    __m512 signal = _mm512_loadu_ps(x_bt + i);
                    __m512 state = _mm512_loadu_ps(h_bt + i);

                    // Decode Opcode from MSB of the signal token (simplification)
                    uint8_t opcode = static_cast<uint8_t>(x_bt[i] > 0.5f ? NIS::OP_ADD : NIS::OP_SCALE);
                    
                    nis_execute_instruction(signal, state, opcode);
                    
                    _mm512_storeu_ps(x_bt + i, signal);
                }
                
                // Morton JMP Simulation: Increment pointer along the "BIOS" curve
                h_bt += (R * D * C / L); // Step through layers
            }
        }
    }

    return {y_seq, h_next, torch::zeros({B, T, 1}, x_c.options())};
}

// -------------------------------------------------------------------------
// KERNEL: Liquid Geometric Hypernetwork (LGH) AVX512 Path
// -------------------------------------------------------------------------

namespace {
inline int64_t lgh_clamp_row(int64_t row, int64_t n_rows) {
    if (n_rows <= 0) {
        return 0;
    }
    if (row < 0) {
        return 0;
    }
    if (row >= n_rows) {
        return n_rows - 1;
    }
    return row;
}

inline int64_t lgh_wrap_row(int64_t row, int64_t n_rows) {
    if (n_rows <= 0) {
        return 0;
    }
    int64_t w = row % n_rows;
    if (w < 0) {
        w += n_rows;
    }
    return w;
}

inline int64_t lgh_map_curve_row(int64_t curve_idx_original, const int64_t* inverse_order, int64_t n_rows) {
    const int64_t original = lgh_clamp_row(curve_idx_original, n_rows);
    return lgh_clamp_row(inverse_order[original], n_rows);
}

inline int64_t lgh_temporal_fold_row(
    int64_t row,
    int64_t delta_t,
    int64_t n_rows,
    int64_t temporal_bins,
    float fold_alpha
) {
    if (n_rows <= 1 || temporal_bins <= 1 || !Core::g_hpc_temporal_folding) {
        return lgh_clamp_row(row, n_rows);
    }
    const int64_t dt = lgh_wrap_row(delta_t, temporal_bins);
    const float alpha = std::max(0.0f, std::min(1.0f, fold_alpha));
    const int64_t span = std::max<int64_t>(1, static_cast<int64_t>(std::llround(static_cast<double>(n_rows - 1) * alpha)));
    const int64_t shift = (dt * span) % n_rows;
    const int64_t mask_bits = std::max<int64_t>(1, std::min<int64_t>(20, 1 + (dt % 20)));
    const int64_t mask = (int64_t(1) << mask_bits) - 1;
    const int64_t rotated = lgh_wrap_row((row ^ mask), n_rows);
    return lgh_wrap_row(rotated + shift, n_rows);
}

inline uint64_t lgh_region_mdna_word(const float* mdna_ptr, int64_t L, int64_t R, int64_t r) {
    const int64_t max_bits = std::min<int64_t>(L, 64);
    uint64_t word = 0ULL;
    for (int64_t l = 0; l < max_bits; l++) {
        if (mdna_ptr[l * R + r] >= 0.5f) {
            word |= (uint64_t(1) << l);
        }
    }
    return word;
}

inline float lgh_region_scale(
    const float* gate_ptr,
    const float* mdna_ptr,
    int64_t L,
    int64_t R,
    int64_t r,
    float thermal_scale
) {
    float sum = 0.0f;
    for (int64_t l = 0; l < L; l++) {
        const float gate_lr = std::max(0.0f, std::min(1.0f, gate_ptr[l * R + r]));
        const float mdna_lr = std::max(0.0f, std::min(1.0f, mdna_ptr[l * R + r]));
        sum += gate_lr * mdna_lr;
    }
    return thermal_scale * (sum / static_cast<float>(std::max<int64_t>(1, L)));
}

#ifdef __AVX512F
inline __mmask16 lgh_mdna_mask16(uint64_t mdna_word, int64_t chunk_index) {
    const int shift = static_cast<int>((chunk_index * 16) & 63);
    const uint64_t rotated = (shift == 0)
        ? mdna_word
        : ((mdna_word >> shift) | (mdna_word << (64 - shift)));
    return static_cast<__mmask16>(rotated & 0xFFFFULL);
}

inline __m512 lgh_pulse_eval_avx512(__m512 merged, __m512 scale_v, __mmask16 k_mdna) {
    return _mm512_maskz_mul_ps(k_mdna, merged, scale_v);
}
#endif

inline int64_t lgh_popcount64(uint64_t v) {
#if defined(_MSC_VER)
    return static_cast<int64_t>(__popcnt64(v));
#else
    return static_cast<int64_t>(__builtin_popcountll(v));
#endif
}

inline bool lgh_should_temporal_fold(
    const float* p_ptr,
    int64_t p_base_cur,
    int64_t p_base_prev,
    int64_t M,
    float threshold
) {
    if (!(threshold > 0.0f) || p_base_prev < 0) {
        return false;
    }
    const int64_t stride = (M >= 256) ? 8 : ((M >= 64) ? 4 : 1);
    float accum = 0.0f;
    int64_t samples = 0;
    for (int64_t m = 0; m < M; m += stride) {
        accum += std::abs(p_ptr[p_base_cur + m] - p_ptr[p_base_prev + m]);
        samples++;
    }
    const float mean_delta = accum / static_cast<float>(std::max<int64_t>(1, samples));
    return mean_delta < threshold;
}

inline float lgh_trace_decay_param(double trace_decay) {
    return static_cast<float>(std::max(0.0, std::min(0.9999, trace_decay)));
}

inline float lgh_trace_gain_param(double trace_gain) {
    return static_cast<float>(std::max(0.0, trace_gain));
}

inline float lgh_trace_fatigue(float trace_value) {
    const float t = std::max(0.0f, trace_value);
    return 1.0f / (1.0f + t);
}

inline float lgh_trace_recover(float trace_value, float trace_decay) {
    return std::max(0.0f, trace_value * trace_decay);
}

inline float lgh_trace_pulse_update(float trace_value, float trace_gain, float activity) {
    return std::min(2.0f, trace_value + trace_gain * std::max(0.0f, activity));
}
} // namespace

// -------------------------------------------------------------------------
// KERNEL: Pulse-Gated SIMD Forward (Phase 2 & 3 Architecture Overhaul)
// -------------------------------------------------------------------------
std::vector<at::Tensor> pulse_gated_forward(
    at::Tensor x_input,      // [B, T, M]
    at::Tensor H_state,      // [B, L, R, D, C]
    at::Tensor mask,          // [L, R] survival mask
    at::Tensor mdna_mask,     // [L, M] or [L, R] active-neuron mask
    at::Tensor curve_indices, // [S] Morton Path indices
    at::Tensor manifold,      // [N, M] One Manifold weights
    at::Tensor workspace,     // [workspace_size] pre-allocated scratchpad
    int64_t time_phase,
    int64_t h_cycles,
    int64_t l_cycles,
    double halt_threshold,
    int64_t prefetch_distance,
    double thermal_penalty,
    double temporal_fold_threshold,
    int64_t temporal_bins
) {
    check_cpu(x_input, "x_input");
    check_cpu(H_state, "H_state");
    check_cpu(mask, "mask");
    check_cpu(mdna_mask, "mdna_mask");
    check_cpu(curve_indices, "curve_indices");
    check_cpu(manifold, "manifold");
    check_cpu(workspace, "workspace");

    auto p_c = ensure_contig(x_input);
    auto h_c = ensure_contig(H_state);
    auto g_c = ensure_contig(mask);
    auto md_c = ensure_contig(mdna_mask);
    auto idx_c = ensure_contig(curve_indices);
    auto m_c = ensure_contig(manifold);
    auto ws_c = ensure_contig(workspace);

    const int64_t B = p_c.size(0);
    const int64_t T = p_c.size(1);
    const int64_t M = p_c.size(2);
    const int64_t L = h_c.size(1);
    const int64_t R = h_c.size(2);
    const int64_t N_atlas = m_c.size(0);
    const int64_t N3 = N_atlas / temporal_bins;
    const int64_t S = idx_c.numel();

    // Zero-Copy Workspace Partitioning (Phase 3)
    // Zone A: H_next [B, L, R, M]
    // Zone B: out [B, T, R, M]
    // Zone C: halt_probs [B, T, 1]
    const int64_t size_A = B * L * R * M;
    const int64_t size_B = B * T * R * M;
    const int64_t size_C = B * T;
    TORCH_CHECK(ws_c.numel() >= (size_A + size_B + size_C), "Workspace too small for configured batch/sequence.");

    float* ws_ptr = ws_c.data_ptr<float>();
    float* h_next_ptr = ws_ptr;
    float* out_ptr = ws_ptr + size_A;
    float* halt_ptr = ws_ptr + size_A + size_B;

    // Map workspace to return tensors (no copy)
    auto h_next = torch::from_blob(h_next_ptr, h_c.sizes(), h_c.options());
    auto out = torch::from_blob(out_ptr, {B, T, R, M}, p_c.options());
    auto halt_probs = torch::from_blob(halt_ptr, {B, T, 1}, p_c.options());

    // Initialize H_next with H_state (Zero-copy initial state)
    std::memcpy(h_next_ptr, h_c.data_ptr<float>(), static_cast<size_t>(size_A) * sizeof(float));
    std::memset(out_ptr, 0, static_cast<size_t>(size_B) * sizeof(float));
    std::memset(halt_ptr, 0, static_cast<size_t>(size_C) * sizeof(float));

    // Internal Pointers
    const float* x_ptr = p_c.data_ptr<float>();
    const float* g_ptr = g_c.data_ptr<float>();
    const float* md_ptr = md_c.data_ptr<float>();
    const int64_t* idx_ptr = idx_c.data_ptr<int64_t>();
    const float* man_ptr = m_c.data_ptr<float>();

    const float thermal_scale = 1.0f - 0.5f * static_cast<float>(std::max(0.0, std::min(0.95, thermal_penalty)));
    const float cycle_scale = static_cast<float>(std::max<int64_t>(1, h_cycles) * std::max<int64_t>(1, l_cycles));
    const float eps = 1e-8f;
    const int64_t atlas_bins = std::max<int64_t>(1, N_atlas / N3);
    const int64_t dt = lgh_wrap_row(time_phase, temporal_bins);

    Perf::g_lgh_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const uint64_t tsc_start = __rdtsc();

    // 1. BUILDS PROTOTYPE AND APPLIES PULSE (Consolidated AVX-512 Loop)
    // For simplicity in this unified kernel, we build the prototype thread-locally or once per T.
    // However, to strictly follow the "Hardware Gate" speed, we use ZMM registers for weights.
    
    std::vector<uint64_t> mdna_words(static_cast<size_t>(R), 0ULL);
    for (int64_t r = 0; r < R; r++) {
        mdna_words[static_cast<size_t>(r)] = lgh_region_mdna_word(md_ptr, L, R, r);
    }

    #pragma omp parallel
    {
        std::vector<float> proto_local(static_cast<size_t>(M), 0.0f);
        float* proto_ptr = proto_local.data();

        #pragma omp for schedule(dynamic)
        for (int64_t b = 0; b < B; b++) {
            for (int64_t t = 0; t < T; t++) {
                std::fill(proto_local.begin(), proto_local.end(), 0.0f);
                float contrib_count = 0.0f;

                // --- PROTOTYPE BUILDING (Morton sequential stream) ---
                for (int64_t s = 0; s < S; s++) {
                    const int64_t row_base = lgh_clamp_row(idx_ptr[s], N3);
                    const int64_t row3 = lgh_temporal_fold_row(row_base, dt, N3, temporal_bins, Core::g_hpc_fold_alpha);
                    const int64_t row = row3 * atlas_bins + (dt % atlas_bins);
                    const float* weight_row = man_ptr + row * M;

                    // Morton Prefetch N+prefetch_distance
                    if (s + prefetch_distance < S) {
                        const int64_t next_base = lgh_clamp_row(idx_ptr[s + prefetch_distance], N3);
                        const int64_t next_row3 = lgh_temporal_fold_row(next_base, dt, N3, temporal_bins, Core::g_hpc_fold_alpha);
                        const int64_t next_row = next_row3 * atlas_bins + (dt % atlas_bins);
                        _mm_prefetch(reinterpret_cast<const char*>(man_ptr + next_row * M), _MM_HINT_T0);
                    }

                    // AVX-512 Accumulation
#ifdef __AVX512F
                    int64_t m = 0;
                    for (; m + 16 <= M; m += 16) {
                        __m512 w_v = _mm512_loadu_ps(weight_row + m);
                        __m512 p_v = _mm512_loadu_ps(proto_ptr + m);
                        _mm512_storeu_ps(proto_ptr + m, _mm512_add_ps(p_v, w_v));
                    }
                    for (; m < M; m++) proto_ptr[m] += weight_row[m];
#else
                    for (int64_t m = 0; m < M; m++) proto_ptr[m] += weight_row[m];
#endif
                    contrib_count += 1.0f;
                }

                const float inv_s = 1.0f / std::max(1e-6f, contrib_count);
#ifdef __AVX512F
                int64_t mq = 0;
                __m512 inv_v = _mm512_set1_ps(inv_s);
                for (; mq + 16 <= M; mq += 16) {
                    __m512 v = _mm512_loadu_ps(proto_ptr + mq);
                    _mm512_storeu_ps(proto_ptr + mq, _mm512_mul_ps(v, inv_v));
                }
                for (; mq < M; mq++) proto_ptr[mq] *= inv_s;
#else
                for (int64_t mq = 0; mq < M; mq++) proto_ptr[mq] *= inv_s;
#endif

                // --- PULSE GATED EVALUATION ---
                float halt_acc = 0.0f;
                for (int64_t r = 0; r < R; r++) {
                    const float scale = lgh_region_scale(g_ptr, md_ptr, L, R, r, thermal_scale) * cycle_scale;
                    const int64_t out_base = ((b * T + t) * R + r) * M;
                    const uint64_t mdna_word = mdna_words[static_cast<size_t>(r)];
                    const int64_t x_base = (b * T + t) * M;

#ifdef __AVX512F
                    const __m512 scale_vec = _mm512_set1_ps(scale);
                    int64_t m = 0;
                    int64_t chunk = 0;
                    for (; m + 16 <= M; m += 16, chunk++) {
                        __mmask16 k_mdna = lgh_mdna_mask16(mdna_word, chunk);
                        __m512 x_v = _mm512_loadu_ps(x_ptr + x_base + m);
                        __m512 p_v = _mm512_loadu_ps(proto_ptr + m);
                        __m512 merged = _mm512_add_ps(x_v, p_v);
                        // Heart of the pulse: _mm512_maskz_mul_ps
                        __m512 gated = _mm512_maskz_mul_ps(k_mdna, merged, scale_vec);
                        _mm512_storeu_ps(out_ptr + out_base + m, gated);
                        __m512 abs_v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), gated);
                        halt_acc += _mm512_reduce_add_ps(abs_v);
                    }
                    for (; m < M; m++) {
                        const float merged = x_ptr[x_base + m] + proto_ptr[m];
                        const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                        const float v = (bit != 0) ? (merged * scale) : 0.0f;
                        out_ptr[out_base + m] = v;
                        halt_acc += std::abs(v);
                    }
#else
                    for (int64_t m = 0; m < M; m++) {
                        const float merged = x_ptr[x_base + m] + proto_ptr[m];
                        const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                        const float v = (bit != 0) ? (merged * scale) : 0.0f;
                        out_ptr[out_base + m] = v;
                        halt_acc += std::abs(v);
                    }
#endif
                }
                
                // Halting proxy (average energy)
                const float denom = static_cast<float>(std::max<int64_t>(1, R * M));
                const float raw_halt = (halt_acc / denom) / ((halt_acc / denom) + static_cast<float>(halt_threshold) + 1e-6f);
                halt_ptr[b * T + t] = std::max(0.0f, std::min(1.0f, raw_halt * (1.0f - 0.35f * static_cast<float>(thermal_penalty))));
            }
        }
    }

    // 2. STATE UPDATE (Last step injection into H_next)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t r = 0; r < R; r++) {
            const int64_t src_base = ((b * T + (T - 1)) * R + r) * M;
            for (int64_t l = 0; l < L; l++) {
                const float gate_lr = std::max(0.0f, std::min(1.0f, g_ptr[l * R + r]));
                const float mdna_lr = std::max(0.0f, std::min(1.0f, md_ptr[l * R + r]));
                const float mix = gate_lr * mdna_lr * thermal_scale;
                const float decay = 0.90f + 0.08f * (1.0f - mix);
                const int64_t h_base = (((b * L + l) * R + r) * M);
                
                float* h_row = h_next_ptr + h_base;
                const float* s_row = out_ptr + src_base;

#ifdef __AVX512F
                const __m512 decay_v = _mm512_set1_ps(decay);
                const __m512 mix_v = _mm512_set1_ps(mix);
                int64_t m = 0;
                for (; m + 16 <= M; m += 16) {
                    __m512 old_v = _mm512_loadu_ps(h_row + m);
                    __m512 src_v = _mm512_loadu_ps(s_row + m);
                    __m512 updated = _mm512_add_ps(_mm512_mul_ps(old_v, decay_v), _mm512_mul_ps(src_v, mix_v));
                    _mm512_storeu_ps(h_row + m, updated);
                }
                for (; m < M; m++) h_row[m] = (decay * h_row[m]) + (mix * s_row[m]);
#else
                for (int64_t m = 0; m < M; m++) h_row[m] = (decay * h_row[m]) + (mix * s_row[m]);
#endif
            }
        }
    }

    const uint64_t tsc_end = __rdtsc();
    Perf::g_lgh_tsc_cycles.fetch_add((tsc_end >= tsc_start) ? (tsc_end - tsc_start) : 0ULL, std::memory_order_relaxed);
    
    return {out, h_next, halt_probs};
}

std::vector<at::Tensor> geometric_manifold_forward_avx512(
    at::Tensor p_brain,          // [B, T, M]
    at::Tensor H_state,          // [B, L, R, D, C]
    at::Tensor gate,             // [L, R]
    at::Tensor manifold_morton,  // [N, M]
    at::Tensor curve_indices,    // [S]
    at::Tensor prefetch_curve_indices, // [S_prefetch]
    at::Tensor mdna_mask,        // [L, R]
    at::Tensor synaptic_trace,   // [N] short-term trace per manifold row
    int64_t time_phase,
    int64_t h_cycles,
    int64_t l_cycles,
    double halt_threshold,
    int64_t prefetch_distance,
    double thermal_penalty,
    double temporal_fold_threshold,
    int64_t wave_radius,
    double wave_decay,
    double trace_decay,
    double trace_gain,
    int64_t temporal_bins
) {
    check_cpu(p_brain, "p_brain");
    check_cpu(H_state, "H_state");
    check_cpu(gate, "gate");
    check_cpu(manifold_morton, "manifold_morton");
    check_cpu(curve_indices, "curve_indices");
    check_cpu(prefetch_curve_indices, "prefetch_curve_indices");
    check_cpu(mdna_mask, "mdna_mask");
    check_cpu(synaptic_trace, "synaptic_trace");
    check_dim(p_brain, 3, "p_brain");
    check_dim(H_state, 5, "H_state");
    check_dim(gate, 2, "gate");
    check_dim(manifold_morton, 2, "manifold_morton");
    check_dim(curve_indices, 1, "curve_indices");
    check_dim(prefetch_curve_indices, 1, "prefetch_curve_indices");
    check_dim(mdna_mask, 2, "mdna_mask");
    check_dim(synaptic_trace, 1, "synaptic_trace");
    check_dtype(p_brain, at::kFloat, "p_brain");
    check_dtype(H_state, at::kFloat, "H_state");
    check_dtype(gate, at::kFloat, "gate");
    check_dtype(manifold_morton, at::kFloat, "manifold_morton");
    check_dtype(curve_indices, at::kLong, "curve_indices");
    check_dtype(prefetch_curve_indices, at::kLong, "prefetch_curve_indices");
    check_dtype(mdna_mask, at::kFloat, "mdna_mask");
    check_dtype(synaptic_trace, at::kFloat, "synaptic_trace");

    auto p_c = ensure_contig(p_brain);
    auto h_c = ensure_contig(H_state);
    auto g_c = ensure_contig(gate);
    auto m_c = ensure_contig(manifold_morton);
    auto idx_c = ensure_contig(curve_indices);
    auto pidx_c = ensure_contig(prefetch_curve_indices);
    auto mdna_c = ensure_contig(mdna_mask);
    auto trace_c = ensure_contig(synaptic_trace);

    const int64_t B = p_c.size(0);
    const int64_t T = p_c.size(1);
    const int64_t M = p_c.size(2);
    const int64_t L = h_c.size(1);
    const int64_t R = h_c.size(2);
    const int64_t D = h_c.size(3);
    const int64_t C = h_c.size(4);
    const int64_t N_atlas = m_c.size(0);
    const int64_t N3 = N_atlas / temporal_bins; // Derived from temporal bins and atlas size
    TORCH_CHECK(M == D * C, "p_brain feature dim must match H_state D*C.");
    TORCH_CHECK(m_c.size(1) == M, "manifold_morton feature dim must match p_brain feature dim.");
    TORCH_CHECK(N3 > 0, "N3 must be positive (manifold size / temporal bins).");
    TORCH_CHECK((N_atlas % temporal_bins) == 0, "manifold_morton rows must be a multiple of temporal_bins.");
    TORCH_CHECK(trace_c.size(0) == N_atlas, "synaptic_trace must have length equal to manifold rows.");
    TORCH_CHECK(g_c.size(0) == L && g_c.size(1) == R, "gate must match [L, R].");
    TORCH_CHECK(mdna_c.size(0) == L && mdna_c.size(1) == R, "mdna_mask must match [L, R].");

    auto out = torch::zeros({B, T, R, M}, p_c.options());
    auto h_next = h_c.clone();
    auto halt_probs = torch::zeros({B, T, 1}, p_c.options());
    auto proto = torch::zeros({M}, p_c.options());
    auto region_scale_t = torch::zeros({R}, p_c.options());

    float* out_ptr = out.data_ptr<float>();
    float* h_ptr = h_next.data_ptr<float>();
    float* halt_ptr = halt_probs.data_ptr<float>();
    float* proto_ptr = proto.data_ptr<float>();
    float* region_scale_ptr = region_scale_t.data_ptr<float>();
    const float* p_ptr = p_c.data_ptr<float>();
    const float* g_ptr = g_c.data_ptr<float>();
    const float* m_ptr = m_c.data_ptr<float>();
    const int64_t* idx_ptr = idx_c.data_ptr<int64_t>();
    const int64_t* pidx_ptr = pidx_c.data_ptr<int64_t>();
    const float* mdna_ptr = mdna_c.data_ptr<float>();
    float* trace_ptr = trace_c.data_ptr<float>();

    const int64_t S = idx_c.numel();
    const int64_t S_prefetch = pidx_c.numel();
    const int64_t atlas_bins = std::max<int64_t>(1, N_atlas / N3);
    const int64_t runtime_bins = std::max<int64_t>(1, std::min<int64_t>(atlas_bins, std::max<int64_t>(1, temporal_bins)));
    const int64_t base_phase = lgh_wrap_row(time_phase, runtime_bins);
    const int64_t prefetch_d = std::max<int64_t>(1, prefetch_distance);
    const float thermal = static_cast<float>(std::max(0.0, std::min(0.95, thermal_penalty)));
    const float thermal_scale = 1.0f - 0.5f * thermal;
    const float cycle_scale = static_cast<float>(std::max<int64_t>(1, h_cycles) * std::max<int64_t>(1, l_cycles));
    const float eps = 1e-8f;
    const float temporal_fold = static_cast<float>(std::max(0.0, temporal_fold_threshold));
    const int64_t wave_r = std::max<int64_t>(0, wave_radius);
    const float wave_decay_f = static_cast<float>(std::max(0.01, wave_decay));
    const float trace_decay_f = lgh_trace_decay_param(trace_decay);
    const float trace_gain_f = lgh_trace_gain_param(trace_gain);
    const int64_t t_bins = runtime_bins;
    const int64_t dt = base_phase;

    Perf::g_lgh_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const uint64_t tsc_start = __rdtsc();

    // Synaptic trace decay (recovery).
    for (int64_t i = 0; i < N_atlas; i++) {
        trace_ptr[i] = lgh_trace_recover(trace_ptr[i], trace_decay_f);
    }

    // Curve prototype in Morton space with prefetching.
    if (S > 0) {
        float contrib_count = 0.0f;
        for (int64_t s = 0; s < S; s++) {
            const int64_t tbin = lgh_wrap_row(dt, t_bins);
            const int64_t row_base = lgh_clamp_row(idx_ptr[s], N3);
            const int64_t row = lgh_temporal_fold_row(row_base, dt, N3, t_bins, Core::g_hpc_fold_alpha);
            if (S_prefetch > 0) {
                const int64_t p_dt = dt + prefetch_d;
                const int64_t p_tbin = lgh_wrap_row(p_dt, t_bins);
                const int64_t p_base = lgh_clamp_row(pidx_ptr[s % S_prefetch], N3);
                const int64_t p_row3 = lgh_temporal_fold_row(p_base, p_dt, N3, t_bins, Core::g_hpc_fold_alpha);
                const int64_t p_row = p_row3 * atlas_bins + (p_tbin % atlas_bins);
                const float* p_ptr_prefetch = m_ptr + p_row * M;
                _mm_prefetch(reinterpret_cast<const char*>(p_ptr_prefetch), _MM_HINT_T0);
            }
            if (s + prefetch_d < S) {
                const int64_t n_dt = dt + prefetch_d;
                const int64_t n_tbin = lgh_wrap_row(n_dt, t_bins);
                const int64_t next_base = lgh_clamp_row(idx_ptr[s + prefetch_d], N3);
                const int64_t next_row3 = lgh_temporal_fold_row(next_base, n_dt, N3, t_bins, Core::g_hpc_fold_alpha);
                const int64_t next_row = next_row3 * atlas_bins + (n_tbin % atlas_bins);
                const float* next_ptr = m_ptr + next_row * M;
                _mm_prefetch(reinterpret_cast<const char*>(next_ptr), _MM_HINT_T0);
            }
            for (int64_t off = -wave_r; off <= wave_r; off++) {
                const int64_t w_row3 = lgh_wrap_row(row + off, N3);
                const int64_t w_row = w_row3 * atlas_bins + (tbin % atlas_bins);
                const float phase = 0.5f + 0.5f * std::cos(0.35f * static_cast<float>(dt + off));
                const float ripple = std::exp(-std::abs(static_cast<float>(off)) * wave_decay_f);
                const float trace_v = trace_ptr[w_row];
                const float fatigue = lgh_trace_fatigue(trace_v);
                const float coeff = ripple * phase * fatigue;
                const float* row_ptr = m_ptr + w_row * M;
#ifdef __AVX512F
                int64_t m = 0;
                const __m512 coeff_v = _mm512_set1_ps(coeff);
                for (; m + 16 <= M; m += 16) {
                    __m512 a = _mm512_loadu_ps(proto_ptr + m);
                    __m512 b = _mm512_loadu_ps(row_ptr + m);
                    _mm512_storeu_ps(proto_ptr + m, _mm512_fmadd_ps(b, coeff_v, a));
                }
                for (; m < M; m++) {
                    proto_ptr[m] += coeff * row_ptr[m];
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    proto_ptr[m] += coeff * row_ptr[m];
                }
#endif
                trace_ptr[w_row] = lgh_trace_pulse_update(trace_v, trace_gain_f, std::abs(coeff));
                contrib_count += std::max(0.001f, ripple * phase);
            }
        }
        const float inv_s = 1.0f / std::max(1e-6f, contrib_count);
#ifdef __AVX512F
        int64_t m = 0;
        __m512 inv = _mm512_set1_ps(inv_s);
        for (; m + 16 <= M; m += 16) {
            __m512 v = _mm512_loadu_ps(proto_ptr + m);
            _mm512_storeu_ps(proto_ptr + m, _mm512_mul_ps(v, inv));
        }
        for (; m < M; m++) {
            proto_ptr[m] *= inv_s;
        }
#else
        for (int64_t m = 0; m < M; m++) {
            proto_ptr[m] *= inv_s;
        }
#endif
    }

    // Region-level scale from averaged mDNA*gate.
    std::vector<uint64_t> mdna_words(static_cast<size_t>(R), 0ULL);
    std::vector<int64_t> mdna_active_lanes(static_cast<size_t>(R), 0);
    for (int64_t r = 0; r < R; r++) {
        mdna_words[static_cast<size_t>(r)] = lgh_region_mdna_word(mdna_ptr, L, R, r);
        mdna_active_lanes[static_cast<size_t>(r)] =
            std::max<int64_t>(1, (lgh_popcount64(mdna_words[static_cast<size_t>(r)]) * M) / 64);
        region_scale_ptr[r] = lgh_region_scale(g_ptr, mdna_ptr, L, R, r, thermal_scale);
    }

    std::vector<float> halt_acc(static_cast<size_t>(B * T), 0.0f);
    uint64_t pulse_ops = 0ULL;
    uint64_t folded_steps = 0ULL;

    // Pulse-gated AVX512 evaluation.
    for (int64_t b = 0; b < B; b++) {
        for (int64_t t = 0; t < T; t++) {
            const int64_t p_base = (b * T + t) * M;
            const int64_t p_prev_base = (t > 0) ? ((b * T + (t - 1)) * M) : -1;
            if (t > 0 && lgh_should_temporal_fold(p_ptr, p_base, p_prev_base, M, temporal_fold)) {
                const int64_t bt_prev = b * T + (t - 1);
                const int64_t bt_cur = b * T + t;
                for (int64_t r = 0; r < R; r++) {
                    const int64_t src_base = ((b * T + (t - 1)) * R + r) * M;
                    const int64_t dst_base = ((b * T + t) * R + r) * M;
                    std::memcpy(out_ptr + dst_base, out_ptr + src_base, static_cast<size_t>(M) * sizeof(float));
                }
                halt_acc[static_cast<size_t>(bt_cur)] = halt_acc[static_cast<size_t>(bt_prev)];
                folded_steps++;
                continue;
            }
            for (int64_t r = 0; r < R; r++) {
                const float scale = region_scale_ptr[r] * cycle_scale;
                const int64_t out_base = ((b * T + t) * R + r) * M;
                const uint64_t mdna_word = mdna_words[static_cast<size_t>(r)];
                if (scale > eps) {
                    pulse_ops += static_cast<uint64_t>(mdna_active_lanes[static_cast<size_t>(r)]);
                }
#ifdef __AVX512F
                const __m512 scale_v = _mm512_set1_ps(scale);
                int64_t m = 0;
                int64_t chunk = 0;
                for (; m + 16 <= M; m += 16, chunk++) {
                    __mmask16 k_mdna = lgh_mdna_mask16(mdna_word, chunk);
                    if (!(scale > eps)) {
                        k_mdna = static_cast<__mmask16>(0x0000);
                    }
                    __m512 p_v = _mm512_loadu_ps(p_ptr + p_base + m);
                    __m512 proto_v = _mm512_loadu_ps(proto_ptr + m);
                    __m512 merged = _mm512_add_ps(p_v, proto_v);
                    __m512 gated = lgh_pulse_eval_avx512(merged, scale_v, k_mdna);
                    _mm512_storeu_ps(out_ptr + out_base + m, gated);
                    __m512 abs_v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), gated);
                    halt_acc[static_cast<size_t>(b * T + t)] += _mm512_reduce_add_ps(abs_v);
                }
                for (; m < M; m++) {
                    const float merged = p_ptr[p_base + m] + proto_ptr[m];
                    const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                    const float v = ((scale > eps) && (bit != 0)) ? (merged * scale) : 0.0f;
                    out_ptr[out_base + m] = v;
                    halt_acc[static_cast<size_t>(b * T + t)] += std::abs(v);
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    const float merged = p_ptr[p_base + m] + proto_ptr[m];
                    const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                    const float v = ((scale > eps) && (bit != 0)) ? (merged * scale) : 0.0f;
                    out_ptr[out_base + m] = v;
                    halt_acc[static_cast<size_t>(b * T + t)] += std::abs(v);
                }
#endif
            }
        }
    }

    // Update H state from last-step LGH output.
    for (int64_t b = 0; b < B; b++) {
        for (int64_t r = 0; r < R; r++) {
            const int64_t src_base = ((b * T + (T - 1)) * R + r) * M;
            for (int64_t l = 0; l < L; l++) {
                const float gate_lr = std::max(0.0f, std::min(1.0f, g_ptr[l * R + r]));
                const float mdna_lr = std::max(0.0f, std::min(1.0f, mdna_ptr[l * R + r]));
                const float mix = std::max(0.0f, std::min(1.0f, gate_lr * mdna_lr * thermal_scale));
                const float decay = 0.90f + 0.08f * (1.0f - mix);
                const int64_t h_base = (((b * L + l) * R + r) * M);
                const uint64_t mdna_word = mdna_words[static_cast<size_t>(r)];
                const int mdna_active = (l < 64) ? static_cast<int>((mdna_word >> l) & 1ULL) : static_cast<int>(mix > eps);
#ifdef __AVX512F
                const __m512 decay_v = _mm512_set1_ps(decay);
                const __m512 mix_v = _mm512_set1_ps(mix);
                const __mmask16 k_mdna = ((mix > eps) && (mdna_active != 0))
                    ? static_cast<__mmask16>(0xFFFF)
                    : static_cast<__mmask16>(0x0000);
                int64_t m = 0;
                for (; m + 16 <= M; m += 16) {
                    __m512 old_v = _mm512_loadu_ps(h_ptr + h_base + m);
                    __m512 src_v = _mm512_loadu_ps(out_ptr + src_base + m);
                    __m512 decayed = _mm512_mul_ps(old_v, decay_v);
                    __m512 injected = lgh_pulse_eval_avx512(src_v, mix_v, k_mdna);
                    _mm512_storeu_ps(h_ptr + h_base + m, _mm512_add_ps(decayed, injected));
                }
                for (; m < M; m++) {
                    const float old_v = h_ptr[h_base + m];
                    const float src_v = out_ptr[src_base + m];
                    const int bit = mdna_active != 0;
                    h_ptr[h_base + m] = (decay * old_v) + ((bit != 0) ? (mix * src_v) : 0.0f);
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    const float old_v = h_ptr[h_base + m];
                    const float src_v = out_ptr[src_base + m];
                    const int bit = mdna_active != 0;
                    h_ptr[h_base + m] = (decay * old_v) + ((bit != 0) ? (mix * src_v) : 0.0f);
                }
#endif
            }
        }
    }

    // Halting proxy from average activated energy.
    const float denom = static_cast<float>(std::max<int64_t>(1, R * M));
    const float halt_denom = static_cast<float>(std::max(1e-6, halt_threshold));
    for (int64_t bt = 0; bt < B * T; bt++) {
        const float mean_abs = halt_acc[static_cast<size_t>(bt)] / denom;
        const float raw = mean_abs / (mean_abs + halt_denom + 1e-6f);
        const float cool = std::max(0.0f, std::min(1.0f, raw * (1.0f - 0.35f * thermal)));
        halt_ptr[bt] = cool;
    }

    const long double bt_ld = static_cast<long double>(B) * static_cast<long double>(T);
    const long double r_ld = static_cast<long double>(R);
    const long double m_ld = static_cast<long double>(M);
    const long double l_ld = static_cast<long double>(L);
    const long double flops = bt_ld * r_ld * m_ld * 4.0L + bt_ld * l_ld * r_ld * m_ld * 3.0L;
    const long double bytes = bt_ld * r_ld * m_ld * 16.0L + bt_ld * l_ld * r_ld * m_ld * 8.0L;
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    const uint64_t tsc_end = __rdtsc();
    Perf::g_lgh_tsc_cycles.fetch_add((tsc_end >= tsc_start) ? (tsc_end - tsc_start) : 0ULL, std::memory_order_relaxed);
    Perf::g_lgh_pulse_ops.fetch_add(pulse_ops, std::memory_order_relaxed);
    Perf::g_lgh_temporal_folds.fetch_add(folded_steps, std::memory_order_relaxed);

    auto out_5d = out.view({B, T, R, D, C});
    return {out_5d, h_next, halt_probs};
}

std::vector<at::Tensor> geometric_manifold_forward_avx512_int8(
    at::Tensor p_brain,          // [B, T, M]
    at::Tensor H_state,          // [B, L, R, D, C]
    at::Tensor gate,             // [L, R]
    at::Tensor manifold_morton_q, // [N, M] int8
    at::Tensor manifold_scale,   // [N] float32
    at::Tensor curve_indices,    // [S] original topology indices
    at::Tensor prefetch_curve_indices, // [S_prefetch] original topology indices
    at::Tensor mdna_mask,        // [L, R]
    at::Tensor synaptic_trace,   // [N]
    int64_t time_phase,
    int64_t h_cycles,
    int64_t l_cycles,
    double halt_threshold,
    int64_t prefetch_distance,
    double thermal_penalty,
    double temporal_fold_threshold,
    int64_t wave_radius,
    double wave_decay,
    double trace_decay,
    double trace_gain,
    int64_t temporal_bins
) {
    check_cpu(p_brain, "p_brain");
    check_cpu(H_state, "H_state");
    check_cpu(gate, "gate");
    check_cpu(manifold_morton_q, "manifold_morton_q");
    check_cpu(manifold_scale, "manifold_scale");
    check_cpu(curve_indices, "curve_indices");
    check_cpu(prefetch_curve_indices, "prefetch_curve_indices");
    check_cpu(mdna_mask, "mdna_mask");
    check_cpu(synaptic_trace, "synaptic_trace");
    check_dim(p_brain, 3, "p_brain");
    check_dim(H_state, 5, "H_state");
    check_dim(gate, 2, "gate");
    check_dim(manifold_morton_q, 2, "manifold_morton_q");
    check_dim(manifold_scale, 1, "manifold_scale");
    check_dim(curve_indices, 1, "curve_indices");
    check_dim(prefetch_curve_indices, 1, "prefetch_curve_indices");
    check_dim(mdna_mask, 2, "mdna_mask");
    check_dim(synaptic_trace, 1, "synaptic_trace");
    check_dtype(p_brain, at::kFloat, "p_brain");
    check_dtype(H_state, at::kFloat, "H_state");
    check_dtype(gate, at::kFloat, "gate");
    check_dtype(manifold_morton_q, at::kChar, "manifold_morton_q");
    check_dtype(manifold_scale, at::kFloat, "manifold_scale");
    check_dtype(curve_indices, at::kLong, "curve_indices");
    check_dtype(prefetch_curve_indices, at::kLong, "prefetch_curve_indices");
    check_dtype(mdna_mask, at::kFloat, "mdna_mask");
    check_dtype(synaptic_trace, at::kFloat, "synaptic_trace");

    auto p_c = ensure_contig(p_brain);
    auto h_c = ensure_contig(H_state);
    auto g_c = ensure_contig(gate);
    auto q_c = ensure_contig(manifold_morton_q);
    auto sc_c = ensure_contig(manifold_scale);
    auto idx_c = ensure_contig(curve_indices);
    auto pidx_c = ensure_contig(prefetch_curve_indices);
    auto mdna_c = ensure_contig(mdna_mask);
    auto trace_c = ensure_contig(synaptic_trace);

    const int64_t B = p_c.size(0);
    const int64_t T = p_c.size(1);
    const int64_t M = p_c.size(2);
    const int64_t L = h_c.size(1);
    const int64_t R = h_c.size(2);
    const int64_t D = h_c.size(3);
    const int64_t C = h_c.size(4);
    const int64_t N_atlas = q_c.size(0);
    const int64_t N3 = N_atlas / temporal_bins;
    TORCH_CHECK(M == D * C, "p_brain feature dim must match H_state D*C.");
    TORCH_CHECK(q_c.size(1) == M, "manifold_morton_q feature dim must match p_brain feature dim.");
    TORCH_CHECK(sc_c.size(0) == N_atlas, "manifold_scale must have shape [N_atlas].");
    TORCH_CHECK(N3 > 0, "N3 must be positive (manifold size / temporal bins).");
    TORCH_CHECK((N_atlas % temporal_bins) == 0, "manifold_morton_q rows must be a multiple of temporal_bins.");
    TORCH_CHECK(trace_c.size(0) == N_atlas, "synaptic_trace must have length equal to manifold rows.");
    TORCH_CHECK(g_c.size(0) == L && g_c.size(1) == R, "gate must match [L, R].");
    TORCH_CHECK(mdna_c.size(0) == L && mdna_c.size(1) == R, "mdna_mask must match [L, R].");

    auto out = torch::zeros({B, T, R, M}, p_c.options());
    auto h_next = h_c.clone();
    auto halt_probs = torch::zeros({B, T, 1}, p_c.options());
    auto proto = torch::zeros({M}, p_c.options());
    auto region_scale_t = torch::zeros({R}, p_c.options());

    float* out_ptr = out.data_ptr<float>();
    float* h_ptr = h_next.data_ptr<float>();
    float* halt_ptr = halt_probs.data_ptr<float>();
    float* proto_ptr = proto.data_ptr<float>();
    float* region_scale_ptr = region_scale_t.data_ptr<float>();
    const float* p_ptr = p_c.data_ptr<float>();
    const float* g_ptr = g_c.data_ptr<float>();
    const int8_t* q_ptr = q_c.data_ptr<int8_t>();
    const float* sc_ptr = sc_c.data_ptr<float>();
    const int64_t* idx_ptr = idx_c.data_ptr<int64_t>();
    const int64_t* pidx_ptr = pidx_c.data_ptr<int64_t>();
    const float* mdna_ptr = mdna_c.data_ptr<float>();
    float* trace_ptr = trace_c.data_ptr<float>();

    const int64_t S = idx_c.numel();
    const int64_t S_prefetch = pidx_c.numel();
    const int64_t atlas_bins = std::max<int64_t>(1, N_atlas / N3);
    const int64_t runtime_bins = std::max<int64_t>(1, std::min<int64_t>(atlas_bins, std::max<int64_t>(1, temporal_bins)));
    const int64_t base_phase = lgh_wrap_row(time_phase, runtime_bins);
    const int64_t prefetch_d = std::max<int64_t>(1, prefetch_distance);
    const float thermal = static_cast<float>(std::max(0.0, std::min(0.95, thermal_penalty)));
    const float thermal_scale = 1.0f - 0.5f * thermal;
    const float cycle_scale = static_cast<float>(std::max<int64_t>(1, h_cycles) * std::max<int64_t>(1, l_cycles));
    const float eps = 1e-8f;
    const float temporal_fold = static_cast<float>(std::max(0.0, temporal_fold_threshold));
    const int64_t wave_r = std::max<int64_t>(0, wave_radius);
    const float wave_decay_f = static_cast<float>(std::max(0.01, wave_decay));
    const float trace_decay_f = lgh_trace_decay_param(trace_decay);
    const float trace_gain_f = lgh_trace_gain_param(trace_gain);
    const int64_t t_bins = runtime_bins;
    const int64_t dt = base_phase;

    Perf::g_lgh_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const uint64_t tsc_start = __rdtsc();

    // Synaptic trace decay (recovery).
    for (int64_t i = 0; i < N_atlas; i++) {
        trace_ptr[i] = lgh_trace_recover(trace_ptr[i], trace_decay_f);
    }

    // Build float prototype from Morton-ordered int8 manifold.
    if (S > 0) {
        float contrib_count = 0.0f;
        for (int64_t s = 0; s < S; s++) {
            const int64_t tbin = lgh_wrap_row(dt, t_bins);
            const int64_t row_base = lgh_clamp_row(idx_ptr[s], N3);
            const int64_t row = lgh_temporal_fold_row(row_base, dt, N3, t_bins, Core::g_hpc_fold_alpha);
            if (S_prefetch > 0) {
                const int64_t p_dt = dt + prefetch_d;
                const int64_t p_tbin = lgh_wrap_row(p_dt, t_bins);
                const int64_t p_base = lgh_clamp_row(pidx_ptr[s % S_prefetch], N3);
                const int64_t p_row3 = lgh_temporal_fold_row(p_base, p_dt, N3, t_bins, Core::g_hpc_fold_alpha);
                const int64_t p_row = p_row3 * atlas_bins + (p_tbin % atlas_bins);
                const int8_t* p_ptr_prefetch = q_ptr + p_row * M;
                _mm_prefetch(reinterpret_cast<const char*>(p_ptr_prefetch), _MM_HINT_T0);
            }
            if (s + prefetch_d < S) {
                const int64_t n_dt = dt + prefetch_d;
                const int64_t n_tbin = lgh_wrap_row(n_dt, t_bins);
                const int64_t next_base = lgh_clamp_row(idx_ptr[s + prefetch_d], N3);
                const int64_t next_row3 = lgh_temporal_fold_row(next_base, n_dt, N3, t_bins, Core::g_hpc_fold_alpha);
                const int64_t next_row = next_row3 * atlas_bins + (n_tbin % atlas_bins);
                const int8_t* next_ptr = q_ptr + next_row * M;
                _mm_prefetch(reinterpret_cast<const char*>(next_ptr), _MM_HINT_T0);
            }
            for (int64_t off = -wave_r; off <= wave_r; off++) {
                const int64_t w_row3 = lgh_wrap_row(row + off, N3);
                const int64_t w_row = w_row3 * atlas_bins + (tbin % atlas_bins);
                const float phase = 0.5f + 0.5f * std::cos(0.35f * static_cast<float>(dt + off));
                const float ripple = std::exp(-std::abs(static_cast<float>(off)) * wave_decay_f);
                const float trace_v = trace_ptr[w_row];
                const float fatigue = lgh_trace_fatigue(trace_v);
                const float coeff = ripple * phase * fatigue;
                const int8_t* row_ptr = q_ptr + w_row * M;
                const float row_scale = sc_ptr[w_row];
                const float total_scale = coeff * row_scale;
#ifdef __AVX512F
                int64_t m = 0;
                const __m512 scale_v = _mm512_set1_ps(total_scale);
                for (; m + 16 <= M; m += 16) {
                    __m128i row_int8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + m));
                    __m512 row_f32 = _mm512_cvtepi8_ps(row_int8);
                    __m512 a = _mm512_loadu_ps(proto_ptr + m);
                    _mm512_storeu_ps(proto_ptr + m, _mm512_fmadd_ps(row_f32, scale_v, a));
                }
                for (; m < M; m++) {
                    proto_ptr[m] += static_cast<float>(row_ptr[m]) * total_scale;
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    proto_ptr[m] += static_cast<float>(row_ptr[m]) * total_scale;
                }
#endif
                trace_ptr[w_row] = lgh_trace_pulse_update(trace_v, trace_gain_f, std::abs(coeff));
                contrib_count += std::max(0.001f, ripple * phase);
            }
        }
        const float inv_s = 1.0f / std::max(1e-6f, contrib_count);
#ifdef __AVX512F
        int64_t m = 0;
        const __m512 inv = _mm512_set1_ps(inv_s);
        for (; m + 16 <= M; m += 16) {
            __m512 v = _mm512_loadu_ps(proto_ptr + m);
            _mm512_storeu_ps(proto_ptr + m, _mm512_mul_ps(v, inv));
        }
        for (; m < M; m++) {
            proto_ptr[m] *= inv_s;
        }
#else
        for (int64_t m = 0; m < M; m++) {
            proto_ptr[m] *= inv_s;
        }
#endif
    }

    std::vector<uint64_t> mdna_words(static_cast<size_t>(R), 0ULL);
    std::vector<int64_t> mdna_active_lanes(static_cast<size_t>(R), 0);
    for (int64_t r = 0; r < R; r++) {
        mdna_words[static_cast<size_t>(r)] = lgh_region_mdna_word(mdna_ptr, L, R, r);
        mdna_active_lanes[static_cast<size_t>(r)] =
            std::max<int64_t>(1, (lgh_popcount64(mdna_words[static_cast<size_t>(r)]) * M) / 64);
        region_scale_ptr[r] = lgh_region_scale(g_ptr, mdna_ptr, L, R, r, thermal_scale);
    }

    std::vector<float> halt_acc(static_cast<size_t>(B * T), 0.0f);
    uint64_t pulse_ops = 0ULL;
    uint64_t folded_steps = 0ULL;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t t = 0; t < T; t++) {
            const int64_t p_base = (b * T + t) * M;
            const int64_t p_prev_base = (t > 0) ? ((b * T + (t - 1)) * M) : -1;
            if (t > 0 && lgh_should_temporal_fold(p_ptr, p_base, p_prev_base, M, temporal_fold)) {
                const int64_t bt_prev = b * T + (t - 1);
                const int64_t bt_cur = b * T + t;
                for (int64_t r = 0; r < R; r++) {
                    const int64_t src_base = ((b * T + (t - 1)) * R + r) * M;
                    const int64_t dst_base = ((b * T + t) * R + r) * M;
                    std::memcpy(out_ptr + dst_base, out_ptr + src_base, static_cast<size_t>(M) * sizeof(float));
                }
                halt_acc[static_cast<size_t>(bt_cur)] = halt_acc[static_cast<size_t>(bt_prev)];
                folded_steps++;
                continue;
            }
            for (int64_t r = 0; r < R; r++) {
                const float scale = region_scale_ptr[r] * cycle_scale;
                const int64_t out_base = ((b * T + t) * R + r) * M;
                const uint64_t mdna_word = mdna_words[static_cast<size_t>(r)];
                if (scale > eps) {
                    pulse_ops += static_cast<uint64_t>(mdna_active_lanes[static_cast<size_t>(r)]);
                }
#ifdef __AVX512F
                const __m512 scale_v = _mm512_set1_ps(scale);
                int64_t m = 0;
                int64_t chunk = 0;
                for (; m + 16 <= M; m += 16, chunk++) {
                    __mmask16 k_mdna = lgh_mdna_mask16(mdna_word, chunk);
                    if (!(scale > eps)) {
                        k_mdna = static_cast<__mmask16>(0x0000);
                    }
                    __m512 p_v = _mm512_loadu_ps(p_ptr + p_base + m);
                    __m512 proto_v = _mm512_loadu_ps(proto_ptr + m);
                    __m512 merged = _mm512_add_ps(p_v, proto_v);
                    __m512 gated = lgh_pulse_eval_avx512(merged, scale_v, k_mdna);
                    _mm512_storeu_ps(out_ptr + out_base + m, gated);
                    __m512 abs_v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), gated);
                    halt_acc[static_cast<size_t>(b * T + t)] += _mm512_reduce_add_ps(abs_v);
                }
                for (; m < M; m++) {
                    const float merged = p_ptr[p_base + m] + proto_ptr[m];
                    const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                    const float v = ((scale > eps) && (bit != 0)) ? (merged * scale) : 0.0f;
                    out_ptr[out_base + m] = v;
                    halt_acc[static_cast<size_t>(b * T + t)] += std::abs(v);
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    const float merged = p_ptr[p_base + m] + proto_ptr[m];
                    const int bit = static_cast<int>((mdna_word >> (m & 63)) & 1ULL);
                    const float v = ((scale > eps) && (bit != 0)) ? (merged * scale) : 0.0f;
                    out_ptr[out_base + m] = v;
                    halt_acc[static_cast<size_t>(b * T + t)] += std::abs(v);
                }
#endif
            }
        }
    }

    for (int64_t b = 0; b < B; b++) {
        for (int64_t r = 0; r < R; r++) {
            const int64_t src_base = ((b * T + (T - 1)) * R + r) * M;
            for (int64_t l = 0; l < L; l++) {
                const float gate_lr = std::max(0.0f, std::min(1.0f, g_ptr[l * R + r]));
                const float mdna_lr = std::max(0.0f, std::min(1.0f, mdna_ptr[l * R + r]));
                const float mix = std::max(0.0f, std::min(1.0f, gate_lr * mdna_lr * thermal_scale));
                const float decay = 0.90f + 0.08f * (1.0f - mix);
                const int64_t h_base = (((b * L + l) * R + r) * M);
                const uint64_t mdna_word = mdna_words[static_cast<size_t>(r)];
                const int mdna_active = (l < 64) ? static_cast<int>((mdna_word >> l) & 1ULL) : static_cast<int>(mix > eps);
#ifdef __AVX512F
                const __m512 decay_v = _mm512_set1_ps(decay);
                const __m512 mix_v = _mm512_set1_ps(mix);
                const __mmask16 k_mdna = ((mix > eps) && (mdna_active != 0))
                    ? static_cast<__mmask16>(0xFFFF)
                    : static_cast<__mmask16>(0x0000);
                int64_t m = 0;
                for (; m + 16 <= M; m += 16) {
                    __m512 old_v = _mm512_loadu_ps(h_ptr + h_base + m);
                    __m512 src_v = _mm512_loadu_ps(out_ptr + src_base + m);
                    __m512 decayed = _mm512_mul_ps(old_v, decay_v);
                    __m512 injected = lgh_pulse_eval_avx512(src_v, mix_v, k_mdna);
                    _mm512_storeu_ps(h_ptr + h_base + m, _mm512_add_ps(decayed, injected));
                }
                for (; m < M; m++) {
                    const float old_v = h_ptr[h_base + m];
                    const float src_v = out_ptr[src_base + m];
                    const int bit = mdna_active != 0;
                    h_ptr[h_base + m] = (decay * old_v) + ((bit != 0) ? (mix * src_v) : 0.0f);
                }
#else
                for (int64_t m = 0; m < M; m++) {
                    const float old_v = h_ptr[h_base + m];
                    const float src_v = out_ptr[src_base + m];
                    const int bit = mdna_active != 0;
                    h_ptr[h_base + m] = (decay * old_v) + ((bit != 0) ? (mix * src_v) : 0.0f);
                }
#endif
            }
        }
    }

    const float denom = static_cast<float>(std::max<int64_t>(1, R * M));
    const float halt_denom = static_cast<float>(std::max(1e-6, halt_threshold));
    for (int64_t bt = 0; bt < B * T; bt++) {
        const float mean_abs = halt_acc[static_cast<size_t>(bt)] / denom;
        const float raw = mean_abs / (mean_abs + halt_denom + 1e-6f);
        const float cool = std::max(0.0f, std::min(1.0f, raw * (1.0f - 0.35f * thermal)));
        halt_ptr[bt] = cool;
    }

    const long double bt_ld = static_cast<long double>(B) * static_cast<long double>(T);
    const long double r_ld = static_cast<long double>(R);
    const long double m_ld = static_cast<long double>(M);
    const long double l_ld = static_cast<long double>(L);
    const long double flops = bt_ld * r_ld * m_ld * 4.0L + bt_ld * l_ld * r_ld * m_ld * 3.0L;
    const long double bytes = bt_ld * r_ld * m_ld * 13.0L + bt_ld * l_ld * r_ld * m_ld * 8.0L;
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    const uint64_t tsc_end = __rdtsc();
    Perf::g_lgh_tsc_cycles.fetch_add((tsc_end >= tsc_start) ? (tsc_end - tsc_start) : 0ULL, std::memory_order_relaxed);
    Perf::g_lgh_pulse_ops.fetch_add(pulse_ops, std::memory_order_relaxed);
    Perf::g_lgh_temporal_folds.fetch_add(folded_steps, std::memory_order_relaxed);

    auto out_5d = out.view({B, T, R, D, C});
    return {out_5d, h_next, halt_probs};
}

void configure_hpc(float target_error, float ema_decay, float halt_gain) {
    Core::g_hpc_target_error = target_error;
    Core::g_hpc_error_ema_decay = ema_decay;
    Core::g_hpc_halt_gain = halt_gain;
}

float get_hpc_error_ema() {
    return Core::g_hpc_error_ema;
}

void configure_runtime(
    bool temporal_folding,
    float fold_alpha,
    int64_t event_mode,
    float event_density_threshold,
    bool ttfs_enabled,
    float ttfs_slope_threshold
) {
    Core::set_hpc_temporal_folding(temporal_folding, fold_alpha);
    Core::set_event_runtime(
        static_cast<int>(event_mode),
        event_density_threshold,
        ttfs_enabled,
        ttfs_slope_threshold
    );
}

void set_perf_counters_enabled(bool enabled) {
    Perf::set_enabled(enabled);
}

void reset_perf_counters() {
    Perf::reset();
}

py::dict get_perf_counters() {
    return Perf::snapshot();
}

// -------------------------------------------------------------------------
// KERNEL: Morton Curve Generation (Firmware Authority)
// -------------------------------------------------------------------------
torch::Tensor make_morton_order(int64_t L, int64_t R, int64_t Depth, int64_t align_multiple) {
    int64_t N = L * R * Depth;
    std::vector<std::pair<uint64_t, int64_t>> map;
    map.reserve(N);

    // Iterate through all coordinates
    for (int64_t l = 0; l < L; l++) {
        for (int64_t r = 0; r < R; r++) {
            for (int64_t d = 0; d < Depth; d++) {
                // Compute 4D Morton code (with t=0) using firmware intrinsic
                // Casting to uint32_t for PDEP
                uint64_t code = NIS::morton_jump_4d((uint32_t)l, (uint32_t)r, (uint32_t)d, 0);
                // Calculate original index assuming row-major (L, R, Depth)
                int64_t original_idx = l * (R * Depth) + r * Depth + d;
                map.push_back({code, original_idx});
            }
        }
    }

    // Sort by Morton code
    std::sort(map.begin(), map.end(), [](const std::pair<uint64_t, int64_t>& a, const std::pair<uint64_t, int64_t>& b) {
        return a.first < b.first;
    });

    // Extract sorted indices
    std::vector<int64_t> result;
    result.reserve(std::max(N, N + align_multiple));
    for (const auto& p : map) {
        result.push_back(p.second);
    }

    // Alignment padding (if needed)
    if (align_multiple > 1) {
        int64_t remainder = N % align_multiple;
        if (remainder != 0) {
            int64_t pad = align_multiple - remainder;
            for (int64_t i = 0; i < pad; i++) {
                result.push_back(result[i % N]); 
            }
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    return torch::from_blob(result.data(), {(int64_t)result.size()}, opts).clone();
}

// -------------------------------------------------------------------------
// PYBIND DEFINITIONS
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_scan", &parallel_scan);
    m.def("quantized_matmul", &quantized_matmul);
    m.def("rms_norm", &rms_norm);
    m.def("fused_rms_mean", &fused_rms_mean);
    m.def("dcls_ram_addresses", &dcls_ram_addresses);
    m.def("dcls_ram_lookup", &dcls_ram_lookup);
    m.def("dcls_ram_lookup_int8", &dcls_ram_lookup_int8);
    m.def("fused_lif_ram_lookup", &fused_lif_ram_lookup);
    m.def("dcls_backward", &dcls_backward);

    m.def("neural_cache_lookup_fast", &neural_cache_lookup_fast);
    m.def("forward_stack_io", &forward_stack_io);
    m.def("mes_super_step_io", &mes_super_step_io);
    m.def("survival_update_io", &survival_update_io);
    m.def("survival_mask_io", &survival_mask_io);
    m.def("survival_losses_io", &survival_losses_io);
    m.def("unified_dispatch_io", &unified_dispatch_io);
    m.def("configure_hpc", &configure_hpc);
    m.def("get_hpc_error_ema", &get_hpc_error_ema);
    m.def("configure_runtime", &configure_runtime);
    m.def("set_perf_counters_enabled", &set_perf_counters_enabled);
    m.def("reset_perf_counters", &reset_perf_counters);
    m.def("get_perf_counters", &get_perf_counters);

    m.def("ademamix_update", &ademamix_update);
    m.def("batched_ademamix_update", &batched_ademamix_update);
    m.def("fused_cognitive_step", &fused_cognitive_step);
    m.attr("CMD_FUSED_STEP") = (int)CMD_FUSED_STEP;
    m.def("fused_cognitive_cycle", &fused_cognitive_cycle_ultra);
    m.def("fused_cognitive_cycle_ultra", &fused_cognitive_cycle_ultra);
    m.def("survival_losses", &survival_losses);
    m.def("calculate_nis_sparsity", &calculate_nis_sparsity, py::arg("x"), py::arg("threshold") = 1e-7f);
    m.def("make_morton_order", &make_morton_order);

    // --- NIS FIRMWARE (Total Neural Unification) ---
    m.attr("L") = NIS::L;
    m.attr("R") = NIS::R;
    m.attr("WORKING_DIM") = NIS::WORKING_DIM;
    m.attr("C") = NIS::C;
    m.attr("MEMORY_DEPTH") = NIS::MEMORY_DEPTH;
    m.attr("H_CYCLES") = NIS::H_CYCLES;
    m.attr("L_CYCLES") = NIS::L_CYCLES;
    m.attr("RMS_NORM_EPS") = NIS::RMS_NORM_EPS;
    m.attr("ROPE_THETA") = NIS::ROPE_THETA;
    m.attr("HALT_THRESHOLD") = NIS::HALT_THRESHOLD;

    m.attr("BATCH_SIZE") = NIS::BATCH_SIZE;
    m.attr("SEQ_LEN") = NIS::SEQ_LEN;
    m.attr("LEARNING_RATE") = NIS::LEARNING_RATE;
    m.attr("EPOCHS") = NIS::EPOCHS;
    m.attr("SEED") = NIS::SEED;

    m.attr("ADEMAMIX_BETA1_FAST") = NIS::ADEMAMIX_BETA1_FAST;
    m.attr("ADEMAMIX_BETA1_SLOW") = NIS::ADEMAMIX_BETA1_SLOW;
    m.attr("ADEMAMIX_BETA2") = NIS::ADEMAMIX_BETA2;
    m.attr("WEIGHT_DECAY") = NIS::WEIGHT_DECAY;
    m.attr("AGC_CLIP_FACTOR") = NIS::AGC_CLIP_FACTOR;

    m.attr("INIT_SCALE") = NIS::INIT_SCALE;
    m.attr("DECAY_INIT_OFFSET") = NIS::DECAY_INIT_OFFSET;
    m.attr("DECAY_INIT_SCALE") = NIS::DECAY_INIT_SCALE;
    m.attr("DELAY_INIT_STD") = NIS::DELAY_INIT_STD;
    m.attr("DELAY_MIN") = NIS::DELAY_MIN;
    m.attr("DELAY_MAX") = NIS::DELAY_MAX;
    m.attr("RAM_INIT_SCALE") = NIS::RAM_INIT_SCALE;
    m.attr("DELREC_INIT_MAX") = NIS::DELREC_INIT_MAX;

    m.attr("LIF_DECAY") = NIS::LIF_DECAY;
    m.attr("LIF_THRESHOLD") = NIS::LIF_THRESHOLD;
    m.attr("H_CYCLE_THRESHOLD") = NIS::H_CYCLE_THRESHOLD;

    m.attr("GLOBAL_BACKPROP") = NIS::GLOBAL_BACKPROP;
    m.attr("LOCAL_LR_RATIO") = NIS::LOCAL_LR_RATIO;
    m.attr("MES_LOCAL_L1") = NIS::MES_LOCAL_L1;
    m.attr("SURPRISE_REWIRE_THRESHOLD") = NIS::SURPRISE_REWIRE_THRESHOLD;
    m.attr("DISSONANCE_PENALTY") = NIS::DISSONANCE_PENALTY;
    m.attr("METABOLIC_TAX_RATE") = NIS::METABOLIC_TAX_RATE;

    m.attr("SURVIVAL_GAMMA") = NIS::SURVIVAL_GAMMA;
    m.attr("SURVIVAL_UPDATE_EVERY") = NIS::SURVIVAL_UPDATE_EVERY;
    m.attr("TARGET_SPARSITY") = NIS::TARGET_SPARSITY;
    m.attr("LAMBDA_COST") = NIS::LAMBDA_COST;
    m.attr("LAMBDA_STABILITY") = NIS::LAMBDA_STABILITY;
    m.attr("LAMBDA_ENERGY") = NIS::LAMBDA_ENERGY;

    m.attr("PARAM_COST_SCALE") = NIS::PARAM_COST_SCALE;
    m.attr("MEMORY_COST_SCALE") = NIS::MEMORY_COST_SCALE;
    m.attr("FAST_PATH_COST") = NIS::FAST_PATH_COST;

    m.attr("BYPASS_H_DECAY") = NIS::BYPASS_H_DECAY;
    m.attr("CURIOSITY_EXPLORE_PROB") = NIS::CURIOSITY_EXPLORE_PROB;
    m.attr("ENGAGEMENT_THRESHOLD_MIN") = NIS::ENGAGEMENT_THRESHOLD_MIN;
    m.attr("ENGAGEMENT_THRESHOLD_MAX") = NIS::ENGAGEMENT_THRESHOLD_MAX;
    m.attr("EFFICIENCY_BONUS_CAP") = NIS::EFFICIENCY_BONUS_CAP;

    m.attr("TRAIN_SAMPLES_PER_EPOCH") = NIS::TRAIN_SAMPLES_PER_EPOCH;
    m.attr("VAL_SAMPLES_PER_EPOCH") = NIS::VAL_SAMPLES_PER_EPOCH;
    m.attr("DREAM_REPLAY_BATCHES") = NIS::DREAM_REPLAY_BATCHES;
    m.attr("MES_ENABLED") = (bool)NIS::MES_ENABLED;
    m.attr("COHERENCE_WEIGHT") = NIS::COHERENCE_WEIGHT;
    m.attr("AGC_CLIP_FACTOR") = NIS::AGC_CLIP_FACTOR;

    m.attr("CPP_OMP_THREADS") = NIS::CPP_OMP_THREADS;
    m.attr("TORCH_NUM_THREADS") = NIS::TORCH_NUM_THREADS;
    m.attr("TORCH_INTEROP_THREADS") = NIS::TORCH_INTEROP_THREADS;

    m.attr("TRAIN_LOSS_EMA_DECAY") = NIS::TRAIN_LOSS_EMA_DECAY;
    m.attr("SPARSITY_LOG_EVERY") = NIS::SPARSITY_LOG_EVERY;
    m.attr("KNOWLEDGE_GAMMA") = NIS::KNOWLEDGE_GAMMA;
    m.attr("IMPORTANCE_EVERY") = NIS::IMPORTANCE_EVERY;
    m.attr("IMPORTANCE_RATIO") = NIS::IMPORTANCE_RATIO;
    m.attr("GATE_UPDATE_EVERY") = NIS::GATE_UPDATE_EVERY;
    m.attr("RAM_PRUNE_FRACTION") = NIS::RAM_PRUNE_FRACTION;
    m.attr("RAM_CRITICAL_THRESHOLD") = NIS::RAM_CRITICAL_THRESHOLD;
    m.attr("SLEEP_INTERVAL_STEPS") = NIS::SLEEP_INTERVAL_STEPS;
    m.attr("WARM_UP_STEPS") = NIS::WARM_UP_STEPS;

    m.attr("SPARSITY_THRESHOLD") = NIS::SPARSITY_THRESHOLD;
    m.attr("MAX_LOG_ENTRIES") = NIS::MAX_LOG_ENTRIES;
    m.attr("EPSILON") = NIS::EPSILON;
    m.attr("CONFUSION_NORM") = NIS::CONFUSION_NORM;
    m.attr("SURVIVAL_WEIGHT") = NIS::SURVIVAL_WEIGHT;
    m.attr("OP_ADD") = NIS::OP_ADD;
    m.attr("OP_SCALE") = NIS::OP_SCALE;
    m.attr("OP_GATE") = NIS::OP_GATE;
    m.attr("OP_REFLECT") = NIS::OP_REFLECT;
    m.attr("OP_JMP") = NIS::OP_JMP;



    m.attr("STRICT_CPU_ONLY") = (bool)NIS::STRICT_CPU_ONLY;
    m.attr("DEVICE") = NIS::DEVICE;
    m.attr("VIRTUAL_LAB_ENABLED") = (bool)NIS::VIRTUAL_LAB_ENABLED;
    m.attr("PREFLIGHT_ENABLED") = (bool)NIS::PREFLIGHT_ENABLED;
    m.attr("NEURAL_CACHE_ENABLED") = (bool)NIS::NEURAL_CACHE_ENABLED;
    m.attr("RAM_INT8_INFER") = (bool)NIS::RAM_INT8_INFER;
    m.attr("USE_FORWARD_STACK") = (bool)NIS::USE_FORWARD_STACK;
    m.attr("USE_FUSED_COGNITIVE_CYCLE") = (bool)NIS::USE_FUSED_COGNITIVE_CYCLE;
    m.attr("TTFS_ENABLED") = (bool)NIS::TTFS_ENABLED;
    m.attr("TTFS_SLOPE_THRESHOLD") = NIS::TTFS_SLOPE_THRESHOLD;
    m.attr("EVENT_DENSITY_THRESHOLD") = NIS::EVENT_DENSITY_THRESHOLD;
    m.attr("EVENT_DRIVEN_MODE") = NIS::EVENT_DRIVEN_MODE;
    m.attr("PHASE_0_KEEP_RATIO") = NIS::PHASE_0_KEEP_RATIO;
    m.attr("DISSONANCE_CONFIDENCE_THRESHOLD") = NIS::DISSONANCE_CONFIDENCE_THRESHOLD;
    m.attr("LGH_ENABLED") = (bool)NIS::LGH_ENABLED;
    m.def("geometric_manifold_forward_avx512", [](torch::Tensor p, torch::Tensor m, torch::Tensor s, torch::Tensor c, float f) {
        return geometric_manifold_recall_avx512(p, m, s, c, f);
    });
    m.def("geometric_manifold_recall_avx512", [](torch::Tensor p, torch::Tensor m, torch::Tensor s, torch::Tensor c, float f) {
        return geometric_manifold_recall_avx512(p, m, s, c, f);
    });
    m.attr("EVENT_DRIVEN_ENABLED") = (bool)NIS::EVENT_DRIVEN_ENABLED;

    m.attr("CACHE_HASH_BITS") = NIS::CACHE_HASH_BITS;
    m.attr("LGH_CURVE_LENGTH") = NIS::LGH_CURVE_LENGTH;
    m.attr("LGH_TRACE_DECAY") = NIS::LGH_TRACE_DECAY;
    m.attr("LGH_TRACE_GAIN") = NIS::LGH_TRACE_GAIN;
    m.attr("LGH_FOCUS_STRENGTH") = NIS::LGH_FOCUS_STRENGTH;
    m.attr("LGH_FOCUS_SHARPNESS") = NIS::LGH_FOCUS_SHARPNESS;

    m.attr("HPC_ENABLED") = (bool)NIS::HPC_ENABLED;
    m.attr("HPC_HIDDEN") = NIS::HPC_HIDDEN;
    m.attr("HPC_TARGET_ERROR") = NIS::HPC_TARGET_ERROR;
    m.attr("HPC_HALT_GAIN") = NIS::HPC_HALT_GAIN;
    m.attr("HPC_ERROR_EMA_DECAY") = NIS::HPC_ERROR_EMA_DECAY;
    m.attr("HPC_TEMPORAL_FOLDING") = (bool)NIS::HPC_TEMPORAL_FOLDING;
    m.attr("HPC_FOLD_ALPHA") = NIS::HPC_FOLD_ALPHA;
    m.attr("HPC_SURPRISE_THRESHOLD") = NIS::HPC_SURPRISE_THRESHOLD;
    m.attr("HPC_TEMPORAL_THRESHOLD") = NIS::HPC_TEMPORAL_THRESHOLD;

    m.attr("LGH_REPLACE_FORWARD_STACK") = (bool)NIS::LGH_REPLACE_FORWARD_STACK;
    m.attr("LGH_CURVE_WRAP") = (bool)NIS::LGH_CURVE_WRAP;
    m.attr("LGH_MASK_MIN_KEEP") = NIS::LGH_MASK_MIN_KEEP;
    m.attr("LGH_MASK_MAX_KEEP") = NIS::LGH_MASK_MAX_KEEP;
    m.attr("LGH_PREFETCH_DISTANCE") = NIS::LGH_PREFETCH_DISTANCE;
    m.attr("LGH_THERMAL_FREQ_MIN_GHZ") = NIS::LGH_THERMAL_FREQ_MIN_GHZ;
    m.attr("LGH_THERMAL_EMA_DECAY") = NIS::LGH_THERMAL_EMA_DECAY;
    m.attr("LGH_THERMAL_PENALTY_WEIGHT") = NIS::LGH_THERMAL_PENALTY_WEIGHT;
    m.attr("LGH_THERMAL_SAMPLE_EVERY_S") = NIS::LGH_THERMAL_SAMPLE_EVERY_S;
    m.attr("LGH_SIMD_CYCLE_PENALTY_WEIGHT") = NIS::LGH_SIMD_CYCLE_PENALTY_WEIGHT;
    m.attr("LGH_SIMD_STARVATION_THRESHOLD") = NIS::LGH_SIMD_STARVATION_THRESHOLD;
    m.attr("LGH_ALIGN_MULTIPLE") = NIS::LGH_ALIGN_MULTIPLE;
    m.attr("LGH_TEMPORAL_BINS") = NIS::LGH_TEMPORAL_BINS;
    m.attr("LGH_TEMPORAL_FOLD_ALPHA") = NIS::LGH_TEMPORAL_FOLD_ALPHA;
    m.attr("LGH_WAVE_RADIUS") = NIS::LGH_WAVE_RADIUS;
    m.attr("LGH_WAVE_DECAY") = NIS::LGH_WAVE_DECAY;
    m.attr("LGH_LOW_ENTROPY_FOLD_THRESHOLD") = NIS::LGH_LOW_ENTROPY_FOLD_THRESHOLD;
    m.attr("LGH_INT4_UNCERTAINTY_THRESHOLD") = NIS::LGH_INT4_UNCERTAINTY_THRESHOLD;
    m.attr("LGH_FP32_UNCERTAINTY_THRESHOLD") = NIS::LGH_FP32_UNCERTAINTY_THRESHOLD;

    m.attr("HPC_CYCLE_MIN_SCALE") = NIS::HPC_CYCLE_MIN_SCALE;
    m.attr("HPC_CYCLE_MAX_SCALE") = NIS::HPC_CYCLE_MAX_SCALE;
    m.attr("HPC_H_CYCLES_MAX") = NIS::HPC_H_CYCLES_MAX;
    m.attr("HPC_L_CYCLES_MAX") = NIS::HPC_L_CYCLES_MAX;
    m.attr("HPC_MONITOR_EVERY") = NIS::HPC_MONITOR_EVERY;
    m.attr("HPC_LOCAL_LOSS_WEIGHT") = NIS::HPC_LOCAL_LOSS_WEIGHT;
    m.attr("HPC_SURPRISE_GATE") = (bool)NIS::HPC_SURPRISE_GATE;
    m.attr("HPC_SURPRISE_MIN_SCALE") = NIS::HPC_SURPRISE_MIN_SCALE;
    m.attr("HPC_SURPRISE_SKIP_ENABLED") = (bool)NIS::HPC_SURPRISE_SKIP_ENABLED;
    m.attr("HPC_SURPRISE_SKIP_SCALE") = NIS::HPC_SURPRISE_SKIP_SCALE;
    m.attr("HPC_TEMPORAL_GATE_ENABLED") = (bool)NIS::HPC_TEMPORAL_GATE_ENABLED;
    m.attr("HPC_TEMPORAL_GATE_THRESHOLD") = NIS::HPC_TEMPORAL_GATE_THRESHOLD;
    m.attr("HPC_TEMPORAL_GATE_MIN_SCALE") = NIS::HPC_TEMPORAL_GATE_MIN_SCALE;
    m.attr("HPC_TEMPORAL_GATE_SKIP_ENABLED") = (bool)NIS::HPC_TEMPORAL_GATE_SKIP_ENABLED;
    m.attr("HPC_TEMPORAL_GATE_SKIP_SCALE") = NIS::HPC_TEMPORAL_GATE_SKIP_SCALE;
    m.attr("HPC_TEMPORAL_GATE_WINDOW") = NIS::HPC_TEMPORAL_GATE_WINDOW;
    m.attr("HPC_SURPRISE_LOSS_WEIGHT") = NIS::HPC_SURPRISE_LOSS_WEIGHT;

    m.attr("AUDIT_PERIOD_STEPS") = NIS::AUDIT_PERIOD_STEPS;
    m.attr("AUDIT_RANDOM_PROB") = NIS::AUDIT_RANDOM_PROB;
    m.attr("IMPORTANCE_STD_FACTOR") = NIS::IMPORTANCE_STD_FACTOR;
    m.attr("IMPORTANCE_EMA_DECAY") = NIS::IMPORTANCE_EMA_DECAY;
    m.attr("MES_SUPER_KERNEL") = (bool)NIS::MES_SUPER_KERNEL;
    m.attr("MES_SKIP_STEP") = NIS::MES_SKIP_STEP;
    m.attr("CONSOLIDATE_EVERY") = NIS::CONSOLIDATE_EVERY;
    m.attr("PHENOTYPE_UPDATE_EVERY") = NIS::PHENOTYPE_UPDATE_EVERY;
    m.attr("OMEGA_STEP") = NIS::OMEGA_STEP;
    m.attr("OMEGA_STEP_UPDATE_EVERY") = NIS::OMEGA_STEP_UPDATE_EVERY;
    m.attr("GENOME_STEP_UPDATE_EVERY") = NIS::GENOME_STEP_UPDATE_EVERY;
    m.attr("MERGE_POLICY") = NIS::MERGE_POLICY;
    m.attr("DYNAMIC_ENERGY_SCALE") = NIS::DYNAMIC_ENERGY_SCALE;
    m.attr("REFLEX_DROPOUT_RATE") = NIS::REFLEX_DROPOUT_RATE;
    m.attr("GLOBAL_PULSE_EVERY") = NIS::GLOBAL_PULSE_EVERY;
    m.attr("GLOBAL_PULSE_WEIGHT") = NIS::GLOBAL_PULSE_WEIGHT;


}
