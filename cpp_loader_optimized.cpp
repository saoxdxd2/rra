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

#ifdef __AVX512F
#include <immintrin.h>
#endif

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
namespace py = pybind11;

// =========================================================================
// CORE UTILITIES (Internal)
// =========================================================================
namespace Core {
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
    static std::atomic<uint64_t> g_titans_forward_calls{0};
    static std::atomic<uint64_t> g_titans_update_calls{0};
    static std::atomic<uint64_t> g_cache_lookup_calls{0};
    static std::atomic<uint64_t> g_ademamix_calls{0};
    static std::atomic<uint64_t> g_batched_ademamix_calls{0};
    static std::atomic<uint64_t> g_mes_super_calls{0};
    static std::atomic<uint64_t> g_survival_update_calls{0};
    static std::atomic<uint64_t> g_survival_mask_calls{0};
    static std::atomic<uint64_t> g_survival_losses_calls{0};
    static std::atomic<uint64_t> g_fused_cognitive_calls{0};
    static std::atomic<uint64_t> g_lgh_calls{0};

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
        g_titans_forward_calls.store(0ULL, std::memory_order_relaxed);
        g_titans_update_calls.store(0ULL, std::memory_order_relaxed);
        g_cache_lookup_calls.store(0ULL, std::memory_order_relaxed);
        g_ademamix_calls.store(0ULL, std::memory_order_relaxed);
        g_batched_ademamix_calls.store(0ULL, std::memory_order_relaxed);
        g_mes_super_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_update_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_mask_calls.store(0ULL, std::memory_order_relaxed);
        g_survival_losses_calls.store(0ULL, std::memory_order_relaxed);
        g_fused_cognitive_calls.store(0ULL, std::memory_order_relaxed);
        g_lgh_calls.store(0ULL, std::memory_order_relaxed);
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
        out["titans_forward_calls"] = py::int_(g_titans_forward_calls.load(std::memory_order_relaxed));
        out["titans_update_calls"] = py::int_(g_titans_update_calls.load(std::memory_order_relaxed));
        out["cache_lookup_calls"] = py::int_(g_cache_lookup_calls.load(std::memory_order_relaxed));
        out["ademamix_calls"] = py::int_(g_ademamix_calls.load(std::memory_order_relaxed));
        out["batched_ademamix_calls"] = py::int_(g_batched_ademamix_calls.load(std::memory_order_relaxed));
        out["mes_super_calls"] = py::int_(g_mes_super_calls.load(std::memory_order_relaxed));
        out["survival_update_calls"] = py::int_(g_survival_update_calls.load(std::memory_order_relaxed));
        out["survival_mask_calls"] = py::int_(g_survival_mask_calls.load(std::memory_order_relaxed));
        out["survival_losses_calls"] = py::int_(g_survival_losses_calls.load(std::memory_order_relaxed));
        out["fused_cognitive_calls"] = py::int_(g_fused_cognitive_calls.load(std::memory_order_relaxed));
        out["lgh_calls"] = py::int_(g_lgh_calls.load(std::memory_order_relaxed));
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
// KERNEL: Titans Memory Forward (nearest-memory associative recall)
// -------------------------------------------------------------------------
std::vector<torch::Tensor> titans_memory_forward(
    torch::Tensor x,            // [B, D]
    torch::Tensor memory_keys,  // [N, D]
    torch::Tensor memory_values,// [N, D_out]
    float temperature
) {
    (void)temperature;
    check_cpu(x, "x");
    check_cpu(memory_keys, "memory_keys");
    check_cpu(memory_values, "memory_values");
    check_dim(x, 2, "x");
    check_dim(memory_keys, 2, "memory_keys");
    check_dim(memory_values, 2, "memory_values");
    check_dtype(x, at::kFloat, "x");
    check_dtype(memory_keys, at::kFloat, "memory_keys");
    check_dtype(memory_values, at::kFloat, "memory_values");

    auto x_c = ensure_contig(x);
    auto k_c = ensure_contig(memory_keys);
    auto v_c = ensure_contig(memory_values);

    const int64_t B = x_c.size(0);
    const int64_t D = x_c.size(1);
    const int64_t N = k_c.size(0);
    const int64_t Dk = k_c.size(1);
    Perf::g_titans_forward_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(D == Dk, "x and memory_keys feature dimensions must match.");
    TORCH_CHECK(v_c.size(0) == N, "memory_values first dimension must match memory_keys.");
    const int64_t Dv = v_c.size(1);

    auto output = torch::zeros({B, Dv}, x_c.options());
    auto scores = torch::zeros({B}, x_c.options());
    if (N == 0) {
        Perf::add(0ULL, 0ULL);
        return {output, scores};
    }

    const float* x_ptr = x_c.data_ptr<float>();
    const float* k_ptr = k_c.data_ptr<float>();
    const float* v_ptr = v_c.data_ptr<float>();
    float* o_ptr = output.data_ptr<float>();
    float* s_ptr = scores.data_ptr<float>();

    std::vector<float> key_norms(static_cast<size_t>(N), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n++) {
        const int64_t base = n * D;
        double sq = 0.0;
        for (int64_t d = 0; d < D; d++) {
            const float kv = k_ptr[base + d];
            sq += static_cast<double>(kv) * static_cast<double>(kv);
        }
        key_norms[static_cast<size_t>(n)] = static_cast<float>(std::sqrt(std::max(0.0, sq)) + 1e-8);
    }

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        const int64_t xb = b * D;
        double x_sq = 0.0;
        for (int64_t d = 0; d < D; d++) {
            const float xv = x_ptr[xb + d];
            x_sq += static_cast<double>(xv) * static_cast<double>(xv);
        }
        const float x_norm = static_cast<float>(std::sqrt(std::max(0.0, x_sq)) + 1e-8);

        float best_sim = -std::numeric_limits<float>::infinity();
        int64_t best_idx = -1;
        for (int64_t n = 0; n < N; n++) {
            const int64_t kb = n * D;
            double dot = 0.0;
            for (int64_t d = 0; d < D; d++) {
                dot += static_cast<double>(x_ptr[xb + d]) * static_cast<double>(k_ptr[kb + d]);
            }
            const float sim = static_cast<float>(dot / (static_cast<double>(x_norm) * static_cast<double>(key_norms[static_cast<size_t>(n)])));
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = n;
            }
        }

        s_ptr[b] = std::isfinite(best_sim) ? best_sim : 0.0f;
        if (best_idx >= 0) {
            const int64_t vb = best_idx * Dv;
            const int64_t ob = b * Dv;
            for (int64_t d = 0; d < Dv; d++) {
                o_ptr[ob + d] = v_ptr[vb + d];
            }
        }
    }

    const long double b_ld = static_cast<long double>(B);
    const long double d_ld = static_cast<long double>(D);
    const long double n_ld = static_cast<long double>(N);
    const long double dv_ld = static_cast<long double>(Dv);
    const long double flops = n_ld * (2.0L * d_ld + 1.0L)
        + b_ld * (2.0L * d_ld + 1.0L + n_ld * (2.0L * d_ld + 2.0L));
    const long double bytes =
        n_ld * d_ld * 4.0L +
        b_ld * d_ld * 4.0L +
        b_ld * n_ld * d_ld * 8.0L +
        b_ld * n_ld * 4.0L +
        b_ld * dv_ld * 8.0L +
        b_ld * 4.0L;
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return {output, scores};
}

// -------------------------------------------------------------------------
// KERNEL: Titans Memory Update (append + keep latest max_memories)
// -------------------------------------------------------------------------
std::vector<torch::Tensor> titans_memory_update(
    torch::Tensor memory_keys,   // [N, D] or empty
    torch::Tensor memory_values, // [N, Dv] or empty
    torch::Tensor new_keys,      // [B, D] or [D]
    torch::Tensor new_values,    // [B, Dv] or [Dv]
    int64_t max_memories
) {
    check_cpu(memory_keys, "memory_keys");
    check_cpu(memory_values, "memory_values");
    check_cpu(new_keys, "new_keys");
    check_cpu(new_values, "new_values");
    check_dtype(memory_keys, at::kFloat, "memory_keys");
    check_dtype(memory_values, at::kFloat, "memory_values");
    check_dtype(new_keys, at::kFloat, "new_keys");
    check_dtype(new_values, at::kFloat, "new_values");
    TORCH_CHECK(max_memories > 0, "max_memories must be > 0.");

    auto k_new = ensure_contig(new_keys.dim() == 1 ? new_keys.unsqueeze(0) : new_keys);
    auto v_new = ensure_contig(new_values.dim() == 1 ? new_values.unsqueeze(0) : new_values);
    check_dim(k_new, 2, "new_keys");
    check_dim(v_new, 2, "new_values");
    TORCH_CHECK(k_new.size(0) == v_new.size(0), "new_keys and new_values batch dimension must match.");

    const int64_t B = k_new.size(0);
    const int64_t D = k_new.size(1);
    const int64_t Dv = v_new.size(1);
    Perf::g_titans_update_calls.fetch_add(1ULL, std::memory_order_relaxed);

    auto k_norm = k_new.clone();
    float* kn_ptr = k_norm.data_ptr<float>();
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        const int64_t base = b * D;
        float sum_sq = 0.0f;
        for (int64_t d = 0; d < D; d++) {
            const float v = kn_ptr[base + d];
            sum_sq += v * v;
        }
        const float inv = 1.0f / std::sqrt(sum_sq + 1e-8f);
        for (int64_t d = 0; d < D; d++) {
            kn_ptr[base + d] *= inv;
        }
    }

    at::Tensor k_old = memory_keys;
    at::Tensor v_old = memory_values;
    if (memory_keys.numel() == 0) {
        k_old = torch::empty({0, D}, k_norm.options());
    } else {
        check_dim(memory_keys, 2, "memory_keys");
        TORCH_CHECK(memory_keys.size(1) == D, "memory_keys feature dimension must match new_keys.");
        k_old = ensure_contig(memory_keys);
    }
    if (memory_values.numel() == 0) {
        v_old = torch::empty({0, Dv}, v_new.options());
    } else {
        check_dim(memory_values, 2, "memory_values");
        TORCH_CHECK(memory_values.size(1) == Dv, "memory_values feature dimension must match new_values.");
        TORCH_CHECK(memory_values.size(0) == k_old.size(0), "memory_values row count must match memory_keys.");
        v_old = ensure_contig(memory_values);
    }

    auto k_cat = torch::cat({k_old, k_norm}, 0);
    auto v_cat = torch::cat({v_old, v_new}, 0);
    const int64_t total = k_cat.size(0);
    if (total > max_memories) {
        const int64_t start = total - max_memories;
        k_cat = k_cat.narrow(0, start, max_memories).contiguous();
        v_cat = v_cat.narrow(0, start, max_memories).contiguous();
    }
    const long double b_ld = static_cast<long double>(B);
    const long double d_ld = static_cast<long double>(D);
    Perf::add(
        Perf::sat_from_ld(b_ld * (3.0L * d_ld + 1.0L)),
        Perf::sat_from_ld(b_ld * (8.0L * d_ld))
    );
    return {k_cat, v_cat};
}

// -------------------------------------------------------------------------
// KERNEL: Neural Cache Fast Lookup
// -------------------------------------------------------------------------
std::vector<torch::Tensor> neural_cache_lookup_fast(
    torch::Tensor query,      // [B, D] float32
    torch::Tensor keys,       // [TBL, RAM, D] bf16/float
    torch::Tensor values,     // [TBL, RAM, O] bf16/float
    torch::Tensor addresses,  // [B, TBL] long
    torch::Tensor valid,      // [TBL, RAM] bool
    float key_similarity_threshold,
    bool use_avx512
) {
    (void)use_avx512;
    check_cpu(query, "query");
    check_cpu(keys, "keys");
    check_cpu(values, "values");
    check_cpu(addresses, "addresses");
    check_cpu(valid, "valid");
    check_dim(query, 2, "query");
    check_dim(keys, 3, "keys");
    check_dim(values, 3, "values");
    check_dim(addresses, 2, "addresses");
    check_dim(valid, 2, "valid");
    check_dtype(query, at::kFloat, "query");
    TORCH_CHECK(keys.scalar_type() == at::kFloat || keys.scalar_type() == at::kBFloat16, "keys must be float32 or bfloat16.");
    TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kBFloat16, "values must be float32 or bfloat16.");
    check_dtype(addresses, at::kLong, "addresses");
    check_dtype(valid, at::kBool, "valid");

    auto q_c = ensure_contig(query);
    auto k_c = ensure_contig(keys.to(torch::kFloat32));
    auto v_c = ensure_contig(values.to(torch::kFloat32));
    auto a_c = ensure_contig(addresses);
    auto m_c = ensure_contig(valid);

    const int64_t B = q_c.size(0);
    const int64_t D = q_c.size(1);
    const int64_t TBL = k_c.size(0);
    const int64_t RAM = k_c.size(1);
    const int64_t KD = k_c.size(2);
    Perf::g_cache_lookup_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(KD == D, "keys last dim must match query dim.");
    TORCH_CHECK(v_c.size(0) == TBL && v_c.size(1) == RAM, "values first two dims must match keys.");
    TORCH_CHECK(a_c.size(0) == B && a_c.size(1) == TBL, "addresses must be [B, TBL].");
    TORCH_CHECK(m_c.size(0) == TBL && m_c.size(1) == RAM, "valid must be [TBL, RAM].");
    const int64_t O = v_c.size(2);

    auto out = torch::zeros({B, O}, q_c.options());
    auto hit_mask = torch::zeros({B}, q_c.options().dtype(torch::kBool));

    const float* q_ptr = q_c.data_ptr<float>();
    const float* k_ptr = k_c.data_ptr<float>();
    const float* val_ptr = v_c.data_ptr<float>();
    const int64_t* addr_ptr = a_c.data_ptr<int64_t>();
    const bool* valid_ptr = m_c.data_ptr<bool>();
    float* out_ptr = out.data_ptr<float>();
    bool* hit_ptr = hit_mask.data_ptr<bool>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; b++) {
        const int64_t q_base = b * D;
        float best_sim = -std::numeric_limits<float>::infinity();
        int64_t best_tbl = -1;
        int64_t best_slot = 0;

        float q_norm_sq = 0.0f;
        for (int64_t d = 0; d < D; d++) {
            const float qv = q_ptr[q_base + d];
            q_norm_sq += qv * qv;
        }
        const float q_norm = std::sqrt(q_norm_sq) + 1e-8f;

        for (int64_t t = 0; t < TBL; t++) {
            int64_t addr = addr_ptr[b * TBL + t];
            if (addr < 0) {
                addr = 0;
            }
            int64_t slot = addr % RAM;
            if (slot < 0) {
                slot += RAM;
            }
            if (!valid_ptr[t * RAM + slot]) {
                continue;
            }

            const int64_t k_base = (t * RAM + slot) * D;
            float dot = 0.0f;
            float k_norm_sq = 0.0f;
            for (int64_t d = 0; d < D; d++) {
                const float kv = k_ptr[k_base + d];
                dot += q_ptr[q_base + d] * kv;
                k_norm_sq += kv * kv;
            }
            const float sim = dot / (q_norm * (std::sqrt(k_norm_sq) + 1e-8f));
            if (sim > best_sim) {
                best_sim = sim;
                best_tbl = t;
                best_slot = slot;
            }
        }

        const bool hit = (best_tbl >= 0) && std::isfinite(best_sim) && (best_sim >= key_similarity_threshold);
        hit_ptr[b] = hit;
        if (hit) {
            const int64_t src = (best_tbl * RAM + best_slot) * O;
            const int64_t dst = b * O;
            for (int64_t o = 0; o < O; o++) {
                out_ptr[dst + o] = val_ptr[src + o];
            }
        }
    }

    const long double b_ld = static_cast<long double>(B);
    const long double d_ld = static_cast<long double>(D);
    const long double tbl_ld = static_cast<long double>(TBL);
    const long double o_ld = static_cast<long double>(O);
    const long double flops = b_ld * (2.0L * d_ld + 1.0L + tbl_ld * (2.0L * d_ld + 2.0L));
    const long double bytes = b_ld * (
        4.0L * d_ld +
        tbl_ld * (8.0L + 1.0L + 8.0L * d_ld) +
        o_ld * 8.0L + 1.0L
    );
    Perf::add(Perf::sat_from_ld(flops), Perf::sat_from_ld(bytes));
    return {out, hit_mask};
}

// -------------------------------------------------------------------------
// KERNEL: AdEMAMix (Optimizer)
// -------------------------------------------------------------------------
void ademamix_update(
    torch::Tensor p, torch::Tensor grad, 
    torch::Tensor m_fast, torch::Tensor m_slow, torch::Tensor v,
    float lr, float beta1_fast, float beta1_slow, float beta2,
    float alpha, float eps, float weight_decay, int step
) {
    check_cpu(p, "p");
    check_cpu(grad, "grad");
    check_cpu(m_fast, "m_fast");
    check_cpu(m_slow, "m_slow");
    check_cpu(v, "v");
    check_dtype(p, at::kFloat, "p");
    check_dtype(grad, at::kFloat, "grad");
    check_dtype(m_fast, at::kFloat, "m_fast");
    check_dtype(m_slow, at::kFloat, "m_slow");
    check_dtype(v, at::kFloat, "v");
    TORCH_CHECK(
        p.numel() == grad.numel() && p.numel() == m_fast.numel() && p.numel() == m_slow.numel() && p.numel() == v.numel(),
        "All optimizer tensors must have the same number of elements."
    );

    auto p_c = ensure_contig(p);
    auto g_c = ensure_contig(grad);
    auto mf_c = ensure_contig(m_fast);
    auto ms_c = ensure_contig(m_slow);
    auto v_c = ensure_contig(v);
    
    int64_t N = p_c.numel();
    Perf::g_ademamix_calls.fetch_add(1ULL, std::memory_order_relaxed);
    float* p_ptr = p_c.data_ptr<float>();
    float* g_ptr = g_c.data_ptr<float>();
    float* mf_ptr = mf_c.data_ptr<float>();
    float* ms_ptr = ms_c.data_ptr<float>();
    float* v_ptr = v_c.data_ptr<float>();
    
    float bc1_f = 1.0f - std::pow(beta1_fast, (float)step + 1.0f);
    float bc1_s = 1.0f - std::pow(beta1_slow, (float)step + 1.0f);
    float bc2 = 1.0f - std::pow(beta2, (float)step + 1.0f);
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) {
        float g = g_ptr[i];
        
        // Weight Decay
        if (weight_decay != 0.0f) {
            p_ptr[i] *= (1.0f - lr * weight_decay);
        }
        
        mf_ptr[i] = beta1_fast * mf_ptr[i] + (1.0f - beta1_fast) * g;
        ms_ptr[i] = beta1_slow * ms_ptr[i] + (1.0f - beta1_slow) * g;
        v_ptr[i]   = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
        
        float mf_hat = mf_ptr[i] / bc1_f;
        float ms_hat = ms_ptr[i] / bc1_s;
        float v_hat  = v_ptr[i] / bc2;
        
        float m_mixed = alpha * ms_hat + (1.0f - alpha) * mf_hat;
        p_ptr[i] -= lr * (m_mixed / (std::sqrt(v_hat) + eps));
    }
    const long double n = static_cast<long double>(N);
    Perf::add(
        Perf::sat_from_ld(14.0L * n),
        Perf::sat_from_ld(36.0L * n)
    );
}

// -------------------------------------------------------------------------
// KERNEL: Batched AdEMAMix (List of Parameters)
// -------------------------------------------------------------------------
void batched_ademamix_update(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> m_fast_list,
    std::vector<torch::Tensor> m_slow_list,
    std::vector<torch::Tensor> v_list,
    float lr, float beta1_fast, float beta1_slow, float beta2,
    float alpha, float eps, int step
) {
    Perf::g_batched_ademamix_calls.fetch_add(1ULL, std::memory_order_relaxed);
    const size_t n_params = params.size();
    TORCH_CHECK(
        grads.size() == n_params &&
        m_fast_list.size() == n_params &&
        m_slow_list.size() == n_params &&
        v_list.size() == n_params,
        "All tensor lists must have the same length."
    );

    const float bc1_f = 1.0f - std::pow(beta1_fast, (float)step + 1.0f);
    const float bc1_s = 1.0f - std::pow(beta1_slow, (float)step + 1.0f);
    const float bc2 = 1.0f - std::pow(beta2, (float)step + 1.0f);

    long double total_elems = 0.0L;
    for (size_t i = 0; i < n_params; i++) {
        auto& p = params[i];
        auto& g = grads[i];
        auto& mf = m_fast_list[i];
        auto& ms = m_slow_list[i];
        auto& v = v_list[i];

        check_cpu(p, "params[i]");
        check_cpu(g, "grads[i]");
        check_cpu(mf, "m_fast_list[i]");
        check_cpu(ms, "m_slow_list[i]");
        check_cpu(v, "v_list[i]");
        check_dtype(p, at::kFloat, "params[i]");
        check_dtype(g, at::kFloat, "grads[i]");
        check_dtype(mf, at::kFloat, "m_fast_list[i]");
        check_dtype(ms, at::kFloat, "m_slow_list[i]");
        check_dtype(v, at::kFloat, "v_list[i]");
        TORCH_CHECK(
            p.is_contiguous() && g.is_contiguous() && mf.is_contiguous() && ms.is_contiguous() && v.is_contiguous(),
            "batched_ademamix_update expects contiguous tensors."
        );
        TORCH_CHECK(
            p.numel() == g.numel() && p.numel() == mf.numel() && p.numel() == ms.numel() && p.numel() == v.numel(),
            "Each parameter and its state tensors must have the same number of elements."
        );

        const int64_t N = p.numel();
        total_elems += static_cast<long double>(N);
        float* p_ptr = p.data_ptr<float>();
        float* g_ptr = g.data_ptr<float>();
        float* mf_ptr = mf.data_ptr<float>();
        float* ms_ptr = ms.data_ptr<float>();
        float* v_ptr = v.data_ptr<float>();

        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < N; j++) {
            const float grad = g_ptr[j];

            mf_ptr[j] = beta1_fast * mf_ptr[j] + (1.0f - beta1_fast) * grad;
            ms_ptr[j] = beta1_slow * ms_ptr[j] + (1.0f - beta1_slow) * grad;
            v_ptr[j]  = beta2 * v_ptr[j] + (1.0f - beta2) * grad * grad;

            const float mf_hat = mf_ptr[j] / bc1_f;
            const float ms_hat = ms_ptr[j] / bc1_s;
            const float v_hat  = v_ptr[j] / bc2;
            const float m_mix = alpha * ms_hat + (1.0f - alpha) * mf_hat;
            p_ptr[j] -= lr * (m_mix / (std::sqrt(v_hat) + eps));
        }
    }
    Perf::add(
        Perf::sat_from_ld(12.0L * total_elems),
        Perf::sat_from_ld(36.0L * total_elems)
    );
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
    int64_t dummy_0,
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
    TORCH_CHECK(params.size() == 6, "forward_stack_io expects 6 params: delays,tables,conns,decays,halt_w,halt_b.");
    check_cpu(scalars, "scalars");
    check_dim(scalars, 1, "scalars");
    check_dtype(scalars, at::kFloat, "scalars");
    auto s_c = ensure_contig(scalars);
    TORCH_CHECK(s_c.numel() >= 5, "forward_stack_io scalars must contain [dummy0,lif_decay,lif_threshold,halt_threshold,steps].");
    const float* s_ptr = s_c.data_ptr<float>();
    const int64_t dummy_0 = static_cast<int64_t>(std::llround(static_cast<double>(s_ptr[0])));
    const double lif_decay = static_cast<double>(s_ptr[1]);
    const double lif_threshold = static_cast<double>(s_ptr[2]);
    const double halt_threshold = static_cast<double>(s_ptr[3]);
    const int64_t steps = static_cast<int64_t>(std::llround(static_cast<double>(s_ptr[4])));
    return forward_stack(
        x_input, H_state,
        params[0], params[1], params[2], params[3], params[4], params[5],
        dummy_0, lif_decay, lif_threshold, halt_threshold, steps
    );
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
    (void)params;
    (void)scalars;
    return survival_losses(x_input, state);
}

// -------------------------------------------------------------------------
// KERNEL: Fused Cognitive Cycle (Legacy / Full)
// -------------------------------------------------------------------------
std::vector<at::Tensor> _fused_cognitive_cycle_impl(
    at::Tensor x_input, at::Tensor H_state, at::Tensor mask,
    at::Tensor all_delays, at::Tensor all_tables, at::Tensor all_connections,
    at::Tensor all_decays, at::Tensor all_halt_w, at::Tensor all_halt_b,
    int64_t H_cycles, int64_t L_cycles,
    double lif_decay, double lif_threshold, double halt_threshold,
    at::Tensor H_out
) {
    check_cpu(x_input, "x_input");
    check_cpu(H_state, "H_state");
    check_cpu(mask, "mask");
    check_cpu(all_delays, "all_delays");
    check_cpu(all_tables, "all_tables");
    check_cpu(all_connections, "all_connections");
    check_cpu(all_decays, "all_decays");
    check_cpu(all_halt_w, "all_halt_w");
    check_cpu(all_halt_b, "all_halt_b");
    check_dim(x_input, 3, "x_input");
    check_dim(H_state, 5, "H_state");
    check_dim(mask, 2, "mask");
    check_dim(all_delays, 3, "all_delays");
    check_dim(all_tables, 3, "all_tables");
    check_dim(all_connections, 3, "all_connections");
    check_dim(all_decays, 3, "all_decays");
    check_dim(all_halt_w, 2, "all_halt_w");
    check_dim(all_halt_b, 1, "all_halt_b");
    check_dtype(x_input, at::kFloat, "x_input");
    check_dtype(H_state, at::kFloat, "H_state");
    check_dtype(mask, at::kFloat, "mask");
    check_dtype(all_delays, at::kFloat, "all_delays");
    check_dtype(all_tables, at::kFloat, "all_tables");
    check_dtype(all_connections, at::kLong, "all_connections");
    check_dtype(all_decays, at::kFloat, "all_decays");
    check_dtype(all_halt_w, at::kFloat, "all_halt_w");
    check_dtype(all_halt_b, at::kFloat, "all_halt_b");
    if (H_out.defined() && H_out.numel() > 0) {
        check_cpu(H_out, "H_out");
        check_dim(H_out, 5, "H_out");
        check_dtype(H_out, at::kFloat, "H_out");
    }

    // 1. Contiguity Check (API Boundary)
    auto x_c = ensure_contig(x_input);
    auto H_c = ensure_contig(H_state);
    auto m_c = ensure_contig(mask);
    auto d_c = ensure_contig(all_delays);
    auto t_c = ensure_contig(all_tables);
    auto c_c = ensure_contig(all_connections);
    auto dec_c = ensure_contig(all_decays);
    auto hw_c = ensure_contig(all_halt_w);
    auto hb_c = ensure_contig(all_halt_b);
    
    int B = x_c.size(0);
    int T = x_c.size(1);
    int D_in = x_c.size(2);
    int L = H_c.size(1);
    int R = H_c.size(2);
    int M = d_c.size(1); 
    int K = d_c.size(2);
    Perf::g_fused_cognitive_calls.fetch_add(1ULL, std::memory_order_relaxed);
    TORCH_CHECK(K > 0 && K <= 30, "K must be in [1, 30] for safe bit addressing.");
    TORCH_CHECK(m_c.size(0) == L && m_c.size(1) == R, "mask must have shape [L, R].");
    TORCH_CHECK(d_c.size(0) == L, "all_delays first dimension must match L.");
    TORCH_CHECK(t_c.size(0) == L && t_c.size(1) == M, "all_tables must have shape [L, M, RAM].");
    TORCH_CHECK(c_c.size(0) == L && c_c.size(1) == M && c_c.size(2) == K, "all_connections must match delays shape.");
    TORCH_CHECK(dec_c.size(0) == L && dec_c.size(1) == R, "all_decays must have shape [L, R, M].");
    TORCH_CHECK(hw_c.size(0) == L, "all_halt_w first dimension must match L.");
    TORCH_CHECK(hb_c.size(0) == L, "all_halt_b must have length L.");
    // Assuming d_c is [L, M, K] or [M, K]? 
    // In organism.py: delays is [L, M, K].
    // Let's verify M vs d_c.size(1).
    // Original code: int M = d_contig.size(1). Correct.
    
    int ram_size = t_c.size(2);
    int block_size = H_c.size(3) * H_c.size(4); // D*C

    // ---------------------------------------------------------------------
    // ENTERPRISE SAFETY CHECKS
    int64_t flattened_dim = H_c.size(3) * H_c.size(4); // M = D*C
    TORCH_CHECK(M == flattened_dim, "M Neuron Mismatch");
    
    at::Tensor H_next;
    if (H_out.defined() && H_out.numel() > 0 && H_out.sizes() == H_c.sizes()) {
        H_next = H_out;
        H_next.copy_(H_c);
    } else {
        H_next = H_c.clone();
    }
    
    auto halt_probs = torch::zeros({B}, x_c.options());
    auto output = torch::zeros({B, R * flattened_dim}, x_c.options());
    auto converged = torch::zeros({B}, torch::kBool);
    
    // Raw Pointers
    float* x_ptr = x_c.data_ptr<float>();
    float* h_ptr = H_next.data_ptr<float>();
    float* m_ptr = m_c.data_ptr<float>();
    float* d_ptr = d_c.data_ptr<float>();
    float* t_ptr = t_c.data_ptr<float>();
    int64_t* c_ptr = c_c.data_ptr<int64_t>();
    float* dec_ptr = dec_c.data_ptr<float>();
    float* hw_ptr = hw_c.data_ptr<float>();
    float* hb_ptr = hb_c.data_ptr<float>();
    float* hp_ptr = halt_probs.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    bool* conv_ptr = converged.data_ptr<bool>();
    (void)hw_ptr;
    (void)hb_ptr;
    (void)hp_ptr;
    
    int total_steps = (int)(H_cycles * L_cycles);

    // Precompute RAM slots once per [B, L, M] to avoid repeated address recomputation
    // inside every cognitive step.
    const int64_t lm_stride = static_cast<int64_t>(L) * static_cast<int64_t>(M);
    const int64_t addr_count = static_cast<int64_t>(B) * lm_stride;
    std::vector<int32_t> ram_slot_cache(static_cast<size_t>(addr_count), 0);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < addr_count; idx++) {
        const int b = static_cast<int>(idx / lm_stride);
        const int lm = static_cast<int>(idx % lm_stride);
        const int l = lm / M;
        const int m = lm % M;
        const int neuron_params_base = l * M * K + m * K;
        const uint32_t addr = Core::compute_ram_address(
            x_ptr + b * T * D_in,
            d_ptr + neuron_params_base,
            c_ptr + neuron_params_base,
            T - 1,
            T,
            D_in,
            K,
            D_in,
            1
        );
        ram_slot_cache[static_cast<size_t>(idx)] = static_cast<int32_t>(addr % static_cast<uint32_t>(ram_size));
    }

    std::vector<std::vector<int>> active_regions(static_cast<size_t>(L));
    for (int l = 0; l < L; l++) {
        auto& active = active_regions[static_cast<size_t>(l)];
        active.reserve(static_cast<size_t>(R));
        for (int r = 0; r < R; r++) {
            if (m_ptr[l * R + r] >= 0.5f) {
                active.push_back(r);
            }
        }
    }
    
    #pragma omp parallel
    {
        // Thread-Local State
        static thread_local std::vector<float> h_avg;
        static thread_local std::vector<float> s_spikes;
        if (h_avg.size() < (size_t)M) {
            h_avg.resize(M);
            s_spikes.resize(M);
        }
        
        #pragma omp for schedule(static)
        for (int b = 0; b < B; b++) {
            // Base offset for this batch
            const int64_t batch_offset = static_cast<int64_t>(b) * static_cast<int64_t>(L) * static_cast<int64_t>(R) * static_cast<int64_t>(M);
            
            for (int step = 0; step < total_steps; step++) {
                if (conv_ptr[b]) break;
                
                float max_delta = 0.0f;
                
                for (int l = 0; l < L; l++) {
                    const auto& active_rs = active_regions[static_cast<size_t>(l)];
                    if (active_rs.empty()) {
                        continue;
                    }
                    const int64_t layer_offset = batch_offset + static_cast<int64_t>(l) * static_cast<int64_t>(R) * static_cast<int64_t>(M);
                
                    // 1. Average State (Across R)
                    for (int m = 0; m < M; m++) {
                        float sum = 0.0f;
                        for (int r = 0; r < R; r++) {
                            sum += h_ptr[layer_offset + r * M + m];
                        }
                        h_avg[m] = sum / (float)R;
                    }
                    
                    // 2. LIF Activation
                    for (int m = 0; m < M; m++) {
                        const int64_t addr_idx = static_cast<int64_t>(b) * lm_stride + static_cast<int64_t>(l) * static_cast<int64_t>(M) + static_cast<int64_t>(m);
                        const int32_t slot = ram_slot_cache[static_cast<size_t>(addr_idx)];
                        float u_ff = t_ptr[l * M * ram_size + m * ram_size + slot];
                        float v_curr = h_avg[m];
                        float v_next = (float)lif_decay * v_curr + u_ff;
                        
                        s_spikes[m] = (v_next > lif_threshold) ? 1.0f : 0.0f;
                    }
                    
                    // 3. Update State (Decay + Spike Injection)
                    for (size_t ri = 0; ri < active_rs.size(); ri++) {
                        int r = active_rs[ri];
                        const int64_t state_base = layer_offset + static_cast<int64_t>(r) * static_cast<int64_t>(M);
                        const int64_t decay_base = static_cast<int64_t>(l) * static_cast<int64_t>(R) * static_cast<int64_t>(M) + static_cast<int64_t>(r) * static_cast<int64_t>(M); 
                        
                        for (int m = 0; m < M; m++) {
                            float s = s_spikes[m];
                            float decay = dec_ptr[decay_base + m];
                            float old_val = h_ptr[state_base + m];
                            float new_val = decay * old_val + s;
                            
                            h_ptr[state_base + m] = new_val;
                            
                            float delta = std::abs(new_val - old_val);
                            if (delta > max_delta) max_delta = delta;
                        }
                    }
                } // End L
                
                if (max_delta < 1e-5f) {
                    conv_ptr[b] = true;
                    break;
                }
            } // End Steps
        }
    }
    
    // Output aggregation (sum over layers, then average).
    const float inv_L = 1.0f / static_cast<float>(std::max(1, L));
    const int64_t out_blocks = static_cast<int64_t>(B) * static_cast<int64_t>(R) * static_cast<int64_t>(M);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < out_blocks; idx++) {
        const int b = static_cast<int>(idx / (static_cast<int64_t>(R) * M));
        const int rm = static_cast<int>(idx % (static_cast<int64_t>(R) * M));
        float sum = 0.0f;
        const int64_t base = static_cast<int64_t>(b) * static_cast<int64_t>(L) * static_cast<int64_t>(R) * static_cast<int64_t>(M) + static_cast<int64_t>(rm);
        for (int l = 0; l < L; l++) {
            sum += h_ptr[base + l * R * M];
        }
        out_ptr[b * R * M + rm] = sum * inv_L;
    }

    const long double b_ld = static_cast<long double>(B);
    const long double l_ld = static_cast<long double>(L);
    const long double r_ld = static_cast<long double>(R);
    const long double m_ld = static_cast<long double>(M);
    const long double k_ld = static_cast<long double>(K);
    const long double steps_ld = static_cast<long double>(std::max(0, total_steps));
    const long double precompute_flops = b_ld * l_ld * m_ld * (8.0L * k_ld);
    const long double cycle_flops = b_ld * steps_ld * l_ld * m_ld * (4.0L * r_ld + 5.0L);
    const long double reduce_flops = b_ld * r_ld * m_ld * (l_ld + 1.0L);
    const long double precompute_bytes = b_ld * l_ld * m_ld * (16.0L * k_ld + 4.0L);
    const long double cycle_bytes = b_ld * steps_ld * l_ld * m_ld * (12.0L * r_ld + 16.0L);
    const long double reduce_bytes = b_ld * r_ld * m_ld * (4.0L * l_ld + 4.0L);
    Perf::add(
        Perf::sat_from_ld(precompute_flops + cycle_flops + reduce_flops),
        Perf::sat_from_ld(precompute_bytes + cycle_bytes + reduce_bytes)
    );

    // Explicitly return output to avoid garbage collection issues
    return {output, H_next, halt_probs};
}

// Wrapper for organism.py compatibility
std::vector<at::Tensor> forward_stack(
    at::Tensor x, at::Tensor H,
    at::Tensor delays, at::Tensor tables, at::Tensor conns,
    at::Tensor decays, at::Tensor hw, at::Tensor hb,
    int64_t dummy_0, // Maps to '0' passed by organism
    double lif_decay,
    double lif_threshold,
    double halt_threshold,
    int64_t steps // Maps to 'L_cycles'
) {
    at::Tensor x_in = x;
    if (x.dim() == 5) {
        // Python path uses [B, T, R, D, C]; fused kernel expects [B, T, D*C].
        int64_t B = x.size(0);
        int64_t T = x.size(1);
        int64_t D = x.size(3);
        int64_t C = x.size(4);
        x_in = x.mean(2).reshape({B, T, D * C}).contiguous();
    } else {
        TORCH_CHECK(x.dim() == 3, "x must be [B,T,D] or [B,T,R,D,C] for forward_stack.");
        x_in = x.contiguous();
    }

    // Create default mask of ones [L, R] on same device/dtype as x
    // impl expects mask to be indexable as m_ptr[l*R + r]
    // H is [B, L, R, D, C]
    int64_t L = H.size(1);
    int64_t R = H.size(2);
    auto mask = torch::ones({L, R}, x_in.options());

    // Call implementation
    // We pass steps as H_cycles, and 1 as L_cycles -> total = steps * 1
    return _fused_cognitive_cycle_impl(
        x_in, H, mask,
        delays, tables, conns,
        decays, hw, hb,
        steps, 1, 
        lif_decay, lif_threshold, halt_threshold,
        at::Tensor() // H_out empty
    );
}

void configure_hpc(bool temporal_folding, float fold_alpha) {
    Core::set_hpc_temporal_folding(temporal_folding, fold_alpha);
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
    m.def("titans_memory_forward", &titans_memory_forward);
    m.def("titans_memory_update", &titans_memory_update);
    m.def("neural_cache_lookup_fast", &neural_cache_lookup_fast);
    m.def("forward_stack_io", &forward_stack_io);
    m.def("mes_super_step_io", &mes_super_step_io);
    m.def("survival_update_io", &survival_update_io);
    m.def("survival_mask_io", &survival_mask_io);
    m.def("survival_losses_io", &survival_losses_io);
    m.def("configure_hpc", &configure_hpc);
    m.def("configure_runtime", &configure_runtime);
    m.def("set_perf_counters_enabled", &set_perf_counters_enabled);
    m.def("reset_perf_counters", &reset_perf_counters);
    m.def("get_perf_counters", &get_perf_counters);
    m.def("ademamix_update", &ademamix_update);
    m.def("batched_ademamix_update", &batched_ademamix_update);
    m.def("fused_cognitive_cycle", &_fused_cognitive_cycle_impl);
}
