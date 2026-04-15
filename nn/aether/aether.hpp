#pragma once

#include "../../include/neural_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rra::nn::aether {

struct AetherConfig {
    size_t d_model = 0;
    size_t max_scales = 6;
    size_t modes_per_scale = 8;
    float diffusion_tau = 0.35f;
    float scale_decay = 0.72f;
    float resonance_gamma = 0.08f;
    float transport_eta = 0.04f;
    float renorm_beta = 0.12f;
    float complexity_lambda = 0.05f;
    float survival_gamma = 0.08f;
    float memory_decay = 0.98f;
    bool persistent_memory = false;
    size_t probe_count = 32;
    size_t probe_taps = 8;
    size_t probe_radius = 16;
    float probe_memory_decay = 0.95f;
    float noise_memory_decay = 0.90f;
    float beamforming_gain = 0.12f;
    float cancellation_gain = 0.08f;
    float residual_surprise_gain = 0.05f;
    bool execution_cache_enabled = true;
    float cache_hit_threshold = 0.015f;
    float gate_update_gain = 8.0f;
    float gate_decay = 0.90f;
    float min_update_gate = 0.05f;

    static AetherConfig for_model(size_t d_model);
};

struct MortonCell4D {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t w = 0;
};

uint64_t morton_encode4(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t bits = 16);
MortonCell4D morton_decode4(uint64_t code, uint32_t bits = 16);

struct RenormalizedLevel {
    size_t scale = 0;
    size_t group_size = 1;
    s4m::Tensor state;
    std::vector<float> survival;
};

class SpectralGeometricPropagator {
public:
    explicit SpectralGeometricPropagator(size_t d_model);
    explicit SpectralGeometricPropagator(AetherConfig config);

    const AetherConfig& config() const { return config_; }
    void reset_memory();

    // Returns a field-update delta. The caller owns the residual connection.
    s4m::Tensor forward(const s4m::Tensor& x);

    // Adaptive renormalization R_k: H_k -> H_{k+1}.
    std::vector<RenormalizedLevel> renormalize(const s4m::Tensor& x) const;

private:
    struct ScalePlan {
        size_t scale = 0;
        size_t group_size = 1;
        size_t mode_count = 1;
        float scale_weight = 1.0f;
        std::vector<float> basis;       // [mode][local]
        std::vector<float> eigenvalues; // graph Laplacian eigenvalues
        std::vector<float> memory;      // [mode][d_model]
        std::vector<float> group_signature;
        std::vector<float> group_gate;
        std::vector<float> group_cache; // [group][local][d_model]
        std::vector<uint8_t> group_cache_valid;
    };

    struct ProbePlan {
        size_t center = 0;
        size_t radius = 1;
        uint64_t morton_code = 0;
        std::vector<size_t> feature_index;
        std::vector<float> feature_sign;
        float signal_memory = 0.0f;
        float noise_memory = 0.0f;
    };

    AetherConfig config_;
    size_t cached_seq_len_ = 0;
    std::vector<ScalePlan> plans_;
    std::vector<ProbePlan> probes_;

    void rebuild_plans(size_t seq_len);
    void rebuild_probes(size_t seq_len);
    float compute_group_signature(
        const s4m::Tensor& x,
        size_t start,
        size_t end
    ) const;
    float update_execution_gate(ScalePlan& plan, size_t group, float signature);
    void add_cached_group(
        const ScalePlan& plan,
        size_t group,
        size_t start,
        size_t len,
        float gate,
        s4m::Tensor& propagated
    ) const;
    void store_cached_group(
        ScalePlan& plan,
        size_t group,
        const std::vector<float>& group_out
    );
    void apply_probe_array(
        const s4m::Tensor& x,
        const s4m::Tensor& propagated,
        s4m::Tensor& delta
    );
    static float basis_value(size_t local, size_t group_size, size_t mode);
    static float laplacian_eigenvalue(size_t group_size, size_t mode);
};

} // namespace rra::nn::aether
