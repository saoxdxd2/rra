#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <array>
#include <random>
#include <deque>
#include <mutex>
#include <atomic>
#include <complex>

#include "byte_field.hpp"
#include "sao_core.hpp"
#include "neural_types.hpp"
#include "neuron.hpp"

namespace s4m::core {

// Using AlignedAllocator and aligned_vector from neural_types.hpp

/**
 * @brief Manifold Cycle State: 16-Subspace Multivector Representation
 */
struct CycleState {
    // SC-V8: 512-bit states backed by 64-bit aligned storage
    aligned_vector<uint64_t> query_seeds_planes;

    // NIS-V2: Shared Latent Codebook and Slates
    aligned_vector<uint64_t> codebook_raw;
    aligned_vector<uint64_t> slates_raw;

    void reserve(size_t n) {
        query_seeds_planes.reserve(n * 8);
        slates_raw.reserve(5 * 8);
        codebook_raw.reserve(65536 * 8);
    }

    void reinit(size_t n, float initial_threshold, float initial_decay, float initial_rate) {
        query_seeds_planes.assign(n * 8, 0ULL);
        slates_raw.assign(5 * 8, 0ULL);
        codebook_raw.assign(65536 * 8, 0ULL);
    }
};

struct EngineOutput {
    uint8_t byte = 0;
    uint32_t winning_bucket = 0;
    bool ready_to_speak = false;
    float confidence = 0.0f;
    std::array<float, 256> probs = {0.0f};
};

struct EngineDiagnostics {
    uint64_t taylor_saturation_count = 0;
    float    hdc_mean_targets_per_key = 0.0f;
    uint32_t dead_neuron_count        = 0;
    float    gradient_max             = 0.0f;
    float    gain_gradient_max        = 0.0f;
    uint32_t hdc_pool_size            = 0;
};

class NISEngine {
public:
    NISEngine();
    explicit NISEngine(const std::vector<NodeParams>& genome);
    ~NISEngine();

    NISEngine(const NISEngine& other);
    NISEngine& operator=(const NISEngine& other);
    NISEngine(NISEngine&& other) noexcept;
    NISEngine& operator=(NISEngine&& other) noexcept;

    // Execution
    void execute_cycle(const uint8_t* bytes, uint64_t count, TickMode mode, float global_surprise = 0.0f);
    void execute_tick(TickMode mode = TickMode::Standard);
    void project_onto_manifold(const uint64_t* mvs, const float* pressures, size_t count);
    float minimize_manifold_energy(float learning_rate);

    void clear_rewards();
    void clear_stats() { last_cognitive_spikes = 0; }

    // Output
    EngineOutput read_output(size_t num_anchors = 256);
    void set_output_anchors(const std::vector<size_t>& anchors) { output_anchors_ = anchors; }

    // Manifold Management
    void reserve_nodes(size_t n);
    void set_node_at(size_t i, uint64_t spatial_id, Node::NodeType type, FunctionalRegion region);
    void add_node(uint64_t spatial_id, Node::NodeType type = Node::NodeType::Excitatory, FunctionalRegion region = FunctionalRegion::Integrator);
    void organic_growth();
    void axonal_sprouting(size_t index, int count);
    void force_wire(size_t source, size_t target, float weight);
    void spatial_sort();
    void compile_topology();
    void rehydrate_runtime_state_from_nodes();

    // Checkpointing
    bool save_state(const std::string& path) const;
    bool load_state(const std::string& path);
    bool load_checkpoint(const std::string& path) { return load_state(path); }

    // Accessors
    const EngineConfig& get_config() const { return cfg_; }
    EngineConfig& get_config() { return cfg_; }
    const CycleState& get_state() const { return state_; }
    const std::vector<Node>& get_nodes() const { return nodes_; }
    uint32_t get_last_cognitive_spikes() const { return last_cognitive_spikes; }

    // Supervised Learning
    void extrinsic_supervision(const BitVector512& target, float learning_rate);

    // Meta-Consistency Field (Layer-3)
    float compute_global_consistency_energy() const;
    void update_temporal_stability();
    void form_consensus_clusters(float threshold = 0.7f);
    void apply_consistency_penalties(float learning_rate = 0.01f);
    uint32_t get_cluster_count() const { return cluster_count_; }

    // Per-node consistency accessors
    float get_node_consistency_score(size_t i) const {
        return (i < nodes_.size()) ? nodes_[i].consistency_score : 0.0f;
    }
    uint32_t get_node_cluster_id(size_t i) const {
        return (i < nodes_.size()) ? nodes_[i].cluster_id : 0;
    }

    WeightStats get_weight_stats() const;
    EngineDiagnostics get_diagnostics() const { return diagnostics_; }
    BitVector512 get_byte_morton(uint8_t b) const { return byte_field_.morton_keys[b]; }
    void wake_node(size_t i) { /* reserved for future queue-based activation */ }
    void reset_binders() { binder_micro_.reset(); binder_meso_.reset(); binder_macro_.reset(); }
    void inject_global_dopamine(float amount);
    void add_membrane_noise(FunctionalRegion region, float intensity);
    void set_inference_sensitivity(float scale);

    ManifoldView get_manifold_view();

private:
    EngineConfig cfg_;
    CycleState state_;
    size_t active_nodes_ = 0;
    std::vector<Node> nodes_;
    std::vector<size_t> output_anchors_;
    uint32_t tick_ = 0;
    uint32_t last_cognitive_spikes = 0;
    std::mt19937_64 rng_{42};
    gnf::GaussianBinder binder_micro_{2};
    gnf::GaussianBinder binder_meso_{4};
    gnf::GaussianBinder binder_macro_{8};
    gnf::ByteField byte_field_;
    HolographicFrame last_positive_frame_{};
    bool has_bound_frame_ = false;
    EngineDiagnostics diagnostics_;

    // Meta-Consistency Field state
    uint32_t cluster_count_ = 0;

    void initialize_cycle_buffers();

    // Internal helpers for consistency field
    BitVector512 get_node_state_vector(size_t i) const;
    void update_node_consistency(size_t i);
};

} // namespace s4m::core