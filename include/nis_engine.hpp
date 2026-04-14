#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <random>
#include <deque>
#include <mutex>
#include <atomic>

#include "neural_types.hpp"
#include "synapse.hpp"
#include "neuron.hpp"
#include "lateral_table.hpp"
#include "sao_core.hpp"

namespace rra::nis_engine {

/**
 * @brief Memory Alignment Allocator
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T, 32>>;

struct Scored {
    float score;
    int64_t row_idx;
};

/**
 * @brief Structure of Arrays (SoA) for fast SIMD traversal.
 */
struct CycleState {
    aligned_vector<float> membrane;
    aligned_vector<float> spikes;
    aligned_vector<uint8_t> spike_state;
    aligned_vector<uint8_t> spike_state_back; // New: Double-buffer for synchrony
    aligned_vector<float> firing_rate_ema;
    aligned_vector<float> eligibility_trace;
    aligned_vector<uint32_t> last_spike_ticks;
    aligned_vector<float> adaptive_thresholds;
    aligned_vector<float> base_thresholds;
    aligned_vector<float> leak_rates;
    aligned_vector<float> stdp_rates;
    aligned_vector<float> metabolic_caps;
    aligned_vector<uint8_t> node_types;
    aligned_vector<uint8_t> node_regions;
    aligned_vector<uint8_t> refractory_timer;
    aligned_vector<float> sensory_pressure;
    
    // Self-Optimizing Additions
    aligned_vector<float> dopamine;            
    aligned_vector<float> dopamine_back;       
    aligned_vector<uint64_t> local_ex_masks;   // Bit-packed excitatory local neighbors
    aligned_vector<uint64_t> local_in_masks;   // Bit-packed inhibitory local neighbors
    aligned_vector<uint8_t>   local_weights_q4;  // N*64 weights packed as 4-bit nibbles (8x smaller)
    std::array<float, 16>     weight_codebook;   // 16-entry TurboQuant codebook shared across all weights
    
    float external_surprise = 0.0f;
    std::array<float, 16> octant_biases;
    std::array<float, 16> region_biases;
    bool is_thinking = false;

    void reserve(size_t n) {
        membrane.reserve(n); spikes.reserve(n); spike_state.reserve(n);
        firing_rate_ema.reserve(n); eligibility_trace.reserve(n);
        last_spike_ticks.reserve(n); adaptive_thresholds.reserve(n);
        base_thresholds.reserve(n); leak_rates.reserve(n);
        stdp_rates.reserve(n); metabolic_caps.reserve(n);
        node_types.reserve(n); node_regions.reserve(n); refractory_timer.reserve(n);
        sensory_pressure.reserve(n); dopamine.reserve(n);
        dopamine_back.reserve(n); 
        local_ex_masks.reserve(n);
        local_in_masks.reserve(n);
        local_weights_q4.reserve((n * 64 + 1) / 2); // 2 nibbles per byte
    }

    void reinit(size_t n, float initial_threshold, float initial_decay, float initial_rate) {
        membrane.assign(n, 0.0f);
        spikes.assign(n, 0.0f);
        spike_state.assign(n, 0U);
        firing_rate_ema.assign(n, initial_rate);
        eligibility_trace.assign(n, 0.0f);
        last_spike_ticks.assign(n, 0U);
        adaptive_thresholds.assign(n, initial_threshold);
        base_thresholds.assign(n, initial_threshold);
        leak_rates.assign(n, initial_decay);
        stdp_rates.assign(n, 0.01f);
        metabolic_caps.assign(n, 1.0f);
        node_types.assign(n, 0U);
        node_regions.assign(n, 0U);
        refractory_timer.assign(n, 0U);
        sensory_pressure.assign(n, 0.0f);
        dopamine.assign(n, 0.0f);
        dopamine_back.assign(n, 0.0f);
        local_ex_masks.assign(n, 0ULL);
        local_in_masks.assign(n, 0ULL);
        local_weights_q4.assign((n * 64 + 1) / 2, 0x88U); // 0x88 = nibble 8 = codebook[8] ≈ 0.0
        q4_build_codebook(weight_codebook.data());
        octant_biases.fill(0.0f);
        region_biases.fill(1.0f);
    }

    void add_node_state(float initial_threshold, float initial_rate) {
        membrane.push_back(0.0f);
        spikes.push_back(0.0f);
        spike_state.push_back(0U);
        firing_rate_ema.push_back(initial_rate);
        eligibility_trace.push_back(0.0f);
        last_spike_ticks.push_back(0U);
        adaptive_thresholds.push_back(initial_threshold);
        base_thresholds.push_back(initial_threshold);
        leak_rates.push_back(0.5f);
        stdp_rates.push_back(0.01f);
        metabolic_caps.push_back(1.0f);
        node_types.push_back(0U);
        node_regions.push_back(0U);
        refractory_timer.push_back(0U);
        sensory_pressure.push_back(0.0f);
        dopamine.push_back(0.0f);
        dopamine_back.push_back(0.0f);
        local_ex_masks.push_back(0ULL);
        local_in_masks.push_back(0ULL);
        for(int i=0; i<32; ++i) local_weights_q4.push_back(0x88U); // 64 nibbles = 32 bytes
    }
};

struct EngineOutput {
    uint8_t byte = 0;
    uint32_t winning_bucket = 0;
    bool ready_to_speak = false;
    std::vector<float> action_probabilities;
};

/**
 * @brief Primary GNF Spiking Engine
 */
class NISEngine {
public:
    NISEngine();
    explicit NISEngine(const std::vector<NodeParams>& genome);
    ~NISEngine();

    NISEngine(const NISEngine& other);
    NISEngine& operator=(const NISEngine& other);
    NISEngine(NISEngine&& other) noexcept;
    NISEngine& operator=(NISEngine&& other) noexcept;

    // Core Execution
    void execute_cycle(const aligned_vector<float>& input, uint64_t flags, TickMode mode);
    void execute_tick(TickMode mode = TickMode::Standard);
    void diffuse_dopamine(); // New: Controlled diffusion
    float engine_compute_endogenous_reward(); // New: Objective Function Autonomy
    void consolidate_learning(float learning_rate);
    void clear_rewards();
    void clear_stats() { last_cognitive_spikes = 0; }

    // Checkpointing
    bool save_state(const std::string& path) const;
    bool load_state(const std::string& path);
    bool load_checkpoint(const std::string& path);

    // Mutation & Growth
    void add_node(uint64_t spatial_id, Node::NodeType type = Node::NodeType::Excitatory, FunctionalRegion region = FunctionalRegion::Integrator);
    void organic_growth(); // New: Continuous organic neurogenesis
    void axonal_sprouting(size_t index, int count); // New: Life-support sprouting
    void force_wire(size_t source, size_t target, float weight);
    void kill_neuron(size_t index);
    void prune_nodes(float usage_threshold);
    void apply_weight_decay(float factor);
    void reset_state();
    void set_plasticity(float p);
    void set_emotion(Emotion e);
    void inject_signals(const std::vector<std::pair<size_t, float>>& signals);
    void prune_weak_synapses(float threshold);
    void inject_global_dopamine(float amount);
    void add_membrane_noise(FunctionalRegion region, float intensity);
    
    // Methods for generate.cpp
    void inject_bit_stream(const std::vector<uint8_t>& bits);
    void forward_input(const std::vector<InputEvent>& data);
    void set_output_anchors(const std::vector<size_t>& anchors) { output_anchors_ = anchors; }
    EngineOutput read_output(size_t num_anchors = 256);

    // Accessors
    [[nodiscard]] const EngineConfig& get_config() const { return cfg_; }
    [[nodiscard]] EngineConfig& get_config() { return cfg_; }
    [[nodiscard]] const CycleState& get_state() const { return state_; }
    [[nodiscard]] const std::vector<Node>& get_nodes() const { return nodes_; }
    [[nodiscard]] uint32_t get_last_cognitive_spikes() const { return last_cognitive_spikes; }
    [[nodiscard]] WeightStats get_weight_stats() const;
    [[nodiscard]] float get_surprise() const;
    [[nodiscard]] const std::array<float, 16>& get_octant_biases() const { return state_.octant_biases; }
    [[nodiscard]] const std::vector<uint32_t>& get_active_queue() const { return active_nodes_read_; }
    
    // Physics Interface
    sao1::sao_manifold get_manifold_view();
    void wake_node(size_t i) { if (i < is_in_queue_.size() && !is_in_queue_[i]) { active_nodes_read_.push_back(static_cast<uint32_t>(i)); is_in_queue_[i] = true; } }
    void add_reward(size_t i, float amount);

    // Initialization logic
    void rehydrate_runtime_state_from_nodes();
    void rebuild_region_index_cache();
    void spatial_sort();
    void compile_topology(); // New: The NREM Sleep Compiler

private:
    EngineConfig cfg_;
    CycleState state_;
    size_t active_nodes_ = 0;
    std::vector<Node> nodes_;
    SynapsePool synapse_pool_;
    std::vector<size_t> output_anchors_;
    
    // Topology Compiler Helpers
    std::vector<uint32_t> old_to_new_idx_;
    std::vector<float> barycenters_;
    
    // Sparse Routing Queues
    std::vector<uint32_t> active_nodes_read_;
    std::vector<uint32_t> active_nodes_write_;
    std::vector<bool> is_in_queue_;

    uint32_t current_time_step_ = 0;
    uint32_t last_cognitive_spikes = 0;
    std::mt19937 rng_{42};

    aligned_vector<float> zero_buffer_; // Scratchpad to avoid allocations every tick

    void initialize_cycle_buffers();
    void force_wire_paged(size_t source, size_t target, float weight);
};

} // namespace rra::nis_engine
