#pragma once

#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>
#include <cstdint>
#include <random>
#include <string>
#include <torch/torch.h>
#include "language_io.hpp"

namespace rra::nn::topology {

enum class NeuronType {
    EXCITATORY,
    INHIBITORY
};

// ---------------------------------------------------------
// Frozen Constants for Embodiment Engine
// ---------------------------------------------------------
constexpr float V_REST = -70.0f;
constexpr float V_THRESH = -56.5f;
constexpr float TAU_M = 22.0f;

// Trace Constants (Derived from TPU phase 3)
constexpr float TAU_TRACE_PRE = 11.0f;
constexpr float TAU_TRACE_POST = 11.0f;
constexpr float TAU_TRACE_CA = 11.0f;

// Homeostasis & I/O Constants (Derived from TPU phase 5)
constexpr float GABA_VOLTAGE_DROP = -5.1202f;
constexpr float EXCITATORY_SCALE = 0.4809f;
constexpr float INHIBITORY_SCALE = 1.3955f;
constexpr float SENSORY_GAIN = 13.1159f;
constexpr float MOTOR_DECAY_TAU = 3.7471f;

struct UniversalPlasticity {
    static const float w1[4][16];
    static const float b1[16];
    static const float w2[16][16];
    static const float b2[16];
    static const float w3[16][1];
    static const float b3[1];

    static torch::Tensor w1_t;
    static torch::Tensor b1_t;
    static torch::Tensor w2_t;
    static torch::Tensor b2_t;
    static torch::Tensor w3_t;
    static torch::Tensor b3_t;
    static bool tensors_initialized;

    static void init_tensors();
    static torch::Tensor evaluate_delta_w_batch(const torch::Tensor& inputs);
};

class CorticalTissue;
class EventDrivenNeuron;

struct SpikeEvent {
    float arrival_time;
    uint64_t target_neuron_id;
    float neurotransmitter_quanta;
    NeuronType source_type;

    bool operator>(const SpikeEvent& other) const {
        return arrival_time > other.arrival_time;
    }
};

class EventDrivenSynapse {
public:
    static constexpr float CALCIUM_INFLUX = 10.0f;
    static constexpr float CALCIUM_DECAY = 5.3f;
    static constexpr float FUSION_RATE = 0.0001f;
    static constexpr float VESICLE_REFILL_RATE = 0.005f;
    static constexpr float AMPA_VOLTAGE_JUMP = 5.0f;

    uint64_t post_synaptic_id;
    EventDrivenNeuron* parent_neuron;
    
    float last_update_time;
    float pre_calcium_concentration;
    float vesicle_pool;
    float ampa_receptor_count;
    float axonal_delay;
    float trace_v_pre;
    float trace_ca;

    EventDrivenSynapse(uint64_t post_id, EventDrivenNeuron* parent, float delay = 2.0f);

    float process_presynaptic_spike(float current_time, CorticalTissue* tissue);
    void apply_dopamine_wave(float current_time, float global_reward, CorticalTissue* tissue);
};

class EventDrivenNeuron {
public:
    uint64_t morton_code;
    uint64_t morton_seed[8];
    NeuronType type;
    float V_m;
    float last_update_time;
    int total_spikes;
    float trace_v_post;
    float last_spike_time;

    std::vector<std::shared_ptr<EventDrivenSynapse>> outgoing_synapses;

    EventDrivenNeuron(uint64_t id, NeuronType t = NeuronType::EXCITATORY);

    void update_to_time(float current_time);
    float process_incoming_neurotransmitter(float current_time, float quanta, NeuronType source_type);
};

class CorticalTissue {
private:
    std::unordered_map<uint64_t, std::shared_ptr<EventDrivenNeuron>> neurons_;
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, std::greater<SpikeEvent>> event_queue_;
    float global_time_;

public:
    CorticalTissue();

    void add_neuron(uint64_t morton_code, NeuronType type = NeuronType::EXCITATORY);
    void connect_neurons(uint64_t pre_id, uint64_t post_id, float delay = 2.0f);
    void force_spike(uint64_t id, float time, float quanta = 100.0f);
    void run_until(float target_time_ms);
    void inject_dopamine(float reward, float time);
    EventDrivenNeuron* get_neuron(uint64_t id);
    std::vector<float> get_weights() const;
    void set_weights(const std::vector<float>& w);
    float get_neuron_voltage(uint64_t id) const;
    int get_neuron_spikes(uint64_t id) const;
    int get_total_network_spikes() const;
    float get_global_time() const { return global_time_; }
};

// =============================================================================
// LIBTORCH-POWERED NLP RESERVOIR ENGINE
// =============================================================================

constexpr int N_S = 32;
constexpr int N_EXC = 48;
constexpr int N_INH = 16;
constexpr int N_M = 32;
constexpr int N = 128;
constexpr int TICKS = 30;

struct ReservoirGenome {
    float sensory_gain = 5.10f;
    float binder_alpha = 0.98f;
    float weight_scale = 0.40f;
    float ampa_jump = 0.32f;
    float gaba_drop = 11.54f;
    float lr_scale = 0.31f; 
    float thresh_offset = 0.0f;
    float tau_m = 32.66f;
};

class ReservoirEngine : public torch::nn::Module {
public:
    ReservoirEngine(const ReservoirGenome& genome = ReservoirGenome(), uint32_t seed = 42);

    // High-Level Training & Prediction via Autograd
    void train_sequence(const std::string& text, float learning_rate = 0.1f);
    std::string predict_sequence(const std::string& text);
    char process_token(char token);
    void reset_state();

    // Serialization using LibTorch format
    void save_checkpoint(const std::string& filepath);
    void load_checkpoint(const std::string& filepath);

private:
    ReservoirGenome genome_;
    
    // LibTorch Tensors
    torch::Tensor W_frozen;
    torch::Tensor sign;
    torch::Tensor v;
    torch::Tensor bd;
    torch::Tensor grid;
    
    // Trainable parameters
    torch::Tensor R;
    torch::Tensor R_bias;
    
    void init_grid();
    void init_topology(uint32_t seed);
    torch::Tensor step(const torch::Tensor& target_co);
};

} // namespace rra::nn::topology
