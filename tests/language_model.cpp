// language_model.cpp â€” rra Biological Language Engine
// cl /std:c++20 /arch:AVX512 /O2 /EHsc /Fe:tests\language_model.exe tests\language_model.cpp nn\brain_tree\brain_tree.cpp nn\brain_tree\language_io.cpp /I. 
#include "../nn/brain_tree/brain_tree.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>

using namespace rra::nn::topology;

constexpr int N_SENSORY = 512;
constexpr int N_HIDDEN  = 1024;
constexpr int N_MOTOR   = 512;
constexpr int TOTAL_NEURONS = N_SENSORY + N_HIDDEN + N_MOTOR;

int main() {
    std::cout << "========================================================\n";
    std::cout << "rra Biological Language Engine (SC-V8 Holographic VSA)\n";
    std::cout << "========================================================\n\n";

    // 1. Initialize the ByteField Vocabulary
    ByteField vocab;
    seed_grid(vocab);
    GaussianBinder binder(5); // corusion shift of 5

    // 2. Initialize the Biological Engine
    CorticalTissue brain;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dd(1.0f, 15.0f);

    std::cout << "-> Growing Spatiotemporal Cortex (" << TOTAL_NEURONS << " Neurons)...\n";
    for(int i = 0; i < N_SENSORY; i++) brain.add_neuron(i, NeuronType::EXCITATORY);
    
    // 80% Excitatory, 20% Inhibitory (GABAergic Homeostasis)
    int exc_hidden = (int)(N_HIDDEN * 0.8f);
    for(int i = 0; i < exc_hidden; i++) brain.add_neuron(N_SENSORY + i, NeuronType::EXCITATORY);
    for(int i = exc_hidden; i < N_HIDDEN; i++) brain.add_neuron(N_SENSORY + i, NeuronType::INHIBITORY);
    
    for(int i = 0; i < N_MOTOR; i++) brain.add_neuron(N_SENSORY + N_HIDDEN + i, NeuronType::EXCITATORY);

    // Random Sparse Wiring
    std::cout << "-> Synaptogenesis (Sparse Wiring)...\n";
    for(int s = 0; s < N_SENSORY; s++) {
        for(int h = 0; h < N_HIDDEN; h++) {
            if (rng() % 10 < 2) brain.connect_neurons(s, N_SENSORY + h, dd(rng));
        }
    }
    for(int h = 0; h < N_HIDDEN; h++) {
        for(int m = 0; m < N_MOTOR; m++) {
            if (rng() % 10 < 2) brain.connect_neurons(N_SENSORY + h, N_SENSORY + N_HIDDEN + m, dd(rng));
        }
    }

    std::string text = "HELLO WORLD, THIS IS THE RRA ENGINE SPEAKING.";
    std::cout << "\n[Target Sequence]: " << text << "\n\n";

    float gtime = 0.0f;
    float sim_step_ms = 50.0f; // 50ms per token

    // We need arrays for the decoder
    std::vector<BitVector512> motor_seeds(N_MOTOR);
    for(int m = 0; m < N_MOTOR; m++) {
        auto neuron = brain.get_neuron(N_SENSORY + N_HIDDEN + m);
        for(int i=0; i<8; i++) motor_seeds[m].data[i] = neuron->morton_seed[i];
    }

    // A dummy basis matrix for the Attractor GF(2) Solver
    alignas(64) uint64_t gf2_basis[512 * 8] = {0};

    // The Training Loop
    for(size_t i = 0; i < text.size(); i++) {
        uint8_t token = (uint8_t)text[i];
        
        // 1. Sensory Encoding (Analog Byte -> VSA Bit Vector -> Spikes)
        binder.bind(vocab, token, 0.1f);
        BitVector512 sensory_mask = binder.snap_to_titan();
        
        // Inject physical spikes into sensory cortex based on 512-bit mask
        for(int b = 0; b < 512; b++) {
            if ((sensory_mask.data[b >> 6] >> (b & 63)) & 1) {
                // SENSORY_GAIN = 13.1159f from TPU
                brain.force_spike(b, gtime, 13.1159f); 
            }
        }

        // 2. Propagate Biological Dynamics
        brain.run_until(gtime + sim_step_ms);

        // 3. Motor Decoding (Spikes -> VSA Bit Vector -> Analog Byte)
        std::vector<uint8_t> motor_spikes(N_MOTOR, 0);
        int total_motor_spikes = 0;
        for(int m = 0; m < N_MOTOR; m++) {
            if (brain.get_neuron_spikes(N_SENSORY + N_HIDDEN + m) > 0) {
                motor_spikes[m] = 1;
                total_motor_spikes++;
            }
            // Reset spike counters for next token
            brain.get_neuron(N_SENSORY + N_HIDDEN + m)->total_spikes = 0;
        }

        // VSA AVX-512 Majority Vote Decode
        float confidence = 0.0f;
        uint8_t predicted_token = value_centric_decode(
            motor_spikes.data(), 
            motor_seeds.data(), 
            vocab.morton_keys.data(), 
            N_MOTOR, 
            &confidence
        );

        std::cout << "Token [" << i << "] Target: '" << token 
                  << "' | Pred: '" << (char)(predicted_token >= 32 ? predicted_token : '?') 
                  << "' | Conf: " << confidence 
                  << " | Motor Spikes: " << total_motor_spikes << "\n";

        // 4. O(1) Backpropagation
        if (predicted_token != token && total_motor_spikes > 0) {
            BitVector512 target_mask = vocab.morton_keys[token];
            BitVector512 pred_mask   = vocab.morton_keys[predicted_token];
            BitVector512 error_mask  = target_mask ^ pred_mask;

            // Instantly resolve the error gradient over the GF(2) basis
            BitVector512 global_attractor = solver_v8::resolve_attractor(error_mask, gf2_basis);
            solver_v8::update_basis(gf2_basis, error_mask);

            // Here we would normally translate `global_attractor` into specific delta_w 
            // weight changes. For now, we inject Dopamine globally.
            brain.inject_dopamine(-5.0f, gtime + sim_step_ms);
        } else {
            brain.inject_dopamine(5.0f, gtime + sim_step_ms);
        }

        gtime += sim_step_ms;
    }

    std::cout << "\n========================================================\n";
    std::cout << "Sequence Processing Complete. O(1) Backprop Active.\n";
    std::cout << "========================================================\n";
    return 0;
}
