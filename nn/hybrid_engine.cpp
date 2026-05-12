#include "hybrid_engine.hpp"

namespace s4m {

HybridBlock::HybridBlock(size_t num_chunks)
    : cafe_field(num_chunks) {
    // Initialize AETHER Propagator with d_model = 512 (32 DualVectors * 16 floats)
    aether_propagator = std::make_unique<rra::nn::aether::SpectralGeometricPropagator>(512);
}

void HybridBlock::forward(core::NISEngine& nis_engine) {
    // Stage 1: Context Mixing (CAFE)
    // The continuous magnitudes and phases are smoothly propagated across the sequence
    cafe_field.apply_stencil_convolution();

    // Stage 2: Phase Alignment (NIS)
    auto& chunks = cafe_field.get_chunks();
    for (auto& chunk : chunks) {
        // Extract Phase: Instant conversion of signs to BitVector512 (Zero Latency)
        BitVector512 phase_bv = cafe::extract_to_bitvector(chunk.state);

        // MAP Inference (NIS): Find the nearest global memory attractor
        BitVector512 attractor = nis_engine.map_inference(phase_bv);

        // Phase Injection: Apply the returned attractor to flip/reinforce signs
        // *Crucially*, this does not reset the continuous magnitudes
        cafe::inject_from_bitvector(chunk.state, attractor);
    }

    // Stage 3: Optimal Transport Stability Breakthrough
    // Dynamically route compute resolution (\mu) based on conserved mass
    float total_mass = static_cast<float>(chunks.size()); 
    cafe_field.apply_optimal_transport(total_mass);
    
    // Stage 4: Aether Spectral Propagation
    if (aether_propagator && !chunks.empty()) {
        s4m::Tensor t = cafe_field.to_tensor();
        s4m::Tensor delta = aether_propagator->forward(t);
        cafe_field.apply_delta(delta);
    }
}

HybridEngine::HybridEngine(size_t num_blocks, size_t num_chunks) {
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks.emplace_back(num_chunks);
    }
}

void HybridEngine::forward() {
    for (auto& block : blocks) {
        block.forward(global_memory);
    }
}

void HybridEngine::backward(const std::vector<float>& chunk_energy_gradients) {
    if (blocks.empty()) return;
    
    // Inject gradients into the last block for Optimal Transport to use in the next forward pass
    auto& last_chunks = blocks.back().cafe_field.get_chunks();
    for (size_t i = 0; i < last_chunks.size() && i < chunk_energy_gradients.size(); ++i) {
        last_chunks[i].energy_gradient = chunk_energy_gradients[i];
        
        // Learning the Memory (NIS): 
        // If the prediction error is high, expand the global memory bank via online mapping.
        if (chunk_energy_gradients[i] > 0.5f) { // Simple threshold for memorization
            BitVector512 current_phase = cafe::extract_to_bitvector(last_chunks[i].state);
            global_memory.expand_memory(current_phase);
        }
    }
}

void HybridEngine::ingest(const uint8_t* data, size_t len) {
    if (blocks.empty() || len == 0) return;
    auto& first_layer_chunks = blocks.front().cafe_field.get_chunks();
    size_t num_chunks = first_layer_chunks.size();
    
    for (size_t i = 0; i < len && i < num_chunks; ++i) {
        float val = static_cast<float>(data[i]) / 255.0f;
        // Broadcast byte value across the entire first dual vector of the chunk
        first_layer_chunks[i].state[0].data = _mm512_set1_ps(val);
    }
}

uint8_t HybridEngine::predict() const {
    auto logits = get_logits();
    uint8_t best_byte = 0;
    float max_val = -1e9f;
    for (int i = 0; i < 256; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_byte = static_cast<uint8_t>(i);
        }
    }
    return best_byte;
}

std::array<float, 256> HybridEngine::get_logits() const {
    std::array<float, 256> logits = {0.0f};
    if (blocks.empty()) return logits;
    auto& last_layer_chunks = blocks.back().cafe_field.get_chunks();
    if (last_layer_chunks.empty()) return logits;

    // Direct structural routing from 512-dim continuous state to 256 logits
    for (int i = 0; i < 256; ++i) {
        int vec_idx = i / 16;
        int float_idx = i % 16;
        if (vec_idx < 32) {
            __m512 out_data = last_layer_chunks.back().state[vec_idx].data;
            logits[i] = ((float*)&out_data)[float_idx];
        }
    }
    return logits;
}

} // namespace s4m
