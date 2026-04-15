#include "hybrid_engine.hpp"

namespace s4m {

HybridBlock::HybridBlock(size_t num_chunks, size_t num_nis_planes)
    : cafe_field(num_chunks), num_planes(num_nis_planes) {
    // 8 uint64_t per 512-bit plane
    nis_planes.resize(num_nis_planes * 8, 0); 
    
    // Initialize AETHER Propagator with d_model = 512 (32 DualVectors * 16 floats)
    aether_propagator = std::make_unique<rra::nn::aether::SpectralGeometricPropagator>(512);
}

void HybridBlock::forward() {
    // Stage 1: Context Mixing (CAFE)
    // The continuous magnitudes and phases are smoothly propagated across the sequence
    cafe_field.apply_stencil_convolution();

    // Stage 2: Phase Alignment (NIS)
    auto& chunks = cafe_field.get_chunks();
    for (auto& chunk : chunks) {
        // Extract Phase: Instant conversion of signs to BitVector512 (Zero Latency)
        BitVector512 phase_bv = cafe::extract_to_bitvector(chunk.state);

        // Resolve (NIS): Instantly find the nearest global memory attractor via XOR row-reduction
        BitVector512 attractor = core::solver_v8::resolve_attractor(phase_bv, nis_planes.data(), num_planes);

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
        size_t seq_len = chunks.size();
        size_t d_model = 512;
        s4m::Tensor t({seq_len, d_model});
        
        for (size_t i = 0; i < seq_len; ++i) {
            float* dst = t.ptr() + i * d_model;
            for (size_t j = 0; j < 32; ++j) {
                _mm512_storeu_ps(dst + j * 16, chunks[i].state[j].data);
            }
        }
        
        s4m::Tensor delta = aether_propagator->forward(t);
        
        for (size_t i = 0; i < seq_len; ++i) {
            const float* src = delta.ptr() + i * d_model;
            for (size_t j = 0; j < 32; ++j) {
                __m512 d = _mm512_loadu_ps(src + j * 16);
                chunks[i].state[j].data = _mm512_add_ps(chunks[i].state[j].data, d);
            }
        }
    }
}

HybridEngine::HybridEngine(size_t num_blocks, size_t num_chunks, size_t num_nis_planes) {
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks.emplace_back(num_chunks, num_nis_planes);
    }
}

void HybridEngine::forward() {
    for (auto& block : blocks) {
        block.forward();
    }
}

void HybridEngine::backward(const std::vector<float>& chunk_energy_gradients) {
    if (blocks.empty()) return;
    
    // Inject gradients into the last block for Optimal Transport to use in the next forward pass
    auto& last_chunks = blocks.back().cafe_field.get_chunks();
    for (size_t i = 0; i < last_chunks.size() && i < chunk_energy_gradients.size(); ++i) {
        last_chunks[i].energy_gradient = chunk_energy_gradients[i];
        
        // Learning the Memory (NIS): 
        // If the prediction error is high, expand the Gaussian rank of the NIS memory planes
        // Instant, one-shot memorization without SGD.
        if (chunk_energy_gradients[i] > 0.5f) { // Simple threshold for memorization
            BitVector512 current_phase = cafe::extract_to_bitvector(last_chunks[i].state);
            core::solver_v8::update_basis(blocks.back().nis_planes.data(), current_phase, blocks.back().num_planes);
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
    if (blocks.empty()) return 0;
    auto& last_layer_chunks = blocks.back().cafe_field.get_chunks();
    if (last_layer_chunks.empty()) return 0;
    
    __m512 out_data = last_layer_chunks.back().state[0].data;
    float p_act = std::abs(((float*)&out_data)[0]);
    p_act = std::min(1.0f, std::max(1e-8f, p_act));
    return static_cast<uint8_t>(p_act * 255.0f);
}

} // namespace s4m
