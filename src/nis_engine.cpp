#include "nis_engine.hpp"
#include "byte_field.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <omp.h>
#include <bit>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <fstream>
#include <string_view>
#include <cstdio>
#include <vector>
#include <array>
#include <utility>

namespace s4m::core {

template<typename T>
void write_pod(std::ostream& out, const T& val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
void read_pod(std::istream& in, T& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
}

static const std::array<std::array<uint64_t, 16>, 256> g_vsa_dictionary = []() {
    std::array<std::array<uint64_t, 16>, 256> d{};
    std::mt19937_64 rng(42);
    for (int b = 0; b < 256; ++b) for (int w = 0; w < 16; ++w) d[b][w] = rng();
    return d;
}();

constexpr std::string_view kSparseMagic = "S4MSPAR\0";
constexpr uint32_t kSparseVersion = 8; // V8: Attractor Singularity

NISEngine::NISEngine() : tick_(0), has_bound_frame_(false), last_cognitive_spikes(0) {
    gnf::seed_grid(byte_field_);
    initialize_cycle_buffers();
}

NISEngine::NISEngine(const std::vector<NodeParams>& genome) : NISEngine() {
    reserve_nodes(genome.size());
    for (size_t i = 0; i < genome.size(); ++i) {
        set_node_at(i, genome[i].spatial_id, static_cast<Node::NodeType>(genome[i].type_id), static_cast<FunctionalRegion>(genome[i].region_id));
    }
    compile_topology();
}

NISEngine::~NISEngine() = default;

void NISEngine::rehydrate_runtime_state_from_nodes() {
    active_nodes_ = nodes_.size();
    state_.reinit(active_nodes_, cfg_.default_threshold, cfg_.default_decay, cfg_.homeos_target_rate);
    std::fill(state_.query_seeds_planes.begin(), state_.query_seeds_planes.end(), 0ULL);
    for (size_t i = 0; i < active_nodes_; ++i) {
        BitVector512 q(nodes_[i].spatial_id);
        solver_v8::update_basis(state_.query_seeds_planes.data(), q, active_nodes_);
    }
}

void NISEngine::execute_cycle(const uint8_t* bytes, uint64_t count, TickMode m, float global_surprise) {
    if (count == 0 || !bytes) return;
    for (size_t batch_start = 0; batch_start < count; batch_start += 64) {
        size_t current_len = std::min<size_t>(64, count - batch_start);
        HolographicFrame frame; vsa_bind(&frame, bytes + batch_start, current_len, &g_vsa_dictionary[0][0]);

        // 1. GNF Encoding (Spatial Ingest)
        BitVector512 current_input_mv;
        for (size_t k = 0; k < current_len; ++k) {
            uint8_t b = bytes[batch_start + k];
            binder_micro_.bind(byte_field_, b, global_surprise);
            current_input_mv = byte_field_.morton_keys[b];
        }
        BitVector512 m_micro = binder_micro_.snap_to_titan();

        // [UTM-SINGULARITY]: Direct Attractor Resolution
        // Bypasses all iterative physics. Finds the manifold attractor state in GF(2).
        BitVector512 attractor = solver_v8::resolve_attractor(m_micro, state_.query_seeds_planes.data(), active_nodes_);

        // Slate 0 acts as the "Resolved State" for holographic prediction
        for (int w = 0; w < 8; ++w) state_.slates_raw[0 * 8 + w] = attractor.data[w];

        // 2. Gaussian Rank Learning (Rank-Update)
        // If surprise is high, we expand the basis
        if (global_surprise > 0.5f) {
            solver_v8::update_basis(state_.query_seeds_planes.data(), m_micro, active_nodes_);
        }

        last_positive_frame_ = frame; has_bound_frame_ = true;
        tick_++;
    }
}

void NISEngine::execute_tick(TickMode m) {} // Solver is instantaneous

void NISEngine::project_onto_manifold(const uint64_t* mvs, const float* pressures, size_t count) {}
float NISEngine::minimize_manifold_energy(float lr) { return 0.0f; } // Unified in execute_cycle
void NISEngine::clear_rewards() {}

EngineOutput NISEngine::read_output(size_t num_anchors) {
    EngineOutput out;
    BitVector512 query; std::memcpy(query.data, &state_.slates_raw[0], 64);

    // Nearest-neighbor in Morton Field to decode the attractor state
    uint8_t best_b = 0; int min_d = 513;
    for (int i = 0; i < 256; ++i) {
        int d = query.popcount_xor(byte_field_.morton_keys[i]);
        if (d < min_d) { min_d = d; best_b = static_cast<uint8_t>(i); }
    }
    out.byte = best_b; out.confidence = 1.0f - (static_cast<float>(min_d) / 512.0f);
    out.ready_to_speak = (out.confidence > 0.01f);

    // Distribute probs based on distance
    for(int b=0; b<256; ++b) {
        int d = query.popcount_xor(byte_field_.morton_keys[b]);
        out.probs[b] = (512.0f - d) / 512.0f;
    }
    return out;
}

void NISEngine::reserve_nodes(size_t n) { nodes_.resize(n); state_.reserve(n); active_nodes_ = n; }
void NISEngine::set_node_at(size_t i, uint64_t sid, Node::NodeType t, FunctionalRegion r) {
    if (i >= nodes_.size()) return;
    nodes_[i].spatial_id = sid; nodes_[i].type = t; nodes_[i].region = r;
}

void NISEngine::compile_topology() {
    std::sort(nodes_.begin(), nodes_.end(), [](const Node& a, const Node& b) { return a.spatial_id < b.spatial_id; });
    rehydrate_runtime_state_from_nodes();
}

bool NISEngine::save_state(const std::string& p) const {
    std::ofstream out(p, std::ios::binary); if (!out.is_open()) return false;
    out.write(kSparseMagic.data(), 8); write_pod(out, kSparseVersion); write_pod(out, static_cast<uint64_t>(nodes_.size()));
    uint64_t n_planes = 512 * ((nodes_.size() + 63) / 64); write_pod(out, n_planes);
    out.write(reinterpret_cast<const char*>(state_.query_seeds_planes.data()), n_planes * 8);
    return true;
}

bool NISEngine::load_state(const std::string& p) {
    std::ifstream in(p, std::ios::binary); if (!in.is_open()) return false;
    std::array<char, 8> m; in.read(m.data(), 8); if (std::string_view(m.data(), 8) != kSparseMagic) return false;
    uint32_t v; read_pod(in, v); uint64_t nn; read_pod(in, nn); nodes_.resize(nn);
    uint64_t n_planes; read_pod(in, n_planes);
    state_.query_seeds_planes.resize(n_planes);
    in.read(reinterpret_cast<char*>(state_.query_seeds_planes.data()), n_planes * 8);
    active_nodes_ = nn;
    return true;
}

void NISEngine::extrinsic_supervision(const BitVector512& target, float lr) {
    // Supervision = Basis rank expansion
    solver_v8::update_basis(state_.query_seeds_planes.data(), target, active_nodes_);
}

void NISEngine::organic_growth() {}
void NISEngine::axonal_sprouting(size_t i, int c) {}
void NISEngine::force_wire(size_t s, size_t t, float w) {}

WeightStats NISEngine::get_weight_stats() const {
    WeightStats s; s.count = active_nodes_; s.sparsity = 0.5f; return s;
}

void NISEngine::inject_global_dopamine(float amount) {}
void NISEngine::add_membrane_noise(FunctionalRegion region, float intensity) {}
void NISEngine::set_inference_sensitivity(float scale) {}

ManifoldView NISEngine::get_manifold_view() {
    return ManifoldView {
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, state_.query_seeds_planes.data(), nullptr,
        nullptr, nullptr, nullptr, nullptr,
        state_.codebook_raw.data(), state_.slates_raw.data(), nullptr, nullptr,
        nullptr, nullptr, nullptr,
        active_nodes_, tick_
    };
}

BitVector512 NISEngine::get_node_state_vector(size_t i) const { return BitVector512(); }
void NISEngine::update_node_consistency(size_t i) {}
float NISEngine::compute_global_consistency_energy() const { return 0.0f; }
void NISEngine::update_temporal_stability() {}
void NISEngine::form_consensus_clusters(float threshold) {}
void NISEngine::apply_consistency_penalties(float learning_rate) {}
void NISEngine::initialize_cycle_buffers() {
    tick_ = 0; last_positive_frame_.emv = BitVector512(); last_positive_frame_.imv = BitVector512(); has_bound_frame_ = false;
}

} // namespace s4m::core
