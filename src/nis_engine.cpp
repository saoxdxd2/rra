#include "nis_engine.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include "core_math.hpp"

namespace rra::nis_engine {

namespace {
constexpr std::array<char, 8> kSparseMagic = {'R', 'R', 'A', 'S', 'P', 'R', '5', '\0'}; 
constexpr uint32_t kSparseVersion = 5U;

template <typename T> bool write_pod(std::ofstream& out, const T& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(T)); return (bool)out; }
template <typename T> bool read_pod(std::ifstream& in, T& v) { return (bool)in.read(reinterpret_cast<char*>(&v), sizeof(T)); }
}

NISEngine::NISEngine() : NISEngine(std::vector<NodeParams>{}) {}
NISEngine::NISEngine(const std::vector<NodeParams>& genome) : synapse_pool_(genome.size() * 4 + 1024) {
    state_.reserve(32768);
    is_in_queue_.resize(32768, false);
    for (const auto& gp : genome) {
        Node n; n.spatial_id = gp.spatial_id; n.pool_syn_offset = 0xFFFFFFFF; n.pool_syn_count = 0;
        n.base_threshold = gp.base_threshold; n.leak_rate = gp.leak_rate; n.plasticity = gp.plasticity;
        n.type = static_cast<Node::NodeType>(gp.type_id); n.region = static_cast<FunctionalRegion>(gp.region_id);
        nodes_.push_back(n);
        for (const auto& s : gp.initial_weights) force_wire(nodes_.size()-1, s.target_idx, s.weight);
    }
    rehydrate_runtime_state_from_nodes();
    initialize_cycle_buffers();
}

NISEngine::~NISEngine() = default;

NISEngine::NISEngine(const NISEngine& o) : cfg_(o.cfg_), state_(o.state_), active_nodes_(o.active_nodes_), nodes_(o.nodes_), synapse_pool_(o.synapse_pool_.size()), active_nodes_read_(o.active_nodes_read_), active_nodes_write_(o.active_nodes_write_), is_in_queue_(o.is_in_queue_), current_time_step_(o.current_time_step_), last_cognitive_spikes(o.last_cognitive_spikes), zero_buffer_(o.zero_buffer_) {
    for (size_t p = 0; p < o.synapse_pool_.size(); ++p) synapse_pool_.get_page(static_cast<uint32_t>(p)) = o.synapse_pool_.get_page(static_cast<uint32_t>(p));
}
NISEngine& NISEngine::operator=(const NISEngine& o) {
    if (this == &o) return *this; cfg_ = o.cfg_; state_ = o.state_; active_nodes_ = o.active_nodes_; nodes_ = o.nodes_; synapse_pool_.clear();
    for (size_t p = 0; p < o.synapse_pool_.size(); ++p) { uint32_t np = synapse_pool_.allocate_page(); synapse_pool_.get_page(np) = o.synapse_pool_.get_page(static_cast<uint32_t>(p)); }
    active_nodes_read_ = o.active_nodes_read_; active_nodes_write_ = o.active_nodes_write_; is_in_queue_ = o.is_in_queue_; current_time_step_ = o.current_time_step_; last_cognitive_spikes = o.last_cognitive_spikes; zero_buffer_ = o.zero_buffer_; return *this;
}
NISEngine::NISEngine(NISEngine&& o) noexcept = default;
NISEngine& NISEngine::operator=(NISEngine&& o) noexcept = default;

void NISEngine::rehydrate_runtime_state_from_nodes() {
    active_nodes_ = nodes_.size();
    state_.reinit(active_nodes_, cfg_.default_threshold, cfg_.default_decay, cfg_.homeos_target_rate);
    state_.spike_state_back.assign(active_nodes_, 0U);
    is_in_queue_.assign(active_nodes_, false);
    zero_buffer_.assign(active_nodes_, 0.0f);
    for (size_t i = 0; i < active_nodes_; ++i) {
        state_.node_types[i] = static_cast<uint8_t>(nodes_[i].type);
        state_.node_regions[i] = static_cast<uint8_t>(nodes_[i].region);
        state_.base_thresholds[i] = nodes_[i].base_threshold;
        state_.leak_rates[i] = nodes_[i].leak_rate;
    }
}

void NISEngine::initialize_cycle_buffers() {
    current_time_step_ = 0;
    active_nodes_read_.clear();
    active_nodes_write_.clear();
}

void NISEngine::execute_cycle(const aligned_vector<float>& in, uint64_t f, TickMode m) {
    state_.spike_state_back = state_.spike_state;
    for (auto& rb : state_.region_biases) rb = rra::core_math::apply_attractor(rb, 1.0f, 0.05f); // Relax biases
    auto man = get_manifold_view();
    sao1::sao_tick_physics(&man, in.data());

    active_nodes_write_.clear();
    for (size_t i = 0; i < active_nodes_; ++i) {
        uint8_t sp = state_.spike_state[i];
        if (m == TickMode::Cognitive && sp) last_cognitive_spikes++;
        if (sp) {
            uint32_t pi = nodes_[i].pool_syn_offset;
            int safety = 0;
            while (pi != 0xFFFFFFFF && safety++ < 1000) {
                auto& pg = synapse_pool_.get_page(pi);
                for (int s = 0; s < 8; ++s) {
                    if (pg.synapses[s].is_active()) {
                        uint32_t t = pg.synapses[s].target_idx();
                        if (t < active_nodes_) {
                            uint32_t st = pg.synapses[s].state();
                            float mult = static_cast<float>(st == 1) - static_cast<float>(st == 2);
                            if (state_.node_regions[i] == 4 /* Modulatory */) {
                                state_.region_biases[t % 16] += mult * 0.15f; 
                            } else {
                                state_.membrane[t] += mult * 4.0f; // High-gain paged routing
                                if (!is_in_queue_[t]) { active_nodes_write_.push_back(t); is_in_queue_[t] = true; }
                            }
                        }
                    }
                }
                pi = pg.next_page_idx;
            }
            state_.refractory_timer[i] = static_cast<uint8_t>(cfg_.homeos_refractory);
        }
        state_.firing_rate_ema[i] = rra::core_math::apply_attractor(state_.firing_rate_ema[i], (float)sp, 0.01f);
        if (!std::isfinite(state_.membrane[i])) state_.membrane[i] = 0.0f;
        if (std::abs(state_.membrane[i]) > 1e-3f || state_.refractory_timer[i] > 0 || state_.dopamine[i] > 1e-3f) {
            if (!is_in_queue_[i]) { active_nodes_write_.push_back((uint32_t)i); is_in_queue_[i] = true; }
        } else {
            is_in_queue_[i] = false;
        }
    }
    active_nodes_read_ = std::move(active_nodes_write_);
    current_time_step_++;
}

void NISEngine::diffuse_dopamine() {
    // DEPRECATED: Gas diffusion removed. Morton geometric credit is the sole
    // per-neuron signal. Dopamine decay is now inlined into consolidate_learning.
    // This function is kept as a no-op stub for API compatibility.
}

void NISEngine::consolidate_learning(float lr) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        float dopamine_i = std::clamp(state_.dopamine[i], -1.0f, 1.0f); // Removed * 0.01f scale
        if (std::abs(dopamine_i) < 0.001f) continue;

        // 1. Float-Weight STDP for Local Causal Masks
        for (int j = 0; j < 64; ++j) {
            int source_idx = i - j; // STRICT BPTT CAUSALITY
            if (source_idx >= 0) {
                // -- Branchless SIMD Weight STDP & Mask Flattening --
                // 3-factor true E-Prop: LR × Dopamine × Source_Eligibility × Target_Eligibility
                // This now matches the paged synapse formula exactly.
                float source_elig = state_.eligibility_trace[source_idx];
                float delta = lr * dopamine_i * source_elig * state_.eligibility_trace[i];
                
                // Q4 TurboQuant decode -> update -> re-encode
                float w = std::clamp(q4_get(state_.local_weights_q4.data(), i * 64 + j, state_.weight_codebook.data()) + delta, -3.0f, 3.0f);
                q4_set(state_.local_weights_q4.data(), i * 64 + j, w, state_.weight_codebook.data());
                
                uint64_t mask = 1ULL << j;
                uint64_t is_ex = static_cast<uint64_t>(w >  0.1f);
                uint64_t is_in = static_cast<uint64_t>(w < -0.1f);
                
                // Straight-Through Estimator (STE): weights in the dead-zone [-0.1, +0.1]
                // with a nonzero gradient get a 3x boost to escape faster.
                // Prevents ternary dead gradients where w floats change but bits never flip.
                if (is_ex == 0 && is_in == 0 && std::abs(delta) > 1e-5f) {
                    float w_ste = std::clamp(w + delta * 3.0f, -3.0f, 3.0f);
                    q4_set(state_.local_weights_q4.data(), i * 64 + j, w_ste, state_.weight_codebook.data());
                }
                
                // Clear the bit first, then conditionally set it without branching
                state_.local_ex_masks[i] &= ~mask;
                state_.local_in_masks[i] &= ~mask;
                state_.local_ex_masks[i] |= (is_ex << j);
                state_.local_in_masks[i] |= (is_in << j);
            }
        }

        // 2. Float-Weight STDP for Paged Synapses
        if (state_.eligibility_trace[i] < 1e-4f) continue;
        uint32_t pi = nodes_[i].pool_syn_offset;
        int safety = 0;
        while (pi != 0xFFFFFFFF && safety++ < 1000) { 
            auto& pg = synapse_pool_.get_page(pi); 
            auto& wp = synapse_pool_.get_weight_page(pi);
            for (uint32_t s = 0; s < 8; ++s) {
                if (pg.synapses[s].is_active()) {
                    uint32_t target_idx = pg.synapses[s].target_idx();
                    if (target_idx < active_nodes_) {
                        float reward_at_target = std::clamp(state_.dopamine[target_idx], -1.0f, 1.0f);
                        wp.weights[s] += lr * reward_at_target * state_.eligibility_trace[target_idx] * state_.eligibility_trace[i];
                        wp.weights[s] = std::clamp(wp.weights[s], -3.0f, 3.0f);
                        if (wp.weights[s] > 0.1f) pg.synapses[s].set(target_idx, 1);
                        else if (wp.weights[s] < -0.1f) pg.synapses[s].set(target_idx, 2);
                        else pg.synapses[s].set(target_idx, 0);
                    }
                }
            }
            pi = pg.next_page_idx; 
        }
    }
    // Dopamine temporal decay: credits from previous tick lose 5% potency.
    // Inlined here so NREM is a single sequential pass — no separate O(N) copy.
    for (auto& d : state_.dopamine) d *= 0.95f;
}

void NISEngine::clear_rewards() {
    std::fill(state_.dopamine.begin(), state_.dopamine.end(), 0.0f);
    std::fill(state_.dopamine_back.begin(), state_.dopamine_back.end(), 0.0f);
}

void NISEngine::add_reward(size_t i, float a) { if (i < state_.dopamine.size()) state_.dopamine[i] += a; }

void NISEngine::force_wire(size_t s, size_t t, float w) {
    if (s >= nodes_.size() || t >= nodes_.size()) return;
    int dist = (int)s - (int)t; // REALIGNMENT: Pull architecture
    if (std::abs(dist) < 32) {
        int bit = dist + 32;
        q4_set(state_.local_weights_q4.data(), static_cast<int>(t * 64 + bit), w, state_.weight_codebook.data());
        if (w < 0.0f) { state_.local_in_masks[t] |= (1ULL << bit); state_.local_ex_masks[t] &= ~(1ULL << bit); }
        else { state_.local_ex_masks[t] |= (1ULL << bit); state_.local_in_masks[t] &= ~(1ULL << bit); }
    } else {
        force_wire_paged(s, t, w);
    }
}

void NISEngine::force_wire_paged(size_t s, size_t t, float w) {
    uint8_t st = (w < 0.0f) ? 2 : 1; 
    uint32_t pi = nodes_[s].pool_syn_offset, lp = 0xFFFFFFFF;
    int safety = 0;
    while (pi != 0xFFFFFFFF && safety++ < 1000) { 
        auto& pg = synapse_pool_.get_page(pi); 
        auto& wp = synapse_pool_.get_weight_page(pi);
        for (uint32_t i = 0; i < 8; ++i) {
            if (pg.synapses[i].target_idx() == (uint32_t)t && pg.synapses[i].is_active()) {
                pg.synapses[i].set((uint32_t)t, st); wp.weights[i] = w; return; 
            }
        }
        lp = pi; pi = pg.next_page_idx; 
    }
    pi = nodes_[s].pool_syn_offset; bool found = false; safety = 0;
    while (pi != 0xFFFFFFFF && safety++ < 1000) { 
        auto& pg = synapse_pool_.get_page(pi); 
        auto& wp = synapse_pool_.get_weight_page(pi);
        for (uint32_t i = 0; i < 8; ++i) if (!pg.synapses[i].is_active()) { pg.synapses[i].set((uint32_t)t, st); wp.weights[i] = w; found = true; break; }
        if (found) break; pi = pg.next_page_idx; 
    }
    if (!found) { 
        uint32_t np = synapse_pool_.allocate_page(); 
        if (nodes_[s].pool_syn_offset == 0xFFFFFFFF) nodes_[s].pool_syn_offset = np; 
        else if (lp != 0xFFFFFFFF) synapse_pool_.get_page(lp).next_page_idx = np; 
        synapse_pool_.get_page(np).synapses[0].set((uint32_t)t, st); 
        synapse_pool_.get_weight_page(np).weights[0] = w;
    }
    nodes_[s].pool_syn_count++;
}

void NISEngine::compile_topology() {
    const int N = static_cast<int>(nodes_.size()); if (N < 512) return;
    const int INPUT_COUNT = 256; const int OUTPUT_COUNT = 8;
    
    // 0. Fix Topological Amnesia: Promote local bits to paged pool before sort
    // so learned weights survive the coordinate shift.
    for (int i = 0; i < N; ++i) {
        uint64_t combined = state_.local_ex_masks[i] | state_.local_in_masks[i];
        for (int j = 0; j < 64; ++j) {
            if ((combined >> j) & 1ULL) {
                int source_idx = i - j; // STRICT BPTT CAUSALITY
                if (source_idx >= 0) {
                    float lw = q4_get(state_.local_weights_q4.data(), i * 64 + j, state_.weight_codebook.data());
                    force_wire_paged(source_idx, i, lw);
                }
            }
        }
    }
    std::fill(state_.local_ex_masks.begin(), state_.local_ex_masks.end(), 0ULL);
    std::fill(state_.local_in_masks.begin(), state_.local_in_masks.end(), 0ULL);
    std::fill(state_.local_weights_q4.begin(), state_.local_weights_q4.end(), 0x88U);
    q4_build_codebook(state_.weight_codebook.data());

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (i < INPUT_COUNT || i >= (N - OUTPUT_COUNT)) { nodes_[i].desired_1d_pos = static_cast<float>(i); continue; }
        double ms = 0.0, ws = 0.0; uint32_t pi = nodes_[i].pool_syn_offset;
        int safety = 0;
        while (pi != 0xFFFFFFFF && safety++ < 1000) { 
            auto& pg = synapse_pool_.get_page(pi); 
            for (int s = 0; s < 8; ++s) {
                if (pg.synapses[s].is_active()) { 
                    ms += static_cast<double>(pg.synapses[s].target_idx()); 
                    ws += 1.0; 
                } 
            }
            pi = pg.next_page_idx; 
        }
        if (ws > 1e-6) { float target = static_cast<float>(ms / ws); nodes_[i].desired_1d_pos = i + 0.20f * (target - i); } 
        else { nodes_[i].desired_1d_pos = static_cast<float>(i); }
    }

    std::vector<uint32_t> old_to_new(N); std::iota(old_to_new.begin(), old_to_new.end(), 0);
    std::sort(old_to_new.begin() + INPUT_COUNT, old_to_new.end() - OUTPUT_COUNT, [&](uint32_t a, uint32_t b) { return nodes_[a].desired_1d_pos < nodes_[b].desired_1d_pos; });
    
    auto pf = [&](aligned_vector<float>& v) { aligned_vector<float> t = v; for (int i = 0; i < N; ++i) v[old_to_new[i]] = t[i]; };
    auto pu8 = [&](aligned_vector<uint8_t>& v) { aligned_vector<uint8_t> t = v; for (int i = 0; i < N; ++i) v[old_to_new[i]] = t[i]; };
    auto pu32 = [&](aligned_vector<uint32_t>& v) { aligned_vector<uint32_t> t = v; for (int i = 0; i < N; ++i) v[old_to_new[i]] = t[i]; };
    auto pu64 = [&](aligned_vector<uint64_t>& v) { aligned_vector<uint64_t> t = v; for (int i = 0; i < N; ++i) v[old_to_new[i]] = t[i]; };

    pf(state_.membrane); pf(state_.firing_rate_ema); pf(state_.eligibility_trace); pf(state_.adaptive_thresholds); pf(state_.base_thresholds); pf(state_.leak_rates); pf(state_.sensory_pressure); 
    pu8(state_.spike_state); pu8(state_.refractory_timer); pu8(state_.node_regions); pu32(state_.last_spike_ticks);
    pu64(state_.local_ex_masks); pu64(state_.local_in_masks); pf(state_.dopamine); pf(state_.dopamine_back); pu8(state_.spike_state_back);

    #pragma omp parallel for
    for (int p = 0; p < static_cast<int>(synapse_pool_.size()); ++p) { 
        auto& pg = synapse_pool_.get_page(static_cast<uint32_t>(p)); 
        for (int s = 0; s < 8; ++s) if (pg.synapses[s].is_active()) { uint32_t ot = pg.synapses[s].target_idx(); if (ot < static_cast<uint32_t>(N)) pg.synapses[s].data = (old_to_new[ot] & 0x3FFFFFFF) | (pg.synapses[s].data & 0xC0000000); }
    }
    std::vector<Node> nn(N); for (int i = 0; i < N; ++i) nn[old_to_new[i]] = std::move(nodes_[i]); nodes_ = std::move(nn);
    
    // Rebuild local masks from paged pool to maximize L1 performance for short-range links
    for (int i = 0; i < N; ++i) {
        uint32_t pi = nodes_[i].pool_syn_offset;
        int safety = 0;
        while (pi != 0xFFFFFFFF && safety++ < 1000) {
            auto& pg = synapse_pool_.get_page(pi);
            auto& wp = synapse_pool_.get_weight_page(pi);
            for (int s = 0; s < 8; ++s) {
                if (!pg.synapses[s].is_active()) continue;
                int t_idx = static_cast<int>(pg.synapses[s].target_idx());
                // Physics kernel (pull) looks backwards chronologically from target "t_idx".
                // bit 0 = self, bit 1 = t_idx-1, so bit = target_idx - source_idx
                int bit = t_idx - i; 
                if (bit >= 0 && bit < 64) { 
                    q4_set(state_.local_weights_q4.data(), static_cast<int>(t_idx * 64 + bit), wp.weights[s], state_.weight_codebook.data());
                    if (wp.weights[s] < 0.0f) state_.local_in_masks[t_idx] |= (1ULL << bit); 
                    else state_.local_ex_masks[t_idx] |= (1ULL << bit); 
                    pg.synapses[s].clear(); 
                    wp.weights[s] = 0.0f;
                }
            }
            pi = pg.next_page_idx;
        }
        uint64_t combined = state_.local_ex_masks[i] | state_.local_in_masks[i];
        int count = static_cast<int>(__popcnt64(combined)) + nodes_[i].pool_syn_count;
        if (count < 4 && i >= 256 && i < (N - 8)) axonal_sprouting(i, 4 - count);
    }
    for(auto& idx : active_nodes_read_) idx = old_to_new[idx];
    for(auto& idx : active_nodes_write_) idx = old_to_new[idx];
    
    // Rebuild is_in_queue_
    std::fill(is_in_queue_.begin(), is_in_queue_.end(), false);
    for(auto idx : active_nodes_read_) is_in_queue_[idx] = true;
    for(auto idx : active_nodes_write_) is_in_queue_[idx] = true;
}

void NISEngine::spatial_sort() { compile_topology(); }
void NISEngine::add_node(uint64_t f, Node::NodeType type, FunctionalRegion region) { 
    state_.add_node_state(cfg_.default_threshold, cfg_.homeos_target_rate); 
    active_nodes_ = state_.membrane.size(); 
    is_in_queue_.push_back(false); 
    zero_buffer_.push_back(0.0f); 
    state_.spike_state_back.push_back(0U);
    state_.node_regions.back() = static_cast<uint8_t>(region);
    Node n; n.spatial_id = f; n.pool_syn_offset = 0xFFFFFFFF; n.pool_syn_count = 0;
    n.type = type; n.region = region;
    nodes_.push_back(n); 
}
void NISEngine::rebuild_region_index_cache() {}
sao1::sao_manifold NISEngine::get_manifold_view() { return { state_.membrane.data(), state_.base_thresholds.data(), state_.eligibility_trace.data(), state_.last_spike_ticks.data(), state_.sensory_pressure.data(), state_.dopamine.data(), state_.dopamine_back.data(), state_.local_ex_masks.data(), state_.local_in_masks.data(), state_.spike_state.data(), state_.spike_state_back.data(), state_.refractory_timer.data(), state_.leak_rates.data(), nullptr /*local_weights_q4: not decoded in hot-loop*/, state_.firing_rate_ema.data(), state_.adaptive_thresholds.data(), state_.region_biases.data(), state_.node_regions.data(), nodes_.size(), current_time_step_ }; }

bool NISEngine::save_state(const std::string& p) const {
    std::ofstream out(p, std::ios::binary); if (!out.is_open()) return false;
    if (!out.write(kSparseMagic.data(), 8)) return false; write_pod(out, kSparseVersion); write_pod(out, static_cast<uint64_t>(nodes_.size()));
    for (const auto& n : nodes_) {
        write_pod(out, n.spatial_id); write_pod(out, static_cast<uint8_t>(n.type)); write_pod(out, n.group_id); write_pod(out, static_cast<uint8_t>(n.is_lateral_projection)); write_pod(out, static_cast<uint8_t>(n.region));
        write_pod(out, n.base_threshold); write_pod(out, n.leak_rate); write_pod(out, n.plasticity); write_pod(out, n.metabolic_cap); write_pod(out, static_cast<uint8_t>(n.frozen));
        uint64_t sc = 0; uint32_t pi = n.pool_syn_offset; int safety = 0;
        while (pi != 0xFFFFFFFF && safety++ < 1000) { auto& pg = synapse_pool_.get_page(pi); for (int s = 0; s < 8; ++s) if (pg.synapses[s].is_active()) sc++; pi = pg.next_page_idx; }
        write_pod(out, sc); pi = n.pool_syn_offset; safety = 0;
        while (pi != 0xFFFFFFFF && safety++ < 1000) { auto& pg = synapse_pool_.get_page(pi); auto& wp = synapse_pool_.get_weight_page(pi); for (int s = 0; s < 8; ++s) if (pg.synapses[s].is_active()) { write_pod(out, pg.synapses[s]); write_pod(out, wp.weights[s]); } pi = pg.next_page_idx; }
    }
    return true;
}

bool NISEngine::load_state(const std::string& p) {
    std::ifstream in(p, std::ios::binary); if (!in.is_open()) return false;
    std::array<char, 8> m; if (!in.read(m.data(), 8) || m != kSparseMagic) return false;
    uint32_t v; read_pod(in, v); if (v != kSparseVersion) return false;
    uint64_t nn; read_pod(in, nn); nodes_.clear(); synapse_pool_.clear(); nodes_.resize(nn);
    for (uint64_t i = 0; i < nn; ++i) {
        Node& n = nodes_[i]; uint8_t u; read_pod(in, n.spatial_id); read_pod(in, u); n.type = static_cast<Node::NodeType>(u); read_pod(in, n.group_id); read_pod(in, u); n.is_lateral_projection = (u != 0); read_pod(in, u); n.region = static_cast<FunctionalRegion>(u);
        read_pod(in, n.base_threshold); read_pod(in, n.leak_rate); read_pod(in, n.plasticity); read_pod(in, n.metabolic_cap); read_pod(in, u); n.frozen = (u != 0);
        uint64_t sc; read_pod(in, sc); n.pool_syn_offset = 0xFFFFFFFF; n.pool_syn_count = 0;
        if (sc > 2000000) return false; // Sanity check
        for (uint64_t s = 0; s < sc; ++s) { Synapse sy; float w; read_pod(in, sy); read_pod(in, w); force_wire_paged(i, sy.target_idx(), w); }
    }
    rehydrate_runtime_state_from_nodes(); return true;
}

bool NISEngine::load_checkpoint(const std::string& p) { return load_state(p); }
float NISEngine::get_surprise() const { return state_.external_surprise; }

WeightStats NISEngine::get_weight_stats() const { 
    WeightStats s; size_t count = 0; 
    for (size_t p = 0; p < synapse_pool_.size(); ++p) for (uint32_t i = 0; i < 8; ++i) if (synapse_pool_.get_page(static_cast<uint32_t>(p)).synapses[i].is_active()) count++; 
    for (const auto& m : state_.local_ex_masks) count += __popcnt64(m);
    for (const auto& m : state_.local_in_masks) count += __popcnt64(m);
    s.count = count; s.mean_weight = 1.0f; return s; 
}

float NISEngine::engine_compute_endogenous_reward() {
    float total_firing = 0.0f;
    int spiking_nodes = 0;
    for (size_t i = 0; i < active_nodes_; ++i) {
        total_firing += state_.firing_rate_ema[i];
        if (state_.spike_state[i]) spiking_nodes++;
    }
    float avg_firing = total_firing / static_cast<float>(std::max<size_t>(1, active_nodes_));
    float novelty = state_.external_surprise; // 0.0
    float energy_penalty = std::max(0.0f, avg_firing - 0.45f) * 10.0f;
    float stagnation = (spiking_nodes == 0) ? 1.0f : 0.0f;
    
    // Calculate variance of firing rates (Diversity Bonus)
    // Homeostasis keeps firing rate stable, but variance implies organized structure
    float var = 0.0f;
    for(auto m : state_.firing_rate_ema) { float diff = m - avg_firing; var += diff * diff; }
    var /= static_cast<float>(nodes_.size());
    float diversity_bonus = std::min(1.0f, var * 50.0f);
    
    // -----------------------------------------------------------------------------
    // Structural signals only — NOT returned as a learning dopamine injection.
    // Morton per-neuron credit is the primary gradient signal (injected in train.cpp).
    // This function is kept for structural diagnostics: stagnation detection,
    // energy penalty gating, and homeostatic seizure prevention.
    // -----------------------------------------------------------------------------
    
    // Stagnation guard: if no neurons spike, wipe accumulated dopamine to prevent
    // reward hallucination from stale signals.
    if (stagnation > 0.5f) {
        for (auto& d : state_.dopamine) d = 0.0f;
    }
    
    // Energy penalty: clamp dopamine on neurons that are chronically over-firing
    // to prevent runaway excitation -> seizure cascades.
    if (avg_firing > 0.45f) {
        for (size_t i = 0; i < active_nodes_; ++i) {
            if (state_.firing_rate_ema[i] > 0.45f)
                state_.dopamine[i] = std::clamp(state_.dopamine[i] - 0.1f, -1.0f, 1.0f);
        }
    }
    
    // Return diversity bonus as a diagnostic scalar (logged but NOT re-injected globally)
    return (0.5f * diversity_bonus) - (1.0f * energy_penalty) - (1.0f * stagnation);
}

void NISEngine::execute_tick(TickMode m) { 
    execute_cycle(zero_buffer_, 0, m); 
    // Structural homeostasis pass — modifies dopamine array defensively (stagnation wipe,
    // energy penalty clamp) but does NOT add a global scalar reward on top of Morton credit.
    engine_compute_endogenous_reward();
}
void NISEngine::kill_neuron(size_t i) { if (i < nodes_.size()) { nodes_[i].pool_syn_offset = 0xFFFFFFFF; nodes_[i].pool_syn_count = 0; } }
void NISEngine::prune_nodes(float th) {}
void NISEngine::apply_weight_decay(float f) {}
void NISEngine::reset_state() { std::fill(state_.membrane.begin(), state_.membrane.end(), 0.0f); }
void NISEngine::set_plasticity(float p) { cfg_.plasticity_learning_rate = p; }
void NISEngine::set_emotion(Emotion e) {}
void NISEngine::inject_signals(const std::vector<std::pair<size_t, float>>& sigs) { for (const auto& s : sigs) if (s.first < state_.sensory_pressure.size()) state_.sensory_pressure[s.first] += s.second * 5.0f; }
void NISEngine::prune_weak_synapses(float th) {}
void NISEngine::inject_global_dopamine(float a) { for (auto& d : state_.dopamine) d += a; }
void NISEngine::add_membrane_noise(FunctionalRegion r, float i) { for (size_t n = 0; n < nodes_.size(); ++n) if (nodes_[n].region == r) state_.membrane[n] += i * ((float)rand() / (float)RAND_MAX - 0.5f); }
void NISEngine::inject_bit_stream(const std::vector<uint8_t>& bits) { for (size_t i = 0; i < bits.size() && i < 256; ++i) state_.sensory_pressure[i] += (bits[i] ? 1.0f : -1.0f); }
void NISEngine::forward_input(const std::vector<InputEvent>& data) { for (const auto& ev : data) if (ev.x < state_.sensory_pressure.size()) state_.sensory_pressure[ev.x] += ev.current; }
EngineOutput NISEngine::read_output(size_t num_anchors) { EngineOutput out; const int N = static_cast<int>(nodes_.size()); float max_v = -1e9f; for (size_t i = 0; i < num_anchors; ++i) { float v = state_.membrane[N - num_anchors + i]; if (v > max_v) { max_v = v; out.winning_bucket = static_cast<uint32_t>(i); } } out.byte = static_cast<uint8_t>(out.winning_bucket & 0xFF); out.ready_to_speak = (max_v > 0.8f); return out; }
void NISEngine::organic_growth() { for (size_t i = 0; i < nodes_.size(); ++i) if (state_.firing_rate_ema[i] > 0.5f && state_.dopamine[i] > 0.5f) { add_node(nodes_[i].spatial_id ^ (uint64_t)rand()); force_wire(nodes_.size() - 1, (size_t)i, 0.5f); if (nodes_.size() >= 32768) break; } }
void NISEngine::axonal_sprouting(size_t i, int count) { std::uniform_int_distribution<size_t> dist(0, nodes_.size() - 1); for (int k = 0; k < count; ++k) force_wire(i, dist(rng_), 0.5f); }

} // namespace rra::nis_engine
