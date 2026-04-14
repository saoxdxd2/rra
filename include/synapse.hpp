#pragma once

#include <vector>
#include <deque>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <atomic>
#include <mutex>

namespace rra::nis_engine {

#pragma pack(push, 1)
struct Synapse {
    /**
     * True Ternary Synapse:
     * Bits 30-31: State (00 = Dead, 01 = Excitatory, 10 = Inhibitory)
     * Bits 0-29:  Target Index (Supports 1 Billion Neurons)
     */
    uint32_t data = 0x00000000; 

    inline uint32_t state() const { return data >> 30; }
    inline bool is_active() const { return state() != 0; }
    inline bool is_inhibitory() const { return state() == 2; }
    inline uint32_t target_idx() const { return data & 0x3FFFFFFF; }

    inline void set(uint32_t target, uint8_t state) {
        data = (target & 0x3FFFFFFF) | (static_cast<uint32_t>(state & 0x3) << 30);
    }
    
    inline void clear() { data = 0x00000000; }
};
#pragma pack(pop)

struct SynapseWeightPage {
    float weights[8];
    SynapseWeightPage() { for(int i=0; i<8; ++i) weights[i] = 0.0f; }
};

// ---------------------------------------------------------------------------
// Q4WeightStore — TurboQuant-Style 4-Bit Latent Weight Compression
//
// The latent weight array (N * 64 floats) dominates NREM memory bandwidth.
// Weights are clamped to [-3.0, 3.0] and statistically cluster near the
// ternary boundaries {-1, 0, +1} after learning, giving an ideal distribution
// for 4-bit (16-centroid) vector quantization.
//
// Memory savings: N*64*4 bytes (float32) -> N*32 bytes (nibble) = 8x reduction.
// For 32K neurons: 8MB -> 1MB — fully L2-cache resident.
//
// API:
//   q4_get(nibbles, idx)          -> decode nibble to float via codebook
//   q4_set(nibbles, idx, val, cb) -> quantize val to nearest codebook entry
//   q4_build_codebook(cb)         -> initialize linearly-spaced codebook
//   q4_refine_codebook(nibbles, n64, cb) -> online codebook update (optional)
// ---------------------------------------------------------------------------

// Default codebook: 16 linearly spaced values in [-3.0, 3.0]
inline void q4_build_codebook(float cb[16]) {
    for (int i = 0; i < 16; ++i)
        cb[i] = -3.0f + static_cast<float>(i) * (6.0f / 15.0f);
}

// Decode: nibble index -> float (upper nibble = even idx, lower = odd idx)
inline float q4_get(const uint8_t* nibbles, int idx, const float* cb) {
    uint8_t byte = nibbles[idx >> 1];
    uint8_t nib  = (idx & 1) ? (byte & 0x0F) : (byte >> 4);
    return cb[nib];
}

// Encode: float -> nearest codebook nibble index, stored in-place
inline void q4_set(uint8_t* nibbles, int idx, float val, const float* cb) {
    // Find nearest codebook entry (branchless linear scan, 16 entries fits in cache)
    int best = 0;
    float best_dist = std::abs(val - cb[0]);
    for (int k = 1; k < 16; ++k) {
        float d = std::abs(val - cb[k]);
        // Branchless update
        int closer = static_cast<int>(d < best_dist);
        best      = closer * k + (1 - closer) * best;
        best_dist = closer * d + (1 - closer) * best_dist;
    }
    uint8_t nib = static_cast<uint8_t>(best);
    uint8_t& byte = nibbles[idx >> 1];
    if (idx & 1) byte = (byte & 0xF0) | nib;
    else         byte = (byte & 0x0F) | (nib << 4);
}

// Optional: refine codebook centroids using online mean (call during NREM)
// n_nibbles = N * 64 (total number of logical weights)
inline void q4_refine_codebook(const uint8_t* nibbles, int n_nibbles, float* cb) {
    float sums[16] = {};
    int   counts[16] = {};
    for (int i = 0; i < n_nibbles; ++i) {
        uint8_t byte = nibbles[i >> 1];
        int nib = (i & 1) ? (byte & 0x0F) : (byte >> 4);
        float decoded_val = cb[nib];
        sums[nib]   += decoded_val;
        counts[nib] += 1;
    }
    for (int k = 0; k < 16; ++k)
        if (counts[k] > 0) cb[k] = sums[k] / static_cast<float>(counts[k]);
}

struct SynapsePage {
    static constexpr uint32_t PAGE_SIZE = 8;
    Synapse synapses[PAGE_SIZE];
    uint32_t next_page_idx;
    SynapsePage() {
        next_page_idx = 0xFFFFFFFF;
        for (uint32_t i = 0; i < PAGE_SIZE; ++i) synapses[i].clear();
    }
};

class GlobalSynapseHeap {
public:
    GlobalSynapseHeap(size_t initial_pages = 1024) {
        pages.resize(initial_pages);
        weight_pages.resize(initial_pages);
        free_head.store(0);
        for (size_t i = 0; i < initial_pages - 1; ++i) pages[i].next_page_idx = static_cast<uint32_t>(i + 1);
        pages.back().next_page_idx = 0xFFFFFFFF;
    }
    GlobalSynapseHeap(GlobalSynapseHeap&& o) noexcept : pages(std::move(o.pages)), weight_pages(std::move(o.weight_pages)) { free_head.store(o.free_head.load()); }
    GlobalSynapseHeap& operator=(GlobalSynapseHeap&& o) noexcept { if (this != &o) { pages = std::move(o.pages); weight_pages = std::move(o.weight_pages); free_head.store(o.free_head.load()); } return *this; }
    GlobalSynapseHeap(const GlobalSynapseHeap&) = delete;
    GlobalSynapseHeap& operator=(const GlobalSynapseHeap&) = delete;

    uint32_t allocate_page() {
        uint32_t head = free_head.load();
        while (head != 0xFFFFFFFF) {
            uint32_t next = pages[head].next_page_idx;
            if (free_head.compare_exchange_weak(head, next)) { pages[head] = SynapsePage(); weight_pages[head] = SynapseWeightPage(); return head; }
        }
        std::lock_guard<std::mutex> lock(expansion_mutex);
        size_t old_size = pages.size();
        pages.resize(old_size * 2);
        weight_pages.resize(old_size * 2);
        for (size_t i = old_size; i < pages.size() - 1; ++i) pages[i].next_page_idx = static_cast<uint32_t>(i + 1);
        pages.back().next_page_idx = 0xFFFFFFFF;
        free_head.store(static_cast<uint32_t>(old_size + 1));
        pages[old_size] = SynapsePage();
        weight_pages[old_size] = SynapseWeightPage();
        return static_cast<uint32_t>(old_size);
    }

    void free_page(uint32_t page_idx) {
        if (page_idx == 0xFFFFFFFF) return;
        uint32_t head = free_head.load();
        do { pages[page_idx].next_page_idx = head; } while (!free_head.compare_exchange_weak(head, page_idx));
    }

    SynapsePage& get_page(uint32_t idx) { return pages[idx]; }
    const SynapsePage& get_page(uint32_t idx) const { return pages[idx]; }
    
    SynapseWeightPage& get_weight_page(uint32_t idx) { return weight_pages[idx]; }
    const SynapseWeightPage& get_weight_page(uint32_t idx) const { return weight_pages[idx]; }
    
    void clear() {
        pages.clear(); pages.resize(1024);
        weight_pages.clear(); weight_pages.resize(1024);
        free_head.store(0);
        for (size_t i = 0; i < 1023; ++i) pages[i].next_page_idx = (uint32_t)i + 1;
        pages.back().next_page_idx = 0xFFFFFFFF;
    }
    
    size_t size() const { return pages.size(); }

private:
    std::deque<SynapsePage> pages;
    std::deque<SynapseWeightPage> weight_pages;
    std::atomic<uint32_t> free_head;
    std::mutex expansion_mutex;
};

using SynapsePool = GlobalSynapseHeap;

} // namespace rra::nis_engine
