#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstdint>
#include <windows.h>
#include <atomic>

namespace rra::gnf::ipc {

// Memory-mapped file name for the 4D manifold renderer
static const char* IPC_MAP_NAME = "Local\\RRA_Manifold_IPC_v1";

// Zero-copy representation of the continuous 4D embedding space
struct ManifoldIPCData {
    std::atomic<uint64_t> frame_counter{0};

    // Extracted byte embeddings
    struct {
        float x, y, z, w;
        float velocity[4];
    } byte_coords[256];

    // Extracted neuron anchors (reduced array for visualization)
    struct {
        float x, y, z, w;
        float variance_score;
        bool is_anchor;
    } neuron_points[512];

    std::atomic<float> current_ce_loss{0.0f};
    std::atomic<float> swarm_entropy{0.0f};
};

} // namespace rra::gnf::ipc
