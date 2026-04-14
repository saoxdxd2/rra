#pragma once

#include <vector>
#include <deque>
#include <memory>

namespace rra::memory {

/**
 * @brief Represents a single time-step snapshot of the neural state.
 * Stored in the ReplayBuffer for offline "sleep" consolidation.
 */
struct EpisodeTrace {
    std::vector<float> spikes;
    std::vector<float> membrane;
    std::vector<float> pred_error;
};

/**
 * @brief Experience Replay Buffer.
 *
 * Maintains a circular buffer of recent network activations.
 * During training phases, these traces are replayed to reinforce 
 * connections via the plasticity rules, stabilizing the network state.
 */
class ReplayBuffer {
public:
    static constexpr std::size_t MAX_RING_BUFFER = 64;

    explicit ReplayBuffer()
        : max_size_(MAX_RING_BUFFER) {}

    /**
     * @brief Pushes a new state snapshot into the buffer.
     * Evicts the oldest entry if max_size_ is exceeded.
     */
    void push(const std::vector<float>& spikes,
              const std::vector<float>& membrane,
              const std::vector<float>& pred_error) {
        if (buffer_.size() >= max_size_) {
            buffer_.pop_front();
        }
        buffer_.push_back({spikes, membrane, pred_error});
    }

    /**
     * @brief Returns a reference to the trace at the given index.
     * Index 0 is the oldest, back() is the newest.
     */
    const EpisodeTrace& at(std::size_t index) const {
        return buffer_[index];
    }

    std::size_t size() const { return buffer_.size(); }
    bool empty() const { return buffer_.empty(); }

    void clear() { buffer_.clear(); }

private:
    std::size_t max_size_;
    std::deque<EpisodeTrace> buffer_;
};

} // namespace rra::memory
