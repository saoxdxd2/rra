#include "aether.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rra::nn::aether {

namespace {

constexpr float kPi = 3.14159265358979323846f;

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t clamp_group_size(size_t requested, size_t seq_len) {
    if (seq_len == 0) return 1;
    return std::max<size_t>(1, std::min(requested, seq_len));
}

float safe_tanh(float x) {
    return std::tanh(std::clamp(x, -16.0f, 16.0f));
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

} // namespace

AetherConfig AetherConfig::for_model(size_t d_model) {
    AetherConfig cfg;
    cfg.d_model = d_model;
    return cfg;
}

uint64_t morton_encode4(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t bits) {
    bits = std::min<uint32_t>(bits, 16);
    uint64_t code = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        code |= (static_cast<uint64_t>((x >> bit) & 1U) << (4 * bit));
        code |= (static_cast<uint64_t>((y >> bit) & 1U) << (4 * bit + 1));
        code |= (static_cast<uint64_t>((z >> bit) & 1U) << (4 * bit + 2));
        code |= (static_cast<uint64_t>((w >> bit) & 1U) << (4 * bit + 3));
    }
    return code;
}

MortonCell4D morton_decode4(uint64_t code, uint32_t bits) {
    bits = std::min<uint32_t>(bits, 16);
    MortonCell4D cell;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        cell.x |= static_cast<uint32_t>((code >> (4 * bit)) & 1ULL) << bit;
        cell.y |= static_cast<uint32_t>((code >> (4 * bit + 1)) & 1ULL) << bit;
        cell.z |= static_cast<uint32_t>((code >> (4 * bit + 2)) & 1ULL) << bit;
        cell.w |= static_cast<uint32_t>((code >> (4 * bit + 3)) & 1ULL) << bit;
    }
    return cell;
}

SpectralGeometricPropagator::SpectralGeometricPropagator(size_t d_model)
    : SpectralGeometricPropagator(AetherConfig::for_model(d_model)) {}

SpectralGeometricPropagator::SpectralGeometricPropagator(AetherConfig config)
    : config_(config) {
    if (config_.d_model == 0) {
        throw std::invalid_argument("AetherConfig.d_model must be non-zero");
    }
    config_.max_scales = std::max<size_t>(1, config_.max_scales);
    config_.modes_per_scale = std::max<size_t>(1, config_.modes_per_scale);
    config_.probe_count = std::max<size_t>(1, config_.probe_count);
    config_.probe_taps = std::max<size_t>(1, config_.probe_taps);
    config_.probe_radius = std::max<size_t>(1, config_.probe_radius);
}

void SpectralGeometricPropagator::reset_memory() {
    for (auto& plan : plans_) {
        std::fill(plan.memory.begin(), plan.memory.end(), 0.0f);
    }
    for (auto& probe : probes_) {
        probe.signal_memory = 0.0f;
        probe.noise_memory = 0.0f;
    }
}

float SpectralGeometricPropagator::basis_value(size_t local, size_t group_size, size_t mode) {
    if (group_size <= 1 || mode == 0) {
        return 1.0f / std::sqrt(static_cast<float>(std::max<size_t>(1, group_size)));
    }
    const float norm = std::sqrt(2.0f / static_cast<float>(group_size));
    const float angle = kPi * (static_cast<float>(local) + 0.5f) * static_cast<float>(mode) /
                        static_cast<float>(group_size);
    return norm * std::cos(angle);
}

float SpectralGeometricPropagator::laplacian_eigenvalue(size_t group_size, size_t mode) {
    if (group_size <= 1 || mode == 0) return 0.0f;
    const float angle = kPi * static_cast<float>(mode) / (2.0f * static_cast<float>(group_size));
    const float s = std::sin(angle);
    return 4.0f * s * s;
}

void SpectralGeometricPropagator::rebuild_plans(size_t seq_len) {
    if (seq_len == cached_seq_len_ && !plans_.empty()) return;

    cached_seq_len_ = seq_len;
    plans_.clear();
    if (seq_len == 0) return;

    size_t group_size = 2;
    float scale_weight = 1.0f;
    for (size_t scale = 0; scale < config_.max_scales; ++scale) {
        const size_t active_group = clamp_group_size(group_size, seq_len);
        const size_t mode_count = std::min(config_.modes_per_scale, active_group);

        ScalePlan plan;
        plan.scale = scale;
        plan.group_size = active_group;
        plan.mode_count = mode_count;
        plan.scale_weight = scale_weight;
        plan.basis.assign(mode_count * active_group, 0.0f);
        plan.eigenvalues.assign(mode_count, 0.0f);
        plan.memory.assign(mode_count * config_.d_model, 0.0f);

        for (size_t m = 0; m < mode_count; ++m) {
            plan.eigenvalues[m] = laplacian_eigenvalue(active_group, m);
            for (size_t i = 0; i < active_group; ++i) {
                plan.basis[m * active_group + i] = basis_value(i, active_group, m);
            }
        }

        plans_.push_back(std::move(plan));
        if (active_group == seq_len) break;
        group_size *= 2;
        scale_weight *= config_.scale_decay;
    }

    rebuild_probes(seq_len);
}

void SpectralGeometricPropagator::rebuild_probes(size_t seq_len) {
    probes_.clear();
    if (seq_len == 0) return;

    const size_t count = std::min(config_.probe_count, std::max<size_t>(1, seq_len * 2));
    probes_.reserve(count);
    const size_t scale_count = std::max<size_t>(1, plans_.size());

    for (size_t p = 0; p < count; ++p) {
        const uint64_t h = mix64(0xA37C9E3779B97F4AULL ^ static_cast<uint64_t>(p));
        const size_t center = static_cast<size_t>(h % seq_len);
        const size_t scale = p % scale_count;
        const size_t scaled_radius = config_.probe_radius << std::min<size_t>(scale, 3);

        ProbePlan probe;
        probe.center = center;
        probe.radius = std::max<size_t>(1, std::min(scaled_radius, seq_len));
        probe.morton_code = static_cast<uint64_t>(center);
        probe.feature_index.reserve(config_.probe_taps);
        probe.feature_sign.reserve(config_.probe_taps);

        for (size_t tap = 0; tap < config_.probe_taps; ++tap) {
            const uint64_t th = mix64(h + 0x9E3779B97F4A7C15ULL * (tap + 1));
            probe.feature_index.push_back(static_cast<size_t>(th % config_.d_model));
            probe.feature_sign.push_back((th & 1ULL) ? 1.0f : -1.0f);
        }

        probes_.push_back(std::move(probe));
    }
}

void SpectralGeometricPropagator::apply_probe_array(
    const s4m::Tensor& x,
    const s4m::Tensor& propagated,
    s4m::Tensor& delta
) {
    if (probes_.empty()) return;

    const size_t seq_len = x.shape[0];
    const size_t d_model = x.shape[1];
    const float* x_ptr = x.ptr();
    const float* prop_ptr = propagated.ptr();
    float* delta_ptr = delta.ptr();

    for (auto& probe : probes_) {
        const size_t radius = std::max<size_t>(1, std::min(probe.radius, seq_len));
        const size_t start = (probe.center > radius) ? (probe.center - radius) : 0;
        const size_t end = std::min(seq_len, probe.center + radius + 1);
        const float inv_norm = 1.0f / static_cast<float>((end - start) * probe.feature_index.size());

        float signal = 0.0f;
        float prediction = 0.0f;
        for (size_t pos = start; pos < end; ++pos) {
            const int64_t offset = static_cast<int64_t>(pos) - static_cast<int64_t>(probe.center);
            const float phase = std::cos(kPi * static_cast<float>(offset) / static_cast<float>(radius + 1));
            const float locality = 1.0f - (std::abs(static_cast<float>(offset)) / static_cast<float>(radius + 1));
            const float weight = phase * locality;

            const float* x_row = x_ptr + pos * d_model;
            const float* p_row = prop_ptr + pos * d_model;
            for (size_t tap = 0; tap < probe.feature_index.size(); ++tap) {
                const size_t d = probe.feature_index[tap];
                const float sign = probe.feature_sign[tap];
                signal += sign * weight * x_row[d];
                prediction += sign * weight * p_row[d];
            }
        }

        signal *= inv_norm;
        prediction *= inv_norm;
        const float residual = signal - prediction;

        probe.signal_memory = config_.probe_memory_decay * probe.signal_memory +
                              (1.0f - config_.probe_memory_decay) * signal;
        probe.noise_memory = config_.noise_memory_decay * probe.noise_memory +
                             (1.0f - config_.noise_memory_decay) * residual;

        const float coherent = config_.beamforming_gain * probe.signal_memory;
        const float cancellation = config_.cancellation_gain * probe.noise_memory;
        const float surprise = config_.residual_surprise_gain * residual;
        const float amplitude = safe_tanh(coherent - cancellation + surprise);

        for (size_t pos = start; pos < end; ++pos) {
            const int64_t offset = static_cast<int64_t>(pos) - static_cast<int64_t>(probe.center);
            const float phase = std::cos(kPi * static_cast<float>(offset) / static_cast<float>(radius + 1));
            const float locality = 1.0f - (std::abs(static_cast<float>(offset)) / static_cast<float>(radius + 1));
            float* out_row = delta_ptr + pos * d_model;

            for (size_t tap = 0; tap < probe.feature_index.size(); ++tap) {
                const size_t d = probe.feature_index[tap];
                const float sign = probe.feature_sign[tap];
                out_row[d] += amplitude * sign * phase * locality;
            }
        }
    }
}

s4m::Tensor SpectralGeometricPropagator::forward(const s4m::Tensor& x) {
    if (x.shape.size() != 2) {
        throw std::invalid_argument("SpectralGeometricPropagator expects a rank-2 tensor");
    }

    const size_t seq_len = x.shape[0];
    const size_t d_model = x.shape[1];
    if (d_model != config_.d_model) {
        throw std::invalid_argument("AETHER d_model does not match input tensor width");
    }

    s4m::Tensor propagated({seq_len, d_model});
    s4m::Tensor delta({seq_len, d_model});
    if (seq_len == 0) return delta;

    rebuild_plans(seq_len);

    const float* x_ptr = x.ptr();
    float* prop_ptr = propagated.ptr();
    std::vector<float> coeff(d_model, 0.0f);

    for (auto& plan : plans_) {
        const size_t group_size = plan.group_size;
        const size_t group_count = ceil_div(seq_len, group_size);

        for (size_t group = 0; group < group_count; ++group) {
            const size_t start = group * group_size;
            const size_t end = std::min(start + group_size, seq_len);
            const size_t len = end - start;
            if (len == 0) continue;

            float energy = 0.0f;
            for (size_t i = start; i < end; ++i) {
                const float* row = x_ptr + i * d_model;
                for (size_t d = 0; d < d_model; ++d) {
                    energy += row[d] * row[d];
                }
            }
            const float curvature = std::clamp(energy / static_cast<float>(len * d_model), 0.0f, 32.0f);

            for (size_t mode = 0; mode < plan.mode_count; ++mode) {
                std::fill(coeff.begin(), coeff.end(), 0.0f);

                for (size_t local = 0; local < len; ++local) {
                    const float phi = (len == group_size)
                        ? plan.basis[mode * group_size + local]
                        : basis_value(local, len, mode);
                    const float* row = x_ptr + (start + local) * d_model;
                    for (size_t d = 0; d < d_model; ++d) {
                        coeff[d] += row[d] * phi;
                    }
                }

                float lambda = plan.eigenvalues[mode];
                if (len != group_size) {
                    lambda = laplacian_eigenvalue(len, mode);
                }

                const float heat = std::exp(-config_.diffusion_tau * plan.scale_weight * lambda * (1.0f + 0.05f * curvature));
                const float survival = 1.0f / (1.0f + config_.complexity_lambda * lambda + config_.renorm_beta * curvature);
                const float gain = plan.scale_weight * heat * survival;

                if (config_.persistent_memory) {
                    float* mem = plan.memory.data() + mode * d_model;
                    for (size_t d = 0; d < d_model; ++d) {
                        mem[d] = config_.memory_decay * mem[d] + (1.0f - config_.memory_decay) * coeff[d];
                        coeff[d] = 0.5f * coeff[d] + 0.5f * mem[d];
                    }
                }

                for (size_t local = 0; local < len; ++local) {
                    const float phi = (len == group_size)
                        ? plan.basis[mode * group_size + local]
                        : basis_value(local, len, mode);
                    float* row = prop_ptr + (start + local) * d_model;
                    const float alpha = gain * phi;
                    for (size_t d = 0; d < d_model; ++d) {
                        row[d] += alpha * coeff[d];
                    }
                }
            }
        }
    }

    float* delta_ptr = delta.ptr();
    for (size_t i = 0; i < seq_len; ++i) {
        const size_t left = (i == 0) ? i : i - 1;
        const size_t right = (i + 1 == seq_len) ? i : i + 1;

        const float* center = x_ptr + i * d_model;
        const float* left_row = x_ptr + left * d_model;
        const float* right_row = x_ptr + right * d_model;
        const float* prop_row = prop_ptr + i * d_model;
        float* out_row = delta_ptr + i * d_model;

        for (size_t d = 0; d < d_model; ++d) {
            const float lap = left_row[d] + right_row[d] - 2.0f * center[d];
            const float heat_delta = prop_row[d] - center[d];
            const float resonance = config_.resonance_gamma * safe_tanh(prop_row[d] * (1.0f + std::abs(center[d])));
            const float transport = config_.transport_eta * lap;
            out_row[d] = heat_delta + resonance + transport;
        }
    }

    apply_probe_array(x, propagated, delta);

    return delta;
}

std::vector<RenormalizedLevel> SpectralGeometricPropagator::renormalize(const s4m::Tensor& x) const {
    if (x.shape.size() != 2) {
        throw std::invalid_argument("renormalize expects a rank-2 tensor");
    }

    const size_t seq_len = x.shape[0];
    const size_t d_model = x.shape[1];
    std::vector<RenormalizedLevel> levels;
    if (seq_len == 0 || d_model == 0) return levels;

    size_t group_size = 2;
    for (size_t scale = 0; scale < config_.max_scales; ++scale) {
        const size_t active_group = clamp_group_size(group_size, seq_len);
        const size_t group_count = ceil_div(seq_len, active_group);

        RenormalizedLevel level;
        level.scale = scale;
        level.group_size = active_group;
        level.state = s4m::Tensor({group_count, d_model});
        level.survival.assign(group_count, 0.0f);

        for (size_t group = 0; group < group_count; ++group) {
            const size_t start = group * active_group;
            const size_t end = std::min(start + active_group, seq_len);
            const size_t len = end - start;
            float* dst = level.state.ptr() + group * d_model;

            float energy = 0.0f;
            for (size_t i = start; i < end; ++i) {
                const float* src = x.ptr() + i * d_model;
                for (size_t d = 0; d < d_model; ++d) {
                    dst[d] += src[d];
                    energy += src[d] * src[d];
                }
            }

            const float inv_len = 1.0f / static_cast<float>(std::max<size_t>(1, len));
            for (size_t d = 0; d < d_model; ++d) {
                dst[d] *= inv_len;
            }

            const float complexity = energy / static_cast<float>(std::max<size_t>(1, len * d_model));
            level.survival[group] = 1.0f / (1.0f + config_.complexity_lambda * complexity);
        }

        levels.push_back(std::move(level));
        if (active_group == seq_len) break;
        group_size *= 2;
    }

    return levels;
}

} // namespace rra::nn::aether
