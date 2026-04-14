#include "nis_engine.hpp"
#include "core_math.hpp"
#include "byte_field.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <vector>
#include <deque>
#include <windows.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <set>

using namespace rra::nis_engine;
using namespace rra::core_math;

// ---------------------------------------------------------------------------
// Zero-Copy Kernel-Bypass Loader (Windows API)
// ---------------------------------------------------------------------------
class ZeroCopyDataset {
private:
    const uint8_t* data_ptr_ = nullptr;
    size_t file_size_ = 0;
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMap_  = NULL;

public:
    ZeroCopyDataset(const std::string& filepath) {
        hFile_ = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open dataset file.");

        LARGE_INTEGER size;
        GetFileSizeEx(hFile_, &size);
        file_size_ = static_cast<size_t>(size.QuadPart);

        hMap_ = CreateFileMappingA(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap_) { CloseHandle(hFile_); throw std::runtime_error("Failed to create file mapping."); }

        data_ptr_ = static_cast<const uint8_t*>(MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0));
        if (!data_ptr_) { CloseHandle(hMap_); CloseHandle(hFile_); throw std::runtime_error("Failed to map view of dataset."); }
        
        std::cout << "[INFO] Dataset Kernel-Bypass Active: " << filepath << " (" << file_size_ << " bytes)\n";
    }

    ~ZeroCopyDataset() {
        if (data_ptr_) UnmapViewOfFile(data_ptr_);
        if (hMap_) CloseHandle(hMap_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
    }

    const uint8_t* data() const { return data_ptr_; }
    size_t size() const { return file_size_; }
};

// ---------------------------------------------------------------------------
// Trainer config & scoring
// ---------------------------------------------------------------------------
struct TrainerV2Config {
    int64_t elite_count = 1;
    double tps_guardrail = 1.0;
    int64_t low_tps_patience = 100;
    double capability_floor = 0.05;
    double tps_ema_decay = 0.1;
    double teacher_ema_decay = 0.1;
    double uncertainty_ema_decay = 0.1;
    int64_t checkpoint_interval = 100;
    void validate() { elite_count = (std::max)(static_cast<int64_t>(1), elite_count); }
};

struct ScoredCandidate { int64_t index = -1; double score = 0.0; };

struct StepOut {
    int64_t generation = 0;
    int64_t candidate_index = 0;
    double best_obj = 0.0;
    bool checkpoint_due = false;
    bool learning_confirmed = false;
    double score = 0.0;
    double tps_ema = 0.0;
    double loss_mse_ema = 0.0;
    double swarm_entropy_ema = 0.0;
};

class PARL_Trainer {
public:
    explicit PARL_Trainer(const TrainerV2Config& cfg, uint64_t seed = 1337ULL) : cfg_(cfg) { cfg_.validate(); reset(seed); }
    void reset(uint64_t seed = 1337ULL) { generation_ = 0; best_obj_ = -std::numeric_limits<double>::infinity(); low_tps_streak_ = 0; learning_confirmed_ = false; tps_ema_ = 0.0; swarm_entropy_ema_ = 0.0; teacher_mse_ema_ = 100.0; has_ema_ = false; elites_.clear(); (void)seed; }
    static double score_parl_agent(double alignment, double diversity, double tps, double entropy) { double structural_bonus = (entropy < 0.20) ? 5.0 : 0.0; return (alignment * 3.5) + (diversity * 2.0) + (tps * 0.5) + structural_bonus; }
    StepOut step(int64_t candidate_index, double tps, double teacher_mse_loss, double agent_diversity, bool force_checkpoint = false) {
        tps = (std::max)(0.0, std::isfinite(tps) ? tps : 0.0);
        teacher_mse_loss = (std::max)(0.0, std::isfinite(teacher_mse_loss) ? teacher_mse_loss : 1.0);
        agent_diversity = clamp(agent_diversity, 0.0, 1.0);
        ++generation_;
        double teacher_alignment = 1.0 / (1.0 + teacher_mse_loss);
        if (teacher_alignment > best_obj_) best_obj_ = teacher_alignment;
        tps_ema_ = ema(tps_ema_, tps, cfg_.tps_ema_decay, has_ema_);
        swarm_entropy_ema_ = ema(swarm_entropy_ema_, agent_diversity, 0.10, has_ema_);
        teacher_mse_ema_ = ema(teacher_mse_ema_, teacher_mse_loss, cfg_.teacher_ema_decay, has_ema_);
        has_ema_ = true;
        if (tps < cfg_.tps_guardrail) ++low_tps_streak_; else low_tps_streak_ = 0;
        learning_confirmed_ = (teacher_mse_ema_ <= 0.25) && (swarm_entropy_ema_ > 0.35);
        const double score = score_parl_agent(teacher_alignment, agent_diversity, tps, teacher_mse_ema_);
        update_elites(candidate_index, score);
        const bool checkpoint_due = force_checkpoint || ((cfg_.checkpoint_interval > 0) && ((generation_ % cfg_.checkpoint_interval) == 0));
        StepOut out; out.generation = generation_; out.candidate_index = candidate_index; out.best_obj = best_obj_; out.checkpoint_due = checkpoint_due; out.learning_confirmed = learning_confirmed_; out.score = score; out.tps_ema = tps_ema_; out.loss_mse_ema = teacher_mse_ema_; out.swarm_entropy_ema = swarm_entropy_ema_; return out;
    }
    const std::vector<ScoredCandidate>& elite() const { return elites_; }
private:
    static bool ranked_before(const ScoredCandidate& a, const ScoredCandidate& b) { return (a.score == b.score) ? a.index < b.index : a.score > b.score; }
    void update_elites(int64_t idx, double score) {
        score = std::isfinite(score) ? score : -std::numeric_limits<double>::infinity();
        const std::size_t keep = static_cast<std::size_t>((std::max)(static_cast<int64_t>(1), cfg_.elite_count));
        const ScoredCandidate cand{idx, score};
        auto pos = std::lower_bound(elites_.begin(), elites_.end(), cand, [](const ScoredCandidate& lhs, const ScoredCandidate& rhs) { return ranked_before(lhs, rhs); });
        if (elites_.size() < keep) { elites_.insert(pos, cand); return; }
        if (pos == elites_.end()) return;
        elites_.insert(pos, cand); elites_.pop_back();
    }
    TrainerV2Config cfg_; int64_t generation_ = 0; double best_obj_ = -std::numeric_limits<double>::infinity(); int64_t low_tps_streak_ = 0; bool learning_confirmed_ = false; bool has_ema_ = false; double tps_ema_ = 0.0; double swarm_entropy_ema_ = 0.0; double teacher_mse_ema_ = 0.0; std::vector<ScoredCandidate> elites_; std::mt19937 rng_{1337};
public:
    uint32_t rng() { return rng_(); }
};

#include "byte_field.hpp"
#include "manifold_ipc.hpp"
#include <queue>
#include <numeric>

struct ReplaySample {
    std::vector<uint8_t> context;
    uint8_t target_byte;
    uint64_t target_morton; 
    float ce_loss;          
    bool operator<(const ReplaySample& other) const { return ce_loss < other.ce_loss; }
};

namespace {
std::atomic<bool> g_shutdown_requested{false};

// The Two-Strike rule for Hard Terminate
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType) {
    static int strikes = 0;
    if (fdwCtrlType == CTRL_C_EVENT || fdwCtrlType == CTRL_CLOSE_EVENT) {
        strikes++;
        if (strikes > 1) {
            std::cerr << "\n[TERMINATE] Forced exit by user.\n";
            _exit(0); 
        }
        std::cout << "\n[SHUTDOWN] Intercepted CTRL+C. Initiating graceful shutdown... (Strike 1)\n";
        g_shutdown_requested = true;
        return TRUE;
    }
    return FALSE;
}

void print_usage() { std::cout << "[S4M] Usage: launcher [--dataset <path>]\n"; }

enum class ParseResult { Ok, Help, Error };

class VirtualEnvironment {
public:
    VirtualEnvironment() = default;
    
    std::vector<std::pair<size_t, float>> step(const EngineOutput& out, size_t sensorium_size) {
        if (out.winning_bucket == 0) pos_x += 0.05f;
        else if (out.winning_bucket == 1) pos_x -= 0.05f;
        else if (out.winning_bucket == 2) pos_y += 0.05f;
        else if (out.winning_bucket == 3) pos_y -= 0.05f;
        
        pos_x = std::clamp(pos_x, 0.0f, 1.0f);
        pos_y = std::clamp(pos_y, 0.0f, 1.0f);
        
        std::vector<std::pair<size_t, float>> senses;
        for (size_t i = 0; i < sensorium_size; ++i) {
            float noise = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;
            senses.push_back({i, ((i % 2 == 0) ? pos_x : pos_y) * 2.0f + noise});
        }
        return senses;
    }
private:
    float pos_x = 0.5f;
    float pos_y = 0.5f;
};

int run_launcher(const std::string& dataset_path) {
    std::cout << "[S4M] Autonomous Golden Loop Pre-Training (GNF)\n";
    
    ZeroCopyDataset dataset(dataset_path);
    EngineOutput last_output;

    // Initialize the Geometric Neural Field embedding for byte-to-Morton mapping
    rra::gnf::ByteField gnf_field;
    rra::gnf::seed_grid(gnf_field);

    NISEngine sde_engine;
    TrainerV2Config cfg; cfg.elite_count = 5; PARL_Trainer trainer(cfg);
    
    // Optimized for 8GB RAM
    const int TOTAL_INITIAL_NEURONS = 512; 
    const int SENSORIUM_SIZE = 256; 
    const int OUTPUT_ANCHORS = 8;

    if (sde_engine.load_checkpoint("gnf_engine.bin")) {
        std::cout << "[INFO] Loaded existing NISEngine checkpoint." << std::endl;
    } else {
        std::cout << "[INFO] Seeding Primordial Soup (Small-World Graph)..." << std::endl;
        for (int i = 0; i < TOTAL_INITIAL_NEURONS; ++i) {
            if (g_shutdown_requested) break;
            
            // 75% Excitatory, 25% Inhibitory balance
            Node::NodeType type = (trainer.rng() % 100 < 25) ? Node::NodeType::Inhibitory : Node::NodeType::Excitatory;
            FunctionalRegion region = (i < SENSORIUM_SIZE) ? FunctionalRegion::Input : FunctionalRegion::Integrator;
            
            sde_engine.add_node(static_cast<uint64_t>(i), type, region);
            
            if (i % 128 == 0) std::cout << "[DEBUG] Seeding node " << i << "..." << std::endl;
        }
        
        for (int i = 0; i < TOTAL_INITIAL_NEURONS && !g_shutdown_requested; ++i) {
            // 1. Local Ring
            for (int neighbor = 1; neighbor <= 8; ++neighbor) {
                sde_engine.force_wire(i, (i + neighbor) % TOTAL_INITIAL_NEURONS, 0.5f);
                sde_engine.force_wire(i, (i - neighbor + TOTAL_INITIAL_NEURONS) % TOTAL_INITIAL_NEURONS, 0.5f);
            }
            // 2. Random Wormholes
            for (int w = 0; w < 4; ++w) {
                sde_engine.force_wire(i, trainer.rng() % TOTAL_INITIAL_NEURONS, 0.2f);
            }
            if (i % 128 == 0) std::cout << "[DEBUG] Wiring soup... " << (i * 100 / TOTAL_INITIAL_NEURONS) << "%" << std::endl;
        }
    }

    std::vector<size_t> output_anchors(OUTPUT_ANCHORS);
    size_t n_total = sde_engine.get_nodes().size();
    for (int i = 0; i < OUTPUT_ANCHORS; ++i) {
        output_anchors[i] = (n_total - OUTPUT_ANCHORS) + i;
    }

    SetConsoleCtrlHandler(CtrlHandler, TRUE);
    const auto& e_cfg = sde_engine.get_config();
    
    size_t cursor = 0;
    uint64_t stream_bytes_processed = 0; 
    uint64_t last_topology_compile = 0;
    double raw_loss_ema = 0.0; 
    bool ema_init = false;
    double period_raw_loss = 0.0; int period_count = 0;
    auto interval_start = std::chrono::steady_clock::now();
    float previous_prob_1[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    // Temporal delay-line: holds the credit buffer from the previous tick.
    // Injected as reward at the NEXT tick so neurons are rewarded for predicting
    // the incoming byte rather than simply matching the current one.
    std::vector<float> prev_credit_buf;

    std::cout << "[INFO] Autonomous pre-training loop active." << std::endl;

    int heartbeat = 0;
    while (!g_shutdown_requested) {
        // 1. Sensory Injection (Closed Loop via ZeroCopyDataset)
        std::vector<std::pair<size_t, float>> bulk_sigs;
        for (size_t i = 0; i < SENSORIUM_SIZE; ++i) {
            if (cursor >= dataset.size()) cursor = 0; // Loop dataset
            float val = static_cast<float>(dataset.data()[cursor]) / 255.0f;
            bulk_sigs.push_back({i, val});
            cursor++;
        }
        
        sde_engine.inject_signals(bulk_sigs);
        stream_bytes_processed += SENSORIUM_SIZE;

        std::cout << "[CYCLE " << ++heartbeat << "] Closed Loop Tick | Streamed: " << stream_bytes_processed << " bytes" << std::endl;

        for (int i = 0; i < 8; ++i) sde_engine.wake_node(output_anchors[i]);
        
        int k_ticks = 0;
        while (k_ticks < e_cfg.physics_max_k_ticks && !g_shutdown_requested) {
            sde_engine.execute_tick(TickMode::Standard); 
            sde_engine.execute_tick(TickMode::Cognitive); 
            k_ticks++;
            const auto& cur_state = sde_engine.get_state(); 
            float max_m = 0.0f;
            for (auto a_idx : output_anchors) max_m = (std::max)(max_m, std::abs(cur_state.membrane[a_idx]));
            // Break early if an output neuron is close to spiking, bypassing the impossible math
            if (max_m > 0.85f * e_cfg.default_threshold) break; 
        }
        
        last_output = sde_engine.read_output();

        // 2. Emergency Spark: Removed. Relying on Homeostatic Plasticity.

        // Dopamine decay is now inlined into consolidate_learning (NREM pass).

        // --- Temporal Delay-Line Morton Predictive Coding ---
        // THEORY: We reward neurons for predicting the NEXT byte, not matching the
        // current one. This is true Predictive Coding / Sequence Prediction.
        //
        // IMPLEMENTATION: O(N) per tick (no O(V*N²) enumeration of possible tokens).
        // - credit_buf[t] = GNF proximity to B_t  (computed now)
        // - reward injected = credit_buf[t-1]      (delayed one tick)
        //
        // When B_{t+1} arrives at tick t+1, neurons that were geometrically close to
        // B_{t+1}'s Morton key had PREDICTED this byte. They get rewarded.
        // This mirrors next-token prediction exactly, using Morton Hamming as the
        // free energy / surprise metric.
        {
            const auto& nodes = sde_engine.get_nodes();
            const auto& state = sde_engine.get_state();
            size_t n_neurons = nodes.size();

            // Build current-tick credit from the just-arrived batch (B_t)
            std::vector<float> credit_now(n_neurons, 0.0f);
            std::vector<uint64_t> neuron_mortons(n_neurons);
            for (size_t n = 0; n < n_neurons; ++n)
                neuron_mortons[n] = nodes[n].spatial_id;

            size_t batch_start = cursor >= SENSORIUM_SIZE ? cursor - SENSORIUM_SIZE : 0;
            for (size_t b = batch_start; b < cursor; ++b) {
                uint8_t byte_val = dataset.data()[b % dataset.size()];
                uint64_t b_morton = gnf_field.morton_keys[byte_val];
                rra::gnf::gnf_activate(
                    b_morton, neuron_mortons.data(), state.membrane.data(),
                    static_cast<int>(n_neurons),
                    rra::gnf::DEFAULT_TAU, rra::gnf::DEFAULT_RADIUS,
                    credit_now.data()
                );
            }

            // Inject PREVIOUS tick's credit: neurons are rewarded for having
            // predicted the bytes that just arrived (true next-token prediction)
            if (!prev_credit_buf.empty() && prev_credit_buf.size() == n_neurons) {
                float max_c = *std::max_element(prev_credit_buf.begin(), prev_credit_buf.end());
                if (max_c > 1e-6f) {
                    for (size_t n = 0; n < n_neurons; ++n)
                        sde_engine.add_reward(n, prev_credit_buf[n] / max_c);
                }
            }

            // Roll forward: current credit becomes the prediction target for next tick
            prev_credit_buf = std::move(credit_now);
        }

        sde_engine.consolidate_learning(e_cfg.train_lr);
        
        if (stream_bytes_processed - last_topology_compile >= 50000) {
            std::cout << "[INFO] Sleep Cycle: Compiling Topology..." << std::endl;
            // organic_growth() first: new integrator nodes are appended, then
            // compile_topology() sorts them while pinning the true last-8 output neurons.
            sde_engine.organic_growth();
            sde_engine.compile_topology();
            n_total = sde_engine.get_nodes().size();
            for (int i = 0; i < OUTPUT_ANCHORS; ++i)
                output_anchors[i] = (n_total - OUTPUT_ANCHORS) + i;
            // Re-wake output anchors after compile: compile clears the active-node queue,
            // leaving output neurons silent for several ticks and creating a false reward dip.
            for (int i = 0; i < OUTPUT_ANCHORS; ++i)
                sde_engine.wake_node(output_anchors[i]);
            // Re-inject last sensory frame so the path from input to output is live.
            sde_engine.inject_signals(bulk_sigs);
            sde_engine.save_state("gnf_engine.bin");
            last_topology_compile = stream_bytes_processed;
        }

        if (heartbeat % 10 == 0) {
            WeightStats wstats = sde_engine.get_weight_stats();
            // Sample dopamine as mean over output anchors — these indices are always stable
            // (pinned by compile_topology), unlike the old hardcoded index 256 which was
            // in the sortable region and got reshuffled on every compile.
            const auto& cs = sde_engine.get_state();
            float current_reward = 0.0f;
            for (auto a : output_anchors) current_reward += cs.dopamine[a];
            current_reward /= static_cast<float>(OUTPUT_ANCHORS);
            if (!ema_init) { raw_loss_ema = current_reward; ema_init = true; } 
            else { raw_loss_ema = raw_loss_ema * 0.95 + current_reward * 0.05; }
            auto interval_end = std::chrono::steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(interval_end - interval_start).count();
            std::cout << "[METRICS] Endogenous Reward Proxy: " << current_reward << " | W: " << wstats.count << " | Spikes: " << sde_engine.get_last_cognitive_spikes() << " | " << ms << "ms" << std::endl;
            sde_engine.clear_stats(); interval_start = std::chrono::steady_clock::now();
        }
    }
    if (g_shutdown_requested) std::cout << "[INFO] Saving state and exiting...\n";
    sde_engine.save_state("gnf_engine.bin"); 
    return 0;
}

ParseResult parse_args(int argc, char** argv, std::string& path) { for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--dataset" && i + 1 < argc) path = argv[++i]; } return ParseResult::Ok; }
} // namespace

int main(int argc, char** argv) {
    std::string path = "dataset.txt"; if (parse_args(argc, argv, path) != ParseResult::Ok) return 1;
    try { return run_launcher(path); } catch (const std::exception& e) { std::cerr << "[ERROR] " << e.what() << "\n"; return 1; }
}
