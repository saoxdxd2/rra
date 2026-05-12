#include "nn/hybrid_engine.hpp"
#include "include/core_math.hpp"
#include "include/dataset.hpp"
#include "include/manifold_ipc.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <format>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <filesystem>
#include <random>

using namespace s4m;
using namespace s4m::core;
namespace fs = std::filesystem;

struct SimpleStopSource {
    std::atomic<bool> stop_requested{false};
    void request_stop() { stop_requested = true; }
    bool is_stop_requested() const { return stop_requested.load(); }
};

SimpleStopSource g_stop_source;

#ifdef _WIN32
BOOL WINAPI CtrlHandler(DWORD fdw) {
    if (fdw == CTRL_C_EVENT) { g_stop_source.request_stop(); return TRUE; }
    return FALSE;
}
#endif

void run_launcher(rra::Dataset& dataset) {
    try {
        // Dual-Vector CAFE-NIS Engine
        const size_t NUM_BLOCKS = 4;
        const size_t NUM_CHUNKS = 64; // 64 chunks * 32 vec * 16 floats = context capacity
        const size_t NUM_PLANES = 512; // 512-bit attractor memory
        
        HybridEngine engine(NUM_BLOCKS, NUM_CHUNKS);

        float bpc_ema = 8.0f, accuracy_ema = 0.0f;
        uint64_t processed = 0;

        std::cout << "[SYNC] Dual-Vector Engine Online. Initiating Hybrid Sweep.\n";
        
        // Setup IPC
        HANDLE hIpcMap = nullptr;
        rra::gnf::ipc::ManifoldIPCData* ipc_data = nullptr;
#ifdef _WIN32
        hIpcMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(rra::gnf::ipc::ManifoldIPCData), rra::gnf::ipc::IPC_MAP_NAME);
        if (hIpcMap) {
            ipc_data = static_cast<rra::gnf::ipc::ManifoldIPCData*>(MapViewOfFile(hIpcMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(rra::gnf::ipc::ManifoldIPCData)));
            if (ipc_data) std::cout << "[IPC] Manifold UI link established.\n";
        }
#endif

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> energy_gradients(NUM_CHUNKS, 0.0f);

        while (!g_stop_source.is_stop_requested()) {
            const uint8_t* ctx_ptr = nullptr;
            uint8_t target = 0;
            dataset.fetch_batch(ctx_ptr, 64, target);

            // 1. Unified Ingestion (CAFE-NIS)
            engine.ingest(ctx_ptr, 64);

            // 2. Forward Pass (Continuous -> Discrete -> Spectral AETHER)
            engine.forward();

            // 3. Unified Prediction (CrossEntropy)
            auto logits = engine.get_logits();
            
            // Softmax
            float max_logit = -1e9f;
            for (int i = 0; i < 256; ++i) {
                if (logits[i] > max_logit) max_logit = logits[i];
            }
            
            float sum_exp = 0.0f;
            std::array<float, 256> probs;
            for (int i = 0; i < 256; ++i) {
                probs[i] = std::exp(logits[i] - max_logit);
                sum_exp += probs[i];
            }
            
            uint8_t predicted_byte = 0;
            float max_prob = -1.0f;
            for (int i = 0; i < 256; ++i) {
                probs[i] /= sum_exp;
                if (probs[i] > max_prob) {
                    max_prob = probs[i];
                    predicted_byte = static_cast<uint8_t>(i);
                }
            }
            
            float target_prob = probs[target];
            if (target_prob == 0.0f) target_prob = 1e-8f;
            float loss = -std::log(target_prob);
            
            // 4. Backward Pass / Energy Shaping
            // We use the cross entropy loss magnitude as the global error signal for the last chunk
            energy_gradients.back() = loss;
            engine.backward(energy_gradients);

            bpc_ema = bpc_ema * 0.995f + (-std::log2f(target_prob)) * 0.005f;
            accuracy_ema = accuracy_ema * 0.99f + ((predicted_byte == target) ? 0.01f : 0.0f);

            if (processed % 4096 == 0) { 
                auto now = std::chrono::high_resolution_clock::now();
                float fps = 4096.0f / std::chrono::duration<float>(now - start_time).count();
                start_time = now;
                std::printf("\r[TMUL] Loss: %.3f | BPC: %.3f | Acc: %5.2f%% | MemBank: %zu | Speed: %5.0f iter/s ",
                            loss, bpc_ema, accuracy_ema * 100.0f, engine.global_memory.get_memory_size(), fps);
                std::fflush(stdout);
                
                if (ipc_data) {
                    ipc_data->current_ce_loss.store(loss, std::memory_order_relaxed);
                    ipc_data->frame_counter.fetch_add(1, std::memory_order_relaxed);
                    for(int i=0; i<256; ++i) {
                        ipc_data->byte_coords[i].x = probs[i];
                        ipc_data->byte_coords[i].y = static_cast<float>(i) / 255.0f;
                    }
                }
            }
            processed += 64;
        }
        
#ifdef _WIN32
        if (ipc_data) UnmapViewOfFile(ipc_data);
        if (hIpcMap) CloseHandle(hIpcMap);
#endif
    } catch (const std::exception& e) { std::cerr << "[ERROR] " << e.what() << "\n"; }
}

int main(int argc, char** argv) {
    std::string path = "dataset.txt";
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--dataset" && i+1 < argc) path = argv[++i];
#ifdef _WIN32
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
#endif
    try {
        rra::Dataset ds(path);
        std::thread t([&ds]() { run_launcher(ds); });
        t.join(); return 0;
    } catch (const std::exception& e) { std::cerr << "[FATAL] " << e.what() << "\n"; return 1; }
}
