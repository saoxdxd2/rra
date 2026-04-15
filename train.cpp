#include "nn/hybrid_engine.hpp"
#include "include/core_math.hpp"
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

// ---------------------------------------------------------------------------
// High-Throughput Memory Mapped Dataset
// ---------------------------------------------------------------------------
class ThreadedMultiFileDataset {
private:
    struct FileEntry {
        std::string path;
        size_t size;
        const uint8_t* data_ptr = nullptr;
#ifdef _WIN32
        HANDLE hFile = INVALID_HANDLE_VALUE;
        HANDLE hMap  = NULL;
#else
        int fd = -1;
#endif
    };
    std::vector<FileEntry> files_;
    size_t total_size_ = 0;
public:
    ThreadedMultiFileDataset(const std::string& directory) {
        if (!fs::exists(directory)) throw std::runtime_error("Dataset path not found.");
        if (fs::is_directory(directory)) {
            for (const auto& entry : fs::recursive_directory_iterator(directory))
                if (entry.is_regular_file() && entry.path().extension() == ".txt") load_file(entry.path().string());
        } else load_file(directory);
        if (files_.empty()) throw std::runtime_error("No files found.");
    }
    ~ThreadedMultiFileDataset() {
        for (auto& f : files_) {
#ifdef _WIN32
            if (f.data_ptr) UnmapViewOfFile(f.data_ptr);
            if (f.hMap) CloseHandle(f.hMap);
            if (f.hFile != INVALID_HANDLE_VALUE) CloseHandle(f.hFile);
#else
            if (f.data_ptr && f.data_ptr != MAP_FAILED) munmap(const_cast<uint8_t*>(f.data_ptr), f.size);
            if (f.fd != -1) close(f.fd);
#endif
        }
    }
    void load_file(const std::string& filepath) {
        FileEntry f; f.path = filepath;
#ifdef _WIN32
        f.hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (f.hFile == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER sz; GetFileSizeEx(f.hFile, &sz); f.size = static_cast<size_t>(sz.QuadPart);
        if (f.size == 0) { CloseHandle(f.hFile); return; }
        f.hMap = CreateFileMappingA(f.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        f.data_ptr = static_cast<const uint8_t*>(MapViewOfFile(f.hMap, FILE_MAP_READ, 0, 0, 0));
#else
        f.fd = open(filepath.c_str(), O_RDONLY); struct stat sb; fstat(f.fd, &sb); f.size = static_cast<size_t>(sb.st_size);
        f.data_ptr = static_cast<const uint8_t*>(mmap(NULL, f.size, PROT_READ, MAP_PRIVATE, f.fd, 0));
#endif
        files_.push_back(f); total_size_ += f.size;
    }
    size_t current_file_idx = 0, current_pos = 0;

    bool fetch_batch(const uint8_t*& out_ptr, size_t batch_size, uint8_t& out_target) {
        const auto* f = &files_[current_file_idx];
        if (current_pos + batch_size >= f->size) {
            current_file_idx = (current_file_idx + 1) % files_.size();
            current_pos = 0;
            f = &files_[current_file_idx];
            out_ptr = f->data_ptr;
            out_target = f->data_ptr[batch_size];
            current_pos = 64; 
            return true;
        }
        out_ptr = f->data_ptr + current_pos;
        out_target = f->data_ptr[current_pos + batch_size];
        current_pos += 64; 
        return false;
    }
    size_t total_size() const { return total_size_; }
};

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

void run_launcher(ThreadedMultiFileDataset& dataset) {
    try {
        // Dual-Vector CAFE-NIS Engine
        const size_t NUM_BLOCKS = 4;
        const size_t NUM_CHUNKS = 16; // 16 chunks * 32 vec * 16 floats = context capacity
        const size_t NUM_PLANES = 512; // 512-bit attractor memory
        
        HybridEngine engine(NUM_BLOCKS, NUM_CHUNKS, NUM_PLANES);

        float bpc_ema = 8.0f, accuracy_ema = 0.0f;
        uint64_t processed = 0;

        std::cout << "[SYNC] Dual-Vector Engine Online. Initiating Hybrid Sweep.\n";
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

            // 3. Unified Prediction
            uint8_t predicted_byte = engine.predict();
            float p_act = static_cast<float>(predicted_byte) / 255.0f;
            if (p_act == 0.0f) p_act = 1e-8f;
            
            // 4. Backward Pass / Optimal Transport Error Feedback
            float target_val = static_cast<float>(target) / 255.0f;
            float error = std::abs(p_act - target_val);
            energy_gradients.back() = error; // Inject error into the last chunk
            engine.backward(energy_gradients);

            bpc_ema = bpc_ema * 0.995f + (-std::log2f(p_act)) * 0.005f;
            accuracy_ema = accuracy_ema * 0.99f + ((predicted_byte == target) ? 0.01f : 0.0f);

            if (processed % 4096 == 0) { 
                auto now = std::chrono::high_resolution_clock::now();
                float fps = 4096.0f / std::chrono::duration<float>(now - start_time).count();
                start_time = now;
                std::printf("\r[TMUL] BPC: %.3f | Acc: %5.2f%% | Speed: %5.0f iter/s ",
                            bpc_ema, accuracy_ema * 100.0f, fps);
                std::fflush(stdout);
            }
            processed += 64;
        }
    } catch (const std::exception& e) { std::cerr << "[ERROR] " << e.what() << "\n"; }
}

int main(int argc, char** argv) {
    std::string path = "dataset.txt";
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--dataset" && i+1 < argc) path = argv[++i];
#ifdef _WIN32
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
#endif
    try {
        ThreadedMultiFileDataset ds(path);
        std::thread t([&ds]() { run_launcher(ds); });
        t.join(); return 0;
    } catch (const std::exception& e) { std::cerr << "[FATAL] " << e.what() << "\n"; return 1; }
}
