#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "nn/hybrid_engine.hpp"
#include "include/core_math.hpp"
#include "include/byte_field.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <format>
#include <numeric>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace s4m;

class MMapDataset {
public:
    MMapDataset(const std::string& path) {
#ifdef _WIN32
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open test file.");
        LARGE_INTEGER size; GetFileSizeEx(hFile_, &size); file_size_ = static_cast<size_t>(size.QuadPart);
        hMap_ = CreateFileMappingA(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
        data_ptr_ = static_cast<const uint8_t*>(MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0));
#else
        fd_ = open(path.c_str(), O_RDONLY);
        struct stat sb; fstat(fd_, &sb); file_size_ = sb.st_size;
        data_ptr_ = static_cast<const uint8_t*>(mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
#endif
    }
    ~MMapDataset() {
#ifdef _WIN32
        UnmapViewOfFile(data_ptr_); CloseHandle(hMap_); CloseHandle(hFile_);
#else
        munmap(const_cast<uint8_t*>(data_ptr_), file_size_); close(fd_);
#endif
    }
    const uint8_t* data() const { return data_ptr_; }
    size_t size() const { return file_size_; }
private:
    const uint8_t* data_ptr_ = nullptr;
    size_t file_size_ = 0;
#ifdef _WIN32
    HANDLE hFile_, hMap_;
#else
    int fd_;
#endif
};

int main(int argc, char** argv) {
    std::string path = "test.txt";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dataset" && i + 1 < argc) path = argv[++i];
    }

    const size_t NUM_BLOCKS = 4;
    const size_t NUM_CHUNKS = 64; 
    const size_t NUM_PLANES = 512; 

    HybridEngine engine(NUM_BLOCKS, NUM_CHUNKS);

    std::cout << "[INFO] Loaded Hybrid Engine for Inference Mode." << std::endl;

    MMapDataset dataset(path);
    size_t total_size = dataset.size();
    const uint8_t* data = dataset.data();

    std::cout << "[INFO] Evaluating " << path << " (" << total_size << " bytes)\n";
    std::cout << "---------------------------------------------------------\n";

    size_t cursor = 0;
    int batch_count = 0;

    // BPC (bits-per-character) accumulator
    double bpc_sum = 0.0;
    int64_t bpc_count = 0;
    float bpc_ema = 8.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (cursor + 64 < total_size) {
        // High-Throughput CAFE-NIS Ingest pass
        engine.ingest(data + cursor, 64);
        engine.forward();

        uint8_t target = data[cursor + 64];

        auto logits = engine.get_logits();
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
        for (int i = 0; i < 256; ++i) probs[i] /= sum_exp;
        
        float p = probs[target];
        if (p == 0.0f) p = 1e-8f;

        bpc_sum += static_cast<double>(-std::log2f(p));
        bpc_count++;
        
        bpc_ema = bpc_ema * 0.99f + static_cast<float>(-std::log2f(p)) * 0.01f;

        if (batch_count % 100 == 0) {
            float progress = (static_cast<float>(cursor) / total_size) * 100.0f;
            std::cout << std::format("[{:>3.0f}%] Cursor: {:>8} | BPC: {:.3f}\n",
                                    progress, cursor, bpc_ema);
        }

        cursor += 64;
        batch_count++;
    }

    auto now = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(now - start_time).count();

    std::cout << "---------------------------------------------------------\n";
    float final_bpc = (bpc_count > 0) ? static_cast<float>(bpc_sum / bpc_count) : 0.0f;
    std::cout << std::format("[RESULT] Final BPC: {:.4f} (over {} bytes)\n", final_bpc, bpc_count);
    std::cout << std::format("[RESULT] Throughput: {:.2f} bytes/sec\n", total_size / total_time);
    std::cout << "[INFO] Evaluation complete.\n";
    return 0;
}
