#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace rra {

class Dataset {
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
    Dataset(const std::string& directory) {
        if (!std::filesystem::exists(directory)) throw std::runtime_error("Dataset path not found.");
        if (std::filesystem::is_directory(directory)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") load_file(entry.path().string());
            }
        } else {
            load_file(directory);
        }
    }

    ~Dataset() {
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
        files_.push_back(f); 
        total_size_ += f.size;
    }

    size_t size() const { return total_size_; }
    
    // Simplistic flat data accessor for demonstration
    const uint8_t* data(size_t index = 0) const { 
        if (files_.empty()) return nullptr;
        return files_[0].data_ptr; 
    }
};

} // namespace rra
