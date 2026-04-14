#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstring>
#include "json.hpp"

namespace rra::utils {

struct TensorInfo {
    std::string dtype;
    std::vector<int> shape;
    std::size_t data_offsets[2];
};

class SafetensorsParser {
public:
    std::map<std::string, TensorInfo> tensors;
    std::string file_path;
    std::size_t data_start_offset = 0;

    bool load(const std::string& path) {
        file_path = path;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Safetensors] Failed to open " << path << "\n";
            return false;
        }

        uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), 8);
        if (!file) return false;

        if (header_size > 100 * 1024 * 1024) { // Sanity check 100MB max header
            std::cerr << "[Safetensors] Header size too large: " << header_size << "\n";
            return false;
        }

        std::string header_json(header_size, '\0');
        file.read(&header_json[0], header_size);
        if (!file) return false;

        data_start_offset = 8 + header_size;

        parse_header(header_json);
        return true;
    }

    std::vector<float> get_tensor_f32(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        const TensorInfo& info = it->second;
        
        std::size_t start = info.data_offsets[0];
        std::size_t end = info.data_offsets[1];
        std::size_t num_bytes = end - start;
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to reopen " + file_path);
        file.seekg(data_start_offset + start, std::ios::beg);
        
        if (info.dtype == "F32") {
            std::size_t num_floats = num_bytes / 4;
            std::vector<float> data(num_floats);
            file.read(reinterpret_cast<char*>(data.data()), num_bytes);
            return data;
        } else if (info.dtype == "BF16") {
            // BF16 is 2 bytes. To convert to F32, we just pad with 0s at the end.
            std::size_t num_elements = num_bytes / 2;
            std::vector<float> data(num_elements);
            std::vector<uint16_t> bf16_raw(num_elements);
            file.read(reinterpret_cast<char*>(bf16_raw.data()), num_bytes);
            
            for (std::size_t i = 0; i < num_elements; ++i) {
                uint32_t val32 = static_cast<uint32_t>(bf16_raw[i]) << 16;
                std::memcpy(&data[i], &val32, 4);
            }
            return data;
        } else if (info.dtype == "F16") {
            // F16 conversion is more complex, but we can do a basic one
            std::size_t num_elements = num_bytes / 2;
            std::vector<uint16_t> f16_raw(num_elements);
            file.read(reinterpret_cast<char*>(f16_raw.data()), num_bytes);
            
            std::vector<float> data(num_elements);
            for (std::size_t i = 0; i < num_elements; ++i) {
                uint16_t h = f16_raw[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h & 0x7C00) >> 10;
                uint32_t mant = (h & 0x03FF);
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) f = sign;
                    else {
                        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
                        exp++; mant &= ~0x0400;
                        f = sign | ((exp + 112) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f = sign | 0x7F800000 | (mant << 13);
                } else {
                    f = sign | ((exp + 112) << 23) | (mant << 13);
                }
                std::memcpy(&data[i], &f, 4);
            }
            return data;
        }
        
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

private:
    void parse_header(const std::string& json_str) {
        try {
            auto root = nlohmann::json::parse(json_str);
            for (auto& [key, value] : root.items()) {
                if (key == "__metadata__") continue;

                TensorInfo info;
                if (value.contains("dtype")) {
                    info.dtype = value["dtype"].get<std::string>();
                }
                if (value.contains("shape") && value["shape"].is_array()) {
                    for (auto& s : value["shape"]) {
                        info.shape.push_back(s.get<int>());
                    }
                }
                if (value.contains("data_offsets") && value["data_offsets"].is_array() && value["data_offsets"].size() >= 2) {
                    info.data_offsets[0] = value["data_offsets"][0].get<std::size_t>();
                    info.data_offsets[1] = value["data_offsets"][1].get<std::size_t>();
                }
                tensors[key] = info;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Safetensors] JSON parse error: " << e.what() << "\n";
        }
    }
};

} // namespace rra::utils
