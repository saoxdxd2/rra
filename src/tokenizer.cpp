#include "tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <optional>

namespace rra::vocabulary {

Tokenizer::Tokenizer() = default;

bool Tokenizer::load(const std::string& path) {
    // S4M Modification: We bypass BPE totally. 
    // Return true natively to spoof the legacy orchestrator.
    std::cout << "[INFO] S4M Engine: BPE bypass active. Streaming raw Native Bytes.\n";
    return true;
}

std::string Tokenizer::decode(const std::vector<uint8_t>& byte_streams) const {
    // Optimization: Pre-allocate string memory to avoid multiple reallocations
    std::string result;
    result.reserve(byte_streams.size());

    for (uint8_t byte_val : byte_streams) {
        // Clamp to 0-255 and push directly to string
        result.push_back(static_cast<char>(byte_val));
    }
    return result;
}

std::vector<uint8_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint8_t> stream;
    stream.reserve(text.size());

    // Using unsigned char in the loop prevents sign-extension issues
    // while maintaining zero-overhead mapping to physical byte state
    for (unsigned char c : text) {
        stream.push_back(static_cast<uint8_t>(c));
    }
    return stream;
}

std::optional<std::string> Tokenizer::get_token_string(int id) const {
    // Direct return of a single-character string if within byte range
    if (id >= 0 && id <= 255) {
        return std::string(1, static_cast<char>(static_cast<unsigned char>(id)));
    }
    return std::nullopt;
}

} // namespace rra::vocabulary