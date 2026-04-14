#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace rra::vocabulary {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer() = default;

    /**
     * @brief S4M Bypass: Vocabulary is inherently 0-255 bytes.
     * No JSON loading occurs.
     */
    bool load(const std::string& path);

    /**
     * @brief Converts a sequence of raw bytes back into text.
     */
    std::string decode(const std::vector<uint8_t>& byte_streams) const;

    /**
     * @brief Casts text physically into its underlying byte values.
     */
    std::vector<uint8_t> encode(const std::string& text) const;

    /**
     * @brief S4M Bit-to-Byte Stream limits vocab to 256 physical states.
     */
    int vocab_size() const { return 256; }

    /**
     * @brief Resolves byte ID back to string char.
     */
    std::optional<std::string> get_token_string(int id) const;
};

} // namespace rra::vocabulary
