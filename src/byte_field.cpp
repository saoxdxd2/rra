#include "byte_field.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace s4m::gnf {

BitVector512 titan_encode_512(float x, float y, float z, float w) {
    alignas(64) float coords[16];
    coords[0] = 1.0f - (x+y+z+w)*0.25f;
    coords[1]=x; coords[2]=y; coords[3]=z; coords[4]=w;
    coords[5]=x*y; coords[6]=x*z; coords[7]=x*w; coords[8]=y*z; coords[9]=y*w; coords[10]=z*w;
    coords[11]=x*y*z; coords[12]=x*y*w; coords[13]=x*z*w; coords[14]=y*z*w;
    coords[15]=x*y*z*w;

    __m512 v_c = _mm512_load_ps(coords);
    __m512 v_val = _mm512_min_ps(_mm512_max_ps(v_c, _mm512_setzero_ps()), _mm512_set1_ps(1.0f));
    __m512i v_count = _mm512_cvtps_epi32(_mm512_mul_ps(v_val, _mm512_set1_ps(32.0f)));

    // Generate thermometer masks: (1 << count) - 1
    // Branchless: mask = (0xFFFFFFFF >> (32 - count)) & -(count > 0)
    __m512i v_all = _mm512_set1_epi32(0xFFFFFFFF);
    __m512i v_shift = _mm512_sub_epi32(_mm512_set1_epi32(32), v_count);
    __m512i v_mask = _mm512_srlv_epi32(v_all, v_shift);

    __mmask16 m_zero = _mm512_cmpgt_epi32_mask(v_count, _mm512_setzero_si512());
    v_mask = _mm512_maskz_mov_epi32(m_zero, v_mask);

    BitVector512 mv; mv.v512 = v_mask;
    return mv;
}

void GaussianBinder::bind(const ByteField& field, uint8_t byte, float surprise) {
    const Coord4D& target = field.coordinates[byte];
    float alpha = 1.0f / static_cast<float>(1ULL << corusion_shift_);
    alpha = std::clamp(alpha * (1.0f + surprise), 0.001f, 0.5f);
    current_.x = (1.0f - alpha) * current_.x + alpha * target.x;
    current_.y = (1.0f - alpha) * current_.y + alpha * target.y;
    current_.z = (1.0f - alpha) * current_.z + alpha * target.z;
    current_.w = (1.0f - alpha) * current_.w + alpha * target.w;
}

BitVector512 GaussianBinder::snap_to_titan() const {
    return titan_encode_512(current_.x, current_.y, current_.z, current_.w);
}

void GaussianBinder::reset() { current_ = {0.5f, 0.5f, 0.5f, 0.5f}; }

void seed_grid(ByteField& field) {
    for (int i = 0; i < 256; ++i) {
        float fx = static_cast<float>(i % 4) / 3.0f;
        float fy = static_cast<float>((i / 4) % 4) / 3.0f;
        float fz = static_cast<float>((i / 16) % 4) / 3.0f;
        float fw = static_cast<float>((i / 64) % 4) / 3.0f;
        field.coordinates[i] = {fx, fy, fz, fw};
        field.morton_keys[i] = titan_encode_512(fx, fy, fz, fw);
    }
}

uint8_t value_centric_decode(const uint8_t* ss, const BitVector512* ns, const BitVector512* bk, int n, float* conf) {
    if (n == 0) return 0;

    alignas(64) int16_t counts[512] = {0};
    int spikers = 0;

    for (int i = 0; i < n; ++i) {
        if (ss[i]) {
            spikers++;
            __m512i v_seed = ns[i].v512;
            for (int w = 0; w < 8; ++w) {
                uint64_t val = reinterpret_cast<const uint64_t*>(&v_seed)[w];
                for (int b = 0; b < 64; ++b) {
                    counts[w * 64 + b] += (val >> b) & 1;
                }
            }
        }
    }
    if (spikers == 0) return 0;

    BitVector512 consensus;
    int threshold = spikers / 2;
    for (int i = 0; i < 8; ++i) {
        uint64_t word = 0;
        for (int b = 0; b < 64; ++b) {
            if (counts[i * 64 + b] > threshold) word |= (1ULL << b);
        }
        consensus.data[i] = word;
    }

    uint8_t best = 0; int min_d = 513;
    for (int i = 0; i < 256; ++i) {
        int d = consensus.popcount_xor(bk[i]);
        if (d < min_d) { min_d = d; best = static_cast<uint8_t>(i); }
    }
    if (conf) *conf = 1.0f - (static_cast<float>(min_d) / 512.0f);
    return best;
}

} // namespace s4m::gnf
