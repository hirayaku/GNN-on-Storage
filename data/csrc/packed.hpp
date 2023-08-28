#pragma once
#include <cstdint>
#include <tuple>

// per-partition data packing size and score: [size: int, score: score]
using packed_t = int64_t;
// need better tests for unpack & pack
inline static int32_t extract_int(packed_t packed) {
    uint32_t upper_bits = (uint64_t)packed >> 32;
    return *(int32_t *)&upper_bits;
}
inline static float extract_float(packed_t packed) {
    uint32_t lower_bits = ((uint64_t)packed << 32) >> 32;
    return *(float *)(&lower_bits);
}
inline static std::tuple<int32_t, float> unpack(packed_t packed) {
    return {extract_int(packed), extract_float(packed)};
}
inline static packed_t pack(int32_t integer, float score) {
    packed_t packed = ((packed_t) integer) << 32;
    uint32_t score_bits = *(uint32_t *)&score;
    return packed | (packed_t)score_bits;
}

struct packed_less {
    bool operator() (packed_t a, packed_t b) {
        auto tpl_a = unpack(a);
        auto tpl_b = unpack(b);
        return std::get<1>(tpl_a) < std::get<1>(tpl_b) || 
            (std::get<1>(tpl_a) == std::get<1>(tpl_b) && std::get<0>(tpl_a) < std::get<0>(tpl_b));
    }
};

// struct if64_t {
//     int32_t i32;
//     float f32;
// };
// union packed_t {
//     int64_t i64;
//     if64_t pair;
// };
