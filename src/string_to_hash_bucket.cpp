// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_to_hash_bucket.hpp"
#include "utils.hpp"

using namespace ov;

namespace {

static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

uint64_t hash_len16(uint64_t u, uint64_t v, uint64_t mul) {
    uint64_t a = (u ^ v) * mul;
    a ^= (a >> 47);
    uint64_t b = (v ^ a) * mul;
    b ^= (b >> 47);
    b *= mul;
    return b;
}

inline uint64_t basic_rotate64(uint64_t val, int shift) {
    return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

inline uint64_t fetch(const char* p) {
    uint64_t result;
    std::memcpy(&result, p, sizeof(result));
    return result;
}

#if defined(_MSC_VER)

uint64_t rotate(uint64_t val, int shift) {
    return sizeof(unsigned long) == sizeof(val) ? _lrotr(val, shift) : basic_rotate64(val, shift);
}

#else

uint64_t rotate(uint64_t val, int shift) {
    return basic_rotate64(val, shift);
}

#endif

uint64_t hash_len17_to_32(const char* s, size_t len) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = fetch(s) * k1;
    uint64_t b = fetch(s + 8);
    uint64_t c = fetch(s + len - 8) * mul;
    uint64_t d = fetch(s + len - 16) * k2;
    return hash_len16(rotate(a + b, 43) + rotate(c, 30) + d, a + rotate(b + k2, 18) + c, mul);
}

inline uint64_t shift_mix(uint64_t val) {
    return val ^ (val >> 47);
}

inline uint32_t fetch32(const char* p) {
    uint32_t result;
    memcpy(&result, p, sizeof(result));
    return result;
}


uint64_t hash_len0_to_16(const char* s, size_t len) {
    if (len >= 8) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = fetch(s) + k2;
        uint64_t b = fetch(s + len - 8);
        uint64_t c = rotate(b, 37) * mul + a;
        uint64_t d = (rotate(a, 25) + b) * mul;
        return hash_len16(c, d, mul);
    }
    if (len >= 4) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = fetch32(s);
        return hash_len16(len + (a << 3), fetch32(s + len - 4), mul);
    }
    if (len > 0) {
        uint8_t a = s[0];
        uint8_t b = s[len >> 1];
        uint8_t c = s[len - 1];
        uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
        uint32_t z = len + (static_cast<uint32_t>(c) << 2);
        return shift_mix(y * k2 ^ z * k0) * k2;
    }
    return k2;
}

uint64_t hash_len33_to_64(const char* s, size_t len) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = fetch(s) * k2;
    uint64_t b = fetch(s + 8);
    uint64_t c = fetch(s + len - 8) * mul;
    uint64_t d = fetch(s + len - 16) * k2;
    uint64_t y = rotate(a + b, 43) + rotate(c, 30) + d;
    uint64_t z = hash_len16(y, a + rotate(b + k2, 18) + c, mul);
    uint64_t e = fetch(s + 16) * mul;
    uint64_t f = fetch(s + 24);
    uint64_t g = (y + fetch(s + len - 32)) * mul;
    uint64_t h = (z + fetch(s + len - 24)) * mul;
    return hash_len16(rotate(e + f, 43) + rotate(g, 30) + h, e + rotate(f + a, 18) + g, mul);
}

std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(uint64_t w,
    uint64_t x,
    uint64_t y,
    uint64_t z,
    uint64_t a,
    uint64_t b) {
    a += w;
    b = rotate(b + a + z, 21);
    uint64_t c = a;
    a += x;
    a += y;
    b += rotate(a, 44);
    return std::make_pair(a + z, b + c);
}

std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(const char* s, uint64_t a, uint64_t b) {
    return weak_hash_len32_with_seeds(fetch(s), fetch(s + 8), fetch(s + 16), fetch(s + 24), a, b);
}

uint64_t hash64(const char* s, size_t len) {
    const uint64_t seed = 81;
    if (len <= 32) {
        if (len <= 16) {
            return hash_len0_to_16(s, len);
        }
        else {
            return hash_len17_to_32(s, len);
        }
    }
    else if (len <= 64) {
        return hash_len33_to_64(s, len);
    }

    // For strings over 64 bytes we loop.  Internal state consists of
    // 56 bytes: v, w, x, y, and z.
    uint64_t x = seed;
    uint64_t y = seed * k1 + 113;
    uint64_t z = shift_mix(y * k2 + 113) * k2;
    std::pair<uint64_t, uint64_t> v = std::make_pair(0, 0);
    std::pair<uint64_t, uint64_t> w = std::make_pair(0, 0);
    x = x * k2 + fetch(s);

    // Set end so that after the loop we have 1 to 64 bytes left to process.
    const char* end = s + ((len - 1) / 64) * 64;
    const char* last64 = end + ((len - 1) & 63) - 63;
    do {
        x = rotate(x + y + v.first + fetch(s + 8), 37) * k1;
        y = rotate(y + v.second + fetch(s + 48), 42) * k1;
        x ^= w.second;
        y += v.first + fetch(s + 40);
        z = rotate(z + w.first, 33) * k1;
        v = weak_hash_len32_with_seeds(s, v.second * k1, x + w.first);
        w = weak_hash_len32_with_seeds(s + 32, z + w.second, y + fetch(s + 16));
        std::swap(z, x);
        s += 64;
    } while (s != end);
    uint64_t mul = k1 + ((z & 0xff) << 1);
    // Make s point to the last 64 bytes of input.
    s = last64;
    w.first += ((len - 1) & 63);
    v.first += w.first;
    w.first += v.first;
    x = rotate(x + y + v.first + fetch(s + 8), 37) * mul;
    y = rotate(y + v.second + fetch(s + 48), 42) * mul;
    x ^= w.second * 9;
    y += v.first * 9 + fetch(s + 40);
    z = rotate(z + w.first, 33) * mul;
    v = weak_hash_len32_with_seeds(s, v.second * mul, x + w.first);
    w = weak_hash_len32_with_seeds(s + 32, z + w.second, y + fetch(s + 16));
    std::swap(z, x);
    return hash_len16(hash_len16(v.first, w.first, mul) + shift_mix(y) * k0 + z,
        hash_len16(v.second, w.second, mul) + x,
        mul);
}

uint64_t hash64(const std::string& str) {
    return hash64(str.data(), str.size());
}

}

void StringToHashBucket::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3);

    auto begins_type = this->get_input_element_type(0);
    auto ends_type = this->get_input_element_type(1);
    auto output_shape = this->get_input_partial_shape(0);

    OPENVINO_ASSERT(begins_type == element::i32 && ends_type == element::i32,
        "Expected an i32 begins and ends for string tensor representation.");
    OPENVINO_ASSERT(m_num_buckets > 0, "num_buckets attribute must be positive");

    set_output_type(0, ov::element::i64, output_shape);
}

bool StringToHashBucket::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends = inputs[1].data<const int32_t>();
    auto chars = inputs[2].data<const char>();

    auto output_shape = inputs[0].get_shape();
    outputs[0].set_shape(output_shape);
    auto result = outputs[0].data<int64_t>();

    auto num_elems = inputs[0].get_size();
    for (size_t ind = 0; ind < num_elems; ++ind) {
        OPENVINO_ASSERT(begins[ind] <= ends[ind]);
        result[ind] = hash64(chars + begins[ind], static_cast<size_t>(ends[ind] - begins[ind])) % m_num_buckets;
    }

    return true;
}
