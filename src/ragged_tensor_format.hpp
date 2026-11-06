// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

// A ragged tensor is a tensor that has a variable number of elements along the first (ragged) dimension.
// OpenVINO represents it in the decomposed form as 3 regular tensors:
//   * begins - i32 tensor with the offset of the first element of every row in the values tensor
//   * ends   - i32 tensor with the offset of the element that goes right after the last element of every row
//   * values - a flat tensor of an arbitrary element type that keeps all the rows of the ragged tensor
//              back to back
//
// RaggedTensorPack converts such a decomposed representation into a single u8 tensor and
// RaggedTensorUnpack converts it back. It allows a ragged tensor to cross a graph boundary, for example
// to be an input or an output of a model, as a single tensor, the same way as a TensorFlow RaggedTensor
// crosses the boundary of a TensorFlow graph.
//
// The packed representation is a versioned, self-describing byte stream:
//
//   struct RaggedTensorHeader {
//       int32_t  version;             // version of the format, ragged_tensor::CURRENT_VERSION for now
//       int32_t  value_type_id;       // element::Type_t of the values tensor
//       uint32_t num_elements;        // number of rows, i.e. the number of elements in begins and ends
//       uint32_t values_byte_size;    // byte size of the values tensor
//   };                               // exactly 16 bytes, see the static_asserts below
//   int32_t begins[num_elements];
//   int32_t ends[num_elements];
//   uint8_t values[values_byte_size];
//
// The byte order is little-endian independently of the host architecture, therefore the format is portable.
// The size of a packed tensor is HEADER_SIZE + 8 * num_elements + values_byte_size.

struct RaggedTensorHeader {
    int32_t version;
    int32_t value_type_id;
    uint32_t num_elements;
    uint32_t values_byte_size;
};

static_assert(sizeof(RaggedTensorHeader) == 16, "RaggedTensorHeader must occupy exactly 16 bytes");
static_assert(alignof(RaggedTensorHeader) == 4, "RaggedTensorHeader must be 4 bytes aligned");
static_assert(offsetof(RaggedTensorHeader, version) == 0, "RaggedTensorHeader::version must go first");
static_assert(offsetof(RaggedTensorHeader, value_type_id) == 4, "RaggedTensorHeader::value_type_id must be at offset 4");
static_assert(offsetof(RaggedTensorHeader, num_elements) == 8, "RaggedTensorHeader::num_elements must be at offset 8");
static_assert(offsetof(RaggedTensorHeader, values_byte_size) == 12, "RaggedTensorHeader::values_byte_size must be at offset 12");

namespace ragged_tensor {

// The version of the packed ragged tensor format that is produced by RaggedTensorPack and the only version
// that is understood by RaggedTensorUnpack.
constexpr int32_t CURRENT_VERSION = 1;

// Number of bytes that precede the payload (begins, ends and values) of a packed ragged tensor.
constexpr size_t HEADER_SIZE = sizeof(RaggedTensorHeader);

// Returns true for the element types whose values can be stored in a packed ragged tensor.
// String tensors are not packable: a byte stream of std::string objects would alias the character buffers
// owned by the values tensor instead of copying them.
inline bool is_packable_type(const ov::element::Type& type) {
    return type.is_static() && type != ov::element::string && type.size() > 0;
}

inline int32_t type_to_id(const ov::element::Type& type) {
    return static_cast<int32_t>(static_cast<ov::element::Type_t>(type));
}

// Converts an element type id stored in a header to an element type.
// Returns element::dynamic if the id doesn't match any element type known to the current version of OpenVINO,
// for example when a packed tensor was produced by a newer or a corrupted implementation.
inline ov::element::Type type_from_id(int32_t type_id) {
    for (const auto* known_type : ov::element::Type::get_known_types()) {
        if (known_type != nullptr && type_to_id(*known_type) == type_id) {
            return *known_type;
        }
    }
    return ov::element::dynamic;
}

// Checks that the number of rows and the byte size of the values tensor fit into the header fields.
inline bool fits_into_header(size_t num_rows, size_t values_byte_size) {
    return num_rows <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
           values_byte_size <= static_cast<size_t>(std::numeric_limits<uint32_t>::max());
}

// Total number of bytes that a packed ragged tensor with the given parameters occupies.
inline size_t packed_byte_size(uint32_t num_rows, uint32_t values_byte_size) {
    return HEADER_SIZE + 2 * sizeof(int32_t) * static_cast<size_t>(num_rows) + static_cast<size_t>(values_byte_size);
}

inline void store_u32_le(uint8_t* dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xFFu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

inline uint32_t load_u32_le(const uint8_t* src) {
    return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) | (static_cast<uint32_t>(src[3]) << 24);
}

// Reads an int32_t value from a little-endian byte stream.
inline int32_t load_i32_le(const uint8_t* src) {
    return static_cast<int32_t>(load_u32_le(src));
}

inline void write_header(uint8_t* dst, const RaggedTensorHeader& header) {
    OPENVINO_ASSERT(dst != nullptr, "Cannot write a header of a packed ragged tensor into a null pointer");
    store_u32_le(dst + 0, static_cast<uint32_t>(header.version));
    store_u32_le(dst + 4, static_cast<uint32_t>(header.value_type_id));
    store_u32_le(dst + 8, header.num_elements);
    store_u32_le(dst + 12, header.values_byte_size);
}

inline RaggedTensorHeader read_header(const uint8_t* src) {
    OPENVINO_ASSERT(src != nullptr, "Cannot read a header of a packed ragged tensor from a null pointer");
    return RaggedTensorHeader{load_i32_le(src + 0), load_i32_le(src + 4), load_u32_le(src + 8), load_u32_le(src + 12)};
}

// A view of a packed ragged tensor. All the pointers point into the memory of the tensor that was passed
// to parse(), so that tensor must outlive the returned object.
struct PackedRaggedTensor {
    RaggedTensorHeader header{};
    ov::element::Type value_type = ov::element::dynamic;
    const uint8_t* begins = nullptr;  // num_elements little-endian int32_t values
    const uint8_t* ends = nullptr;    // num_elements little-endian int32_t values
    const uint8_t* values = nullptr;  // header.values_byte_size raw bytes of the values tensor

    size_t num_rows() const {
        return static_cast<size_t>(header.num_elements);
    }

    size_t num_values() const {
        return static_cast<size_t>(header.values_byte_size) / value_type.size();
    }
};

// Checks that every row [begins[i], ends[i]) of a ragged tensor fits into a values tensor with num_values elements.
inline void validate_rows(const int32_t* begins, const int32_t* ends, size_t num_rows, size_t num_values) {
    for (size_t i = 0; i < num_rows; ++i) {
        const int64_t row_begin = static_cast<int64_t>(begins[i]);
        const int64_t row_end = static_cast<int64_t>(ends[i]);
        OPENVINO_ASSERT(row_begin >= 0 && row_begin <= row_end && row_end <= static_cast<int64_t>(num_values),
                        "Ragged tensor row ",
                        i,
                        " [",
                        row_begin,
                        ", ",
                        row_end,
                        ") doesn't fit into the values tensor with ",
                        num_values,
                        " elements");
    }
}

// Parses and validates the header of a packed ragged tensor that is stored in `packed`.
// Throws an exception if `packed` doesn't hold a valid packed ragged tensor.
inline PackedRaggedTensor parse(const ov::Tensor& packed) {
    OPENVINO_ASSERT(packed.get_element_type() == ov::element::u8,
                    "Expected a u8 tensor that holds a packed ragged tensor, got a tensor with ",
                    packed.get_element_type().to_string(),
                    " elements");

    const size_t byte_size = packed.get_byte_size();
    OPENVINO_ASSERT(byte_size >= HEADER_SIZE,
                    "Incorrect packed ragged tensor: at least ",
                    HEADER_SIZE,
                    " bytes are required for the header, got ",
                    byte_size,
                    " bytes");

    const auto* data = static_cast<const uint8_t*>(packed.data());
    PackedRaggedTensor result;
    result.header = read_header(data);

    OPENVINO_ASSERT(result.header.version == CURRENT_VERSION,
                    "Unsupported version ",
                    result.header.version,
                    " of the packed ragged tensor format, expected version ",
                    CURRENT_VERSION);

    result.value_type = type_from_id(result.header.value_type_id);
    OPENVINO_ASSERT(is_packable_type(result.value_type),
                    "Incorrect packed ragged tensor: values have an unsupported element type id ",
                    result.header.value_type_id);

    OPENVINO_ASSERT(result.header.values_byte_size % result.value_type.size() == 0,
                    "Incorrect packed ragged tensor: values byte size ",
                    result.header.values_byte_size,
                    " is not a multiple of the element size ",
                    result.value_type.size(),
                    " of ",
                    result.value_type.to_string());

    const size_t expected_byte_size = packed_byte_size(result.header.num_elements, result.header.values_byte_size);
    OPENVINO_ASSERT(expected_byte_size == byte_size,
                    "Incorrect packed ragged tensor: the header describes ",
                    expected_byte_size,
                    " bytes but the tensor has ",
                    byte_size,
                    " bytes");

    result.begins = data + HEADER_SIZE;
    result.ends = result.begins + sizeof(int32_t) * static_cast<size_t>(result.header.num_elements);
    result.values = result.ends + sizeof(int32_t) * static_cast<size_t>(result.header.num_elements);

    return result;
}

}  // namespace ragged_tensor