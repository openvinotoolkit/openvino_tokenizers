// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>

#include "ragged_tensor_pack.hpp"
#include "utils.hpp"

using namespace ov;

using namespace ragged_tensor;


void RaggedTensorPack::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3,
                    "RaggedTensorPack expects 3 inputs (begins, ends, values) that represent a ragged tensor in the decomposed form, got ",
                    get_input_size(),
                    " inputs");

    check_ragged_input_any_rank_data(this, 0);

    OPENVINO_ASSERT(get_input_partial_shape(0).compatible(get_input_partial_shape(1)),
                    "RaggedTensorPack: begins and ends must have the same shape, got shapes ",
                    get_input_partial_shape(0).to_string(),
                    " and ",
                    get_input_partial_shape(1).to_string());

    const auto values_type = get_input_element_type(2);
    OPENVINO_ASSERT(is_packable_type(values_type),
                    "RaggedTensorPack cannot serialize a values tensor with ",
                    values_type.to_string(),
                    " elements");

    // The result is a single 1D u8 tensor that keeps the header, begins, ends and values of the input tensors.
    // Its exact length is not reported by type inference: it depends on the element type and the number of elements
    // of the values tensor as they are seen when this operation is executed, and a plugin is allowed to change the
    // precision of an input, for example to convert i64 into i32. The length is set by evaluate() instead.
    set_output_type(0, element::u8, PartialShape{Dimension::dynamic()});
}


bool RaggedTensorPack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& begins_input = inputs[0];
    const auto& ends_input = inputs[1];
    const auto& values_input = inputs[2];

    OPENVINO_ASSERT(begins_input.get_element_type() == element::i32 && ends_input.get_element_type() == element::i32,
                    "RaggedTensorPack expects i32 begins and ends tensors");

    const size_t num_elements = begins_input.get_size();
    OPENVINO_ASSERT(num_elements == ends_input.get_size(),
                    "RaggedTensorPack: begins and ends must have the same number of elements, got ",
                    num_elements,
                    " and ",
                    ends_input.get_size());

    const auto values_type = values_input.get_element_type();
    OPENVINO_ASSERT(is_packable_type(values_type),
                    "RaggedTensorPack cannot serialize a values tensor with ",
                    values_type.to_string(),
                    " elements");

    const size_t values_byte_size = values_input.get_byte_size();
    OPENVINO_ASSERT(fits_into_header(num_elements, values_byte_size),
                    "RaggedTensorPack cannot serialize a ragged tensor with ",
                    num_elements,
                    " rows and ",
                    values_byte_size,
                    " bytes of values, the packed format supports at most 2^32-1 of each");

    // Make sure the ragged structure is consistent before it is serialized
    validate_rows(begins_input.data<const int32_t>(), ends_input.data<const int32_t>(), num_elements, values_input.get_size());

    outputs[0].set_shape(Shape{packed_byte_size(static_cast<uint32_t>(num_elements), static_cast<uint32_t>(values_byte_size))});
    uint8_t* out = outputs[0].data<uint8_t>();

    RaggedTensorHeader header;
    header.version = CURRENT_VERSION;
    header.value_type_id = type_to_id(values_type);
    header.num_elements = static_cast<uint32_t>(num_elements);
    header.values_byte_size = static_cast<uint32_t>(values_byte_size);
    write_header(out, header);
    out += HEADER_SIZE;

    const size_t indices_byte_size = sizeof(int32_t) * num_elements;
    std::memcpy(out, begins_input.data<const int32_t>(), indices_byte_size);
    out += indices_byte_size;
    std::memcpy(out, ends_input.data<const int32_t>(), indices_byte_size);
    out += indices_byte_size;
    std::memcpy(out, values_input.data(), values_byte_size);

    return true;
}