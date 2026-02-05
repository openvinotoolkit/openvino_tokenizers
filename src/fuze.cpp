// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuze.hpp"
#include "utils.hpp"

using namespace ov;

void FuzeRagged::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_element_type(0) == element::i32, "Expected an i32 tensor as the first part of the decomposed ragged string representation");
    OPENVINO_ASSERT(get_input_element_type(1) == element::i32, "Expected an i32 tensor as the second part of the decomposed ragged string representation");
    OPENVINO_ASSERT(get_input_element_type(2) == element::i32, "Expected an i32 tensor as the third part of the decomposed ragged string representation");
    OPENVINO_ASSERT(get_input_element_type(3) == element::i32, "Expected an i32 tensor as the forth part of the decomposed ragged string representation");

    set_output_type(0, element::i32, get_input_partial_shape(0));
    set_output_type(1, element::i32, get_input_partial_shape(0));
}

bool FuzeRagged::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_rows = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    uint32_t char_pointer = 0;

    for(size_t row = 0; row < num_rows; ++row) {
        new_begins[row] = begins[ragged_begins[row]];
        new_ends[row] = ends[(ragged_ends[row] > ragged_begins[row]) ? (ragged_ends[row] - 1) : ragged_ends[row]];
    }
    return true;
}
