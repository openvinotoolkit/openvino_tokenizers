// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/constant.hpp>

#include "ragged_to_sparse.hpp"
#include "utils.hpp"

using namespace ov;
using op::v0::Constant;

void RaggedToSparse::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2);

    auto starts_type = this->get_input_element_type(0);
    auto ends_type = this->get_input_element_type(1);

    OPENVINO_ASSERT(starts_type == element::i32, "Expected an i32 starts tensor ragged representation.");
    OPENVINO_ASSERT(ends_type == element::i32, "Expected an i32 starts tensor ragged representation.");
    OPENVINO_ASSERT(get_input_partial_shape(0) == get_input_partial_shape(1), "starts and ends tensors should be the same shape.");

    set_output_type(0, get_input_element_type(0), PartialShape({Dimension::dynamic(), 2}));
}


bool RaggedToSparse::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();

    const auto last_element_index = inputs[1].get_size() - 1;
    const size_t num_elements = static_cast<size_t>(ends[last_element_index] - begins[0]);
    outputs[0].set_shape(ov::Shape{num_elements, 2});

    auto batch_size = inputs[0].get_size();

    auto output = outputs[0].data<int32_t>();
    size_t current_idx = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        auto num_row_elements = ends[i] - begins[i];
        for (size_t j = 0; j < num_row_elements; ++j) {
            output[current_idx++] = i;
            output[current_idx++] = j;
        };
    };
    return true;
}
