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

    FRONT_END_GENERAL_CHECK(starts_type == element::i64, "Expected an i64 starts tensor ragged representation.");
    FRONT_END_GENERAL_CHECK(ends_type == element::i64, "Expected an i64 starts tensor ragged representation.");
    FRONT_END_GENERAL_CHECK(starts_type == ends_type, "starts and ends tensors should be the same type.");

    set_output_type(0, get_input_element_type(0), PartialShape({Dimension::dynamic(), 2}));
}


bool RaggedToSparse::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // FIXME: Works for POD types only (not for strings!)
    // FIXME: Output mask is calculated even if there are no consumers
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();

    auto last_element_index = inputs[1].get_size();
    outputs[0].set_shape({ends[last_element_index] - begins[0], 2});

    auto batch_size = inputs[0].get_size()

    auto output = outputs[0].data<const int32_t>();

    for (size_t i = 0; i < batch_size; ++i) {
        auto num_elements = ends[i] - begins[i];
        output[0]
    };
//
//    begins = [0, 3]
//    ends = [3, 4]
//
//    output = [
//        [0, 0],
//        [0, 1],
//        [0, 2],
//        [1, 0]
//    ]

    [
        [*, *, *],
        [*, 0, 0]
    ]

    // Suppose validate was called and set correct output shape
    // Take a target shape value for ragged dimension
//    size_t target_dim = outputs[0].get_shape().back();
//
//    auto out_elems = reinterpret_cast<char*>(outputs[0].data());
//    auto out_mask = outputs[1].data<char>();
//
//    auto out_elem_orig = out_elems;
//    auto out_mask_orig = out_mask;
//
//    for(size_t i = 0; i < nelems; ++i) {
//        auto begin = elems + elem_size*begins[i];
//        auto len = std::min(size_t(ends[i] - begins[i]), target_dim);  // truncation
//        auto end = begin + elem_size*len;
//        out_elems = std::copy(begin, end, out_elems);
//        out_mask = std::fill_n(out_mask, len, char(1));
//        if(len < target_dim)
//            out_mask = std::fill_n(out_mask, target_dim - len, char(0));
//        while(len < target_dim) {
//            out_elems = std::copy(default_value, default_value + elem_size, out_elems);
//            ++len;
//        }
//    }
//
//    OPENVINO_ASSERT(out_elems == out_elem_orig + outputs[0].get_byte_size());
//    OPENVINO_ASSERT(out_mask == out_mask_orig + outputs[1].get_byte_size());
    return true;
}
