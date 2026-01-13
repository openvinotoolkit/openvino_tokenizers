// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/parallel.hpp>

#include "equal_str.hpp"
#include "utils.hpp"

using namespace ov;


void EqualStr::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 6);

    auto begins_type1 = this->get_input_element_type(0);
    auto ends_type1 = this->get_input_element_type(1);
    auto begins_type2 = this->get_input_element_type(3);
    auto ends_type2 = this->get_input_element_type(4);

    OPENVINO_ASSERT(begins_type1 == element::i32 && begins_type2 == element::i32,
        "Expected an i32 begins for string tensor representation.");
    OPENVINO_ASSERT(ends_type1 == element::i32 && ends_type2 == element::i32,
        "Expected an i32 ends for string tensor representation.");

    set_output_type(0, ov::element::i32, PartialShape({ Dimension::dynamic() }));
}

bool EqualStr::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins1 = inputs[0].data<const int32_t>();
    auto ends1 = inputs[1].data<const int32_t>();
    auto chars1 = inputs[2].data<const uint8_t>();
    auto begins2 = inputs[3].data<const int32_t>();
    auto ends2 = inputs[4].data<const int32_t>();
    auto chars2 = inputs[5].data<const uint8_t>();

    size_t num_elems1 = inputs[0].get_size();
    size_t num_elems2 = inputs[3].get_size();

    // in case broadcasting with at least one input empty tensor
    // output tensor must be also empty according to TensorFlow
    size_t num_elems = (num_elems1 == 0 || num_elems2 == 0) ? 0 : std::max(num_elems1, num_elems2);
    outputs[0].set_shape(ov::Shape{ num_elems });
    auto result = outputs[0].data<int32_t>();

    ov::parallel_for(num_elems, [&](size_t idx){
        // handle indices due to broadcasting case
        const size_t idx1 = (idx < num_elems1) ? idx : 0;
        const size_t idx2 = (idx < num_elems2) ? idx : 0;
        const auto begin1 = begins1[idx1];
        const auto begin2 = begins2[idx2];
        const auto end1 = ends1[idx1];
        const auto end2 = ends2[idx2];

        std::string op1(chars1 + begin1, chars1 + end1);
        std::string op2(chars2 + begin2, chars2 + end2);
        result[idx] = (op1 == op2);
    });

    return true;
}
