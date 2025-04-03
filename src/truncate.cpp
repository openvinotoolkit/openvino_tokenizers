// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "truncate.hpp"
#include "utils.hpp"

using namespace ov;

void Truncate::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() > 0);
    OPENVINO_ASSERT((get_input_size() - 3) % 3 == 0);

    size_t num_inputs = (get_input_size() - 3) / 3;
    OPENVINO_ASSERT(1 <= num_inputs <= 2);

    for (size_t i = 0; i < num_inputs; ++i) {
        check_ragged_input(this, 3 * i);
        set_ragged_output(this, i, get_input_partial_shape(3 * i), get_input_element_type(3 * i));
    }

    // Asser that scalar max_length input is a scalar.
    OPENVINO_ASSERT(get_input_element_type(get_input_size() - 3) == element::i32, "Expected an i32 tensor as the max_length input");
    OPENVINO_ASSERT(get_input_partial_shape(get_input_size() - 3).rank().get_length() == 0, "Expected a scalar tensor as the max_length input");

    // Check that last 2 inputs for truncation side and mode are u8 strings.
    check_string_scalar_input(this, get_input_size() - 2);
    check_string_scalar_input(this, get_input_size() - 1);
}

bool Truncate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    size_t num_inputs = (inputs.size() - 3) / 3;
    
    int32_t max_length = inputs[inputs.size() - 3].data<const int32_t>()[0];
    std::string trunc_side = std::string(inputs[inputs.size() - 2].data<const char>(), inputs[inputs.size() - 2].get_size());
    std::string trunc_mode = std::string(inputs[inputs.size() - 1].data<const char>(), inputs[inputs.size() - 1].get_size());
    OPENVINO_ASSERT(trunc_side == "left" || trunc_side == "right", "Unknown truncation side: ", trunc_side);
    
    for(size_t i = 0; i < num_inputs; ++i) {
        outputs[3*i + 0] = inputs[3*i + 0];
        outputs[3*i + 1] = inputs[3*i + 1];
        outputs[3*i + 2] = inputs[3*i + 2];
        
        auto begin = outputs[3*i + 0].data<int32_t>();
        auto end = outputs[3*i + 1].data<int32_t>();
        int32_t begins_size = outputs[3*i + 0].get_size();

        for (size_t i = 0; i < begins_size; ++i) {
            auto truncated_length = std::min(end[i] - begin[i], max_length);

            if (trunc_side == "right") {
                end[i] = begin[i] + truncated_length;
            } else if (trunc_side == "left") {
                begin[i] = end[i] - truncated_length;
            }
        }
    }

    return true;
}
