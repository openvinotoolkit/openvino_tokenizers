// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "truncate.hpp"
#include "utils.hpp"

using namespace ov;

void Truncate::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() > 0);

    m_num_inputs = 0;
    bool max_length_is_set = false;
    for (size_t i = 0; i < get_input_size() / 3; ++i) {
        check_ragged_input(this, 3*i);
        set_ragged_output(this, 3*i, get_input_partial_shape(3 * i), get_input_element_type(3 * i));
        m_num_inputs++;
        // max_length should be a scalar.
        if (get_input_element_type(3*(i+1)) == element::i32 && get_input_partial_shape(3*(i+1)).rank().get_length() == 0) {
            max_length_is_set = true;
            break;
        }
    }
    OPENVINO_ASSERT(max_length_is_set, "Expected a scalar tensor as the max_length input");
    OPENVINO_ASSERT(m_num_inputs >= 1 && m_num_inputs <= 2, "Only single or pair inputs are supported in Truncation op");
    
    // Truncation mode should be u8 strings.
    check_string_scalar_input(this, m_num_inputs*3 + 1);
    
    // If trunc_mode is set, check that it's a u8 string.
    if (get_input_size() > m_num_inputs*3 + 1) {
        check_string_scalar_input(this, get_input_size() - 1);
    }
}

bool Truncate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    
    int32_t max_length = inputs[inputs.size() - 3].data<const int32_t>()[0];
    std::string trunc_side = std::string(inputs[inputs.size() - 2].data<const char>(), inputs[inputs.size() - 2].get_size());
    std::string trunc_mode = std::string(inputs[inputs.size() - 1].data<const char>(), inputs[inputs.size() - 1].get_size());
    OPENVINO_ASSERT(trunc_side == "left" || trunc_side == "right", "Unknown truncation side: ", trunc_side);
    
    if (get_input_size() > m_num_inputs*3 + 1) {
        OPENVINO_ASSERT(trunc_mode == "only_first" || trunc_mode == "only_second" || trunc_mode == "longest_first", "Unknown truncation mode: ", trunc_mode);
    }

    for(size_t i = 0; i < m_num_inputs; ++i) {
        outputs[3*i + 0] = inputs[3*i + 0];
        outputs[3*i + 1] = inputs[3*i + 1];
        outputs[3*i + 2] = inputs[3*i + 2];
    }

    int32_t begins_size = outputs[0].get_size();
    if (m_num_inputs == 1) {
        auto begin = outputs[0].data<int32_t>();
        auto end = outputs[1].data<int32_t>();

        for (size_t i = 0; i < begins_size; ++i) {
            auto truncated_length = std::min(end[i] - begin[i], max_length);
            if (trunc_side == "right") {
                end[i] = begin[i] + truncated_length;
            } else if (trunc_side == "left") {
                begin[i] = end[i] - truncated_length;
            }
        }
        return true;
    }
    OPENVINO_ASSERT(m_num_inputs == 2, "Only single or pair inputs are supported in Truncation op");
    OPENVINO_ASSERT(begins_size == outputs[1].get_size(), "Begin and end tensors should have the same size");
    OPENVINO_ASSERT(begins_size == outputs[3].get_size(), "Shapes of first and second tensors should be the same");
    OPENVINO_ASSERT(begins_size == outputs[4].get_size(), "Begin and end tensors should have the same size");

    for(size_t i = 0; i < begins_size; ++i) {
        auto first_begin = outputs[3*0 + 0].data<int32_t>();
        auto first_end = outputs[3*0 + 1].data<int32_t>();
        auto second_begin = outputs[3*1 + 0].data<int32_t>();
        auto second_end = outputs[3*1 + 1].data<int32_t>();
        auto first_length = first_end[i] - first_begin[i];
        auto second_length = second_end[i] - second_begin[i];

        if (first_length + second_length > max_length) {
            // If bothe lengths are greater than max_length and if max_length is not even,
            // remainder should be added to the longest input to match to HF behavior.
            const int32_t first_remainder = (max_length % 2) * (first_length >= second_length);
            const int32_t second_remainder = (max_length % 2) * (first_length < second_length);

            if (trunc_side == "right") {
                // If it's right truncation then we should modify ends.
                if (trunc_mode == "only_first") {
                    // TODO: HF fails with this value.
                    if (first_length > max_length) {
                        first_end[i] = first_begin[i] + max_length;
                    }
                } else if (trunc_mode == "only_second") {
                    // TODO: HF fails with this value.
                    if (second_length > max_length) {
                        second_end[i] = second_begin[i] + max_length;
                    }
                } else if (trunc_mode == "longest_first") {
                    // if max_length = 10 and first_length = 9, second_length = 2,
                    // then we should get output with first_length = 8, second_length = 2
                    if (first_length >= (max_length / 2 + max_length % 2) && second_length <= max_length / 2) {
                        first_end[i] = first_begin[i] + (max_length - second_length);
                    // if max_length = 10 and first_length = 2, second_length = 9,
                    // then we should get output with first_length = 2, second_length = 8
                    } else if (first_length < (max_length / 2 + max_length % 2) && second_length > max_length / 2) {
                        second_end[i] = second_begin[i] + (max_length - first_length);
                    // if max_length = 10 and first, second inputs sizes exceed that length,
                    // then output should be divided. If max_length is not even
                    // ramainding 1 should be added depending on which inputs was longer.
                    } else {
                        first_end[i] = first_begin[i] + (max_length / 2) + first_remainder;
                        second_end[i] = second_begin[i] + (max_length / 2) + second_remainder;
                    }
                }
            } else {  // "left" truncation
                // If it's left truncation then we should modify begins.
                if (trunc_mode == "only_first") {
                    if (first_length > max_length) {
                        // TODO: HF fails with this value.
                        first_begin[i] = first_end[i] - max_length;
                    }
                } else if (trunc_mode == "only_second") {
                    if (second_length > max_length) {
                        // TODO: HF fails with this value.
                        second_begin[i] = second_end[i] - max_length;
                    }
                } else if (trunc_mode == "longest_first") {
                    if (first_length >= max_length / 2 + max_length % 2 && second_length <= max_length / 2) {
                        first_begin[i] = first_end[i] - (max_length - second_length);
                    } else if (first_length < max_length / 2 + max_length % 2 && second_length > max_length / 2) {
                        second_begin[i] = second_end[i] - (max_length - first_length);
                    } else {
                        // if max_length = 10 and first, second inputs sizes exceed that length,
                        // then output should be divided. If max_length is not even
                        // ramainding 1 should be added depending on which inputs was longer.
                        first_begin[i] = first_end[i] - (max_length / 2 + first_remainder);
                        second_begin[i] = second_end[i] - (max_length / 2 + second_remainder);
                    }
                }
            }
        }
    }

    return true;
}
