// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "special_tokens_split.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace ov::opset13;


void SpecialTokensSplit::compile_pattern_if_necessary(std::string split_pattern) const {
    if (m_split_pattern == nullptr) {
        auto options = re2::RE2::Options();
        options.set_log_errors(false);
        m_split_pattern = std::make_shared<re2::RE2>(split_pattern, options);
    }
}


SpecialTokensSplit::SpecialTokensSplit(const ov::OutputVector& arguments) :
    ov::op::Op(arguments) {
    constructor_validate_and_infer_types();
}


SpecialTokensSplit::SpecialTokensSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<re2::RE2>& m_split_pattern
) :
    ov::op::Op(arguments),
    m_split_pattern(m_split_pattern) {

    auto split_pattern_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
    auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
    auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());
    compile_pattern_if_necessary(std::move(split_pattern));

    constructor_validate_and_infer_types();
}


void SpecialTokensSplit::validate_and_infer_types() {
    auto input_size = get_input_size();
    const bool has_skips = input_size == 7;

    OPENVINO_ASSERT(input_size == 6 || input_size == 7, "Incorrect number of inputs passed to SpecialTokensSplit: " + std::to_string(input_size) +  "; try to reconvert tokenizer with newer version of OpenVINO Tokenizers");
    // input strings
    check_ragged_string_input(this, 0);
    // split pattern
    check_string_scalar_input(this, 5 + has_skips);

    set_ragged_string_output(this, 0, get_input_partial_shape(0));
    if (has_skips) {
        this->set_output_type(5, get_input_element_type(5), get_input_partial_shape(5));
    } else {
        this->set_output_type(5, ov::element::boolean, get_input_partial_shape(2));
    };
}

bool SpecialTokensSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto input_size = get_input_size();
    const bool has_skips = (input_size == 7);

    auto split_pattern = std::string(inputs[5 + has_skips].data<const char>(), inputs[5 + has_skips].get_size());
    compile_pattern_if_necessary(std::move(split_pattern));

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    const size_t batch_size = inputs[0].get_size();
    const size_t num_chars = inputs[4].get_size();

    Tensor skips_alternative;
    const bool *skips;
    if (has_skips) {
        skips = inputs[5].data<bool>();
        outputs[5].set_shape(Shape{num_chars});
    } else {
        outputs[5].set_shape(Shape{num_chars});
        skips_alternative = Tensor(element::boolean, Shape{batch_size});
        skips = std::fill_n(skips_alternative.data<bool>(), batch_size, false) -
                batch_size;
    };

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[2].set_shape(Shape{num_chars});
    outputs[3].set_shape(Shape{num_chars});
    outputs[4] = inputs[4];

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends   = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    auto new_skips = outputs[5].data<bool>();

    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < batch_size; ++seq) {
        new_ragged_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            if (has_skips && skips[ragged_col]) {
                new_begins[ragged_offset] = begins[ragged_col];
                new_skips[ragged_offset] = true;
                new_ends[ragged_offset++] = ends[ragged_col];
            } else {
                auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);

                re2::StringPiece special_token;
                re2::StringPiece input(str);
                size_t curr_start = 0;
                while (m_split_pattern->Match(str, curr_start, str.length(), RE2::UNANCHORED, &special_token, 1)) {
                    const size_t match_start = special_token.data() - str.data();
                    const size_t match_end = match_start + special_token.length();
                    if (curr_start < match_start) {
                        new_begins[ragged_offset] = begins[ragged_col] + curr_start;
                        new_skips[ragged_offset] = false;
                        new_ends[ragged_offset++] = begins[ragged_col] + match_start;
                    };
                    new_begins[ragged_offset] = begins[ragged_col] + match_start;
                    new_skips[ragged_offset] = true;
                    new_ends[ragged_offset++] = begins[ragged_col] + match_end;

                    curr_start = match_end;
                };
                if (curr_start < str.length()) {
                    new_begins[ragged_offset] = begins[ragged_col] + curr_start;
                    new_skips[ragged_offset] = false;
                    new_ends[ragged_offset++] = begins[ragged_col] + str.length();
                }
            }
        }

        new_ragged_ends[seq] = ragged_offset;
    }

    // Fix real shape based on collected results
    outputs[2].set_shape({size_t(ragged_offset)});
    outputs[3].set_shape({size_t(ragged_offset)});
    outputs[5].set_shape({size_t(ragged_offset)});

    return true;
}
