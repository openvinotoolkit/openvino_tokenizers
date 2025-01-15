// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "byte_fallback.hpp"
#include "utils.hpp"
#include "sentence_piece.hpp"

using namespace ov;

void ByteFallback::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool ByteFallback::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[2].set_shape(Shape({inputs[2].get_size()}));
    const size_t num_elems = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_chars  = outputs[2].data<uint8_t>();
    uint32_t char_offset = 0;

    for(size_t i = 0; i < num_elems; ++i) {
        new_begins[i] = char_offset;

        auto token = std::string(chars + begins[i], chars + ends[i]);
        if (token.length() == 6 && token.rfind("<") == 0 && token.rfind(">") == 5) {
            // convert "byte tokens" into bytes
            int ch = sentencepiece::PieceToByte(token);
            new_chars[char_offset++] = ch;
        } else {
            std::copy(token.begin(), token.end(), &new_chars[char_offset]);
            char_offset += token.size();
        }
        new_ends[i] = char_offset;
    }
    outputs[2].set_shape({char_offset});

    return true;
}
