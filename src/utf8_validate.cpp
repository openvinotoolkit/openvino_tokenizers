// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utf8_validate.hpp"
#include "openvino/opsets/opset13.hpp"
using namespace ov;
using namespace ov::opset13;

#undef tokenizer


void UTF8Validate::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool UTF8Validate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto begins = inputs[0].data<int32_t>();
    auto ends   = inputs[1].data<int32_t>();
    uint8_t* bytes  = inputs[2].data<uint8_t>();
    auto begins_shape = inputs[0].get_shape();
    auto chars_shape = inputs[2].get_shape();

    outputs[0].set_shape(begins_shape);
    outputs[1].set_shape(begins_shape);
    outputs[2].set_shape(chars_shape);
    
    auto out_begins = outputs[0].data<int32_t>();
    auto out_ends   = outputs[1].data<int32_t>();
    auto out_bytes  = outputs[2].data<uint8_t>();

    // TODO: Check if tensor is not 1D.
    // TODO: Add replace mode.
    
    // UTF-8 code points should not intersect: 
    // if 2 byte object has code point < 0x80 then it's not valid 2 byte utf-8, 
    // even if it has a valid bit mask.
    const uint64_t code_point_starts[4] = {0x0, 0x80, 0x800, 0x10000};
    uint64_t utf_code_point;
    size_t bytes_to_consume;  // Number of additional 0b10xxxxxx bytes to consume to produce a valid UTF8 symbol.
    size_t num_bytes;

    size_t out_idx = begins[0];
    for (size_t i = 0; i < begins_shape[0]; i++) {
        // Flag indicating whether UTF8 symbol is complete: true means it's complete, false means we expect continuation.
        // bool new_symbol_flag = true;
        bytes_to_consume = 0;
        
        out_begins[i] = out_idx;
        for (size_t j = begins[i]; j < ends[i]; j += 1) {
            // Beggining of the symbol.
            // Check when the last octate of the previous symbol was processed.
            if (!bytes_to_consume && bytes[j] < 128) {
                // A valid single byte symbol.
                // todo: Add byte to the resulting sequence.
                out_bytes[out_idx] = bytes[j];
                out_idx += 1;
                continue;
            } else if (!bytes_to_consume && bytes[j] >> 5 == 0b110) {
                num_bytes = 2;
                bytes_to_consume = 1;
                utf_code_point = (0b11111 & bytes[j]) << 6;
                continue;
            } else if (!bytes_to_consume && bytes[j] >> 4 == 0b1110) {
                num_bytes = 3;
                bytes_to_consume = 2;
                utf_code_point = (0b1111 & bytes[j]) << 6 * bytes_to_consume;
                continue;
            } else if (!bytes_to_consume && bytes[j] >> 3 == 0b11110) {
                num_bytes = 4; 
                bytes_to_consume = 3;
                utf_code_point = (0b111 & bytes[j]) << 6 * bytes_to_consume;
                continue;
            } else if (!bytes_to_consume) {
                // TODO: Incorrect byte. Replace or skip.
                continue;
            }

            // Check when we are continuating a multibyte symbol.
            if (bytes_to_consume > 0 && bytes[j] >> 6 != 0b10) {
                // TODO: Incorrect sequence. Replace or skip.
                bytes_to_consume = 0; // TODO: double check
                continue;
            }

            if (bytes_to_consume > 0) {
                bytes_to_consume -= 1;
                utf_code_point |= (0b111111 & bytes[j]) << 6 * bytes_to_consume;
            }

            if (!bytes_to_consume && utf_code_point < code_point_starts[num_bytes - 1]) {
                // utf_code_point is out of range.
                // TODO: Incorrect sequence. Replace or skip.
                bytes_to_consume = 0;
                continue;
            } else if (!bytes_to_consume) {
                // We formed a new symbols and utf_code_point is complete.
                // Add bytes to the resulting sequence.
                std::copy(bytes + j + 1 - num_bytes, bytes + j + 1, out_bytes + out_idx);
                out_idx += num_bytes;
                // Zeroing so that thr new code point from masks can be formed on the next cycle.
                utf_code_point = 0;
            }
        }
        out_ends[i] = out_idx;
    }

    return true;
}
