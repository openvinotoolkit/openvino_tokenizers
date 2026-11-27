// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utf8_validate.hpp"

#include <algorithm>

using namespace ov;

void UTF8Validate::validate_and_infer_types() {
    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool UTF8Validate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto* begins = inputs[0].data<const int32_t>();
    const auto* ends = inputs[1].data<const int32_t>();
    const auto* bytes = inputs[2].data<const uint8_t>();
    const auto rows = inputs[0].get_size();
    constexpr uint8_t replacement[] = {0xef, 0xbf, 0xbd};

    // Input rows can reference overlapping regions of the byte tensor.
    size_t capacity = 0;
    for (size_t i = 0; i < rows; ++i) {
        capacity += ends[i] - begins[i];
    }
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[2].set_shape({m_replace_mode ? 3 * capacity : capacity});
    auto* out_begins = outputs[0].data<int32_t>();
    auto* out_ends = outputs[1].data<int32_t>();
    auto* out_bytes = outputs[2].data<uint8_t>();

    size_t out_idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        out_begins[i] = out_idx;
        const auto end = static_cast<size_t>(ends[i]);
        for (auto j = static_cast<size_t>(begins[i]); j < end;) {
            const auto lead = bytes[j];
            const size_t width = lead < 0x80 ? 1 :
                                 lead < 0xc2 || lead > 0xf4 ? 0 :
                                 lead < 0xe0 ? 2 :
                                 lead < 0xf0 ? 3 : 4;
            size_t consumed = 1;
            for (; consumed < width && consumed < end - j; ++consumed) {
                const auto next = bytes[j + consumed];
                if (next < 0x80 || next > 0xbf) {
                    break;
                }
                // Restrict second bytes to exclude overlong encodings, surrogates and
                // code points above U+10FFFF as soon as the invalid byte is seen.
                if (consumed == 1 && ((lead == 0xe0 && next < 0xa0) ||
                                      (lead == 0xed && next > 0x9f) ||
                                      (lead == 0xf0 && next < 0x90) ||
                                      (lead == 0xf4 && next > 0x8f))) {
                    break;
                }
            }
            if (consumed == width) {
                std::copy_n(bytes + j, width, out_bytes + out_idx);
                out_idx += width;
            } else if (m_replace_mode) {
                std::copy_n(replacement, sizeof(replacement), out_bytes + out_idx);
                out_idx += sizeof(replacement);
            }
            // Consume the valid prefix of a malformed sequence. The next byte starts
            // a new sequence, matching Python's UTF-8 replace and ignore behavior.
            j += consumed;
        }
        out_ends[i] = out_idx;
    }

    outputs[2].set_shape({out_idx});
    return true;
}
