// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytes_to_chars.hpp"
#include "utils.hpp"

using namespace ov;


const std::array<std::vector<uint8_t>, 256> create_bytes_to_chars_map() {
    return {{
        { 196, 128 },
        { 196, 129 },
        { 196, 130 },
        { 196, 131 },
        { 196, 132 },
        { 196, 133 },
        { 196, 134 },
        { 196, 135 },
        { 196, 136 },
        { 196, 137 },
        { 196, 138 },
        { 196, 139 },
        { 196, 140 },
        { 196, 141 },
        { 196, 142 },
        { 196, 143 },
        { 196, 144 },
        { 196, 145 },
        { 196, 146 },
        { 196, 147 },
        { 196, 148 },
        { 196, 149 },
        { 196, 150 },
        { 196, 151 },
        { 196, 152 },
        { 196, 153 },
        { 196, 154 },
        { 196, 155 },
        { 196, 156 },
        { 196, 157 },
        { 196, 158 },
        { 196, 159 },
        { 196, 160 },
        { 33 },
        { 34 },
        { 35 },
        { 36 },
        { 37 },
        { 38 },
        { 39 },
        { 40 },
        { 41 },
        { 42 },
        { 43 },
        { 44 },
        { 45 },
        { 46 },
        { 47 },
        { 48 },
        { 49 },
        { 50 },
        { 51 },
        { 52 },
        { 53 },
        { 54 },
        { 55 },
        { 56 },
        { 57 },
        { 58 },
        { 59 },
        { 60 },
        { 61 },
        { 62 },
        { 63 },
        { 64 },
        { 65 },
        { 66 },
        { 67 },
        { 68 },
        { 69 },
        { 70 },
        { 71 },
        { 72 },
        { 73 },
        { 74 },
        { 75 },
        { 76 },
        { 77 },
        { 78 },
        { 79 },
        { 80 },
        { 81 },
        { 82 },
        { 83 },
        { 84 },
        { 85 },
        { 86 },
        { 87 },
        { 88 },
        { 89 },
        { 90 },
        { 91 },
        { 92 },
        { 93 },
        { 94 },
        { 95 },
        { 96 },
        { 97 },
        { 98 },
        { 99 },
        { 100 },
        { 101 },
        { 102 },
        { 103 },
        { 104 },
        { 105 },
        { 106 },
        { 107 },
        { 108 },
        { 109 },
        { 110 },
        { 111 },
        { 112 },
        { 113 },
        { 114 },
        { 115 },
        { 116 },
        { 117 },
        { 118 },
        { 119 },
        { 120 },
        { 121 },
        { 122 },
        { 123 },
        { 124 },
        { 125 },
        { 126 },
        { 196, 161 },
        { 196, 162 },
        { 196, 163 },
        { 196, 164 },
        { 196, 165 },
        { 196, 166 },
        { 196, 167 },
        { 196, 168 },
        { 196, 169 },
        { 196, 170 },
        { 196, 171 },
        { 196, 172 },
        { 196, 173 },
        { 196, 174 },
        { 196, 175 },
        { 196, 176 },
        { 196, 177 },
        { 196, 178 },
        { 196, 179 },
        { 196, 180 },
        { 196, 181 },
        { 196, 182 },
        { 196, 183 },
        { 196, 184 },
        { 196, 185 },
        { 196, 186 },
        { 196, 187 },
        { 196, 188 },
        { 196, 189 },
        { 196, 190 },
        { 196, 191 },
        { 197, 128 },
        { 197, 129 },
        { 197, 130 },
        { 194, 161 },
        { 194, 162 },
        { 194, 163 },
        { 194, 164 },
        { 194, 165 },
        { 194, 166 },
        { 194, 167 },
        { 194, 168 },
        { 194, 169 },
        { 194, 170 },
        { 194, 171 },
        { 194, 172 },
        { 197, 131 },
        { 194, 174 },
        { 194, 175 },
        { 194, 176 },
        { 194, 177 },
        { 194, 178 },
        { 194, 179 },
        { 194, 180 },
        { 194, 181 },
        { 194, 182 },
        { 194, 183 },
        { 194, 184 },
        { 194, 185 },
        { 194, 186 },
        { 194, 187 },
        { 194, 188 },
        { 194, 189 },
        { 194, 190 },
        { 194, 191 },
        { 195, 128 },
        { 195, 129 },
        { 195, 130 },
        { 195, 131 },
        { 195, 132 },
        { 195, 133 },
        { 195, 134 },
        { 195, 135 },
        { 195, 136 },
        { 195, 137 },
        { 195, 138 },
        { 195, 139 },
        { 195, 140 },
        { 195, 141 },
        { 195, 142 },
        { 195, 143 },
        { 195, 144 },
        { 195, 145 },
        { 195, 146 },
        { 195, 147 },
        { 195, 148 },
        { 195, 149 },
        { 195, 150 },
        { 195, 151 },
        { 195, 152 },
        { 195, 153 },
        { 195, 154 },
        { 195, 155 },
        { 195, 156 },
        { 195, 157 },
        { 195, 158 },
        { 195, 159 },
        { 195, 160 },
        { 195, 161 },
        { 195, 162 },
        { 195, 163 },
        { 195, 164 },
        { 195, 165 },
        { 195, 166 },
        { 195, 167 },
        { 195, 168 },
        { 195, 169 },
        { 195, 170 },
        { 195, 171 },
        { 195, 172 },
        { 195, 173 },
        { 195, 174 },
        { 195, 175 },
        { 195, 176 },
        { 195, 177 },
        { 195, 178 },
        { 195, 179 },
        { 195, 180 },
        { 195, 181 },
        { 195, 182 },
        { 195, 183 },
        { 195, 184 },
        { 195, 185 },
        { 195, 186 },
        { 195, 187 },
        { 195, 188 },
        { 195, 189 },
        { 195, 190 },
        { 195, 191 },
    }};
}

void BytesToChars::validate_and_infer_types() {
    check_ragged_string_input(this, 0);

    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 5 || input_size == 6, "supported input sizes are 5 or 6");

    set_ragged_string_output(this, 0, get_input_partial_shape(0));
    if (input_size == 6) {
        this->set_output_type(5, get_input_element_type(5),  get_input_partial_shape(5));
    };
}

bool BytesToChars::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    const bool has_skips = inputs.size() == 6;
    bool * skips;
    if (has_skips) {
        skips = inputs[5].data<bool>();
    };

    // Set output shapes
    outputs[0] = inputs[0];
    outputs[1] = inputs[1];
    outputs[2].set_shape(inputs[2].get_shape());
    outputs[3].set_shape(inputs[3].get_shape());
    outputs[4].set_shape(Shape({inputs[4].get_size() * 2}));
    if (has_skips) {
        outputs[5] = inputs[5];
    }
    const size_t num_elems = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    auto new_chars  = outputs[4].data<uint8_t>();
    uint32_t char_pointer = 0;

    for(size_t j = 0; j < num_elems; ++j) {
        for(size_t i = ragged_begins[j]; i < ragged_ends[j]; ++i) {
            const auto word_len = ends[i] - begins[i];
            new_begins[i] = char_pointer;

            if (has_skips) {
                if (skips[i]) {
                      std::copy(chars + begins[i], chars + ends[i], new_chars + char_pointer);
                      char_pointer += word_len;
                } else {
                    for (size_t k = 0; k < word_len; ++k) {
                        for (auto byte : m_bytes_to_chars[chars[begins[i] + k]]) {
                            new_chars[char_pointer++] = byte;
                        }
                    }
                }
            } else {
                for (size_t k = 0; k < word_len; ++k) {
                    for (auto byte : m_bytes_to_chars[chars[begins[i] + k]]) {
                        new_chars[char_pointer++] = byte;
                    }
                }
            };
            new_ends[i] = char_pointer;
        }
    }
    outputs[4].set_shape({char_pointer});
    return true;
}

