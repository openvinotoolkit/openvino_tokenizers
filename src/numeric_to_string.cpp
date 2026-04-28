// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "numeric_to_string.hpp"
#include <openvino/opsets/opset13.hpp>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

using namespace ov;

void NumericToString::validate_and_infer_types() {
    set_output_type(0, element::string, get_input_partial_shape(0));
}

bool NumericToString::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    const auto& input = inputs[0];
    auto& output = outputs[0];
    output.set_shape(input.get_shape());

    const size_t n = shape_size(input.get_shape());
    std::string* out_data = output.data<std::string>();

    const auto type = input.get_element_type();

    if (type == element::i64) {
        const int64_t* in = input.data<int64_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(in[i]);
        }
    } else if (type == element::i32) {
        const int32_t* in = input.data<int32_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(in[i]);
        }
    } else if (type == element::i16) {
        const int16_t* in = input.data<int16_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(static_cast<int32_t>(in[i]));
        }
    } else if (type == element::i8) {
        const int8_t* in = input.data<int8_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(static_cast<int32_t>(in[i]));
        }
    } else if (type == element::u64) {
        const uint64_t* in = input.data<uint64_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(in[i]);
        }
    } else if (type == element::u32) {
        const uint32_t* in = input.data<uint32_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(in[i]);
        }
    } else if (type == element::f32) {
        const float* in = input.data<float>();
        for (size_t i = 0; i < n; ++i) {
            std::ostringstream oss;
            oss << in[i];
            out_data[i] = oss.str();
        }
    } else if (type == element::f64) {
        const double* in = input.data<double>();
        for (size_t i = 0; i < n; ++i) {
            std::ostringstream oss;
            oss << in[i];
            out_data[i] = oss.str();
        }
    } else if (type == element::boolean) {
        const uint8_t* in = input.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = in[i] ? "true" : "false";
        }
    } else if (type == element::u8) {
        const uint8_t* in = input.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(static_cast<uint32_t>(in[i]));
        }
    } else if (type == element::u16) {
        const uint16_t* in = input.data<uint16_t>();
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::to_string(static_cast<uint32_t>(in[i]));
        }
    } else {
        OPENVINO_ASSERT(false, "[NumericToString] Unsupported input type: " + type.get_type_name());
    }

    return true;
}
