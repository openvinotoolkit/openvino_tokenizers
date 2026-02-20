// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/constant.hpp>

#include "ragged_to_dense.hpp"
#include "utils.hpp"

using namespace ov;
using op::v0::Constant;

void RaggedToDense::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 3 + 1 + 1 ||  get_input_size() == 3 + 1 + 1 + 1,
                    "RaggedToDense requires 5 inputs (begins, ends, data, padding_size, value) and 1 optional input (pad_right).");

    // Input ragged tensor (begins, ends, data)
    check_ragged_input_any_rank_data(this, 0);

    // Target size along ragged dimension
    OPENVINO_ASSERT(get_input_element_type(3).is_integral_number());
    auto rank = get_input_partial_shape(3).rank();
    OPENVINO_ASSERT(
        rank.is_dynamic() ||
        rank.get_length() == 0 ||
        rank.get_length() == 1 && get_input_partial_shape(3)[0].compatible(1),
        "Target dense dimension size for RaggedToDense should be a 0D or 1D tensor with a single element");

    // Default value to fill out of ragged range elements in output tensor
    OPENVINO_ASSERT(get_input_element_type(4).compatible(get_input_element_type(2)));
    auto input4_rank = get_input_partial_shape(4).rank();
    OPENVINO_ASSERT(input4_rank.compatible(0));

    set_input_is_relevant_to_shape(3);

    const auto begins_shape = get_input_partial_shape(0);
    const auto data_shape = get_input_partial_shape(2);
    const auto begins_rank = begins_shape.rank();
    const auto data_rank = data_shape.rank();

    if (begins_rank.is_dynamic() || data_rank.is_dynamic()) {
        set_output_type(0, get_input_element_type(2), PartialShape::dynamic());
        set_output_type(1, element::boolean, PartialShape::dynamic());
    } else {
        auto out_shape = begins_shape;
        if (auto target_dim = dynamic_cast<Constant*>(get_input_node_ptr(3))) {
            out_shape.push_back(target_dim->cast_vector<int64_t>()[0]);
        } else {
            out_shape.push_back(Dimension());
        }

        const auto data_rank_len = static_cast<size_t>(data_rank.get_length());
        for (size_t idx = 1; idx < data_rank_len; ++idx) {
            out_shape.push_back(data_shape[idx]);
        }

        set_output_type(0, get_input_element_type(2), out_shape);
        set_output_type(1, element::boolean, out_shape);
    }
    if (get_input_size() == 3 + 1 + 1 + 1) {
        OPENVINO_ASSERT(get_input_partial_shape(5).is_dynamic() || get_input_partial_shape(5).is_static() && get_input_partial_shape(5).rank().get_length() == 0,
                        "RaggedToDense: pad_right should be a boolean scalar.");

        OPENVINO_ASSERT(get_input_element_type(5).is_integral(),
                      "RaggedToDense: pad_right should be a boolean value.");
    }
}


bool RaggedToDense::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // FIXME: Works for POD types only (not for strings!)
    // FIXME: Output mask is calculated even if there are no consumers
    auto begins = inputs[0].data<const int32_t>();
    auto ends = inputs[1].data<const int32_t>();
    const auto nelems = inputs[0].get_size();

    const auto elems = reinterpret_cast<const char*>(inputs[2].data());
    const auto elem_size = inputs[2].get_element_type().size();
    const auto default_value = reinterpret_cast<const char*>(inputs[4].data());

    // Take a target shape value for ragged dimension
    const size_t target_dim = static_cast<size_t>(inputs[3].data<const int32_t>()[0]);

    // If output shape is dynamic at compile-time (e.g. target_dim is not constant),
    // set it at runtime based on actual input values.
    {
        ov::Shape out_shape = inputs[0].get_shape();
        out_shape.push_back(target_dim);
        const auto& data_shape = inputs[2].get_shape();
        for (size_t idx = 1; idx < data_shape.size(); ++idx) {
            out_shape.push_back(data_shape[idx]);
        }
        outputs[0].set_shape(out_shape);
        outputs[1].set_shape(out_shape);
    }

    // Number of dense elements per one ragged element (trailing dense dimensions).
    // For 1D data, this equals 1.
    const auto& data_shape = inputs[2].get_shape();
    size_t inner_elems = 1;
    for (size_t idx = 1; idx < data_shape.size(); ++idx) {
        inner_elems *= data_shape[idx];
    }

    auto out_elems = reinterpret_cast<char*>(outputs[0].data());
    auto out_mask = outputs[1].data<char>();

    auto out_elem_orig = out_elems;
    auto out_mask_orig = out_mask;
    
    bool pad_right = m_pad_right;
    if (get_input_size() == 6) {
        pad_right = inputs[5].data<bool>()[0];
    }

    auto fill_default_block = [&](char*& dst) {
        for (size_t k = 0; k < inner_elems; ++k) {
            dst = std::copy(default_value, default_value + elem_size, dst);
        }
    };

    if (pad_right) {
        for (size_t i = 0; i < nelems; ++i) {
            const size_t data_len = static_cast<size_t>(ends[i] - begins[i]);
            size_t target_len = (std::min(data_len, target_dim) * (1 - m_pad_max_length) +
                                 target_dim * m_pad_max_length);

            const auto begin = elems + elem_size * inner_elems * static_cast<size_t>(begins[i]);
            const auto end = begin + elem_size * inner_elems * target_len;
            out_elems = std::copy(begin, end, out_elems);

            out_mask = std::fill_n(out_mask, target_len * inner_elems, char(1));
            if (target_len < target_dim) {
                out_mask = std::fill_n(out_mask, (target_dim - target_len) * inner_elems, char(0));
            }

            while (target_len < target_dim) {
                fill_default_block(out_elems);
                ++target_len;
            }
        }
    } else {
        for (size_t i = 0; i < nelems; ++i) {
            const size_t data_len = static_cast<size_t>(ends[i] - begins[i]);
            size_t target_len = (std::min(data_len, target_dim) * (1 - m_pad_max_length) +
                                 target_dim * m_pad_max_length);
            const size_t pad_len = target_dim - target_len;

            for (size_t j = 0; j < pad_len; ++j) {
                fill_default_block(out_elems);
            }

            const auto begin = elems + elem_size * inner_elems * static_cast<size_t>(begins[i]);
            const auto end = begin + elem_size * inner_elems * target_len;
            out_elems = std::copy(begin, end, out_elems);

            out_mask = std::fill_n(out_mask, pad_len * inner_elems, char(0));
            out_mask = std::fill_n(out_mask, (target_dim - pad_len) * inner_elems, char(1));
        }
    }

    OPENVINO_ASSERT(out_elems == out_elem_orig + outputs[0].get_byte_size());
    OPENVINO_ASSERT(out_mask == out_mask_orig + outputs[1].get_byte_size());
    return true;
}
