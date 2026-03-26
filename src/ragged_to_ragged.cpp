// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/constant.hpp>

#include "ragged_to_ragged.hpp"
#include "utils.hpp"

using namespace ov;
using op::v0::Constant;

void RaggedToRagged::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2);

    auto rowids_type = this->get_input_element_type(0);
    auto first_dim_size_type = this->get_input_element_type(1);

    OPENVINO_ASSERT(rowids_type == element::i32, "Expected an i32 rowids tensor ragged representation.");
    OPENVINO_ASSERT(first_dim_size_type == element::i32, "Expected an i32 first dim size tensor ragged representation.");

    // Check whether input 1 is a Constant node, otherwise fall back to the lower-bound tensor
    PartialShape out_shape({ Dimension::dynamic() });
    auto infer_from_int32_tensor = [&](const ov::Tensor& t) {
        if (t && t.get_element_type() == element::i32 && t.get_size() >= 1) {
            out_shape = PartialShape({ static_cast<Dimension::value_type>(t.data<const int32_t>()[0]) });
        }
    };
    if (const auto* first_dim_const = dynamic_cast<const Constant*>(get_input_node_ptr(1))) {
        auto vals = first_dim_const->cast_vector<int32_t>();
        if (!vals.empty()) {
            out_shape = PartialShape({ static_cast<Dimension::value_type>(vals[0]) });
        }
    } else {
        infer_from_int32_tensor(get_input_tensor(1).get_lower_value());
    }

    set_output_type(0, get_input_element_type(0), out_shape);
    set_output_type(1, get_input_element_type(0), out_shape);
}


bool RaggedToRagged::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto rowids = inputs[0].data<const int32_t>();
    auto rowids_size = static_cast<int32_t>(inputs[0].get_size());
    auto first_dim_size = inputs[1].data<const int32_t>();

    const size_t batch_size = static_cast<size_t>(first_dim_size[0]);
    outputs[0].set_shape(ov::Shape{ batch_size });
    outputs[1].set_shape(ov::Shape{ batch_size });

    auto begins = outputs[0].data<int32_t>();
    auto ends = outputs[1].data<int32_t>();

    // prev_row_id_idx stores value idx for previous row
    int32_t prev_row_id_idx = 0;
    // prev_row_id stores row id for previous row
    int32_t prev_row_id = -1;
    for (int32_t rowids_idx = 0; rowids_idx < rowids_size; ++rowids_idx) {
        int32_t curr_row_id = rowids[rowids_idx];
        OPENVINO_ASSERT(0 <= curr_row_id, "row id must be non-negative");
        if (curr_row_id >= batch_size) {
            break;
        }

        if (prev_row_id != curr_row_id) {
            if (prev_row_id != -1) {
                begins[prev_row_id] = prev_row_id_idx;
                ends[prev_row_id] = rowids_idx;
            }

            int32_t idx = prev_row_id + 1;
            while (idx < curr_row_id) {
                begins[idx] = rowids_idx;
                ends[idx] = rowids_idx;
                ++idx;
            }

            prev_row_id_idx = rowids_idx;
            prev_row_id = curr_row_id;
        }

        if (rowids_idx + 1 == rowids_size) {
            begins[curr_row_id] = prev_row_id_idx;
            ends[curr_row_id] = rowids_size;
            prev_row_id = curr_row_id;
            prev_row_id_idx = rowids_size;
        }
    }

    prev_row_id = (prev_row_id < 0) ? 0 : prev_row_id + 1;
    for (int32_t batch_idx = prev_row_id; batch_idx < batch_size; ++batch_idx) {
        begins[batch_idx] = prev_row_id_idx;
        ends[batch_idx] = prev_row_id_idx;
    }

    return true;
}
