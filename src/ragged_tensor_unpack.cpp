// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>

#include <openvino/op/constant.hpp>

#include "ragged_tensor_unpack.hpp"
#include "utils.hpp"

using namespace ov;

using namespace ragged_tensor;


void RaggedTensorUnpack::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 1,
                    "RaggedTensorUnpack expects 1 input that holds a packed ragged tensor, got ",
                    get_input_size(),
                    " inputs");

    const auto input_type = get_input_element_type(0);
    OPENVINO_ASSERT(input_type == element::dynamic || input_type == element::u8,
                    "RaggedTensorUnpack expects a u8 tensor that holds a packed ragged tensor as an input, got a tensor with ",
                    input_type.to_string(),
                    " elements");

    const auto input_rank = get_input_partial_shape(0).rank();
    OPENVINO_ASSERT(input_rank.is_dynamic() || input_rank.get_length() == 1,
                    "RaggedTensorUnpack expects a 1D u8 tensor that holds a packed ragged tensor as an input, got shape ",
                    get_input_partial_shape(0).to_string());

    const auto values_type = resolve_value_type();
    // OpenVINO requires a static element type for the outputs of a model, so the type can't be left dynamic
    // and has to be declared at the graph construction time even though it is stored in the data.
    OPENVINO_ASSERT(values_type.is_static(),
                    "RaggedTensorUnpack cannot infer the element type of values from a packed tensor that is not known "
                    "at the graph construction time. Set the 'value_type' attribute of the operation to the element "
                    "type that RaggedTensorPack used for the values tensor, for example 'value_type': 'i32'.");

    // begins and ends are always i32 1D tensors of the same size, while the element type of values comes
    // either from the `value_type` attribute or from the header of a constant input.
    set_output_type(0, element::i32, PartialShape{Dimension::dynamic()});
    set_output_type(1, element::i32, PartialShape{Dimension::dynamic()});
    set_output_type(2, values_type, PartialShape{Dimension::dynamic()});

    // When the packed tensor is already known at the graph construction time, for example when it comes from
    // a Constant, its header gives the exact output shapes.
    if (const auto* packed_constant = dynamic_cast<ov::op::v0::Constant*>(get_input_node_ptr(0))) {
        const auto parsed = parse(packed_constant->get_tensor_view());
        set_output_type(0, element::i32, PartialShape{static_cast<Dimension::value_type>(parsed.num_rows())});
        set_output_type(1, element::i32, PartialShape{static_cast<Dimension::value_type>(parsed.num_rows())});
        set_output_type(2, values_type, PartialShape{static_cast<Dimension::value_type>(parsed.num_values())});
    }
}


element::Type RaggedTensorUnpack::resolve_value_type() const {
    if (m_value_type.is_static()) {
        OPENVINO_ASSERT(is_packable_type(m_value_type),
                        "RaggedTensorUnpack cannot deserialize values with ",
                        m_value_type.to_string(),
                        " elements");
        return m_value_type;
    }

    // The `value_type` attribute is not set, the header of a constant input is the only source of the type
    if (const auto* packed_constant = dynamic_cast<ov::op::v0::Constant*>(get_input_node_ptr(0))) {
        return parse(packed_constant->get_tensor_view()).value_type;
    }

    return element::dynamic;
}


bool RaggedTensorUnpack::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto packed = parse(inputs[0]);

    // The header is self-describing, so it is the source of truth for the element type of values. When the
    // `value_type` attribute is set, it must agree with the header, this catches a mismatched configuration.
    if (m_value_type.is_static()) {
        OPENVINO_ASSERT(m_value_type == packed.value_type,
                        "RaggedTensorUnpack is configured to deserialize values with ",
                        m_value_type.to_string(),
                        " elements, but the packed tensor holds values with ",
                        packed.value_type.to_string(),
                        " elements");
    }

    outputs[0].set_shape(Shape{packed.num_rows()});
    outputs[1].set_shape(Shape{packed.num_rows()});
    outputs[2].set_shape(Shape{packed.num_values()});

    if (packed.num_rows() != 0) {
        const size_t indices_byte_size = sizeof(int32_t) * packed.num_rows();
        std::memcpy(outputs[0].data<int32_t>(), packed.begins, indices_byte_size);
        std::memcpy(outputs[1].data<int32_t>(), packed.ends, indices_byte_size);
    }
    if (packed.header.values_byte_size != 0) {
        std::memcpy(outputs[2].data(), packed.values, packed.header.values_byte_size);
    }

    // Make sure the ragged structure stored in the packed tensor is consistent
    validate_rows(outputs[0].data<const int32_t>(), outputs[1].data<const int32_t>(), packed.num_rows(), packed.num_values());

    return true;
}