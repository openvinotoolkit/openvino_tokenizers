// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"
#include "openvino/runtime/core.hpp"

#define OPENVINO_TOKENIZERS_API OPENVINO_CORE_EXPORTS

namespace ov {
namespace tokenizers {

/**
 * @brief Creates an OpenVINO operation node of the specified type with the given inputs and attributes.
 *
 * This function constructs a node in the OpenVINO computational graph based on the provided operation type,
 * input tensors, and a map of operation-specific attributes. It returns the output(s) produced by the node.
 *
 * @param op_type A string specifying the type of operation to create (e.g., "BPETokenizer").
 * @param inputs A vector of OpenVINO outputs (`ov::OutputVector`) representing the input tensors to the operation.
 * @param attributes A map (`ov::AnyMap`) containing operation-specific attributes.
 *
 * @return ov::OutputVector A vector containing the output(s) of the created node. The number of outputs depends on the
 * operation type.
 */
extern "C" ov::OutputVector OPENVINO_TOKENIZERS_API create_tokenizer_node(const std::string& op_type,
                                                                          const ov::OutputVector& inputs,
                                                                          const ov::AnyMap& attributes);

}  // namespace tokenizers
}  // namespace ov
