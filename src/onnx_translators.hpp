// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/node_context.hpp>

ov::OutputVector
translate_onnx_string_normalizer(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_label_encoder(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_tokenizer(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_tfid_vectorizer(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_contrib_sentencepiece_tokenizer(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_contrib_sentencepiece_decoder(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_contrib_vector_to_string(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_contrib_string_join(const ov::frontend::NodeContext &node);

ov::OutputVector
translate_onnx_contrib_string_split(const ov::frontend::NodeContext &node);
