// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <openvino/runtime/tensor.hpp>
#include <openvino/frontend/node_context.hpp>
#include <pcre2.h>
#include "absl/strings/string_view.h"

#ifndef OPENVINO_ELEMENT_STRING_SUPPORTED
    #define OPENVINO_ELEMENT_STRING_SUPPORTED 0
#endif

#ifndef OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    #define OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK 0
#endif

#define USE_STRING_TENSORS 0    // modify this depending on willingness to use explicit string tensors

#if USE_STRING_TENSORS && !OPENVINO_ELEMENT_STRING_SUPPORTED
    #error "USE_STRING_TENSORS = 1 can be used only when OpenVINO supports element::string that is determined by OPENVINO_ELEMENT_STRING_SUPPORTED == 1"
#endif

#define SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS 0


void parse_packed_strings (
    const ov::Tensor& packed,
    int32_t& batch_size,
    const int32_t*& begin_ids,
    const int32_t*& end_ids,
    const uint8_t*& symbols);


void check_string_input(const ov::Node* node, size_t input_index);

void check_string_scalar_input(const ov::Node* node, size_t input_index);

void check_ragged_input(const ov::Node* node, size_t input_index);

void check_ragged_string_input(const ov::Node* node, size_t input_index);

void set_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape, ov::element::Type type);

void unpack_strings_to_tensors(const std::string* strings, const ov::Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars);

void override_parameter (std::shared_ptr<ov::Node> node, ov::element::Type type, const ov::PartialShape& shape);

ov::OutputVector pre_translate_string_tensor_input(const ov::Output<ov::Node>& input);

ov::OutputVector pre_translate_ragged_tensor_input(ov::Output<ov::Node> input);

ov::OutputVector pre_translate_ragged_string_tensor_input(ov::Output<ov::Node> input);

ov::Output<ov::Node> post_translate_string_tensor_output(const ov::OutputVector& outputs);

ov::Output<ov::Node> post_translate_ragged_tensor_output(const ov::OutputVector& outputs);

bool evaluate_normalization_helper (
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs,
    std::function<std::string(const std::string&)> normalizer,
    const bool has_skips = false);

std::shared_ptr<ov::Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name);

void set_node_name(const std::string& node_name, const std::shared_ptr<ov::Node>& node);

class PCRE2Wrapper {
public:
    pcre2_code* m_compiled = nullptr;
    PCRE2Wrapper(const absl::string_view& pattern);
    std::string substitute(const std::string& orig_str, const absl::string_view& replace_pattern, bool global_replace);
    std::pair<size_t, size_t> match(const std::string& orig_str, size_t curr_start);
    ~PCRE2Wrapper();
};

class Trie {
    public:
        Trie() = default;

        void add(const std::vector<unsigned char>& str, const int value, int idx = 0);
        int find_longest(const std::vector<unsigned char>& str, int& idx);
        int find_longest(const std::string_view& str, int& idx);

    private:
        std::unordered_map<unsigned char, std::unique_ptr<Trie>> m_to;
        int m_value = -1;  // -1 for unset value
};

bool getenv_bool(const char* env_var, bool default_value);
