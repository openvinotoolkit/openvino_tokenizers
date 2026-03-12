// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/hash_table.hpp"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"

#include "onnx_translators.hpp"
#include "utils.hpp"

#include "case_fold.hpp"
#include "equal_str.hpp"
#include "normalize_unicode.hpp"
#include "ragged_to_dense.hpp"
#include "ragged_to_ragged.hpp"
#include "ragged_to_sparse.hpp"
#include "regex_normalization.hpp"
#include "regex_split.hpp"
#include "sentence_piece.hpp"
#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "string_to_hash_bucket.hpp"
#include "vocab_encoder.hpp"
#include "wordpiece_tokenizer.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::frontend;
using namespace ov::opset13;

ov::OutputVector
translate_string_normalizer(const ov::frontend::NodeContext &node) {
  auto node_name = node.get_name();
  FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                          "StringNormalizer expects only 1 input");
  ov::OutputVector inputs =
      pre_translate_string_tensor_input(node.get_input(0));
  // check for attributes
  std::string case_change_action =
      node.get_attribute<std::string>("case_change_action", "NONE");
  std::vector<std::string> stopwords =
      node.get_attribute<std::vector<std::string>>("stopwords");
  bool is_case_sensitive = node.get_attribute<int64_t>("is_case_sensitive", 0);
  ov::Output<ov::Node> string_result;
  // check for stop words
  if (!stopwords.empty()) {
    std::string stopword_pattern;
    for (size_t i = 0; i < stopwords.size(); ++i) {
      if (i > 0)
        stopword_pattern += "|";
      stopword_pattern += stopwords[i];
    }
    std::string pattern_value;
    if (is_case_sensitive != 0) {
      pattern_value = "\\b(" + stopword_pattern + ")\\b\\s*";
    } else {
      pattern_value = "\\b(?i)(" + stopword_pattern + ")\\b\\s*";
    }
    auto pattern_constant =
        std::make_shared<Constant>(element::u8, Shape{pattern_value.length()},
                                   (const void *)pattern_value.data());
    std::string rewrite_value = "";
    auto rewrite_constant =
        std::make_shared<Constant>(element::u8, Shape{rewrite_value.length()},
                                   (const void *)rewrite_value.data());
    inputs.push_back(pattern_constant);
    inputs.push_back(rewrite_constant);
    inputs = std::make_shared<RegexNormalization>(inputs, true)->outputs();
  }
  if (case_change_action == "LOWER") {
    string_result = post_translate_string_tensor_output(
        std::make_shared<CaseFold>(inputs, "")->outputs());
  } else if (case_change_action == "UPPER") {
    string_result = post_translate_string_tensor_output(
        std::make_shared<CaseFold>(inputs, "", false)->outputs());
  } else {
    string_result = post_translate_string_tensor_output(inputs);
  }

  set_node_name(node_name, string_result.get_node_shared_ptr());
  return {string_result};
}
