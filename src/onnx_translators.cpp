// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_translators.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"

#include "case_fold.hpp"
#include "equal_str.hpp"
#include "fuze.hpp"
#include "normalize_unicode.hpp"
#include "openvino/op/one_hot.hpp"
#include "ragged_tensor_pack.hpp"
#include "ragged_to_dense.hpp"
#include "regex_normalization.hpp"
#include "regex_split.hpp"
#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "string_to_hash_bucket.hpp"
#include "vocab_encoder.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::frontend;
using namespace ov::opset13;

ov::OutputVector
translate_onnx_string_normalizer(const ov::frontend::NodeContext &node) {
  auto node_name = node.get_name();
  FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                          "StringNormalizer expects only 1 input");
  ov::OutputVector inputs =
      pre_translate_string_tensor_input(node.get_input(0));
  // check for attributes
  std::string case_change_action =
      node.get_attribute<std::string>("case_change_action", "NONE");
  std::vector<std::string> stopwords;
  if (node.has_attribute("stopwords")) {
    stopwords = node.get_attribute<std::vector<std::string>>("stopwords");
  }
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
        std::make_shared<CaseFold>(inputs)->outputs());
  } else if (case_change_action == "UPPER") {
    string_result = post_translate_string_tensor_output(
        std::make_shared<CaseFold>(inputs, "", false)->outputs());
  } else {
    string_result = post_translate_string_tensor_output(inputs);
  }

  set_node_name(node_name, string_result.get_node_shared_ptr());
  return {string_result};
}

ov::OutputVector
translate_onnx_label_encoder(const ov::frontend::NodeContext &node) {
  auto node_name = node.get_name();
  FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                          "LabelEncoder expects only 1 input");

  bool valid_keys =
      node.has_attribute("keys_strings") || node.has_attribute("keys_tensor");
  bool valid_values = node.has_attribute("values_int64s") ||
                      node.has_attribute("values_tensor");

  FRONT_END_GENERAL_CHECK(valid_keys && valid_values,
                          "[ONNX Frontend] internal error: LabelEncoder OV "
                          "Impl supports only one case: "
                          "2. string keys(/tensor) with i64 values.(/tensor)");

  // check for default value
  ov::Output<ov::Node> default_value;
  int64_t default_int64_value = -1;
  if (node.has_attribute("default_int64")) {
    default_int64_value = node.get_attribute<int64_t>("default_int64", -1);
  } else if (node.has_attribute("default_tensor")) {
    ov::Tensor default_tensor =
        node.get_attribute<ov::Tensor>("default_tensor");
    std::memcpy(&default_int64_value, default_tensor.data(), sizeof(int64_t));
  }
  int32_t default_int32_value = static_cast<int32_t>(default_int64_value);
  default_value =
      std::make_shared<Constant>(ov::element::i32, ov::Shape{1},
                                 std::vector<int32_t>{default_int32_value});

  // check for keys
  ov::Output<ov::Node> all_keys;
  if (node.has_attribute("keys_strings")) {
    std::vector<std::string> keys_strings =
        node.get_attribute<std::vector<std::string>>("keys_strings");
    ov::Shape key_shape = {keys_strings.size()};
    all_keys =
        std::make_shared<Constant>(ov::element::string, key_shape, keys_strings)
            ->output(0);
  } else if (node.has_attribute("keys_tensor")) {
    ov::Tensor keys = node.get_attribute<ov::Tensor>("keys_tensor");
    auto constant = std::make_shared<ov::op::v0::Constant>(keys);
    all_keys =
        std::make_shared<Convert>(constant->output(0), ov::element::string)
            ->output(0);
  }

  // check for values
  ov::Output<ov::Node> all_values;
  if (node.has_attribute("values_int64s")) {
    std::vector<int64_t> values_ints =
        node.get_attribute<std::vector<int64_t>>("values_int64s");
    ov::Shape values_shape = {values_ints.size()};
    all_values =
        std::make_shared<Constant>(ov::element::i32, values_shape, values_ints)
            ->output(0);
  } else if (node.has_attribute("values_tensor")) {
    ov::Tensor values = node.get_attribute<ov::Tensor>("values_tensor");
    auto constant = std::make_shared<ov::op::v0::Constant>(values);
    all_values =
        std::make_shared<Convert>(constant->output(0), ov::element::i32)
            ->output(0);
  }

  // unpack string tensor for required keys and all keys from vocabulary
  ov::OutputVector unpacked_input =
      pre_translate_string_tensor_input(node.get_input(0));
  ov::OutputVector unpacked_all_keys =
      pre_translate_string_tensor_input(all_keys);

  ov::OutputVector arguments = unpacked_input;
  arguments.insert(arguments.end(), unpacked_all_keys.begin(),
                   unpacked_all_keys.end());
  arguments.push_back(all_values);
  arguments.push_back(default_value);
  auto tokens = std::make_shared<VocabEncoder>(arguments)->output(0);

  // convert back to i64
  tokens = std::make_shared<Convert>(tokens, ov::element::i64);

  set_node_name(node.get_name(), tokens.get_node_shared_ptr());
  return {tokens};
}

ov::OutputVector
translate_onnx_tokenizer(const ov::frontend::NodeContext &node) {
  auto node_name = node.get_name();
  FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                          "Tokenizer expects only 1 input");
  FRONT_END_GENERAL_CHECK(
      node.has_attribute("tokenexp"),
      "Frontend tokenizer implementation expects tokenexp attribute");

  int64_t mark = node.get_attribute<int64_t>("mark");
  FRONT_END_GENERAL_CHECK(mark == 1, "Frontend tokenizer implementation only "
                                     "supports mark as False currently");

  int64_t mincharnum = node.get_attribute<int64_t>("mincharnum");
  std::string pad_value = node.get_attribute<std::string>("pad_value");

  auto default_value = std::make_shared<ov::op::v0::Constant>(
      ov::element::string, ov::Shape{}, // scalar shape
      std::vector<std::string>{pad_value});

  auto vocabulary = string_attribute_to_constant(node, "tokenexp")->outputs();
  ov::OutputVector inputs =
      pre_translate_string_tensor_input(node.get_input(0));
  auto begins = inputs[0];

  // compute batch_dim to generate ragged_begins and ragged_ends for RegexSplit
  auto input_shape = std::make_shared<ShapeOf>(begins, element::i32);
  auto axis_0_1d = std::make_shared<Constant>(element::i32, Shape{1},
                                              std::vector<int32_t>{0});
  auto batch_dim = std::make_shared<Squeeze>(input_shape, axis_0_1d);
  auto zero_const = std::make_shared<Constant>(element::i32, Shape{},
                                               std::vector<int32_t>{0});
  auto one_const = std::make_shared<Constant>(element::i32, Shape{},
                                              std::vector<int32_t>{1});
  auto ragged_begins =
      std::make_shared<Range>(zero_const, batch_dim, one_const, element::i32);
  auto ragged_ends = std::make_shared<Add>(ragged_begins, one_const);

  ov::OutputVector ragged_tensor = {ragged_begins, ragged_ends, inputs[0],
                                    inputs[1], inputs[2]};
  ragged_tensor.insert(ragged_tensor.end(), vocabulary.begin(),
                       vocabulary.end());

  auto outputs =
      std::make_shared<RegexSplit>(ragged_tensor, "remove", -1)->outputs();

  // Implement Min Char Num
  if (mincharnum > 1) {
    auto word_size =
        std::make_shared<Subtract>(outputs[3], outputs[2])->output(0);
    auto mincharnum_int = std::make_shared<Constant>(
        element::i64, Shape{}, std::vector<int64_t>{mincharnum});
    auto mincharnum_const =
        std::make_shared<Convert>(mincharnum_int, element::i32);
    auto mask =
        std::make_shared<GreaterEqual>(word_size, mincharnum_const)->output(0);
    auto non_zero = std::make_shared<NonZero>(mask)->output(0);
    auto sq_indices = std::make_shared<Squeeze>(non_zero, axis_0_1d);
    outputs[2] =
        std::make_shared<Gather>(outputs[2], sq_indices, zero_const)->output(0);
    outputs[3] =
        std::make_shared<Gather>(outputs[3], sq_indices, zero_const)->output(0);

    // Recompute Ragged Dimensions
    auto mask_int = std::make_shared<Convert>(mask, element::i32);
    auto cumsum_incl =
        std::make_shared<CumSum>(mask_int, zero_const, false, false);
    auto padded_cumsum = std::make_shared<Concat>(
        OutputVector{axis_0_1d->output(0), cumsum_incl->output(0)}, 0);
    auto cumsum_at_begins =
        std::make_shared<Gather>(padded_cumsum, outputs[0], zero_const);
    auto cumsum_at_ends =
        std::make_shared<Gather>(padded_cumsum, outputs[1], zero_const);
    auto seg_counts =
        std::make_shared<Subtract>(cumsum_at_ends, cumsum_at_begins);
    outputs[0] = std::make_shared<CumSum>(seg_counts, zero_const, true, false)
                     ->output(0);
    outputs[1] = std::make_shared<CumSum>(seg_counts, zero_const, false, false)
                     ->output(0);
  }
  auto flatten_string_tensor =
      post_translate_string_tensor_output({outputs[2], outputs[3], outputs[4]});

  // Implement token exp
  auto longest_row_size =
      std::make_shared<Subtract>(outputs[1], outputs[0])->output(0);
  longest_row_size =
      std::make_shared<ReduceMax>(longest_row_size, axis_0_1d, true);
  auto ragged_to_dense =
      std::make_shared<RaggedToDense>(
          ov::OutputVector({outputs[0], outputs[1], flatten_string_tensor,
                            longest_row_size, default_value}),
          true, false)
          ->output(0);

  set_node_name(node.get_name(), ragged_to_dense.get_node_shared_ptr());
  return {ragged_to_dense};
}

ov::OutputVector
translate_onnx_tfid_vectorizer(const ov::frontend::NodeContext &node) {
  auto node_name = node.get_name();
  FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
                          "TFIDVectorizer expects only 1 input");

  FRONT_END_GENERAL_CHECK(
      node.has_attribute("pool_strings"),
      "TFIDVectorizer Frontend Implementation expects pool strings attribute");

  std::vector<std::string> pool_strings =
      node.get_attribute<std::vector<std::string>>("pool_strings");
  int64_t max_gram_length = node.get_attribute<int64_t>("max_gram_length");
  int64_t min_gram_length = node.get_attribute<int64_t>("min_gram_length");
  int64_t max_skip_count = node.get_attribute<int64_t>("max_skip_count");
  std::vector<int64_t> ngram_counts =
      node.get_attribute<std::vector<int64_t>>("ngram_counts");
  std::vector<int64_t> ngram_indexes =
      node.get_attribute<std::vector<int64_t>>("ngram_indexes");
  std::vector<double> weights =
      node.get_attribute<std::vector<double>>("weights", std::vector<double>());
  std::string mode = node.get_attribute<std::string>("mode");

  // We do not handle multi grams in this implementation
  FRONT_END_GENERAL_CHECK(
      max_gram_length == min_gram_length == 1,
      "TFIDVectorizer Frontend Implementation handle only unigrams");

  FRONT_END_GENERAL_CHECK(
      ngram_counts.size() == 1,
      "TFIDVectorizer Frontend Implementation handle only unigrams");

  // We do not handle skip counts in this implementation
  FRONT_END_GENERAL_CHECK(
      max_skip_count == 0,
      "TFIDVectorizer Frontend Implementation does not handle skip counts");

  size_t vocab_size = pool_strings.size();
  auto vocab = std::make_shared<ov::op::v0::Constant>(
                   ov::element::string, ov::Shape{vocab_size}, pool_strings)
                   ->output(0);

  int32_t vocab_size_int = static_cast<int32_t>(vocab_size);
  auto vocab_size_const = std::make_shared<Constant>(
      ov::element::i32, Shape{}, std::vector<int32_t>{vocab_size_int});

  auto all_values =
      std::make_shared<ov::op::v0::Constant>(
          ov::element::i32, ov::Shape{ngram_indexes.size()}, ngram_indexes)
          ->output(0);

  ov::OutputVector unpacked_input =
      pre_translate_string_tensor_input(node.get_input(0));
  ov::OutputVector unpacked_all_keys = pre_translate_string_tensor_input(vocab);
  auto neg_indice = std::make_shared<Constant>(element::i32, Shape{},
                                               std::vector<int32_t>{-1});

  ov::OutputVector arguments = unpacked_input;
  arguments.insert(arguments.end(), unpacked_all_keys.begin(),
                   unpacked_all_keys.end());
  arguments.push_back(all_values);
  arguments.push_back(neg_indice);
  auto tokens = std::make_shared<VocabEncoder>(arguments)->output(0);

  auto on_value = std::make_shared<Constant>(element::f32, Shape{},
                                             std::vector<float>{1.0f});
  auto off_value = std::make_shared<Constant>(element::f32, Shape{},
                                              std::vector<float>{0.0f});

  auto one_hot = std::make_shared<OneHot>(tokens, vocab_size_const, on_value,
                                          off_value, -1)
                     ->output(0);

  auto reduce_axis = std::make_shared<Constant>(ov::element::i32, Shape{1},
                                                std::vector<int32_t>{1});
  auto tf_counts =
      std::make_shared<ReduceSum>(one_hot, reduce_axis, false)->output(0);

  return {tf_counts};
}
