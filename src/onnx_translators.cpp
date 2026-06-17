// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_translators.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"

#include "case_fold.hpp"
#include "contrib_string_ops.hpp"
#include "equal_str.hpp"
#include "fuze.hpp"
#include "normalize_unicode.hpp"
#include "openvino/op/one_hot.hpp"
#include "ragged_tensor_pack.hpp"
#include "ragged_to_dense.hpp"
#include "regex_normalization.hpp"
#include "regex_split.hpp"
#include "sentence_piece.hpp"
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
  default_value =
      std::make_shared<Constant>(ov::element::i64, ov::Shape{1},
                                 std::vector<int64_t>{default_int64_value});

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
    all_keys = constant->output(0);
  }

  // check for values
  ov::Output<ov::Node> all_values;
  if (node.has_attribute("values_int64s")) {
    std::vector<int64_t> values_ints =
        node.get_attribute<std::vector<int64_t>>("values_int64s");
    ov::Shape values_shape = {values_ints.size()};
    all_values = std::make_shared<Constant>(ov::element::i64, values_shape, values_ints)->output(0);
  } else if (node.has_attribute("values_tensor")) {
    ov::Tensor values = node.get_attribute<ov::Tensor>("values_tensor");
    all_values = std::make_shared<ov::op::v0::Constant>(values)->output(0);
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

  int64_t mark = node.get_attribute<int64_t>("mark", 0);
  FRONT_END_GENERAL_CHECK(mark == 0, "Frontend tokenizer implementation only "
                                     "supports mark==0 currently");

  int64_t mincharnum = node.get_attribute<int64_t>("mincharnum", 1);
  std::string pad_value =
      node.get_attribute<std::string>("pad_value", std::string{});

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

  // We do not handle multi grams in this implementation
  FRONT_END_GENERAL_CHECK(
      max_gram_length == 1 && min_gram_length == 1,
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

  int64_t vocab_size_int = static_cast<int64_t>(vocab_size);
  auto vocab_size_const = std::make_shared<Constant>(
      ov::element::i64, Shape{}, std::vector<int64_t>{vocab_size_int});

  auto all_values =
      std::make_shared<ov::op::v0::Constant>(
          ov::element::i64, ov::Shape{ngram_indexes.size()}, ngram_indexes)
          ->output(0);

  ov::OutputVector unpacked_input =
      pre_translate_string_tensor_input(node.get_input(0));
  ov::OutputVector unpacked_all_keys = pre_translate_string_tensor_input(vocab);
  auto neg_indice = std::make_shared<Constant>(element::i64, Shape{},
                                               std::vector<int64_t>{-1});

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

  auto reduce_axis = std::make_shared<Constant>(ov::element::i64, Shape{1},
                                                std::vector<int64_t>{1});
  auto tf_counts =
      std::make_shared<ReduceSum>(one_hot, reduce_axis, false)->output(0);

  set_node_name(node_name, tf_counts.get_node_shared_ptr());

  return {tf_counts};
}

namespace {

// Extract a scalar value of type T from a constant input. Helper mirrored after
// the tensorflow translator's extract_scalar_const_value.
template <typename T>
T extract_scalar_const(const ov::Output<ov::Node> &input,
                       const std::string &name) {
  auto const_node =
      ov::as_type_ptr<Constant>(input.get_node_shared_ptr());
  FRONT_END_GENERAL_CHECK(const_node,
      "[ONNX Frontend] expected constant for ", name);
  auto values = const_node->cast_vector<T>();
  FRONT_END_GENERAL_CHECK(values.size() == 1,
      "[ONNX Frontend] expected scalar for ", name);
  return values[0];
}

// Build an OV graph that converts the SentencepieceTokenizer OV op's sparse
// outputs (indices [N, 2], values [N] i32, dense_shape [2] i64) into the
// onnxruntime-extensions style outputs:
//   - flat token ids (i32, 1D)
//   - row split indices (i64, 1D length batch_size + 1)
ov::OutputVector make_row_splits_from_sparse(
    const ov::Output<ov::Node> &sparse_indices,
    const ov::Output<ov::Node> &sparse_values,
    const ov::Output<ov::Node> &dense_shape) {
  auto axis0_1d = std::make_shared<Constant>(element::i64, Shape{1},
                                             std::vector<int64_t>{0});
  auto axis1_1d = std::make_shared<Constant>(element::i64, Shape{1},
                                             std::vector<int64_t>{1});
  auto axis0_0d = std::make_shared<Constant>(element::i64, Shape{},
                                             std::vector<int64_t>{0});

  // batch_indices = sparse_indices[:, 0], shape [N]
  auto batch_indices =
      std::make_shared<Gather>(sparse_indices, axis0_0d, axis1_1d);

  // B = dense_shape[0], scalar i64
  auto B = std::make_shared<Gather>(dense_shape, axis0_0d, axis0_0d);

  // range = [0, B+1)
  auto one_i64 = std::make_shared<Constant>(element::i64, Shape{},
                                            std::vector<int64_t>{1});
  auto B_plus_1 = std::make_shared<Add>(B, one_i64);
  auto zero_i64 = std::make_shared<Constant>(element::i64, Shape{},
                                             std::vector<int64_t>{0});
  auto range =
      std::make_shared<Range>(zero_i64, B_plus_1, one_i64, element::i64);

  // mask[n, i] = batch_indices[n] < range[i]
  auto bi_unsq = std::make_shared<Unsqueeze>(batch_indices, axis1_1d);
  auto range_unsq = std::make_shared<Unsqueeze>(range, axis0_1d);
  auto mask = std::make_shared<Less>(bi_unsq, range_unsq);
  auto mask_i64 = std::make_shared<Convert>(mask, element::i64);
  auto row_splits =
      std::make_shared<ReduceSum>(mask_i64, axis0_1d, false)->output(0);

  return {sparse_values, row_splits};
}

}  // namespace

ov::OutputVector
translate_onnx_contrib_sentencepiece_tokenizer(
    const ov::frontend::NodeContext &node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 6 || node.get_input_size() == 7,
        "ai.onnx.contrib.SentencepieceTokenizer expects 6 or 7 inputs "
        "(input, nbest_size, alpha, add_bos, add_eos, reverse[, fairseq])");
    FRONT_END_GENERAL_CHECK(node.has_attribute("model"),
        "ai.onnx.contrib.SentencepieceTokenizer requires the 'model' attribute");

    auto model_bytes = node.get_attribute<std::string>("model");
    auto sp_model_const = std::make_shared<Constant>(
        element::u8, Shape{model_bytes.size()},
        reinterpret_cast<const void *>(model_bytes.data()));

    auto input_strings = node.get_input(0);
    auto nbest_size =
        extract_scalar_const<int32_t>(node.get_input(1), "nbest_size");
    auto alpha = extract_scalar_const<float>(node.get_input(2), "alpha");
    auto add_bos = extract_scalar_const<bool>(node.get_input(3), "add_bos");
    auto add_eos = extract_scalar_const<bool>(node.get_input(4), "add_eos");
    auto reverse = extract_scalar_const<bool>(node.get_input(5), "reverse");
    // The optional 7th input enables fairseq-style id remapping, which the
    // underlying SentencepieceTokenizer op does not implement.
    if (node.get_input_size() == 7) {
        auto fairseq = extract_scalar_const<bool>(node.get_input(6), "fairseq");
        FRONT_END_GENERAL_CHECK(!fairseq,
            "ai.onnx.contrib.SentencepieceTokenizer does not support fairseq mode");
    }

    auto sp_tokenizer = std::make_shared<SentencepieceTokenizer>(
        OutputVector{sp_model_const, input_strings},
        nbest_size, alpha, add_bos, add_eos, reverse);
    FRONT_END_GENERAL_CHECK(sp_tokenizer->get_output_size() == 3,
        "SentencepieceTokenizer must produce three outputs");

    auto outs = make_row_splits_from_sparse(
        sp_tokenizer->output(0),  // sparse_indices i64 [N,2]
        sp_tokenizer->output(1),  // sparse_values  i32 [N]
        sp_tokenizer->output(2)); // dense_shape    i64 [2]

    outs[0].add_names({node_name + ":0"});
    outs[1].add_names({node_name + ":1"});
    return outs;
}

ov::OutputVector
translate_onnx_contrib_sentencepiece_decoder(
    const ov::frontend::NodeContext &node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 2,
        "ai.onnx.contrib.SentencepieceDecoder expects 2 inputs (token ids, fairseq)");
    FRONT_END_GENERAL_CHECK(node.has_attribute("model"),
        "ai.onnx.contrib.SentencepieceDecoder requires the 'model' attribute");
    // The optional fairseq flag enables fairseq-style id remapping, which the
    // underlying SentencepieceDetokenizer op does not implement.
    auto fairseq = extract_scalar_const<bool>(node.get_input(1), "fairseq");
    FRONT_END_GENERAL_CHECK(!fairseq,
        "ai.onnx.contrib.SentencepieceDecoder does not support fairseq mode");

    auto model_bytes = node.get_attribute<std::string>("model");
    auto sp_model_const = std::make_shared<Constant>(
        element::u8, Shape{model_bytes.size()},
        reinterpret_cast<const void *>(model_bytes.data()));

    // SentencepieceDetokenizer expects a 2D [batch, seq_len] i32 tensor of ids.
    auto token_ids = std::make_shared<Convert>(node.get_input(0), element::i32);
    auto sp_detokenizer = std::make_shared<SentencepieceDetokenizer>(
        OutputVector{sp_model_const, token_ids});
    FRONT_END_GENERAL_CHECK(sp_detokenizer->get_output_size() == 3,
        "SentencepieceDetokenizer must produce three outputs");

    auto str_out = post_translate_string_tensor_output(sp_detokenizer->outputs());
    str_out.add_names({node_name + ":0"});
    return {str_out};
}

namespace {

// Parse the `map` attribute of ai.onnx.contrib.VectorToString. Each non-empty
// line has the format "<token>\t<id>". Returns the vocabulary in id order
// (filled with `unk` for missing ids in [0, max_id]).
std::vector<std::string> parse_vector_to_string_map(
    const std::string &text, const std::string &unk) {
    std::vector<std::pair<std::string, int64_t>> pairs;
    int64_t max_id = -1;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        if (nl > pos) {
            auto tab = text.find('\t', pos);
            if (tab != std::string::npos && tab < nl) {
                std::string token = text.substr(pos, tab - pos);
                std::string id_str = text.substr(tab + 1, nl - tab - 1);
                // Strip trailing CR if present.
                if (!id_str.empty() && id_str.back() == '\r') id_str.pop_back();
                try {
                    int64_t id = std::stoll(id_str);
                    if (id >= 0) {
                        pairs.emplace_back(std::move(token), id);
                        if (id > max_id) max_id = id;
                    }
                } catch (...) {
                    // Skip malformed line.
                }
            }
        }
        pos = nl + 1;
    }
    std::vector<std::string> vocab(static_cast<size_t>(max_id + 1), unk);
    for (auto &p : pairs) {
        vocab[static_cast<size_t>(p.second)] = std::move(p.first);
    }
    return vocab;
}

}  // namespace

ov::OutputVector
translate_onnx_contrib_vector_to_string(
    const ov::frontend::NodeContext &node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1,
        "ai.onnx.contrib.VectorToString expects 1 input (token ids)");
    FRONT_END_GENERAL_CHECK(node.has_attribute("map"),
        "ai.onnx.contrib.VectorToString requires the 'map' attribute");

    auto map_text = node.get_attribute<std::string>("map");
    auto unk = node.get_attribute<std::string>("unk", std::string{});

    auto vocab = parse_vector_to_string_map(map_text, unk);
    size_t vocab_size = vocab.size();
    FRONT_END_GENERAL_CHECK(vocab_size > 0,
        "ai.onnx.contrib.VectorToString: parsed empty map");
    // Append unk as a sentinel slot so out-of-range ids can be mapped to it
    // purely via integer Select+Gather (CPU plugin does not support Select on
    // string tensors).
    vocab.push_back(unk);
    size_t vocab_size_with_unk = vocab.size();

    auto vocab_const = std::make_shared<Constant>(
        element::string, Shape{vocab_size_with_unk}, vocab);

    // Lookup: ids may be i64 or i32 in the source model. Cast to i64 and map
    // out-of-range ids to the sentinel unk slot.
    auto ids = node.get_input(0);
    auto ids_i64 = std::make_shared<Convert>(ids, element::i64);
    auto vocab_size_const = std::make_shared<Constant>(
        element::i64, Shape{},
        std::vector<int64_t>{static_cast<int64_t>(vocab_size)});
    auto zero_i64 = std::make_shared<Constant>(element::i64, Shape{},
                                               std::vector<int64_t>{0});
    auto unk_idx_const = std::make_shared<Constant>(
        element::i64, Shape{},
        std::vector<int64_t>{static_cast<int64_t>(vocab_size)});  // last slot

    auto in_range_hi = std::make_shared<Less>(ids_i64, vocab_size_const);
    auto in_range_lo = std::make_shared<GreaterEqual>(ids_i64, zero_i64);
    auto in_range = std::make_shared<LogicalAnd>(in_range_hi, in_range_lo);

    auto safe_ids = std::make_shared<Select>(in_range, ids_i64, unk_idx_const);
    auto axis0 = std::make_shared<Constant>(element::i64, Shape{},
                                            std::vector<int64_t>{0});
    auto result = std::make_shared<Gather>(vocab_const, safe_ids, axis0);

    set_node_name(node_name, result);
    return {result->output(0)};
}

ov::OutputVector
translate_onnx_contrib_string_join(const ov::frontend::NodeContext &node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3,
        "ai.onnx.contrib.StringJoin expects 3 inputs "
        "(input, sep, axis)");

    auto input_unpacked = pre_translate_string_tensor_input(node.get_input(0));
    auto sep_unpacked = pre_translate_string_tensor_input(node.get_input(1));

    // axis: ensure i64
    auto axis_in = node.get_input(2);
    ov::Output<ov::Node> axis_i64;
    if (axis_in.get_element_type() == element::i64) {
        axis_i64 = axis_in;
    } else {
        axis_i64 = std::make_shared<Convert>(axis_in, element::i64);
    }

    OutputVector args;
    args.insert(args.end(), input_unpacked.begin(), input_unpacked.end());
    args.push_back(sep_unpacked[2]);  // pass sep raw chars (u8) directly
    args.push_back(axis_i64);

    auto join = std::make_shared<ContribStringJoin>(args);
    auto packed = post_translate_string_tensor_output(join->outputs());
    set_node_name(node_name, packed.get_node_shared_ptr());
    return {packed};
}

ov::OutputVector
translate_onnx_contrib_string_split(const ov::frontend::NodeContext &node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3,
        "ai.onnx.contrib.StringSplit expects 3 inputs "
        "(input, delimiter, skip_empty)");

    auto input_unpacked = pre_translate_string_tensor_input(node.get_input(0));
    auto delim_unpacked = pre_translate_string_tensor_input(node.get_input(1));

    auto skip_in = node.get_input(2);
    ov::Output<ov::Node> skip_bool;
    if (skip_in.get_element_type() == element::boolean) {
        skip_bool = skip_in;
    } else {
        skip_bool = std::make_shared<Convert>(skip_in, element::boolean);
    }

    OutputVector args;
    args.insert(args.end(), input_unpacked.begin(), input_unpacked.end());
    args.push_back(delim_unpacked[2]);  // pass delim raw chars (u8) directly
    args.push_back(skip_bool);

    auto split = std::make_shared<ContribStringSplit>(args);
    auto indices = split->output(0);
    auto values_packed = post_translate_string_tensor_output(
        OutputVector{split->output(1), split->output(2), split->output(3)});
    auto dense_shape = split->output(4);

    indices.add_names({node_name + ":0"});
    values_packed.add_names({node_name + ":1"});
    dense_shape.add_names({node_name + ":2"});

    return {indices, values_packed, dense_shape};
}
