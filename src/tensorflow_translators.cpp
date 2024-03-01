// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/hash_table.hpp"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"

#include "tensorflow_translators.hpp"
#include "utils.hpp"

#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "sentence_piece.hpp"
#include "case_fold.hpp"
#include "normalize_unicode.hpp"
#include "regex_normalization.hpp"
#include "regex_split.hpp"

#include "wordpiece_tokenizer.hpp"

using namespace TemplateExtension;
using namespace ov;
using namespace ov::frontend;
using namespace ov::opset13;

namespace {
    template<typename T>
    T extract_scalar_const_value(const std::shared_ptr<Node>& node, const std::string& const_name) {
        auto const_node = as_type_ptr<Constant>(node);
        FRONT_END_GENERAL_CHECK(const_node, "Conversion expects " + const_name + " to be constant.");
        std::vector<T> const_value = const_node->cast_vector<T>();
        FRONT_END_GENERAL_CHECK(const_value.size() == 1, "Conversion expects " + const_name + " to be a scalar.");
        return const_value[0];
    }
}  // namespace

OutputVector translate_sentencepiece_op(const NodeContext& node) {
    // extract model to configure SentencePieceTokenizer
    auto sp_model_ov_any = node.get_attribute_as_any("model");
    FRONT_END_GENERAL_CHECK(sp_model_ov_any.is<std::string>(),
        "SentencePieceOp configuration model is in incorrect format");
    auto str_spm_model = sp_model_ov_any.as<std::string>();
    auto sp_model_const = std::make_shared<Constant>(element::u8, Shape{ str_spm_model.size() }, str_spm_model.data());
    return { sp_model_const };
}

NamedOutputVector translate_sentencepiece_tokenizer(const NodeContext& node) {
    // this is custom translator that converts a sub-graph with SentencePieceOp, SentencePieceTokenizer,
    // and RaggedTensorToSparse operation- into a custom operation SentencepieceTokenizerExtensionOp
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 0, "RaggedTensorToSparse expects at least one input.");
    auto node_name = node.get_name();

    // check that producers of RaggedTensorToSparse is SentencePieceTokenizer
    auto sp_tokenize_op = node.get_input(0).get_node_shared_ptr();
    FRONT_END_GENERAL_CHECK(sp_tokenize_op->get_input_size() > 6,
        "SentencepieceTokenizeOp expects at least six inputs");

    // prepare inputs that go to custom operation
    // prepare input 0 - SentencePieceTokenizer configuration model
    auto sp_model_const = as_type_ptr<Constant>(sp_tokenize_op->input_value(0).get_node_shared_ptr());
    FRONT_END_GENERAL_CHECK(sp_model_const, "Conversion expects SentencePiece model to be constant.");

    // prepare input
    auto inputs = sp_tokenize_op->input_value(1);

    // extract values for nbest_size, alpha, add_bos, add_eos, reverse attributes
    auto nbest_size = extract_scalar_const_value<int32_t>(sp_tokenize_op->input_value(2).get_node_shared_ptr(), "nbest_size");
    auto alpha = extract_scalar_const_value<float>(sp_tokenize_op->input_value(3).get_node_shared_ptr(), "alpha");
    auto add_bos = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(4).get_node_shared_ptr(), "add_bos");
    auto add_eos = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(5).get_node_shared_ptr(), "add_eos");
    auto reverse = extract_scalar_const_value<bool>(sp_tokenize_op->input_value(6).get_node_shared_ptr(), "reverse");

    OutputVector inputs_vector = OutputVector{ sp_model_const, inputs };

    // create a node with custom operation
    auto sp_tokenizer_ext = std::make_shared<SentencepieceTokenizer>(inputs_vector, nbest_size, alpha, add_bos, add_eos, reverse);
    FRONT_END_GENERAL_CHECK(sp_tokenizer_ext->get_output_size() == 3,
        "Internal error: SentencepieceTokenizer operation extension must have three outputs.");

    // set tensor names
    sp_tokenizer_ext->output(0).add_names({ node_name + ":0" });
    sp_tokenizer_ext->output(1).add_names({ node_name + ":1" });
    sp_tokenizer_ext->output(2).add_names({ node_name + ":2" });

    // create named outputs for the conversion extension
    NamedOutputVector named_results;
    named_results.push_back({ "sparse_indices", sp_tokenizer_ext->output(0) });
    named_results.push_back({ "sparse_values", sp_tokenizer_ext->output(1) });
    named_results.push_back({ "sparse_dense_shape", sp_tokenizer_ext->output(2) });

    return named_results;
}

ov::OutputVector translate_case_fold_utf8(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "CaseFold expects only 1 input");
    return { post_translate_string_tensor_output(std::make_shared<CaseFold>(
        pre_translate_string_tensor_input(node.get_input(0)))->outputs()) };
}

ov::OutputVector translate_normalize_utf8(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "NormalizeUTF8 expects only 1 input");
    return { post_translate_string_tensor_output(std::make_shared<NormalizeUnicode>(
        pre_translate_string_tensor_input(node.get_input(0)),
        node.get_attribute<std::string>("normalization_form"))->outputs()) };
}

ov::OutputVector translate_static_regex_replace(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "StaticRegexReplace expects only 1 input");
    auto replace_global = node.get_attribute<bool>("replace_global", true);
    ov::OutputVector inputs = pre_translate_string_tensor_input(node.get_input(0));
    inputs.push_back(string_attribute_to_constant(node, "pattern"));
    inputs.push_back(string_attribute_to_constant(node, "rewrite"));
    auto string_pack_result = post_translate_string_tensor_output(std::make_shared<RegexNormalization>(inputs, replace_global)->outputs());
    set_node_name(node_name, string_pack_result.get_node_shared_ptr());
    return { string_pack_result };
}

ov::OutputVector translate_regex_split_with_offsets(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3, "RegexSplitWithOffsets expects 3 inputs");
    ov::OutputVector inputs = pre_translate_string_tensor_input(node.get_input(0));
    auto delim_regex_pattern = node.get_input(1).get_node()->input_value(2);    // use u8 part of packed string tensor as we are expecting a scalar string: TODO: verify it is really there
    inputs.push_back(delim_regex_pattern);
    // TODO: Use node.get_input(2) with keep_delim_regex_pattern, most likely it should be handled in another RegexSplit with `isolate` behaviour
    auto outputs = std::make_shared<RegexSplit>(inputs)->outputs();
    auto flatten_string_tensor = post_translate_string_tensor_output({ outputs[2], outputs[3], outputs[4] });
    return { post_translate_ragged_tensor_output({outputs[0], outputs[1], flatten_string_tensor}) };
}

ov::OutputVector translate_wordpiece_tokenize_with_offsets(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 2, "WordpieceTokenizeWithOffsets expects 2 inputs");
    ov::OutputVector inputs = pre_translate_ragged_string_tensor_input(node.get_input(0));

#if USE_STRING_TENSORS
    // It may seem enough to call pre_translate_string_tensor_input that will override Parameter element
    // type in case if string tensors are not used.
    // But a Parameter is still required to be overridden even if string tensors are used because in TF model
    // it is represented not as a string tensor, but as a resource with hash table for lookup that we cannot interpret
    // and have to replace by 1D string tensor.
    override_parameter(node.get_input(1).get_node_shared_ptr(), element::string, PartialShape{ Dimension() });
#endif

    auto vocab = pre_translate_string_tensor_input(node.get_input(1));
    inputs.insert(inputs.end(), vocab.begin(), vocab.end());
    // FIXME: Cannot set real value for unk_token_id from attributes because it is not known in this operation
    // TODO: Set other attributes.
    auto wp_tokenizer = std::make_shared<WordpieceTokenizer>(
        inputs,
        node.get_attribute<std::string>("suffix_indicator"),
        node.get_attribute<long>("max_bytes_per_word")
    );
    return { post_translate_ragged_tensor_output(wp_tokenizer->outputs()) };
}

ov::OutputVector translate_string_lower(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 1, "StringLower expects only 1 input");
    auto encoding = node.get_attribute<std::string>("encoding", "");
    ov::OutputVector inputs = pre_translate_string_tensor_input(node.get_input(0));
    auto string_lower_result = post_translate_string_tensor_output(std::make_shared<CaseFold>(inputs, encoding)->outputs());
    set_node_name(node_name, string_lower_result.get_node_shared_ptr());
    return { string_lower_result };
}

OutputVector translate_lookup_table_find_op(const ov::frontend::tensorflow::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3, "LookupTableFind or LookupTableFindV2 expects 3 inputs");
    auto table_handle = as_type_ptr<ov::frontend::tensorflow::HashTable>(node.get_input_by_reference(0).get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        table_handle,
        "[TensorFlow Frontend] internal error: LookupTableFind operation expects table_handle by the first input");
    TENSORFLOW_OP_VALIDATION(
        node,
        table_handle->is_initialized(),
        "[TensorFlow Frontend] internal error: LookupTableFind operation expects initialized table_handle");
    auto keys = node.get_input(1);
    auto default_value = node.get_input(2);

    auto key_type = table_handle->get_key_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        key_type.is_integral_number(),
        "[TensorFlow Frontend] internal error: LookupTableFind is only supported for integer keys");

    auto all_keys = table_handle->get_keys();
    auto all_values = table_handle->get_values();

    // reshape both all values and keys to 1D tensor to work it further
    auto target_shape = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{-1});
    all_keys = std::make_shared<Reshape>(all_keys, target_shape, false);
    all_values = std::make_shared<Reshape>(all_values, target_shape, false);

    // update all values with default value and all keys
    auto default_value_shape = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{1});
    default_value = std::make_shared<Reshape>(default_value, default_value_shape, false);
    all_values = std::make_shared<Concat>(OutputVector{ all_values, default_value }, 0);
    auto num_keys = std::make_shared<ShapeOf>(all_keys, element::i64)->output(0);
    auto scalar_shape = std::make_shared<Constant>(element::i32, Shape{ 0 }, std::vector<int32_t>{});
    num_keys = std::make_shared<Reshape>(num_keys, scalar_shape, false);
    num_keys = std::make_shared<Convert>(num_keys, key_type);

    // compute mask which keys are not valid and for which default value must be used
    auto unsqueeze_axis = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{-1});
    auto unsqueeze_keys = std::make_shared<Unsqueeze>(keys, unsqueeze_axis);
    auto equal_mask = std::make_shared<Equal>(all_keys, unsqueeze_keys)->output(0);
    auto reduce_equal_mask = std::make_shared<ReduceLogicalOr>(equal_mask, unsqueeze_axis, false);

    // map keys to new keys from range [0, n], n index will be for out-of-range keys
    // 1. generate mask-01 of shape [keys_shape, len(all_keys)],
    // where 0 - not found key, 1 - found key
    auto const_zero = std::make_shared<Constant>(key_type, Shape{}, 0);
    auto const_one = std::make_shared<Constant>(key_type, Shape{}, 1);
    auto mask01 = std::make_shared<Select>(equal_mask, const_one, const_zero);
    // 2. generate a range [0, n-1] that will be multiplied to mask for computation of new keys
    auto new_all_keys = std::make_shared<Range>(const_zero, num_keys, const_one, key_type);
    // 3. compute new keys
    auto reduce_axis = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{-1});
    auto new_keys = std::make_shared<Multiply>(mask01, new_all_keys)->output(0);
    new_keys = std::make_shared<ReduceMax>(new_keys, reduce_axis, false);

    // replace invalid keys with key_for_default_value
    new_keys = std::make_shared<Select>(reduce_equal_mask, new_keys, num_keys);

    // at this point all keys are sorted and are from the range [0, n]
    // and keys are also mapped to this range
    auto gather_axis = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{0});
    auto lookup_values = std::make_shared<Gather>(all_values, new_keys, gather_axis);
    set_node_name(node.get_name(), lookup_values);

    return { lookup_values };
}
