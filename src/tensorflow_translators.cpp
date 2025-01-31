// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/hash_table.hpp"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"

#include "tensorflow_translators.hpp"
#include "utils.hpp"

#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "sentence_piece.hpp"
#include "equal_str.hpp"
#include "ragged_to_dense.hpp"
#include "ragged_to_sparse.hpp"
#include "ragged_to_ragged.hpp"
#include "regex_normalization.hpp"
#include "regex_split.hpp"
#include "string_to_hash_bucket.hpp"
#include "vocab_encoder.hpp"
#include "wordpiece_tokenizer.hpp"
#include "case_fold.hpp"
#include "normalize_unicode.hpp"

using namespace ov;
using namespace ov::op;
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

Output<Node> compute_subgraph_scalar_rank(const Output<Node>& output, element::Type output_type, bool as_scalar) {
    auto shape_of = std::make_shared<ShapeOf>(output, output_type);
    auto rank_of = std::make_shared<ShapeOf>(shape_of, output_type);

    if (as_scalar) {
        auto const_zero = std::make_shared<Constant>(element::i32, Shape{}, 0);
        return std::make_shared<Squeeze>(rank_of, const_zero);
    }
    return rank_of;
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

NamedOutputVector translate_ragged_tensor_to_sparse(const NodeContext& node) {
    // this is custom translator that converts a sub-graph with SentencePieceOp, SentencePieceTokenizer,
    // and RaggedTensorToSparse operation- into a custom operation SentencepieceTokenizerExtensionOp
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 0, "RaggedTensorToSparse expects at least one input.");
    auto node_name = node.get_name();

    // check that producers of RaggedTensorToSparse is SentencePieceTokenizer
    ov::Output<ov::Node> sparse_indices, sparse_values, sparse_dense_shape;
    if (ov::as_type_ptr<ov::op::util::FrameworkNode>(node.get_input(0).get_node_shared_ptr())) {
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
        sparse_indices = sp_tokenizer_ext->output(0);
        sparse_values = sp_tokenizer_ext->output(1);
        sparse_dense_shape = sp_tokenizer_ext->output(2);
    }
    else {
        FRONT_END_GENERAL_CHECK(node.get_input_size() == 2, "RaggedTensorToSparse is supported only for one dimension raggedness");
        auto rt_nested_splits = node.get_input(0);
        auto rt_dense_values = node.get_input(1);

        rt_nested_splits = std::make_shared<Convert>(rt_nested_splits, ov::element::i32);

        // compute vectors of begins and ends
        auto rpt_shape = std::make_shared<ShapeOf>(rt_nested_splits, ov::element::i32)->output(0);
        auto const_one = std::make_shared<Constant>(ov::element::i32, Shape{}, 1);
        auto rpt_shape_minus_one = std::make_shared<Subtract>(rpt_shape, const_one)->output(0);
        auto begins_start = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0);
        auto ends_start = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1);
        auto step = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1);
        auto begins = std::make_shared<Slice>(rt_nested_splits, begins_start, rpt_shape_minus_one, step);
        auto ends = std::make_shared<Slice>(rt_nested_splits, ends_start, rpt_shape, step);
        auto longest_batch = rpt_shape_minus_one;

        // compute the longest row in a tensor
        auto longest_row_size = std::make_shared<Subtract>(ends, begins)->output(0);
        auto reduce_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0);
        longest_row_size = std::make_shared<ReduceMax>(longest_row_size, reduce_axis, true);

        sparse_dense_shape = std::make_shared<Concat>(ov::OutputVector{ longest_batch, longest_row_size }, 0);
        sparse_indices = std::make_shared<RaggedToSparse>(ov::OutputVector{ begins, ends })->output(0);
        sparse_values = rt_dense_values;

        sparse_indices = std::make_shared<Convert>(sparse_indices, ov::element::i64);
        sparse_dense_shape = std::make_shared<Convert>(sparse_dense_shape, ov::element::i64);
    }

    // set tensor names
    sparse_indices.add_names({ node_name + ":0" });
    if (!ov::as_type_ptr<Parameter>(sparse_values.get_node_shared_ptr())) {
        // for a case without SentencePiece tokenizer
        // we must not corrupt input tensor name due to skip connection
        sparse_values.add_names({ node_name + ":1" });
    }
    sparse_dense_shape.add_names({ node_name + ":2" });

    // create named outputs for the conversion extension
    NamedOutputVector named_results;
    named_results.push_back({ "sparse_indices", sparse_indices });
    named_results.push_back({ "sparse_values", sparse_values });
    named_results.push_back({ "sparse_dense_shape", sparse_dense_shape });

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

OutputVector translate_lookup_table_find_op(const ov::frontend::NodeContext& node) {
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 3, "LookupTableFind or LookupTableFindV2 expects 3 inputs");
    auto table_handle = as_type_ptr<ov::frontend::HashTable>(node.get_input_by_reference(0).get_node_shared_ptr());
    FRONT_END_GENERAL_CHECK(table_handle,
        "[TensorFlow Frontend] internal error: LookupTableFind operation expects table_handle by the first input"
    );
    auto keys = node.get_input(1);
    auto default_value = node.get_input(2);

    auto key_type = table_handle->get_key_type();
    auto value_type = default_value.get_element_type();
    FRONT_END_GENERAL_CHECK(
        key_type.is_integral_number() || (key_type == element::string && value_type == element::i64),
        "[TensorFlow Frontend] internal error: LookupTableFind is supported two cases: "
        "1. integer keys with any value type; 2. string keys with i64 values.");

    auto all_keys = table_handle->get_keys();
    auto all_values = table_handle->get_values();

    // reshape both all values and keys to 1D tensor to work it further
    auto target_shape = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{-1});
    all_keys = std::make_shared<Reshape>(all_keys, target_shape, false);
    all_values = std::make_shared<Reshape>(all_values, target_shape, false);

    if (key_type == element::string && value_type.is_integral_number()) {
        // VocabEncoder has limitation that is support of only i32 value type
        // so prepare values format to i32 on inputs
        // and cast output tensor to i64 as required by TensorFlow
        if (value_type != ov::element::i32) {
            default_value = std::make_shared<Convert>(default_value, element::i32);
            all_values = std::make_shared<Convert>(all_values, element::i32);
        }

        // unpack string tensor for required keys and all keys from vocabulary
        ov::OutputVector unpacked_keys = pre_translate_string_tensor_input(keys);
        ov::OutputVector unpacked_all_keys = pre_translate_string_tensor_input(all_keys);

        ov::OutputVector arguments = unpacked_keys;
        arguments.insert(arguments.end(), unpacked_all_keys.begin(), unpacked_all_keys.end());
        arguments.push_back(all_values);
        arguments.push_back(default_value);
        auto tokens = std::make_shared<VocabEncoder>(arguments)->output(0);

        if (value_type != ov::element::i32) {
            tokens = std::make_shared<Convert>(tokens, value_type);
        }

        set_node_name(node.get_name(), tokens.get_node_shared_ptr());
        return { tokens };
    }
    FRONT_END_GENERAL_CHECK(
        key_type != element::string,
        "[TensorFlow Frontend] internal error: LookupTableFind operation with string key is only supported for integral values");

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

NamedOutputVector translate_string_split(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    FRONT_END_GENERAL_CHECK(node.get_input_size() == 2, "StringSplitV2 expects two inputs (1D input and separator)");
    auto input = node.get_input(0);
    ov::OutputVector unpacked_input = pre_translate_string_tensor_input(input);
    auto begins = unpacked_input[0];
    auto sep_const = ov::as_type_ptr<Constant>(node.get_input(1).get_node_shared_ptr());
    FRONT_END_GENERAL_CHECK(sep_const, "[TensorFlow Frontend] internal error: only constant separator is supported for StringSplitV2");
    auto sep_value = sep_const->cast_vector<std::string>();
    FRONT_END_GENERAL_CHECK(sep_value.size() == 1, "[TensorFlow Frontend] inconsistent model: separator must be a scalar");
    auto sep = std::make_shared<Constant>(element::u8, Shape{ sep_value[0].length() }, (const void*)sep_value[0].data())->output(0);
    if (sep_value[0] == "") {
        // default case that means string elements will be removed from leading and trailing white-space
        std::string pattern_value = "^\\s+|\\s+$";
        auto pattern_constant = std::make_shared<Constant>(element::u8, Shape{ pattern_value.length() }, (const void*)pattern_value.data());
        std::string rewrite_value = "";
        auto rewrite_constant = std::make_shared<Constant>(element::u8, Shape{ rewrite_value.length() }, (const void*)rewrite_value.data());
        ov::OutputVector inputs = unpacked_input;
        inputs.push_back(pattern_constant);
        inputs.push_back(rewrite_constant);
        unpacked_input = std::make_shared<RegexNormalization>(inputs, true)->outputs();
        std::string new_sep_value = "[\\s\\p{Zs}]+";
        sep = std::make_shared<Constant>(element::u8, Shape{ new_sep_value.length() }, (const void*)new_sep_value.data());
    }
    auto maxsplit = node.get_attribute<int64_t>("maxsplit", -1);

    // compute batch_dim to generate ragged_begins and ragged_ends for RegexSplit
    auto input_shape = std::make_shared<ShapeOf>(begins, element::i32);
    auto squeeze_axis = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{0});
    auto batch_dim = std::make_shared<Squeeze>(input_shape, squeeze_axis);
    auto zero_const = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{0});
    auto one_const = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{1});
    auto ragged_begins = std::make_shared<Range>(zero_const, batch_dim, one_const, element::i32);
    auto ragged_ends = std::make_shared<Add>(ragged_begins, one_const);

    auto regex_split_outputs = std::make_shared<RegexSplit>(ov::OutputVector{ ragged_begins, ragged_ends, unpacked_input[0],
        unpacked_input[1], unpacked_input[2], sep }, nullptr, nullptr, "remove", false, maxsplit)->outputs();


    // compute sparse tensor indices
    auto indices = std::make_shared<RaggedToSparse>(ov::OutputVector{ regex_split_outputs[0], regex_split_outputs[1] })->output(0);
    indices = std::make_shared<Convert>(indices, element::i64);
    indices.set_names({ node_name + ":0" });

    // compute values of Sparse Tensor of ov::element::string type
    auto values = post_translate_string_tensor_output(ov::OutputVector{ regex_split_outputs[2], regex_split_outputs[3], regex_split_outputs[4] });
    values.set_names({ node_name + ":1" });

    // compute a shape of output tensor in a dense form
    // compute maximum number of string elements per batch in output tensor after split
    auto max_num_per_batch = std::make_shared<Subtract>(regex_split_outputs[1], regex_split_outputs[0])->output(0);
    auto reduction_axes = std::make_shared<Constant>(element::i32, Shape{ 1 }, std::vector<int32_t>{0});
    max_num_per_batch = std::make_shared<ReduceMax>(max_num_per_batch, reduction_axes, true);
    auto shape = std::make_shared<Concat>(ov::OutputVector{ input_shape, max_num_per_batch }, 0)->output(0);
    shape = std::make_shared<Convert>(shape, element::i64);
    shape.set_names({ node_name + ":2" });

    // create named outputs for the conversion extension
    NamedOutputVector named_results;
    named_results.push_back({ "indices", indices });
    named_results.push_back({ "values", values });
    named_results.push_back({ "shape", shape });

    return named_results;
}

ov::OutputVector translate_ragged_tensor_to_tensor(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    auto node_input_size = node.get_input_size();
    FRONT_END_GENERAL_CHECK(node_input_size == 4 || node_input_size == 5,
        "[TensorFlow Frontend] internal error: RaggedTensorToTensor is supported only with one row partition tensor");
    auto shape = node.get_input(0);
    auto values = node.get_input(1);
    auto default_value = node.get_input(2);
    auto row_partition_types = node.get_attribute<std::vector<std::string>>("row_partition_types");
    FRONT_END_GENERAL_CHECK((row_partition_types == std::vector<std::string>{"ROW_SPLITS"}) ||
        (row_partition_types == std::vector<std::string>{"FIRST_DIM_SIZE", "VALUE_ROWIDS"}),
        "[TensorFlow Frontend] internal error: RaggedTensorToTensor is supported only for ROW_SPLITS type");
    // currently we support only shape for 2D tensor in output
    // for example, shape can be equal to [2, 5] or [-1, 8]
    FRONT_END_GENERAL_CHECK(shape.get_partial_shape().is_static() && shape.get_shape() == ov::Shape{ 2 },
        "[TensorFlow Frontend] internal error: RaggedTensorToTensor is supported only for 2D ragged tensor on input");

    // since begins, ends and target shape are expected to be of int32 type
    shape = std::make_shared<Convert>(shape, ov::element::i32);

    ov::Output<ov::Node> begins, ends;
    ov::Output<ov::Node> longest_batch, longest_row_size;
    if (row_partition_types == std::vector<std::string>{"ROW_SPLITS"}) {
        auto row_partition_tensor = node.get_input(3);
        row_partition_tensor = std::make_shared<Convert>(row_partition_tensor, ov::element::i32);

        // compute vectors of begins and ends
        auto rpt_shape = std::make_shared<ShapeOf>(row_partition_tensor, ov::element::i32)->output(0);
        auto const_one = std::make_shared<Constant>(ov::element::i32, Shape{}, 1);
        auto rpt_shape_minus_one = std::make_shared<Subtract>(rpt_shape, const_one)->output(0);
        auto begins_start = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0);
        auto ends_start = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1);
        auto step = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1);
        begins = std::make_shared<Slice>(row_partition_tensor, begins_start, rpt_shape_minus_one, step);
        ends = std::make_shared<Slice>(row_partition_tensor, ends_start, rpt_shape, step);
        longest_batch = rpt_shape_minus_one;

        // since shape can contain -1 dimension that means dimension size will be defined automatically
        // such shape must be adjusted based on other inputs to RaggedTensorToTensor
        // compute the longest row in a tensor
        longest_row_size = std::make_shared<Subtract>(ends, begins)->output(0);
        auto reduce_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0);
        longest_row_size = std::make_shared<ReduceMax>(longest_row_size, reduce_axis, true);
    }
    else {
        auto first_dim_size = node.get_input(3);
        auto value_rowids = node.get_input(4);

        first_dim_size = std::make_shared<Convert>(first_dim_size, ov::element::i32);
        auto new_first_dim_size_shape = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1);
        first_dim_size = std::make_shared<Reshape>(first_dim_size, new_first_dim_size_shape, false);
        value_rowids = std::make_shared<Convert>(value_rowids, ov::element::i32);

        auto ragged_to_ragged = std::make_shared<RaggedToRagged>(ov::OutputVector{ value_rowids , first_dim_size });
        begins = ragged_to_ragged->output(0);
        ends = ragged_to_ragged->output(1);
        longest_batch = first_dim_size;

        // compute longest_row_size
        auto scalar_shape = std::make_shared<Constant>(ov::element::i32, Shape{ 0 }, std::vector<int32_t>{});
        first_dim_size = std::make_shared<Reshape>(first_dim_size, scalar_shape, false);
        auto const_zero = std::make_shared<Constant>(ov::element::i32, Shape{}, 0);
        auto const_one = std::make_shared<Constant>(ov::element::i32, Shape{}, 1);
        auto range_row_ids = std::make_shared<Range>(const_zero, first_dim_size, const_one, ov::element::i32)->output(0);
        auto unsqueeze_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1)->output(0);
        range_row_ids = std::make_shared<Unsqueeze>(range_row_ids, unsqueeze_axis);
        unsqueeze_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0)->output(0);
        value_rowids = std::make_shared<Unsqueeze>(value_rowids, unsqueeze_axis);
        auto mask = std::make_shared<Equal>(range_row_ids, value_rowids)->output(0);
        mask = std::make_shared<Select>(mask, const_one, const_zero);
        auto reduce_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 1)->output(0);
        longest_row_size = std::make_shared<ReduceSum>(mask, reduce_axis, false);
        reduce_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0)->output(0);
        longest_row_size = std::make_shared<ReduceMax>(longest_row_size, reduce_axis, true);
    }

    auto ragged_to_dense = std::make_shared<RaggedToDense>(ov::OutputVector{ begins, ends, values, longest_row_size, default_value })->output(0);

    // adjust shape value since it can contain -1 value that means a dimension must be deduced based on minimal dimension size
    // to store output tensor
    auto replace_shape = std::make_shared<Concat>(ov::OutputVector{ longest_batch, longest_row_size }, 0)->output(0);
    auto const_zero = std::make_shared<Constant>(ov::element::i32, Shape{}, 0);
    auto shape_less_zero = std::make_shared<Less>(shape, const_zero);
    shape = std::make_shared<Select>(shape_less_zero, replace_shape, shape);

    auto pads_begin = std::make_shared<Constant>(ov::element::i32, Shape{ 2 }, std::vector<int32_t>{0, 0});
    // note that replace_shape to be equal a shape of ragged_to_dense
    // Pad operation removes (or crops) if padding number is negative
    auto pads_end = std::make_shared<Subtract>(shape, replace_shape);
    auto squeeze_axis = std::make_shared<Constant>(ov::element::i32, Shape{ 1 }, 0);
    auto pad_value = std::make_shared<Squeeze>(default_value, squeeze_axis);
    auto result_dense_tensor = std::make_shared<Pad>(ragged_to_dense, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT)->output(0);

    result_dense_tensor.get_node_shared_ptr()->set_friendly_name(node_name);
    result_dense_tensor.set_names({ node_name + ":0" });

    return { result_dense_tensor };
}

ov::OutputVector translate_equal(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    auto node_input_size = node.get_input_size();
    FRONT_END_GENERAL_CHECK(node_input_size == 2,
        "[TensorFlow Frontend] inconsistent model: Equal must have two inputs");
    auto input1 = node.get_input(0);
    auto input2 = node.get_input(1);

    ov::Output<ov::Node> result;
    if (input1.get_element_type() == ov::element::string ||
        input2.get_element_type() == ov::element::string) {
        ov::OutputVector unpacked_input1 = pre_translate_string_tensor_input(input1);
        ov::OutputVector unpacked_input2 = pre_translate_string_tensor_input(input2);
        ov::OutputVector inputs = unpacked_input1;
        inputs.insert(inputs.end(), unpacked_input2.begin(), unpacked_input2.end());

        auto equal_str = std::make_shared<EqualStr>(inputs)->output(0);
        result = std::make_shared<Convert>(equal_str, element::boolean);
    }
    else {
        result = std::make_shared<Equal>(input1, input2)->output(0);
    }

    result.get_node_shared_ptr()->set_friendly_name(node_name);
    result.set_names({ node_name + ":0" });

    return { result };
}

ov::OutputVector translate_string_to_hash_bucket_fast(const ov::frontend::NodeContext& node) {
    auto node_name = node.get_name();
    auto node_input_size = node.get_input_size();
    FRONT_END_GENERAL_CHECK(node_input_size == 1,
        "[TensorFlow Frontend] inconsistent model: StringToHashBucketFast must have one input");
    auto input = node.get_input(0);
    auto num_buckets = node.get_attribute<int64_t>("num_buckets");
    FRONT_END_GENERAL_CHECK(num_buckets > 0,
        "[TensorFlow Frontend] inconsistent model: num_buckets for StringToHashBucketFast must be positive");

    ov::OutputVector unpacked_input = pre_translate_string_tensor_input(input);
    ov::Output<ov::Node> result = std::make_shared<StringToHashBucket>(unpacked_input, num_buckets);

    result.get_node_shared_ptr()->set_friendly_name(node_name);
    result.set_names({ node_name + ":0" });
    return { result };
}

OutputVector translate_squeeze_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    auto node_name = node.get_name();
    std::vector<int64_t> axes;
    if (node.has_attribute("axis")) {
        axes = node.get_attribute<std::vector<int64_t>>("axis", {});
    }
    else {
        // check deprecated name
        axes = node.get_attribute<std::vector<int64_t>>("squeeze_dims", {});
    }
    auto axis_const = std::make_shared<Constant>(element::i32, Shape{ axes.size() }, axes);

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->input_value(0);

        auto input_rank = compute_subgraph_scalar_rank(input, element::i32, true);
        auto const_one = std::make_shared<Constant>(element::i32, Shape{}, 1);
        auto input_rank_minus_one = std::make_shared<Subtract>(input_rank, const_one)->output(0);

        // adjust axis to make them non-negative
        auto axis_complex = std::make_shared<FloorMod>(axis_const, input_rank_minus_one);

        auto squeeze = std::make_shared<Squeeze>(input, axis_complex);
        set_node_name(node_name, squeeze);
        auto squeeze_complex = std::make_shared<ComplexTypeMark>(squeeze, complex_part_type);
        return { squeeze_complex->output(0) };
    }
    else if (input.get_element_type() == element::string) {
        ov::OutputVector unpacked_input = pre_translate_string_tensor_input(input);
        auto begins = unpacked_input[0];
        auto ends = unpacked_input[1];
        auto chars = unpacked_input[2];

        // squeeze begins and ends by given dimensions
        begins = std::make_shared<Squeeze>(begins, axis_const);
        ends = std::make_shared<Squeeze>(ends, axis_const);

        auto string_pack_result = post_translate_string_tensor_output(OutputVector{ begins, ends, chars });
        set_node_name(node_name, string_pack_result.get_node_shared_ptr());
        return { string_pack_result };
    }

    auto squeeze = std::make_shared<Squeeze>(input, axis_const);
    set_node_name(node_name, squeeze);
    return { squeeze };
}
