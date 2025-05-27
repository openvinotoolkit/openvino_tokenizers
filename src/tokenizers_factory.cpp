// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tokenizers_factory.hpp"

#include "openvino/core/except.hpp"
#include "tokenizer.hpp"

namespace ov {
namespace tokenizers {

namespace {
template <typename T>
T get_attribute_value(const ov::AnyMap& attributes, const std::string& attribute_name, const T& default_value) {
    return attributes.count(attribute_name) && attributes.at(attribute_name).is<T>()
               ? attributes.at(attribute_name).as<T>()
               : default_value;
}

}  // namespace

ov::OutputVector create_tokenizer_node(const std::string& op_type,
                                       const ov::OutputVector& inputs,
                                       const ov::AnyMap& attributes) {
    if (op_type == "StringTensorUnpack") {
        return std::make_shared<StringTensorUnpack>(inputs)->outputs();
    } else if (op_type == "SpecialTokensSplit") {
        return std::make_shared<SpecialTokensSplit>(inputs)->outputs();
    } else if (op_type == "RegexSplit") {
        auto behaviour = get_attribute_value<std::string>(attributes, "behaviour", "remove");
        auto invert = get_attribute_value<bool>(attributes, "invert", false);
        return std::make_shared<RegexSplit>(inputs, behaviour, invert)->outputs();
    } else if (op_type == "RaggedToDense") {
        auto pad_right = get_attribute_value<bool>(attributes, "pad_right", true);
        auto pad_max_length = get_attribute_value<bool>(attributes, "pad_max_length", false);
        return std::make_shared<RaggedToDense>(inputs, pad_right, pad_max_length)->outputs();
    } else if (op_type == "VocabDecoder") {
        return std::make_shared<VocabDecoder>(inputs, std::vector<int32_t>{})->outputs();
    } else if (op_type == "FuzeRagged") {
        return std::make_shared<FuzeRagged>(inputs)->outputs();
    } else if (op_type == "StringTensorPack") {
        return std::make_shared<StringTensorPack>(inputs)->outputs();
    } else if (op_type == "BPETokenizer") {
        auto unk_token = get_attribute_value<std::string>(attributes, "unk_token", "");
        auto fuse_unk = get_attribute_value<bool>(attributes, "fuse_unk", false);
        auto suffix_indicator = get_attribute_value<std::string>(attributes, "suffix_indicator", "");
        auto end_suffix = get_attribute_value<std::string>(attributes, "end_suffix", "");
        auto byte_fallback = get_attribute_value<bool>(attributes, "byte_fallback", false);
        return std::make_shared<BPETokenizer>(inputs, unk_token, fuse_unk, suffix_indicator, end_suffix, byte_fallback)
            ->outputs();
    }
    OPENVINO_THROW("Unsupported operation type: `", op_type, "`");
}

}  // namespace tokenizers
}  // namespace ov
