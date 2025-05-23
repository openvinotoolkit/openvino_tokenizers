// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tokenizers_factory.hpp"

#include "tokenizer.hpp"

namespace ov {
namespace tokenizers {

ov::OutputVector create_tokenizer_node(const std::string& op_type,
                                       const ov::OutputVector& inputs,
                                       const ov::AnyMap& attributes) {
    if (op_type == "StringTensorUnpack") {
        return std::make_shared<StringTensorUnpack>(inputs)->outputs();
    } else if (op_type == "SpecialTokensSplit") {
        return std::make_shared<SpecialTokensSplit>(inputs)->outputs();
    } else if (op_type == "RegexSplit") {
        auto behaviour = attributes.at("behaviour").as<std::string>();
        auto invert = attributes.at("invert").as<bool>();
        return std::make_shared<RegexSplit>(inputs, behaviour, invert)->outputs();
    } else if (op_type == "RaggedToDense") {
        auto pad_right = attributes.at("pad_right").as<bool>();
        auto pad_max_length = attributes.at("pad_max_length").as<bool>();
        return std::make_shared<RaggedToDense>(inputs, pad_right, pad_max_length)->outputs();
    } else if (op_type == "VocabDecoder") {
        return std::make_shared<VocabDecoder>(inputs, std::vector<int32_t>{})->outputs();
    } else if (op_type == "FuzeRagged") {
        return std::make_shared<FuzeRagged>(inputs)->outputs();
    } else if (op_type == "StringTensorPack") {
        return std::make_shared<StringTensorPack>(inputs)->outputs();
    } else if (op_type == "BPETokenizer") {
        auto unk_token = attributes.at("unk_token").as<std::string>();
        auto fuse_unk = attributes.at("fuse_unk").as<bool>();
        auto suffix_indicator = attributes.at("suffix_indicator").as<std::string>();
        auto end_suffix = attributes.at("end_suffix").as<std::string>();
        auto byte_fallback = attributes.at("byte_fallback").as<bool>();

        return std::make_shared<BPETokenizer>(inputs, unk_token, fuse_unk, suffix_indicator, end_suffix, byte_fallback)
            ->outputs();
    }
    return ov::OutputVector{};
}

}  // namespace tokenizers
}  // namespace ov
