// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "tokenizer.hpp"

#define OPENVINO_TOKENIZERS_TENSORFLOW_CONVERSION_EXTENSIONS                                                                                     \
     std::make_shared<ov::frontend::ConversionExtension>("WordpieceTokenizeWithOffsets", translate_wordpiece_tokenize_with_offsets),  \
    std::make_shared<ov::frontend::ConversionExtension>("RegexSplitWithOffsets", translate_regex_split_with_offsets),                \
    std::make_shared<ov::frontend::ConversionExtension>("NormalizeUTF8", translate_normalize_utf8),                                  \
    std::make_shared<ov::frontend::ConversionExtension>("CaseFoldUTF8", translate_case_fold_utf8),                                   \
    std::make_shared<ov::frontend::ConversionExtension>("SentencepieceOp", translate_sentencepiece_op),                              \
    std::make_shared<ov::frontend::ConversionExtension>("RaggedTensorToSparse", translate_ragged_tensor_to_sparse),                  \
    std::make_shared<ov::frontend::ConversionExtension>("StringLower", translate_string_lower),                                      \
    std::make_shared<ov::frontend::ConversionExtension>("StaticRegexReplace", translate_static_regex_replace),                       \
    std::make_shared<ov::frontend::ConversionExtension>("LookupTableFind", translate_lookup_table_find_op),                          \
    std::make_shared<ov::frontend::ConversionExtension>("LookupTableFindV2", translate_lookup_table_find_op),                        \
    std::make_shared<ov::frontend::ConversionExtension>("StringSplitV2", translate_string_split),                                    \
    std::make_shared<ov::frontend::ConversionExtension>("RaggedTensorToTensor", translate_ragged_tensor_to_tensor),                  \
    std::make_shared<ov::frontend::ConversionExtension>("Equal", translate_equal),                                                   \
    std::make_shared<ov::frontend::ConversionExtension>("StringToHashBucketFast", translate_string_to_hash_bucket_fast)

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
            std::make_shared<ov::OpExtension<StringTensorPack>>(),
            std::make_shared<ov::OpExtension<RaggedTensorPack>>(),
            std::make_shared<ov::OpExtension<StringTensorUnpack>>(),
            std::make_shared<ov::OpExtension<CaseFold>>(),
            std::make_shared<ov::OpExtension<EqualStr>>(),
            std::make_shared<ov::OpExtension<NormalizeUnicode>>(),
            std::make_shared<ov::OpExtension<RegexNormalization>>(),
            std::make_shared<ov::OpExtension<RegexSplit>>(),
            std::make_shared<ov::OpExtension<WordpieceTokenizer>>(),
            std::make_shared<ov::OpExtension<BPETokenizer>>(),
            std::make_shared<ov::OpExtension<BytesToChars>>(),
            std::make_shared<ov::OpExtension<CombineSegments>>(),
            std::make_shared<ov::OpExtension<RaggedToDense>>(),
            std::make_shared<ov::OpExtension<RaggedToSparse>>(),
            std::make_shared<ov::OpExtension<RaggedToRagged>>(),
            std::make_shared<ov::OpExtension<StringToHashBucket>>(),
            std::make_shared<ov::OpExtension<VocabEncoder>>(),
            std::make_shared<ov::OpExtension<VocabDecoder>>(),
            std::make_shared<ov::OpExtension<CharsToBytes>>(),
            std::make_shared<ov::OpExtension<TrieTokenizer>>(),
            std::make_shared<ov::OpExtension<FuzeRagged>>(),
            std::make_shared<ov::OpExtension<ByteFallback>>(),
            std::make_shared<ov::OpExtension<TemplateExtension::SentencepieceTokenizer>>(),
            std::make_shared<ov::OpExtension<TemplateExtension::SentencepieceDetokenizer>>(),
            std::make_shared<ov::OpExtension<TemplateExtension::SentencepieceStreamDetokenizer>>(),
            OPENVINO_TOKENIZERS_TENSORFLOW_CONVERSION_EXTENSIONS
}));
//! [ov_extension:entry_point]
// clang-format on
