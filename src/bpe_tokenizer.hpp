// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include "utils.hpp"

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#undef tokenizer
#undef m_tokenizer

using TextMerges = std::vector<std::pair<std::string, std::string>>;
using Merges = std::map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>>;
using Vocab = std::unordered_map<std::string, unsigned int>;
using Tokens = std::vector<int64_t>;

class BPETokenizerImpl {
private:
    Vocab m_vocab;
    Vocab m_old_vocab;
    Merges m_merges;
    std::shared_ptr<Trie> m_trie;
    std::pair<std::pair<int64_t, int64_t>, size_t> get_min_rank_pair(Tokens tokens);
public:
    BPETokenizerImpl(Vocab vocab, Merges merges): m_vocab(vocab), m_merges(merges) {};
    BPETokenizerImpl(const Vocab& vocab, const TextMerges& merges);
    Tokens tokenize(std::string& text);
};


class BPETokenizer : public ov::op::Op {
public:
    OPENVINO_OP("BPETokenizer");

    BPETokenizer () = default;
    BPETokenizer(
        const ov::OutputVector& arguments,
        const std::string& unk_token = "",
        bool fuse_unk = false,
        const std::string& suffix_indicator = "",
        const std::string& end_suffix = "",
        bool byte_fallback = false
    ) :
        ov::op::Op(arguments),
        m_unk_token(unk_token),
        m_fuse_unk(fuse_unk),
        m_suffix_indicator(suffix_indicator),
        m_end_suffix(end_suffix),
        m_byte_fallback(byte_fallback) {

        constructor_validate_and_infer_types();
    }
    BPETokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<BPETokenizerImpl>& tokenizer,
        const std::shared_ptr<std::map<std::string, int32_t>>& added_tokens,
        const std::string& unk_token = "",
        bool fuse_unk = false,
        const std::string& suffix_indicator = "",
        const std::string& end_suffix = "",
        bool byte_fallback = false
    ) :
        ov::op::Op(arguments),
        m_tokenizer(tokenizer),
        m_added_tokens(added_tokens),
        m_unk_token(unk_token),
        m_fuse_unk(fuse_unk),
        m_suffix_indicator(suffix_indicator),
        m_end_suffix(end_suffix),
        m_byte_fallback(byte_fallback) {

        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BPETokenizer>(inputs, m_tokenizer, m_added_tokens, m_unk_token, m_fuse_unk, m_suffix_indicator, m_end_suffix, m_byte_fallback);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("unk_token", m_unk_token);
        visitor.on_attribute("fuse_unk", m_fuse_unk);
        visitor.on_attribute("suffix_indicator", m_suffix_indicator);
        visitor.on_attribute("end_suffix", m_end_suffix);
        visitor.on_attribute("byte_fallback", m_byte_fallback);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    mutable std::shared_ptr<BPETokenizerImpl> m_tokenizer;
    mutable std::shared_ptr<std::map<std::string, int32_t>> m_added_tokens;
    std::string m_unk_token;
    bool m_fuse_unk = false;
    std::string m_suffix_indicator;
    std::string m_end_suffix;
    bool m_byte_fallback = false;
};
