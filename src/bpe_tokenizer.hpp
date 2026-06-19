// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include "absl/container/flat_hash_map.h"
#include "utils.hpp"

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#undef tokenizer
#undef m_tokenizer

using TextMerges = std::vector<std::pair<std::string, std::string>>;

// Hash for an ordered pair of token ids. The bundled absl shim aliases
// flat_hash_map to std::unordered_map<K, V, std::hash<K>>, and std has no
// std::hash specialization for std::pair, so the hasher must be explicit.
struct TokenPairHash {
    std::size_t operator()(const std::pair<int32_t, int32_t>& pair) const {
        return (static_cast<std::size_t>(static_cast<uint32_t>(pair.first)) << 32)
             | static_cast<std::size_t>(static_cast<uint32_t>(pair.second));
    }
};

using Merges = absl::flat_hash_map<std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>, TokenPairHash>;
using Vocab = std::unordered_map<std::string, unsigned int>;

// A single symbol in the word being tokenized. The word is held as one
// contiguous vector of Symbols forming a doubly linked list through integer
// indices (-1 == no neighbor). Merges append a new Symbol and mark the two
// operands dead, so indices are stable for the lifetime of a tokenize() call.
struct BPESymbol {
    int32_t id;     // token id
    int32_t prev;   // index of previous live symbol, or -1
    int32_t next;   // index of next live symbol, or -1
    bool alive;
};

class BPETokenizerImpl {
private:
    Vocab m_vocab;
    Merges m_merges;
    std::shared_ptr<Trie> m_trie;
    std::string m_suffix_indicator;
    std::string m_end_suffix;
    bool m_byte_fallback = false;
    int32_t m_unk_token_id = -1;
    bool m_fuse_unk = false;
    size_t m_cache_capacity;
    std::mutex m_mutex;
    std::unordered_map<std::string, std::vector<int32_t>> m_cache;
public:
    BPETokenizerImpl(Vocab vocab, Merges merges): m_vocab(vocab), m_merges(merges) {};
    BPETokenizerImpl(
        const Vocab& vocab, const TextMerges& merges, 
        size_t cache_capacity,
        std::string unk_token,
        std::string suffix_indicator,
        std::string end_suffix,
        bool fuse_unk = false,
        bool byte_fallback = false
    );
    std::vector<int32_t> tokenize(std::string& text);
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
        bool byte_fallback = false,
        size_t cache_capacity = 20000
    ) :
        ov::op::Op(arguments),
        m_tokenizer(tokenizer),
        m_added_tokens(added_tokens),
        m_unk_token(unk_token),
        m_fuse_unk(fuse_unk),
        m_suffix_indicator(suffix_indicator),
        m_end_suffix(end_suffix),
        m_byte_fallback(byte_fallback),
        m_cache_capacity(cache_capacity) {

        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BPETokenizer>(inputs, m_tokenizer, m_added_tokens, m_unk_token, m_fuse_unk, 
                                              m_suffix_indicator, m_end_suffix, m_byte_fallback, m_cache_capacity);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("unk_token", m_unk_token);
        visitor.on_attribute("fuse_unk", m_fuse_unk);
        visitor.on_attribute("suffix_indicator", m_suffix_indicator);
        visitor.on_attribute("end_suffix", m_end_suffix);
        visitor.on_attribute("byte_fallback", m_byte_fallback);
        visitor.on_attribute("cache_capacity", m_cache_capacity);
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
    size_t m_cache_capacity = 20000;
    mutable std::mutex m_mutex;
};
