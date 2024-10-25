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
using Merges = std::map<std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>>;
using Vocab = std::unordered_map<std::string, unsigned int>;

template <typename T = int32_t>
class TokensList {
public:
    struct Node {
        T data;
        std::shared_ptr<Node> prev;
        std::shared_ptr<Node> next;
        Node(const T& data) : data(data), prev(nullptr), next(nullptr) {}
    };

    size_t m_size;

public:
    size_t size() const {
        return m_size;
    }

    std::shared_ptr<Node> head;
    std::shared_ptr<Node> tail;

    TokensList() : head(nullptr), tail(nullptr), m_size(0) {}

    ~TokensList() {
        while (head) {
            head = head->next;
        }
    }

    void insert(const T& data) {
        std::shared_ptr<Node> new_node = std::make_shared<Node>(data);
        if (!head) {
            head = tail = new_node;
        } else {
            tail->next = new_node;
            new_node->prev = tail;
            tail = new_node;
        }
        m_size++;
    }

    std::shared_ptr<Node> merge_neighbors(std::shared_ptr<Node> first, std::shared_ptr<Node> second, const T& new_data) {
        // OPENVINO_ASSERT(!first || !second || first->next != second);
        // OPENVINO_THROW("Nodes must be consecutive and non-null");

        std::shared_ptr<Node> new_node = std::make_shared<Node>(new_data);

        new_node->prev = first->prev;
        new_node->next = second->next;

        if (first->prev) {
            first->prev->next = new_node;
        } else {
            head = new_node;
        }

        if (second->next) {
            second->next->prev = new_node;
        } else {
            tail = new_node;
        }

        // No need to delete first and second as shared_ptr will handle it
        m_size -= 1;
        return new_node;
    }
};

// Define a custom hash function for std::pair
struct NodePairHash {
    std::size_t operator()(const std::pair<std::shared_ptr<TokensList<int32_t>::Node>, std::shared_ptr<TokensList<int32_t>::Node>>& pair) const {
        auto hash1 = std::hash<std::shared_ptr<TokensList<int32_t>::Node>>{}(pair.first);
        auto hash2 = std::hash<std::shared_ptr<TokensList<int32_t>::Node>>{}(pair.second);
        return hash1 ^ (hash2 << 1);  // Combine the two hash values
    }
};

// Define a custom equality function for std::pair
struct NodePairEqual {
    bool operator()(const std::pair<std::shared_ptr<TokensList<int32_t>::Node>, std::shared_ptr<TokensList<int32_t>::Node>>& lhs,
                    const std::pair<std::shared_ptr<TokensList<int32_t>::Node>, std::shared_ptr<TokensList<int32_t>::Node>>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

using TokenNode = std::shared_ptr<TokensList<int32_t>::Node>;

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
