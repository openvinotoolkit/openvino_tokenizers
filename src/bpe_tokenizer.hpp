// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <openvino/op/op.hpp>
#include <mutex>
#include <shared_mutex>
#include "absl/container/flat_hash_map.h"
#include "utils.hpp"

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#undef tokenizer
#undef m_tokenizer

using TextMerges = std::vector<std::pair<std::string, std::string>>;

// Rank (position in the merge list) and resulting token id for a merge pair.
struct MergeValue {
    int32_t rank;
    int32_t new_id;
};

// Open-addressing hash map from a packed (left_id, right_id) token pair to its
// merge rank and result. The vocabulary uses absl::flat_hash_map, but in this
// build that is only a shim over std::unordered_map (see third_party/absl),
// i.e. a node-based chained table. The merges map is the hottest BPE structure
// -- probed for every adjacent pair and again after every merge -- so a single
// contiguous open-addressed array (linear probing, no per-node allocation)
// gives a large cache-locality win over pointer-chasing buckets.
class MergesMap {
public:
    void reserve(size_t num_entries) {
        // Size to the next power of two with load factor < ~0.7.
        size_t needed = static_cast<size_t>(num_entries / 0.7) + 1;
        size_t capacity = 1;
        while (capacity < needed) {
            capacity <<= 1;
        }
        if (capacity < 8) {
            capacity = 8;
        }
        m_slots.clear();
        m_slots.resize(capacity);
        m_mask = capacity - 1;
        m_size = 0;
    }

    void insert(int32_t left, int32_t right, MergeValue value) {
        const uint64_t key = pack(left, right);
        size_t idx = hash(key) & m_mask;
        while (m_slots[idx].occupied) {
            if (m_slots[idx].key == key) {
                m_slots[idx].value = value;
                return;
            }
            idx = (idx + 1) & m_mask;
        }
        Slot& slot = m_slots[idx];
        slot.key = key;
        slot.value = value;
        slot.occupied = true;
        ++m_size;
    }

    // Returns a pointer to the merge value for (left, right), or nullptr.
    const MergeValue* find(int32_t left, int32_t right) const {
        if (m_slots.empty()) {
            return nullptr;
        }
        const uint64_t key = pack(left, right);
        size_t idx = hash(key) & m_mask;
        while (m_slots[idx].occupied) {
            if (m_slots[idx].key == key) {
                return &m_slots[idx].value;
            }
            idx = (idx + 1) & m_mask;
        }
        return nullptr;
    }

    size_t size() const { return m_size; }

private:
    struct Slot {
        uint64_t key = 0;
        MergeValue value{};
        bool occupied = false;
    };

    static uint64_t pack(int32_t left, int32_t right) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32)
             | static_cast<uint64_t>(static_cast<uint32_t>(right));
    }

    // Fibonacci/multiplicative hash; the packed integer key is already dense so
    // a full string-style hash is unnecessary.
    static size_t hash(uint64_t key) {
        key *= 0x9E3779B97F4A7C15ULL;
        return static_cast<size_t>(key >> 32);
    }

    std::vector<Slot> m_slots;
    size_t m_mask = 0;
    size_t m_size = 0;
};

using Merges = MergesMap;
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

using BPEQueueEntry = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>;

struct BPETokenizerScratch {
    std::vector<BPESymbol> symbols;
    std::vector<BPEQueueEntry> queue_storage;
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
    std::shared_mutex m_mutex;
    std::unordered_map<std::string, std::vector<int32_t>> m_cache;
public:
    BPETokenizerImpl(Vocab vocab, Merges merges): m_vocab(std::move(vocab)), m_merges(std::move(merges)) {};
    BPETokenizerImpl(
        Vocab vocab, const TextMerges& merges,
        size_t cache_capacity,
        const std::string& unk_token,
        std::string suffix_indicator,
        std::string end_suffix,
        bool fuse_unk = false,
        bool byte_fallback = false
    );
    std::vector<int32_t> tokenize(std::string& text);
    void tokenize_into(std::string_view text, std::vector<int32_t>& out);
    void tokenize_into(std::string_view text, std::vector<int32_t>& out, BPETokenizerScratch& scratch);
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
    mutable std::once_flag m_init_flag;
};
