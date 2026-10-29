// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>
#include <openvino/runtime/tensor.hpp>
#include <openvino/frontend/node_context.hpp>
#include <pcre2.h>
#include "absl/strings/string_view.h"

#ifndef OPENVINO_ELEMENT_STRING_SUPPORTED
    #define OPENVINO_ELEMENT_STRING_SUPPORTED 0
#endif

#ifndef OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK
    #define OPENVINO_USE_INPUT_OUTPUT_STRING_TENSOR_HACK 0
#endif

#define USE_STRING_TENSORS 0    // modify this depending on willingness to use explicit string tensors

#if USE_STRING_TENSORS && !OPENVINO_ELEMENT_STRING_SUPPORTED
    #error "USE_STRING_TENSORS = 1 can be used only when OpenVINO supports element::string that is determined by OPENVINO_ELEMENT_STRING_SUPPORTED == 1"
#endif

#define SENTENCE_PIECE_EXTENSION_DECOMPOSED_STRINGS 0


void parse_packed_strings (
    const ov::Tensor& packed,
    int32_t& batch_size,
    const int32_t*& begin_ids,
    const int32_t*& end_ids,
    const uint8_t*& symbols);


void check_string_input(const ov::Node* node, size_t input_index);

void check_string_scalar_input(const ov::Node* node, size_t input_index);

void check_ragged_input(const ov::Node* node, size_t input_index);

void check_ragged_input_any_rank_data(const ov::Node* node, size_t input_index);

void check_ragged_string_input(const ov::Node* node, size_t input_index);

void set_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_string_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape);

void set_ragged_output(ov::Node* node, size_t output_index, const ov::PartialShape& shape, ov::element::Type type);

void unpack_strings_to_tensors(const std::string* strings, const ov::Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars);

void override_parameter (std::shared_ptr<ov::Node> node, ov::element::Type type, const ov::PartialShape& shape);

ov::OutputVector pre_translate_string_tensor_input(const ov::Output<ov::Node>& input);

ov::OutputVector pre_translate_ragged_tensor_input(ov::Output<ov::Node> input);

ov::OutputVector pre_translate_ragged_string_tensor_input(ov::Output<ov::Node> input);

ov::Output<ov::Node> post_translate_string_tensor_output(const ov::OutputVector& outputs);

ov::Output<ov::Node> post_translate_ragged_tensor_output(const ov::OutputVector& outputs);

bool evaluate_normalization_helper (
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs,
    std::function<std::string(const std::string&)> normalizer,
    const bool has_skips = false);

std::shared_ptr<ov::Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name);

void set_node_name(const std::string& node_name, const std::shared_ptr<ov::Node>& node);

class PCRE2Wrapper {
    public:
        class MatchData {
            public:
                explicit MatchData(const PCRE2Wrapper& wrapper);
                MatchData(const MatchData&) = delete;
                MatchData& operator=(const MatchData&) = delete;
                MatchData(MatchData&& other) noexcept;
                MatchData& operator=(MatchData&& other) noexcept;
                ~MatchData();

                pcre2_match_data* get() const;

            private:
                pcre2_match_data* m_match_data = nullptr;
        };

        pcre2_code* m_compiled = nullptr;
        PCRE2Wrapper(const absl::string_view& pattern);
        std::string substitute(const std::string& orig_str, const absl::string_view& replace_pattern, bool global_replace) const;
        MatchData create_match_data() const;
        std::pair<size_t, size_t> match(const std::string& orig_str, size_t curr_start) const;
        std::pair<size_t, size_t> match(const std::string_view& str, size_t curr_start) const;
        std::pair<size_t, size_t> match(const std::string_view& str, size_t curr_start, MatchData& match_data) const;
        // Return both full-match offsets and capture-group offsets in one call.
        // Returns {{full_begin, full_end}, {group_begin, group_end}} or {{SIZE_MAX,SIZE_MAX},{SIZE_MAX,SIZE_MAX}} on failure.
        std::pair<std::pair<size_t,size_t>, std::pair<size_t,size_t>> match_and_find_group(const std::string& orig_str, size_t curr_start) const;
        ~PCRE2Wrapper();
    private:
        bool m_is_jit = 0;
};

// Longest-prefix-match lookup table over byte strings.
//
// Layout notes. The first byte is resolved through a flat 256-entry table
// (`m_root`) instead of a binary search, and every deeper node lives in one
// contiguous pool (`m_nodes`) addressed by index rather than through
// `std::unique_ptr`. A depth-1 terminal whose node has no children needs no
// pool access at all, which is the overwhelmingly common case for byte-level
// BPE vocabularies (256 single-byte entries plus a few hundred longer ones).
// Deeper walks, which WordPiece needs, keep a sorted per-node child array but
// with 8-byte entries in pooled nodes instead of 16-byte entries pointing at
// scattered heap allocations.
class Trie {
    public:
        Trie();

        void add(const std::vector<unsigned char>& str, const int value, int idx = 0);

        int find_longest(const std::vector<unsigned char>& str, int& idx) const {
            return find_longest_impl(str.data(), static_cast<int>(str.size()), idx);
        }
        int find_longest(const std::string_view& str, int& idx) const {
            return find_longest_impl(
                reinterpret_cast<const unsigned char*>(str.data()), static_cast<int>(str.size()), idx);
        }

    private:
        // Depth-1 dispatch entry. `value` is the token id of the one-byte key, or
        // -1 when that byte is not a key. `child` is the pool index of the node to
        // continue the walk from, or -1 when nothing longer starts with this byte.
        // Both being -1 means the byte is absent from the table entirely.
        struct RootSlot {
            int32_t value;
            int32_t child;
        };
        struct Child {
            int32_t node;
            unsigned char key;
        };
        struct Node {
            int32_t value = -1;  // -1 for unset value
            std::vector<Child> children;  // sorted by key
        };

        // Hot path: one indexed load, no search, no pointer chase. Kept in the
        // header so the depth-1 case inlines into the tokenizer loops.
        //
        // `idx >= size` returns -1 and leaves `idx` alone. The previous
        // implementation read `str[idx]` unconditionally, so it relied on every
        // caller checking that itself; all four do, and this is a safe superset.
        int find_longest_impl(const unsigned char* data, int size, int& idx) const {
            if (idx >= size) {
                return -1;
            }
            const RootSlot slot = m_root[data[idx]];
            if (slot.child < 0) {
                // No longer key starts with this byte, so the depth-1 entry, present
                // or not, is already the answer.
                idx += (slot.value != -1);
                return slot.value;
            }
            return find_longest_deep(data, size, idx);
        }

        int find_longest_deep(const unsigned char* data, int size, int& idx) const;
        static int32_t find_child(const std::vector<Child>& children, unsigned char ch);

        RootSlot m_root[256];
        std::vector<Node> m_nodes;
        // Pool index of the depth-1 node per first byte, or -1 when that byte
        // begins no key. Used only while building: `m_root[ch].child` mirrors it
        // but deliberately stays -1 until that node actually gains a child, which
        // is what keeps depth-1 terminals off the deep path.
        int32_t m_root_node[256];
        // Value of the empty key. The previous implementation stored it on the root
        // node, where find_longest never read it; retained for the same reason.
        int32_t m_empty_value = -1;
};

bool getenv_bool(const char* env_var, bool default_value);
