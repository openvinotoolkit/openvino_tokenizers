// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Fingerprint matches are confirmed by a full key comparison, so truncation
// affects collision frequency but not correctness.
#ifndef BPE_CACHE_FINGERPRINT_BITS
#    define BPE_CACHE_FINGERPRINT_BITS 32
#endif

// Cache from a pretoken to its BPE token-id sequence.
// Values of up to four token ids are stored inline; longer values use stable,
// append-only arena storage.
class BPEResultCache {
public:
    enum class InsertResult {
        INSERTED,
        EXISTING,
        SATURATED,
    };

    // Valid while the caller holds the lock used for lookup. An empty stored
    // value has non-null data and size zero.
    struct ValueView {
        const int32_t* data = nullptr;
        size_t size = 0;

        const int32_t* begin() const {
            return data;
        }
        const int32_t* end() const {
            return data + size;
        }
        bool found() const {
            return data != nullptr;
        }
    };

    explicit BPEResultCache(size_t max_entries = 0) : m_max_entries(max_entries) {
        if (max_entries == 0) {
            return;
        }

        size_t slot_count = 1;
        while (max_entries > max_entries_for_slots(slot_count)) {
            if (slot_count > std::numeric_limits<size_t>::max() / 2) {
                throw std::length_error("BPE result cache capacity is too large");
            }
            slot_count *= 2;
        }
        m_slots.resize(slot_count);
        m_mask = slot_count - 1;
    }

    // Slots contain pointers into this cache's spill arena.
    BPEResultCache(const BPEResultCache&) = delete;
    BPEResultCache& operator=(const BPEResultCache&) = delete;

    static uint64_t hash(std::string_view key) {
        uint64_t value = 14695981039346656037ULL;
        for (const unsigned char byte : key) {
            value ^= byte;
            value *= 1099511628211ULL;
        }
        return value;
    }

    ValueView find(std::string_view key, uint64_t hash_value) const {
        if (m_slots.empty()) {
            return ValueView{};
        }

        const Fingerprint fingerprint = stored_fingerprint(hash_value);
        size_t slot_idx = static_cast<size_t>(hash_value) & m_mask;
        for (size_t probes = 0; probes < m_slots.size(); ++probes) {
            const Slot& slot = m_slots[slot_idx];
            if (slot.fingerprint == 0) {
                return ValueView{};
            }
            if (slot.fingerprint == fingerprint && keys_equal(slot.key, key)) {
                return ValueView{slot.tokens(), slot.count};
            }
            slot_idx = (slot_idx + 1) & m_mask;
        }
        return ValueView{};
    }

    ValueView find(std::string_view key) const {
        return find(key, hash(key));
    }

    template <typename Iterator>
    InsertResult insert(std::string_view key, uint64_t hash_value, Iterator value_begin, Iterator value_end) {
        if (m_size >= m_max_entries || m_slots.empty()) {
            return InsertResult::SATURATED;
        }

        const Fingerprint fingerprint = stored_fingerprint(hash_value);
        size_t slot_idx = static_cast<size_t>(hash_value) & m_mask;
        for (size_t probes = 0; probes < m_slots.size(); ++probes) {
            Slot& slot = m_slots[slot_idx];
            if (slot.fingerprint == 0) {
                const size_t value_size = static_cast<size_t>(std::distance(value_begin, value_end));
                // Decline values that cannot be represented by the slot count.
                if (value_size > std::numeric_limits<TokenCount>::max()) {
                    return InsertResult::SATURATED;
                }
                // Keep the slot unoccupied until allocation and copying succeed.
                std::string owned_key(key);
                int32_t* storage = nullptr;
                if (value_size > INLINE_TOKENS) {
                    storage = allocate_spill(value_size);
                    slot.value.spill = storage;
                } else {
                    storage = slot.value.inline_tokens;
                }
                std::copy(value_begin, value_end, storage);
                slot.count = static_cast<TokenCount>(value_size);
                slot.key = std::move(owned_key);
                slot.fingerprint = fingerprint;
                ++m_size;
                return InsertResult::INSERTED;
            }
            if (slot.fingerprint == fingerprint && keys_equal(slot.key, key)) {
                return InsertResult::EXISTING;
            }
            slot_idx = (slot_idx + 1) & m_mask;
        }
        return InsertResult::SATURATED;
    }

    size_t size() const {
        return m_size;
    }

    size_t slot_count() const {
        return m_slots.size();
    }

private:
    static constexpr size_t INLINE_TOKENS = 4;

#if BPE_CACHE_FINGERPRINT_BITS == 64
    using Fingerprint = uint64_t;
#elif BPE_CACHE_FINGERPRINT_BITS == 32
    using Fingerprint = uint32_t;
#else
#    error "BPE_CACHE_FINGERPRINT_BITS must be 32 or 64"
#endif
    using TokenCount = uint32_t;

    struct Slot {
        Fingerprint fingerprint = 0;
        TokenCount count = 0;
        union Value {
            int32_t inline_tokens[INLINE_TOKENS];
            const int32_t* spill;
        } value{};
        std::string key;

        const int32_t* tokens() const {
            return count > INLINE_TOKENS ? value.spill : value.inline_tokens;
        }
    };

    // Pointees remain stable when the chunk list grows.
    static constexpr size_t SPILL_CHUNK_TOKENS = 4096;

    int32_t* allocate_spill(size_t value_size) {
        if (m_spill_chunks.empty() || m_spill_used + value_size > m_spill_capacity) {
            const size_t chunk_tokens = std::max(value_size, SPILL_CHUNK_TOKENS);
            m_spill_chunks.push_back(std::make_unique<int32_t[]>(chunk_tokens));
            m_spill_capacity = chunk_tokens;
            m_spill_used = 0;
        }
        int32_t* storage = m_spill_chunks.back().get() + m_spill_used;
        m_spill_used += value_size;
        return storage;
    }

    static size_t max_entries_for_slots(size_t slot_count) {
        return (slot_count / 10) * 7 + ((slot_count % 10) * 7) / 10;
    }

    // Reserve zero as the empty-slot sentinel.
    static Fingerprint stored_fingerprint(uint64_t hash_value) {
        const Fingerprint truncated = static_cast<Fingerprint>(hash_value);
        return truncated == std::numeric_limits<Fingerprint>::max() ? truncated
                                                                   : static_cast<Fingerprint>(truncated + 1);
    }

    static bool keys_equal(const std::string& stored, std::string_view candidate) {
        return stored.size() == candidate.size() &&
               std::equal(stored.begin(), stored.end(), candidate.begin());
    }

    std::vector<Slot> m_slots;
    std::vector<std::unique_ptr<int32_t[]>> m_spill_chunks;
    size_t m_spill_used = 0;
    size_t m_spill_capacity = 0;
    size_t m_mask = 0;
    size_t m_size = 0;
    size_t m_max_entries = 0;
};
