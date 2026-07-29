// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class BPEResultCache {
public:
    enum class InsertResult {
        INSERTED,
        EXISTING,
        SATURATED,
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

    static uint64_t hash(std::string_view key) {
        uint64_t value = 14695981039346656037ULL;
        for (const unsigned char byte : key) {
            value ^= byte;
            value *= 1099511628211ULL;
        }
        return value;
    }

    const std::vector<int32_t>* find(std::string_view key, uint64_t hash_value) const {
        if (m_slots.empty()) {
            return nullptr;
        }

        const uint64_t fingerprint = stored_fingerprint(hash_value);
        size_t slot_idx = static_cast<size_t>(hash_value) & m_mask;
        for (size_t probes = 0; probes < m_slots.size(); ++probes) {
            const Slot& slot = m_slots[slot_idx];
            if (slot.fingerprint == 0) {
                return nullptr;
            }
            if (slot.fingerprint == fingerprint && keys_equal(slot.key, key)) {
                return &slot.value;
            }
            slot_idx = (slot_idx + 1) & m_mask;
        }
        return nullptr;
    }

    const std::vector<int32_t>* find(std::string_view key) const {
        return find(key, hash(key));
    }

    template <typename Iterator>
    InsertResult insert(std::string_view key, uint64_t hash_value, Iterator value_begin, Iterator value_end) {
        if (m_size >= m_max_entries || m_slots.empty()) {
            return InsertResult::SATURATED;
        }

        const uint64_t fingerprint = stored_fingerprint(hash_value);
        size_t slot_idx = static_cast<size_t>(hash_value) & m_mask;
        for (size_t probes = 0; probes < m_slots.size(); ++probes) {
            Slot& slot = m_slots[slot_idx];
            if (slot.fingerprint == 0) {
                std::string owned_key(key);
                std::vector<int32_t> owned_value(value_begin, value_end);
                slot.key = std::move(owned_key);
                slot.value = std::move(owned_value);
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
    struct Slot {
        uint64_t fingerprint = 0;
        std::string key;
        std::vector<int32_t> value;
    };

    static size_t max_entries_for_slots(size_t slot_count) {
        return (slot_count / 10) * 7 + ((slot_count % 10) * 7) / 10;
    }

    static uint64_t stored_fingerprint(uint64_t hash_value) {
        return hash_value == std::numeric_limits<uint64_t>::max() ? hash_value : hash_value + 1;
    }

    static bool keys_equal(const std::string& stored, std::string_view candidate) {
        return stored.size() == candidate.size() &&
               std::equal(stored.begin(), stored.end(), candidate.begin());
    }

    std::vector<Slot> m_slots;
    size_t m_mask = 0;
    size_t m_size = 0;
    size_t m_max_entries = 0;
};
