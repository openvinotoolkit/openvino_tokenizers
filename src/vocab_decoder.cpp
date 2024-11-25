// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include <algorithm>

#include "vocab_decoder.hpp"
#include "utils.hpp"

using namespace ov;

void VocabDecoder::validate_and_infer_types() {
    check_string_input(this, 1);
    const auto shape = get_input_partial_shape(0);
    set_ragged_string_output(this, 0, {shape[0]});
}

bool VocabDecoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto batch_size = inputs[0].get_shape()[0];
    auto seq_len    = inputs[0].get_shape()[1];
    auto input_data = inputs[0].data<const int32_t>();

    auto vocab_size   = inputs[1].get_size();
    auto vocab_begins = inputs[1].data<const int32_t>();
    auto vocab_ends   = inputs[2].data<const int32_t>();
    auto vocab_chars  = inputs[3].data<const uint8_t>();

    OPENVINO_ASSERT(inputs.size() == 4 || inputs.size() == 5, "Too few inputs passed to VocabDecoder, it means it is not converted properly or it is not used in the supported pattern");
    
    // Use skip tokens from input if specified, otherwise use the attribute.
    std::vector<int> skip_tokens;
    if (inputs.size() == 5) {
        skip_tokens = std::vector<int>(inputs[4].data<int32_t>(), inputs[4].data<int32_t>() + inputs[4].get_shape()[0]);
    } else {
        skip_tokens = m_skip_tokens;
    }

    // Set output shapes
    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});
    outputs[2].set_shape({batch_size * ((seq_len > 0) ? seq_len : 1)});
    outputs[3].set_shape({batch_size * ((seq_len > 0) ? seq_len : 1)});
    const size_t num_rows = inputs[0].get_size();

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();

    std::deque<uint8_t> buffer;
    for(size_t batch = 0; batch < batch_size; ++batch) {
        new_ragged_begins[batch] = batch * ((seq_len > 0) ? seq_len : 1);
        new_ragged_ends[batch]   = new_ragged_begins[batch] + ((seq_len > 0) ? seq_len : 1);

        if (seq_len == 0) {
            new_begins[batch] = buffer.size();
            new_ends[batch] = buffer.size();
            continue;
        };

        for(size_t seq = new_ragged_begins[batch]; seq < new_ragged_ends[batch]; ++seq) {
            auto token_id = input_data[seq];
            new_begins[seq] = buffer.size();
            if (
                token_id < vocab_size
                && std::find(skip_tokens.begin(), skip_tokens.end(), token_id) == skip_tokens.end()
            ) {
                buffer.insert(
                    buffer.end(),
                    vocab_chars + vocab_begins[token_id],
                    vocab_chars + vocab_ends[token_id]
                );
            }
            new_ends[seq] = buffer.size();
        }
    }
    outputs[4].set_shape({buffer.size()});
    auto new_chars  = outputs[4].data<uint8_t>();
    std::copy(buffer.begin(), buffer.end(), new_chars);
    return true;
}
