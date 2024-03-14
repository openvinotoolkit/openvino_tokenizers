// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "trie_tokenizer.hpp"

void TrieTokenizer::validate_and_infer_types() {
//    check_ragged_string_input(this, 0);
////    set_ragged_string_output(this, 0, get_input_partial_shape(0));
//    set_string_output(this, 0, get_input_partial_shape(0));
}


bool TrieTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return true;
}
