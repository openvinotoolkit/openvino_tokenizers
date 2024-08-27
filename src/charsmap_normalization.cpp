// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "charsmap_normalization.hpp"
#include "utils.hpp"

using namespace ov;


void CharsMapNormalization::validate_and_infer_types() {
    check_string_input(this, 0);

//    OPENVINO_ASSERT(
//        m_encoding == "" || m_encoding == "utf-8",
//        "CaseFold operation `encoding` attribute must be one of [\"\", \"utf-8\"], got `", m_encoding, "`."
//    );
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool CharsMapNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate_normalization_helper(
        outputs, inputs,
        [](const std::string& str) { return str; });
}
