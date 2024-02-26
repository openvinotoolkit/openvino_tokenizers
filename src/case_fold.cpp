// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "case_fold.hpp"
#include "utils.hpp"

#include "fast_tokenizer/normalizers/normalizers.h"

using namespace ov;


void CaseFold::validate_and_infer_types() {
    check_string_input(this, 0);
    OPENVINO_ASSERT(
        m_encoding == "" || m_encoding == "utf-8",
        "CaseFold operation `encoding` attribute must be one of [\"\", \"utf-8\"], got `" + m_encoding + "`."
    );
    set_string_output(this, 0, get_input_partial_shape(0));
}

bool CaseFold::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate_normalization_helper(
        outputs, inputs,
        [&](const std::string& str) {
            if (m_encoding.empty()) {
                for (char ch : str) {
                    if (ch < 0 || ch > 127) {
                        OPENVINO_THROW(
                            "CaseFold operation works with ascii chars only. "
                            "Use CaseFold with encoding=\"utf-8\" or filter non-ascii chars from the input."
                        );
                    };
                };
            };
            using namespace paddlenlp::fast_tokenizer;
            return normalizers::NormalizedString(str).Lowercase().GetStr();
        });
}
