// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/node_context.hpp>

ov::OutputVector
translate_string_normalizer(const ov::frontend::NodeContext &node);
