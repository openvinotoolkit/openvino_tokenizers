// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "contrib_string_ops.hpp"
#include "utils.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <openvino/op/constant.hpp>

using namespace ov;

void ContribStringJoin::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 5, "ContribStringJoin expects 5 inputs");
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    {
        auto t = get_input_element_type(4);
        OPENVINO_ASSERT(t == element::i64 || t == element::i32 || t.is_dynamic(),
                        "ContribStringJoin axis input must be of integer type, got: ", t);
    }

    auto in_pshape = get_input_partial_shape(0);
    PartialShape out_pshape;
    if (in_pshape.rank().is_static()) {
        auto rank = in_pshape.rank().get_length();
        if (rank <= 1) {
            out_pshape = PartialShape{};
        } else {
            // Try to read axis as a constant so we can preserve known
            // non-axis dimensions in the output shape (important for the
            // CPU plugin which requires static shapes).
            int64_t const_axis = -1;  // -1 means unknown at inference time
            if (auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(
                    input_value(4).get_node_shared_ptr())) {
                const_axis = axis_const->cast_vector<int64_t>()[0];
                if (const_axis < 0) const_axis += rank;
            }
            std::vector<Dimension> dims;
            dims.reserve(rank - 1);
            for (int64_t i = 0; i < rank; ++i) {
                if (i == const_axis) continue;
                // Preserve the known input dimension; fall back to dynamic
                // if axis is not a compile-time constant.
                dims.push_back(const_axis >= 0 ? in_pshape[i] : Dimension());
            }
            out_pshape = PartialShape(dims);
        }
    } else {
        out_pshape = PartialShape::dynamic();
    }

    set_output_type(0, element::i32, out_pshape);
    set_output_type(1, element::i32, out_pshape);
    set_output_type(2, element::u8, PartialShape{Dimension()});
}

bool ContribStringJoin::evaluate(ov::TensorVector& outputs,
                                 const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 5, "ContribStringJoin expects 5 inputs");

    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();
    const auto& in_shape = inputs[0].get_shape();
    size_t in_rank = in_shape.size();

    std::string_view sep(reinterpret_cast<const char*>(inputs[3].data<const uint8_t>()),
                         inputs[3].get_size());

    int64_t axis;
    {
        OPENVINO_ASSERT(inputs[4].get_size() == 1,
                        "ContribStringJoin axis input must be a scalar (single element)");
        const auto& at = inputs[4].get_element_type();
        if (at == element::i64)
            axis = inputs[4].data<const int64_t>()[0];
        else if (at == element::i32)
            axis = static_cast<int64_t>(inputs[4].data<const int32_t>()[0]);
        else
            OPENVINO_THROW("ContribStringJoin: unsupported axis element type ", at);
    }

    if (in_rank > 0) {
        if (axis < 0)
            axis += static_cast<int64_t>(in_rank);
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < in_rank,
                        "ContribStringJoin axis out of range");
    }

    // Handle 0-D / single-string degenerate case: identity copy.
    // shape_size == 0 (empty input, e.g. [0,N]) must NOT enter this branch:
    // the output still has N elements and only initialising index 0 would
    // leave the rest of begins/ends uninitialized.  The general path handles
    // axis_size == 0 correctly by producing N empty strings.
    if (in_rank == 0 || ov::shape_size(in_shape) == 1) {
        ov::Shape out_shape = (in_rank <= 1) ? ov::Shape{} : in_shape;
        if (in_rank > 1) {
            out_shape.erase(out_shape.begin() + axis);
        }
        outputs[0].set_shape(out_shape);
        outputs[1].set_shape(out_shape);

        size_t total_in = (ov::shape_size(in_shape) == 0) ? 0
                                                          : static_cast<size_t>(ends[0] - begins[0]);
        outputs[2].set_shape(ov::Shape{total_in});
        if (ov::shape_size(out_shape) >= 1) {
            outputs[0].data<int32_t>()[0] = 0;
            outputs[1].data<int32_t>()[0] = static_cast<int32_t>(total_in);
            if (total_in > 0) {
                std::copy(chars + begins[0], chars + ends[0],
                          outputs[2].data<uint8_t>());
            }
        }
        return true;
    }

    // Compute strides for input (row-major).
    std::vector<size_t> in_strides(in_rank, 1);
    for (int i = static_cast<int>(in_rank) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    // Output shape = input shape with axis dimension removed.
    ov::Shape out_shape;
    out_shape.reserve(in_rank - 1);
    for (size_t d = 0; d < in_rank; ++d) {
        if (static_cast<int64_t>(d) != axis) {
            out_shape.push_back(in_shape[d]);
        }
    }
    size_t out_size = ov::shape_size(out_shape);
    size_t axis_size = in_shape[axis];
    const size_t axis_stride = in_strides[axis];

    // Output strides (over out_shape) for unraveling output linear index.
    std::vector<size_t> out_strides(out_shape.size(), 1);
    for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Helper: unravel output linear index to compute input base offset.
    std::vector<size_t> coords_out(out_shape.size());
    auto compute_base_offset = [&](size_t oi) -> size_t {
        size_t r = oi;
        for (size_t d = 0; d < out_shape.size(); ++d) {
            coords_out[d] = r / out_strides[d];
            r = r % out_strides[d];
        }
        size_t base = 0, k = 0;
        for (size_t d = 0; d < in_rank; ++d) {
            if (static_cast<int64_t>(d) == axis) continue;
            base += coords_out[k++] * in_strides[d];
        }
        return base;
    };

    // Pass 1: compute total output chars to pre-size the output buffer.
    // Pre-count all separators: each of the out_size joined strings contains
    // (axis_size - 1) separators.  Guard against axis_size == 0 (size_t underflow).
    size_t total_chars = (axis_size > 0) ? out_size * (axis_size - 1) * sep.size() : 0;
    for (size_t oi = 0; oi < out_size; ++oi) {
        size_t base = compute_base_offset(oi);
        for (size_t a = 0; a < axis_size; ++a) {
            size_t idx = base + a * axis_stride;
            total_chars += static_cast<size_t>(ends[idx] - begins[idx]);
        }
    }

    outputs[0].set_shape(out_shape);
    outputs[1].set_shape(out_shape);
    outputs[2].set_shape(ov::Shape{total_chars});
    auto out_begins = outputs[0].data<int32_t>();
    auto out_ends   = outputs[1].data<int32_t>();
    auto out_chars  = outputs[2].data<uint8_t>();

    // Pass 2: write joined strings directly to the output buffer.
    size_t cur = 0;
    for (size_t oi = 0; oi < out_size; ++oi) {
        out_begins[oi] = static_cast<int32_t>(cur);
        size_t base = compute_base_offset(oi);
        for (size_t a = 0; a < axis_size; ++a) {
            size_t idx = base + a * axis_stride;
            if (a > 0) {
                std::copy(sep.data(), sep.data() + sep.size(), out_chars + cur);
                cur += sep.size();
            }
            size_t slen = static_cast<size_t>(ends[idx] - begins[idx]);
            std::copy(chars + begins[idx], chars + ends[idx], out_chars + cur);
            cur += slen;
        }
        out_ends[oi] = static_cast<int32_t>(cur);
    }
    return true;
}

void ContribStringSplit::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 5, "ContribStringSplit expects 5 inputs");
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    {
        auto t = get_input_element_type(4);
        OPENVINO_ASSERT(t == element::boolean || t == element::u8 || t == element::i8 || t.is_dynamic(),
                        "ContribStringSplit skip_empty input must be of boolean/byte type, got: ", t);
    }

    auto in_pshape = get_input_partial_shape(0);
    Dimension out_rank_dim;
    if (in_pshape.rank().is_static()) {
        out_rank_dim = Dimension(in_pshape.rank().get_length() + 1);
    } else {
        out_rank_dim = Dimension();
    }
    set_output_type(0, element::i64, PartialShape{Dimension(), out_rank_dim});
    set_output_type(1, element::i32, PartialShape{Dimension()});
    set_output_type(2, element::i32, PartialShape{Dimension()});
    set_output_type(3, element::u8, PartialShape{Dimension()});
    set_output_type(4, element::i64, PartialShape{out_rank_dim});
}

bool ContribStringSplit::evaluate(ov::TensorVector& outputs,
                                  const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 5, "ContribStringSplit expects 5 inputs");

    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();
    const auto& in_shape = inputs[0].get_shape();
    size_t in_rank = in_shape.size();
    size_t in_size = ov::shape_size(in_shape);
    size_t out_rank = in_rank + 1;

    std::string_view delim(reinterpret_cast<const char*>(inputs[3].data<const uint8_t>()),
                           inputs[3].get_size());
    bool skip_empty;
    {
        OPENVINO_ASSERT(inputs[4].get_size() == 1,
                        "ContribStringSplit skip_empty input must be a scalar (single element)");
        const auto& st = inputs[4].get_element_type();
        if (st == element::boolean)
            skip_empty = inputs[4].data<const bool>()[0];
        else if (st == element::u8)
            skip_empty = inputs[4].data<const uint8_t>()[0] != 0;
        else if (st == element::i8)
            skip_empty = inputs[4].data<const int8_t>()[0] != 0;
        else
            OPENVINO_THROW("ContribStringSplit: unsupported skip_empty element type ", st);
    }

    std::vector<size_t> in_strides(in_rank, 1);
    for (int i = static_cast<int>(in_rank) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    std::vector<int64_t> indices_flat;
    std::vector<std::string> values;
    size_t max_tokens = 0;

    std::vector<int64_t> coord(in_rank);
    for (size_t p = 0; p < in_size; ++p) {
        OPENVINO_ASSERT(ends[p] >= begins[p], "Malformed packed string: ends[", p, "] < begins[", p, "]");
        std::string_view s(reinterpret_cast<const char*>(chars + begins[p]),
                           static_cast<size_t>(ends[p] - begins[p]));

        size_t r = p;
        for (size_t d = 0; d < in_rank; ++d) {
            coord[d] = static_cast<int64_t>(r / in_strides[d]);
            r = r % in_strides[d];
        }

        std::vector<std::string_view> tokens;
        if (delim.empty()) {
            // Split into individual UTF-8 bytes (matches onnxruntime-extensions
            // behavior of splitting on empty delimiter into characters).
            tokens.reserve(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
                tokens.emplace_back(s.data() + i, 1);
            }
        } else {
            size_t pos = 0;
            while (true) {
                size_t found = s.find(delim, pos);
                if (found == std::string_view::npos) {
                    tokens.emplace_back(s.data() + pos, s.size() - pos);
                    break;
                }
                tokens.emplace_back(s.data() + pos, found - pos);
                pos = found + delim.size();
            }
        }

        // tok_pos tracks the original slot position in the split sequence so
        // that sparse COO indices correctly reflect pre-skip positions.
        size_t tok_pos = 0;
        for (auto& t : tokens) {
            if (!skip_empty || !t.empty()) {
                indices_flat.insert(indices_flat.end(), coord.begin(), coord.end());
                indices_flat.push_back(static_cast<int64_t>(tok_pos));
                values.emplace_back(t);
            }
            ++tok_pos;
        }
        if (tok_pos > max_tokens) max_tokens = tok_pos;
    }

    size_t N = values.size();

    outputs[0].set_shape(ov::Shape{N, out_rank});
    if (N > 0) {
        std::copy(indices_flat.begin(), indices_flat.end(),
                  outputs[0].data<int64_t>());
    }

    outputs[1].set_shape(ov::Shape{N});
    outputs[2].set_shape(ov::Shape{N});
    size_t total_chars = 0;
    for (auto& v : values) total_chars += v.size();
    outputs[3].set_shape(ov::Shape{total_chars});

    auto vb = outputs[1].data<int32_t>();
    auto ve = outputs[2].data<int32_t>();
    auto vc = outputs[3].data<uint8_t>();
    size_t cur = 0;
    for (size_t i = 0; i < N; ++i) {
        vb[i] = static_cast<int32_t>(cur);
        std::copy(values[i].begin(), values[i].end(), vc + cur);
        cur += values[i].size();
        ve[i] = static_cast<int32_t>(cur);
    }

    outputs[4].set_shape(ov::Shape{out_rank});
    auto ds = outputs[4].data<int64_t>();
    for (size_t d = 0; d < in_rank; ++d) {
        ds[d] = static_cast<int64_t>(in_shape[d]);
    }
    ds[in_rank] = static_cast<int64_t>(max_tokens);

    return true;
}
