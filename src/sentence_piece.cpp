// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cctype>
#include <algorithm>
#include <functional>

#include "sentencepiece_processor.h"

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"

#include "sentence_piece.hpp"
#include "utils.hpp"

using sentencepiece::SentencePieceProcessor;
using namespace ov;
using namespace ov::frontend;
using namespace ov::opset13;

// copy from 'src' folder of 'sentencepiece' to prevent usage of private API
namespace sentencepiece {
namespace {

// Converts byte (0-255) to piece (e.g., 58 -> "<0x3A>").
std::string ByteToPiece(unsigned char c) {
  return absl::StrFormat("<0x%02X>", c);
}

int PieceToByte(absl::string_view piece) {
  using PieceToByteMap = absl::flat_hash_map<std::string, unsigned char>;
  static const auto *const kMap = []() -> PieceToByteMap * {
    auto *m = new PieceToByteMap();
    for (int i = 0; i < 256; ++i) {
      (*m)[ByteToPiece(i)] = i;
    }
    return m;
  }();
  const auto it = kMap->find(std::string(piece));
  if (it == kMap->end()) {
    return -1;
  } else {
    return it->second;
  }
}

#define CHECK_OK(expr)                                 \
  do {                                                 \
    const auto _status = expr;                         \
    OPENVINO_ASSERT(_status.ok(), _status.ToString()); \
  } while (0)

}  // namespace
}  // sentencepiece

namespace {

std::string form_extra_options(bool add_bos, bool add_eos, bool reverse) {
    std::string extra_options = "";
    if (add_bos) {
        extra_options += "bos";
    }
    if (add_eos) {
        extra_options = extra_options.empty() ? extra_options : extra_options + ":";
        extra_options += "eos";
    }
    if (reverse) {
        extra_options = extra_options.empty() ? extra_options : extra_options + ":";
        extra_options += "reverse";
    }
    return extra_options;
}

void init_sp_model(const OutputVector& args, std::shared_ptr<SentencePieceProcessor>& sp) {
    auto sp_model_const = as_type_ptr<Constant>(args[0].get_node_shared_ptr());
    OPENVINO_ASSERT(sp_model_const, "SentencepieceTokenizer expects SentencePiece model to be constant.");
    auto spm_model = static_cast<const char*>(sp_model_const->get_data_ptr());
    auto spm_model_size = sp_model_const->get_byte_size();

    std::string model_proto(spm_model, spm_model_size);
    CHECK_OK(sp->LoadFromSerializedProto(model_proto));
}

void init_sp_model_in_eval(const TensorVector& inputs, std::shared_ptr<SentencePieceProcessor>& sp) {
    auto spm_model = inputs[0].data<const char>();
    auto spm_model_size = inputs[0].get_size();
    std::string model_proto(spm_model, spm_model_size);
    CHECK_OK(sp->LoadFromSerializedProto(model_proto));
}

} // namespace

SentencepieceTokenizer::SentencepieceTokenizer(const OutputVector& args, int32_t nbest_size, float alpha,
    bool add_bos, bool add_eos, bool reverse) : m_sp(std::make_shared<SentencePieceProcessor>()),
    m_nbest_size(nbest_size), m_alpha(alpha), m_add_bos(add_bos), m_add_eos(add_eos),
    m_reverse(reverse), Op(args) {

    init_sp_model(args, m_sp);
    auto do_reverse = (m_reverse && get_input_size() < 5);  // do not reverse if special_tokens_re is used
    CHECK_OK(m_sp->SetEncodeExtraOptions(form_extra_options(m_add_bos, m_add_eos, do_reverse)));
    constructor_validate_and_infer_types();
}

SentencepieceTokenizer::SentencepieceTokenizer(
    const OutputVector& args,
    const std::shared_ptr<SentencePieceProcessor>& sp,
    const std::shared_ptr<re2::RE2>& special_tokens_re,
    const std::shared_ptr<absl::flat_hash_map<std::string, int32_t>>& special_tokens_map,
    int32_t nbest_size,
    float alpha,
    bool add_bos,
    bool add_eos,
    bool reverse
) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp),
    m_special_tokens_re(special_tokens_re),
    m_special_tokens_map(special_tokens_map),
    m_nbest_size(nbest_size), m_alpha(alpha), m_add_bos(add_bos), m_add_eos(add_eos),
    m_reverse(reverse), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
        init_sp_model(args, m_sp);
        auto do_reverse = (m_reverse && get_input_size() < 5);  // do not reverse if special_tokens_re is used
        CHECK_OK(m_sp->SetEncodeExtraOptions(form_extra_options(m_add_bos, m_add_eos, do_reverse)));
    };
    constructor_validate_and_infer_types();
}

void SentencepieceTokenizer::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_element_type(0) == element::u8, "SentencepieceTokenizer accepts sp model as the first input and it should be of type u8 tensor");

    auto input_size = get_input_size();
    // sentencepiece model, string input, (unpacked special tokens)
    if(input_size == 2 || input_size == 6) {
        OPENVINO_ASSERT(
            // WA: f32 appeared as a placeholder for unknown type during intermediate conversion steps
            get_input_element_type(1) == element::string || get_input_element_type(1) == element::f32,
            "SentencepieceTokenizer accepts sentences as the second input and it should be of type string tensor");
    // sentencepiece model, unpacked string input, (unpacked special tokens)
    } else if (input_size == 4 || input_size == 8) {
        check_string_input(this, 1);
    } else {
        OPENVINO_THROW("Unexpected input format. SentencepieceTokenizer accepts one string input or three decomposed string inputs (begins, ends, symbols)");
    };

    if (input_size == 6 || input_size == 8) {
        // unpacked special tokens
        check_string_input(this, input_size - 4);
        // special tokens ids
        OPENVINO_ASSERT(this->get_input_element_type(input_size - 1) == element::i32, "Expected an i32 tensor for special tokens ids.");
    };

    // The operation SentencepieceTokenizerExtensionOp has three outputs: sparse indices, sparse values
    // and dense shape
    set_output_type(0, element::i64, PartialShape{ Dimension(), Dimension(2) });
    set_output_type(1, element::i32, PartialShape{ Dimension() });
    set_output_type(2, element::i64, PartialShape{ Dimension(2) });
}

bool SentencepieceTokenizer::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("nbest_size", m_nbest_size);
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("add_bos", m_add_bos);
    visitor.on_attribute("add_eos", m_add_eos);
    visitor.on_attribute("reverse", m_reverse);
    return true;
}

bool SentencepieceTokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    auto input_size = get_input_size();
    {
        // Write to common trie structures should be protected to prevent race conditions.
        std::lock_guard<std::mutex> lock(m_mutex);    
        if (m_sp == nullptr) {
            m_sp = std::make_shared<SentencePieceProcessor>();
            init_sp_model_in_eval(inputs, m_sp);
            auto do_reverse = (m_reverse && input_size < 5);  // do not reverse if special_tokens_re is used
            CHECK_OK(m_sp->SetEncodeExtraOptions(form_extra_options(m_add_bos, m_add_eos, do_reverse)));
        }

        if (input_size > 5 && m_special_tokens_re == nullptr) {
            auto special_tokens_begins = inputs[input_size - 4].data<const int32_t>();
            auto special_tokens_ends   = inputs[input_size - 3].data<const int32_t>();
            auto special_tokens_chars  = inputs[input_size - 2].data<const uint8_t>();
            auto special_tokens_ids = inputs[input_size - 1].data<const int32_t>();

            std::string special_tokens;
            m_special_tokens_map = std::make_shared<absl::flat_hash_map<std::string, int32_t>>();
            for (size_t i = 0; i < inputs[input_size - 4].get_size(); ++i) {
                const std::string token = std::string(
                    special_tokens_chars + special_tokens_begins[i],
                    special_tokens_chars + special_tokens_ends[i]
                );
                if (!special_tokens.empty()) {
                    special_tokens += "|";
                };

                if (std::all_of(token.begin(), token.end(), [](char c) { return std::isalpha(c); })) {
                    // have to check if special token is not a part of some word
                    // chatglm2/3 has "sop" and "eop" special tokens that will split words like "people" otherwise
                    special_tokens += ("\\b" + re2::RE2::QuoteMeta(token));
                    special_tokens += "|";
                    special_tokens += (re2::RE2::QuoteMeta(token) + "\\b");
                } else {
                    special_tokens += re2::RE2::QuoteMeta(token);
                };

                m_special_tokens_map->insert(std::pair{token, special_tokens_ids[i]});
            }
            m_special_tokens_re = std::make_shared<re2::RE2>("(" + special_tokens + ")");
        }
}

    std::vector<int64_t> sparse_indices;
    std::vector<int32_t> sparse_values;
    std::vector<int64_t> sparse_dense_shape;

    int32_t batch_size;

    // used in case of string tensors
    const std::string* strings;

    // used in case of u8 packed representation
    const int32_t* begin_ids;
    const int32_t* end_ids;
    const uint8_t* data;

    if (input_size == 2 || input_size == 6) {
        auto input_element_type = get_input_element_type(1);
        if(input_element_type == ov::element::string) {
            strings = inputs[1].data<const std::string>();
            batch_size = static_cast<int32_t>(ov::shape_size(inputs[1].get_shape()));
        } else {
            OPENVINO_THROW("Unexpected input type during inference. SentencepieceTokenizer accepts element::u8 or element::string.");
        }
    } else {
        auto begin_ids = inputs[1].data<const int32_t>();
        auto end_ids = inputs[2].data<const int32_t>();
        auto data = inputs[3].data<const uint8_t>();
        batch_size = shape_size(inputs[1].get_shape());
    };

    size_t max_token_id = 0;
    for (size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
        std::string sentence;
        if (input_size == 2 || input_size == 6) {
            sentence = strings[batch_ind];
        } else {
            auto begin_ind = begin_ids[batch_ind];
            auto end_ind = end_ids[batch_ind];
            sentence = absl::string_view((const char*)data + begin_ind, end_ind - begin_ind);
        };

        std::vector<int32_t> ids;
        if (input_size < 5) {
            CHECK_OK(m_sp->SampleEncode(sentence, m_nbest_size, m_alpha, &ids));
        } else {
            std::string special_token;
            std::vector<int32_t> part_ids;
            re2::StringPiece input(sentence);
            auto cursor = input.begin();
            const auto num_tokens_before = ids.size();
            while (cursor != input.end()) {
                if (re2::RE2::FindAndConsume(&input, *m_special_tokens_re, &special_token)) {
                    auto before_special_token = absl::string_view(cursor, input.begin() - cursor - special_token.size());
                    CHECK_OK(m_sp->SampleEncode(before_special_token, m_nbest_size, m_alpha, &part_ids));
                    ids.insert(ids.end(), part_ids.begin(), part_ids.end());
                    cursor = input.begin();

                    auto token_and_id = m_special_tokens_map->find(special_token);
                    if (token_and_id != m_special_tokens_map->end()) {
                        ids.push_back(token_and_id->second);
                    } else {
                        // fallback to regular tokenization if no special tokens found the map
                        CHECK_OK(m_sp->SampleEncode(before_special_token, m_nbest_size, m_alpha, &part_ids));
                        ids.insert(ids.end(), part_ids.begin(), part_ids.end());
                        cursor = input.begin();
                    };
                } else {
                    CHECK_OK(m_sp->SampleEncode(input, m_nbest_size, m_alpha, &part_ids));
                    ids.insert(ids.end(), part_ids.begin(), part_ids.end());
                    cursor = input.end();
                };
            };

            if (m_reverse && ids.size() - num_tokens_before > 1) {
                std::reverse(ids.begin() + num_tokens_before, ids.end());
            };
        };

        // put into resulted vectors
        for (size_t token_id = 0; token_id < ids.size(); ++token_id) {
            sparse_indices.push_back(static_cast<int64_t>(batch_ind));
            sparse_indices.push_back(static_cast<int64_t>(token_id));
            sparse_values.push_back(static_cast<int32_t>(ids[token_id]));
        };
        max_token_id = max_token_id < ids.size() ? ids.size() : max_token_id;
    }
    sparse_dense_shape.push_back(static_cast<int64_t>(batch_size));
    sparse_dense_shape.push_back(static_cast<int64_t>(max_token_id));

    outputs[0].set_shape({ sparse_indices.size() / 2, 2 });
    memcpy(outputs[0].data(), sparse_indices.data(), sizeof(int64_t) * sparse_indices.size());
    outputs[1].set_shape({ sparse_values.size() });
    memcpy(outputs[1].data(), sparse_values.data(), sizeof(int32_t) * sparse_values.size());
    outputs[2].set_shape({ 2 });
    memcpy(outputs[2].data(), sparse_dense_shape.data(), sizeof(int64_t) * sparse_dense_shape.size());

    return true;
}

bool SentencepieceTokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceTokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceTokenizer>(new_args, m_sp, m_special_tokens_re, m_special_tokens_map, m_nbest_size, m_alpha, m_add_bos, m_add_eos, m_reverse);
}


// Detokenizer

SentencepieceDetokenizer::SentencepieceDetokenizer(const OutputVector& args) :
    m_sp(std::make_shared<SentencePieceProcessor>()), Op(args) {
    init_sp_model(args, m_sp);
    constructor_validate_and_infer_types();
}

SentencepieceDetokenizer::SentencepieceDetokenizer(const OutputVector& args, const std::shared_ptr<SentencePieceProcessor>& sp) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
        init_sp_model(args, m_sp);
    };
    constructor_validate_and_infer_types();
}

void SentencepieceDetokenizer::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "SentencepieceDetokenizer expects two inputs: sp model and token ids");
    OPENVINO_ASSERT(get_input_element_type(0) == element::u8, "SentencepieceDetokenizer accepts sp model as the first input and it should be of type u8 tensor");
    OPENVINO_ASSERT(get_input_partial_shape(1).size() == 2, "SentencepieceDetokenizer expects 2D tensor as second input");

    auto batch_size = PartialShape({get_input_partial_shape(1)[0]});
    set_string_output(this, 0, batch_size);
}

bool SentencepieceDetokenizer::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool SentencepieceDetokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    if (m_sp == nullptr) {
        m_sp = std::make_shared<SentencePieceProcessor>();
        init_sp_model_in_eval(inputs, m_sp);
    };

    auto batch_size = inputs[1].get_shape()[0];
    auto seq_len    = inputs[1].get_shape()[1];
    auto input_data = inputs[1].data<const int32_t>();

    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});

    auto begins = outputs[0].data<int32_t>();
    auto ends   = outputs[1].data<int32_t>();
    std::deque<uint8_t> buffer;

    const auto vocab_size = m_sp->GetPieceSize();
    auto id_filter = [vocab_size](int32_t value) { return value < vocab_size; };

    for(size_t batch = 0; batch < batch_size; ++batch) {
        auto start = batch * seq_len;

        std::vector<int32_t> token_ids;
        token_ids.reserve(seq_len);
        std::copy_if(&input_data[start], &input_data[start] + seq_len, std::back_inserter(token_ids), id_filter);

        std::string detokenized;
        CHECK_OK(m_sp->Decode(token_ids, &detokenized));

        begins[batch] = buffer.size();
        buffer.insert(buffer.end(), detokenized.begin(), detokenized.end());
        ends[batch] = buffer.size();
    }
    outputs[2].set_shape({buffer.size()});
    auto chars  = outputs[2].data<uint8_t>();
    std::copy(buffer.begin(), buffer.end(), chars);
    return true;
}

bool SentencepieceDetokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceDetokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceDetokenizer>(new_args, m_sp);
}


// Stream Detokenizer

SentencepieceStreamDetokenizer::SentencepieceStreamDetokenizer(const OutputVector& args) :
    m_sp(std::make_shared<SentencePieceProcessor>()), Op(args) {
    init_sp_model(args, m_sp);
    constructor_validate_and_infer_types();
}

SentencepieceStreamDetokenizer::SentencepieceStreamDetokenizer(const OutputVector& args, const std::shared_ptr<SentencePieceProcessor>& sp) :
    m_sp((sp == nullptr) ? std::make_shared<SentencePieceProcessor>(): sp), Op(args) {
    // constructor above without sp argument never called when the node is created with python factory, so need to init and cache m_sp here
    if (!m_sp->status().ok()) {
        init_sp_model(args, m_sp);
    };
    constructor_validate_and_infer_types();
}

void SentencepieceStreamDetokenizer::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "SentencepieceDetokenizer expects two inputs: sp model and token ids");
    OPENVINO_ASSERT(get_input_element_type(0) == element::u8, "SentencepieceDetokenizer accepts sp model as the first input and it should be of type u8 tensor");
    OPENVINO_ASSERT(get_input_partial_shape(1).size() == 2, "SentencepieceDetokenizer expects 2D tensor as second input");

    auto batch_size = PartialShape({get_input_partial_shape(1)[0]});
    set_string_output(this, 0, batch_size);
}

bool SentencepieceStreamDetokenizer::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool SentencepieceStreamDetokenizer::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    if (m_sp == nullptr) {
        m_sp = std::make_shared<SentencePieceProcessor>();
        init_sp_model_in_eval(inputs, m_sp);
    };

    auto batch_size = inputs[1].get_shape()[0];
    auto seq_len    = inputs[1].get_shape()[1];
    auto input_data = inputs[1].data<const int32_t>();

    outputs[0].set_shape({batch_size});
    outputs[1].set_shape({batch_size});

    auto begins = outputs[0].data<int32_t>();
    auto ends   = outputs[1].data<int32_t>();
    const auto vocab_size = m_sp->GetPieceSize();
    std::deque<uint8_t> buffer;

    for(size_t batch = 0; batch < batch_size; ++batch) {
        const auto start = batch * seq_len;

        begins[batch] = buffer.size();
        for(size_t seq = start; seq < start + seq_len; ++seq) {
            const auto token_id = input_data[seq];

            if (token_id >= vocab_size) {
                continue;
            };

            const auto token = m_sp->IdToPiece(token_id);
            if(token.rfind("<") == 0 && token.rfind(">") == 5) {
                // convert "byte tokens" into bytes
                int ch = sentencepiece::PieceToByte(token);
                buffer.insert(buffer.end(), ch);
            } else {
                buffer.insert(buffer.end(), token.begin(), token.end());
            };

        };
        ends[batch] = buffer.size();
    }
    outputs[2].set_shape({buffer.size()});
    auto chars  = outputs[2].data<uint8_t>();
    std::copy(buffer.begin(), buffer.end(), chars);
    return true;
}

bool SentencepieceStreamDetokenizer::has_evaluate() const {
    return true;
}

std::shared_ptr<Node> SentencepieceStreamDetokenizer::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<SentencepieceStreamDetokenizer>(new_args, m_sp);
}
