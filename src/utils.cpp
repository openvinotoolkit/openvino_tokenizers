// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"
#include "ragged_tensor_pack.hpp"
#include <cstdlib>
#include <cctype>
#include <algorithm>

using namespace ov;
using namespace ov::opset15;

void parse_packed_strings (const Tensor& packed, int32_t& batch_size, const int32_t*& begin_ids, const int32_t*& end_ids, const uint8_t*& symbols) {
    auto strings = packed.data<const uint8_t>();
    auto bitstream_size = packed.get_byte_size();
    // check the format of the input bitstream representing the string tensor
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4, "Incorrect packed string tensor format: no batch size in the packed string tensor");
    batch_size = *reinterpret_cast<const int32_t*>(strings + 0);
    FRONT_END_GENERAL_CHECK(bitstream_size >= 4 + 4 + 4 * batch_size,
        "Incorrect packed string tensor format: the packed string tensor must contain first string offset and end indices");
    begin_ids = reinterpret_cast<const int32_t*>(strings + 4);
    end_ids = begin_ids + 1;
    symbols = strings + 4 + 4 + 4 * batch_size;
}

void check_string_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed string representation, got ", node->get_input_element_type(input_index+0));
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed string representation, got ", node->get_input_element_type(input_index+1));
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::u8,  "Expected a u8 tensor as the third part of the decomposed string representation, got ", node->get_input_element_type(input_index+2));
}

void check_string_scalar_input(const Node* node, size_t input_index) {
    auto shape = node->get_input_partial_shape(input_index);
    auto element_type = node->get_input_element_type(input_index);

#if false && USE_STRING_TENSORS
    // This block is not used when we convert ops to decomposed representation (and we really do)
    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::string) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 0),
        "string/0D tensor is expected, but observed: ", element_type.get_type_name(), ", ", shape.to_string());

    #else

    OPENVINO_ASSERT(
        (element_type == element::dynamic || element_type == element::u8) &&
        (shape.rank().is_dynamic() || shape.rank().get_length() == 1),
        "u8/1D tensor is expected, got element type ", element_type.to_string(), ", shape ", shape.to_string());

    #endif

}

void check_ragged_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed ragged representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed ragged representation");
    auto rank = node->get_input_partial_shape(input_index+2).rank();
    FRONT_END_GENERAL_CHECK(rank.is_dynamic() || rank.get_length() == 1, "The last tensor in ragged tensor representation should be a 1D tensor");
}

void check_ragged_string_input(const Node* node, size_t input_index) {
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::i32, "Expected an i32 tensor as the third part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+3) == element::i32, "Expected an i32 tensor as the forth part of the decomposed ragged string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+4) == element::u8,  "Expected a u8 tensor as the fifth part of the decomposed ragged string representation");
}

void set_string_output(Node* node, size_t output_index, const PartialShape& shape) {
    node->set_output_type(output_index+0, element::i32, shape);     // byte offset in output[+2] -- begin of each string
    node->set_output_type(output_index+1, element::i32, shape);     // byte offset in output[+2] -- end of each string
    node->set_output_type(output_index+2, element::u8,  PartialShape{Dimension()});     // symbols from all strings concatenated
}

void set_ragged_string_output(Node* node, size_t output_index, const PartialShape& shape) {
    node->set_output_type(output_index+0, element::i32, shape);     // element offset in output[+2] -- begin of each ragged dimension elements
    node->set_output_type(output_index+1, element::i32, shape);     // element offset in output[+3] -- end of each ragged dimension elements
    node->set_output_type(output_index+2, element::i32, PartialShape{Dimension()}); // byte offset in output[+4] -- begin of each string
    node->set_output_type(output_index+3, element::i32, PartialShape{Dimension()}); // byte offset in output[+4] -- end of each string
    node->set_output_type(output_index+4, element::u8,  PartialShape{Dimension()}); // symbols from all strings cnocatenated
}

void set_ragged_output(Node* node, size_t output_index, const PartialShape& shape, element::Type type) {
    node->set_output_type(output_index+0, element::i32, shape);     // element offset in output[+2] -- begin of each ragged dimension elements
    node->set_output_type(output_index+1, element::i32, shape);     // element offset in output[+2] -- end of each ragged dimension elements
    node->set_output_type(output_index+2, type, PartialShape{Dimension()}); // flatten elements
}


void unpack_strings_to_tensors (const std::string* strings, const Shape shape, ov::Tensor& begins, ov::Tensor& ends, ov::Tensor& chars) {
    auto nelements = shape_size(shape);

    size_t total = 0;
    for(size_t i = 0; i < nelements; ++i)
        total += strings[i].length();

    begins.set_shape(shape);
    ends.set_shape(shape);
    chars.set_shape(Shape{total});

    auto pbegins = begins.data<int32_t>();
    auto pends = ends.data<int32_t>();
    auto poutput_symbols = reinterpret_cast<char*>(chars.data<uint8_t>());
    size_t offset = 0;

    for(size_t i = 0; i < nelements; ++i)
    {
        pbegins[i] = offset;
        poutput_symbols = std::copy(strings[i].begin(), strings[i].end(), poutput_symbols);
        offset += strings[i].length();
        pends[i] = offset;
    }
}

void override_parameter (std::shared_ptr<ov::Node> node, element::Type type, const PartialShape& shape) {
    if (auto parameter = std::dynamic_pointer_cast<Parameter>(node)) {
        // TODO: Apply this change conditionally based on real Parameter value
        if (getenv_bool("OPENVINO_TOKENIZERS_PRINT_DEBUG_INFO", false)) {
            std::cerr << "Overriding Parameter element_type to " << type << " and shape " << shape << "\n";
        }
        parameter->set_partial_shape(shape);
        parameter->set_element_type(type);
        parameter->validate_and_infer_types();
    }
}

OutputVector pre_translate_string_tensor_input(const ov::Output<ov::Node>& input) {
    auto input_node = input.get_node_shared_ptr();

    if (auto struct_pack = std::dynamic_pointer_cast<op::v15::StringTensorPack>(input_node)) {
        FRONT_END_GENERAL_CHECK(struct_pack->get_input_size() == 3, "Expected 3 inputs to StringTensorPack which represents a string tensor");
        return struct_pack->input_values();
    }
    else {
        return std::make_shared<op::v15::StringTensorUnpack>(input)->outputs();
    }
}

OutputVector pre_translate_ragged_tensor_input(ov::Output<ov::Node> input) {
    auto ragged_pack = dynamic_cast<RaggedTensorPack*>(input.get_node());
    OPENVINO_ASSERT(ragged_pack, "Expected RaggedTensorPack but didn't find it");
    return ragged_pack->input_values();
}

OutputVector pre_translate_ragged_string_tensor_input(ov::Output<ov::Node> input) {
    auto ragged_inputs = pre_translate_ragged_tensor_input(input);
    auto string_inputs = pre_translate_string_tensor_input(ragged_inputs[2]);
    ragged_inputs.pop_back();
    ragged_inputs.insert(ragged_inputs.end(), string_inputs.begin(), string_inputs.end());
    return ragged_inputs;
}

ov::Output<ov::Node> post_translate_string_tensor_output(const OutputVector& outputs) {
    FRONT_END_GENERAL_CHECK(outputs.size() == 3, "Expected 3 tensors in decomposed string tensor representation");
    return std::make_shared<op::v15::StringTensorPack>(outputs[0], outputs[1], outputs[2]);
}

ov::Output<ov::Node> post_translate_ragged_tensor_output(const OutputVector& outputs) {
    FRONT_END_GENERAL_CHECK(outputs.size() == 3, "Expected 3 tensors in decomposed string tensor representation");
    return std::make_shared<RaggedTensorPack>(outputs);
}

bool evaluate_normalization_helper (ov::TensorVector& outputs, const ov::TensorVector& inputs, std::function<std::string(const std::string&)> normalizer, const bool has_skips) {

    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    auto skips = has_skips ? inputs[3].data<bool>() : nullptr;

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    if (has_skips) {
        outputs[3] = inputs[3];
    }

    const size_t num_elements = inputs[0].get_size();

    // TODO: How to avoid copying from this temporary buffer?
    // TODO: It can be possible to collect output symbols directly in the output tensor memory if `normalizer` has reasonable estimation for the final size.
    std::vector<std::string> buffer;
    buffer.resize(num_elements);

    // For the whole implementation below the input shapes can be ignored, we are working with the flatten representaions
    // and only number of elements in the original tensors matter

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();

    size_t total_size = 0;
    if (has_skips) {
        total_size = ov::parallel_sum(num_elements, total_size, [&](size_t i) -> size_t {
            const std::string input_string = std::string(chars + begins[i], chars + ends[i]);
            const std::string normalized = (skips[i] == 0) ? normalizer(std::move(input_string)) : input_string;
            buffer[i] = normalized;
            return normalized.size();
        });
    } else {
        total_size = ov::parallel_sum(num_elements, total_size, [&](size_t i) -> size_t {
            const std::string normalized = normalizer(std::string(chars + begins[i], chars + ends[i]));
            buffer[i] = normalized;
            return normalized.size();
        });
    };

    outputs[2].set_shape(Shape{total_size});
    auto new_chars  = outputs[2].data<uint8_t>();

    size_t current_size = 0;
    for(size_t i = 0; i < num_elements; ++i) {
        new_begins[i] = current_size;
        std::copy(buffer[i].begin(), buffer[i].end(), new_chars + current_size);
        current_size += buffer[i].size();
        new_ends[i] = current_size;
    }
    return true;
}

std::shared_ptr<Node> string_attribute_to_constant (const ov::frontend::NodeContext& node, const std::string& name) {
    auto value = node.get_attribute<std::string>(name);

    // TODO: How to translate attribute `replace_global`?

    #if USE_STRING_TENSORS
    return std::make_shared<Constant>(element::string, Shape{}, &value);
    #else
    return std::make_shared<Constant>(element::u8, Shape{value.length()}, (const void*)value.data());
    #endif
}

void set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        outputs[idx].get_tensor().add_names({ node_name + ":" + std::to_string(idx) });
    }
}

PCRE2Wrapper::PCRE2Wrapper(const absl::string_view& pattern) {
    int errorcode;
    PCRE2_SIZE erroroffset;
    m_compiled = pcre2_compile((PCRE2_SPTR) pattern.data(), 
                                pattern.size(), PCRE2_UTF | PCRE2_UCP, 
                                &errorcode, &erroroffset, NULL);
    auto jit_code = pcre2_jit_compile(m_compiled, PCRE2_JIT_COMPLETE);
    m_is_jit = (jit_code == 0);
    if (m_compiled == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
        if (getenv_bool("OPENVINO_TOKENIZERS_PRINT_DEBUG_INFO", false)) {
            std::cerr << "PCRE2 compilation failed at offset " << erroroffset << ": " << buffer << std::endl;
        }
        return;
    }
}

PCRE2Wrapper::~PCRE2Wrapper() {
    if (m_compiled != nullptr) {
        pcre2_code_free(m_compiled);
        m_compiled = nullptr;
    }
}

std::string PCRE2Wrapper::substitute (const std::string& orig_str,
                                     const absl::string_view& replace_pattern,
                                     bool global_replace) const {
    if (m_compiled == nullptr) {
        return orig_str;
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = orig_str.size();

    // Check if the string matches the pattern
    const auto match_func = m_is_jit ? pcre2_jit_match : pcre2_match;
    int num_matches = match_func(
        m_compiled,
        (PCRE2_SPTR) orig_str.c_str(), subject_length,
        0,
        PCRE2_NO_UTF_CHECK,
        match_data,
        NULL
    );
    if (num_matches < 0 || num_matches == PCRE2_ERROR_NOMATCH) {
        pcre2_match_data_free(match_data);
        return orig_str;
    }

    // Allocate dynamically since lenght depends dynamically on the lenght of input and replace strings.
    // Allocated memory will be freed at the exit from function.
    size_t buffer_length = sizeof(PCRE2_UCHAR) * 4 * (subject_length + num_matches * replace_pattern.size());
    PCRE2_UCHAR* buffer = (PCRE2_UCHAR*) std::malloc(buffer_length);
    if (buffer == nullptr) {
        if (getenv_bool("OPENVINO_TOKENIZERS_PRINT_DEBUG_INFO", false)) {
            std::cerr << "Memory allocation failed" << std::endl;
        }
        pcre2_match_data_free(match_data);
        return orig_str;
    }

    int rc = pcre2_substitute(
        m_compiled,
        (PCRE2_SPTR) orig_str.c_str(), orig_str.size(),
        0,
        (global_replace ? PCRE2_SUBSTITUTE_GLOBAL : 0) | PCRE2_NO_UTF_CHECK,
        match_data,
        NULL,
        (PCRE2_SPTR) replace_pattern.data(), replace_pattern.size(),
        buffer,
        &buffer_length
    );

    if (rc < 0) {
        if (getenv_bool("OPENVINO_TOKENIZERS_PRINT_DEBUG_INFO", false)) {
            if (rc == PCRE2_ERROR_NOMEMORY) {
                std::cerr << "Buffer overflow" << std::endl;
            } else {
                size_t error_length = sizeof(PCRE2_UCHAR) * 400;
                PCRE2_UCHAR* error_buffer = (PCRE2_UCHAR*) std::malloc(error_length);
                pcre2_get_error_message(rc, error_buffer, error_length);
                std::cerr << "PCRE2 substitution failed with error code " << rc  << ": " << error_buffer << std::endl;
            }
        }
        pcre2_match_data_free(match_data);
        std::free(buffer);
        return orig_str;
    }
    auto res = std::string(reinterpret_cast<char*>(buffer), buffer_length);
    std::free(buffer);
    pcre2_match_data_free(match_data);
    return res;
}

std::pair<size_t, size_t> PCRE2Wrapper::match(const std::string& str, size_t curr_start) const {
    if (m_compiled == nullptr) {
        return {SIZE_MAX, SIZE_MAX};
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = str.length();

    const auto match_func = m_is_jit ? pcre2_jit_match : pcre2_match;
    int match_result = match_func(
        m_compiled,
        (PCRE2_SPTR) str.c_str(), subject_length,
        curr_start,
        0,
        match_data,
        NULL
    );

    if (match_result < 0) {
        pcre2_match_data_free(match_data);
        return {SIZE_MAX, SIZE_MAX};
    }

    // At this point there is at least one match, no out of bound can happen here.
    // If ovector do not contain values early return is done and the code below is not run.
    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    std::pair<size_t, size_t> res = {ovector[0], ovector[1]};

    // Free only after copying results from match_data to res;
    pcre2_match_data_free(match_data);
    return res;
}

std::pair<size_t, size_t> PCRE2Wrapper::match(const std::string_view& str, size_t curr_start) const {
    if (m_compiled == nullptr) {
        return {SIZE_MAX, SIZE_MAX};
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = str.size();

    const auto match_func = m_is_jit ? pcre2_jit_match : pcre2_match;
    int match_result = match_func(
        m_compiled,
        (PCRE2_SPTR) str.data(), subject_length,
        static_cast<PCRE2_SIZE>(curr_start),
        0,
        match_data,
        NULL
    );

    if (match_result < 0) {
        pcre2_match_data_free(match_data);
        return {SIZE_MAX, SIZE_MAX};
    }

    // At this point there is at least one match, no out of bound can happen here.
    // If ovector do not contain values early return is done and the code below is not run.
    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    std::pair<size_t, size_t> res = {ovector[0], ovector[1]};

    // Free only after copying results from match_data to res;
    pcre2_match_data_free(match_data);
    return res;
}

// Return both full-match offsets and capture-group offsets in one call
std::pair<std::pair<size_t,size_t>, std::pair<size_t,size_t>> PCRE2Wrapper::match_and_find_group(const std::string& str, size_t curr_start) const {
    if (m_compiled == nullptr) {
        return {{SIZE_MAX, SIZE_MAX}, {SIZE_MAX, SIZE_MAX}};
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = str.length();

    const auto match_func = m_is_jit ? pcre2_jit_match : pcre2_match;
    int match_result = match_func(
        m_compiled,
        (PCRE2_SPTR) str.c_str(), subject_length,
        curr_start,
        0,
        match_data,
        NULL
    );

    if (match_result < 0) {
        pcre2_match_data_free(match_data);
        return {{SIZE_MAX, SIZE_MAX}, {SIZE_MAX, SIZE_MAX}};
    }

    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    const std::pair<size_t,size_t> full_match = std::make_pair(ovector[0], ovector[1]);
    std::pair<size_t, size_t> group_match = std::make_pair(SIZE_MAX, SIZE_MAX);

    // in the old tokenizers #special tokens == #capture groups, find the only one that is inside full match
    // optimize for hundreds of tokens by using parallel_for
    // newer tokenizers (>2025.3.0) has <= 4 capture groups
    ov::parallel_for(pcre2_get_ovector_count(match_data) - 1, [&](size_t group){
        ++group;  // group 0 is full match
        if (full_match.first <= ovector[2*group] && ovector[2*group] <= full_match.second && ovector[2*group + 1] <= full_match.second) {
            group_match = {ovector[2*group], ovector[2*group + 1]};
        }
    });

    pcre2_match_data_free(match_data);
    return {full_match, group_match};
}


void Trie::add(const std::vector<unsigned char>& str, const int value, int idx) {
    if (idx == str.size()) {
        m_value = value;
    } else {
        auto ch = str[idx];
        if (m_to.count(ch) == 0) {
            m_to[ch] = std::make_unique<Trie>();
        }
        m_to[ch]->add(str, value, idx + 1);
    }
}


int Trie::find_longest(const std::vector<unsigned char>& str, int& idx) const {
    int token_id = -1;  // no token found
    const Trie* current_node = this;

    uint8_t ch = str[idx];
    int end_idx = idx;

    while (current_node->m_to.count(ch)) {
        current_node = current_node->m_to.at(ch).get();
        idx++;
        if (current_node->m_value != -1) {
            token_id = current_node->m_value;
            end_idx = idx;
        }
        if (idx == str.size()) {
            break;
        }
        ch = str[idx];
    }
    idx = end_idx;
    return token_id;
}

int Trie::find_longest(const std::string_view& str, int& idx) const {
    int token_id = -1;  // no token found
    const Trie* current_node = this;

    uint8_t ch = str[idx];
    int end_idx = idx;

    while (current_node->m_to.count(ch)) {
        current_node = current_node->m_to.at(ch).get();
        idx++;
        if (current_node->m_value != -1) {
            token_id = current_node->m_value;
            end_idx = idx;
        }
        if (idx == str.size()) {
            break;
        }
        ch = str[idx];
    }
    idx = end_idx;
    return token_id;
}

bool getenv_bool(const char* env_var, bool default_value) {
    const char* env_p = std::getenv(env_var);
    std::string value = env_p != nullptr ? std::string(env_p) : "";

    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });

    std::set<std::string> off = {"0", "false", "off"};
    std::set<std::string> on = {"1", "true", "on"};
    bool rc;
    if (value == "") {
        rc = default_value;
    } else if (off.find(value) != off.end()) {
        rc = false;
    } else if (on.find(value) != on.end()) {
        rc = true;
    } else {
        std::stringstream ss;
        ss << "environment variable '" << env_var << "' value '" << value << "' invalid. Must be boolean.";
        throw std::runtime_error(ss.str());
    }
    return rc;
}
