// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"
#include "string_tensor_pack.hpp"
#include "string_tensor_unpack.hpp"
#include "ragged_tensor_pack.hpp"

using namespace ov;
using namespace ov::frontend;
using namespace ov::opset13;

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
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+0) == element::i32, "Expected an i32 tensor as the first part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+1) == element::i32, "Expected an i32 tensor as the second part of the decomposed string representation");
    FRONT_END_GENERAL_CHECK(node->get_input_element_type(input_index+2) == element::u8,  "Expected a u8 tensor as the third part of the decomposed string representation");
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
        "u8/1D tensor is expected");

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
        std::cerr << "Overriding Parameter element_type to " << type << " and shape " << shape << "\n";
        parameter->set_partial_shape(shape);
        parameter->set_element_type(type);
        parameter->validate_and_infer_types();
    }
}

OutputVector pre_translate_string_tensor_input(const ov::Output<ov::Node>& input) {
    auto input_node = input.get_node_shared_ptr();

    if (auto struct_pack = std::dynamic_pointer_cast<StringTensorPack>(input_node)) {
        FRONT_END_GENERAL_CHECK(struct_pack->get_input_size() == 3, "Expected 3 inputs to StringTensorPack which represents a string tensor");
        return struct_pack->input_values();
    }
    else {
        return std::make_shared<StringTensorUnpack>(OutputVector{ input }, "begins_ends")->outputs();
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
    return std::make_shared<StringTensorPack>(outputs, "begins_ends");
}

ov::Output<ov::Node> post_translate_ragged_tensor_output(const OutputVector& outputs) {
    FRONT_END_GENERAL_CHECK(outputs.size() == 3, "Expected 3 tensors in decomposed string tensor representation");
    return std::make_shared<RaggedTensorPack>(outputs);
}

bool evaluate_normalization_helper (ov::TensorVector& outputs, const ov::TensorVector& inputs, std::function<std::string(const std::string&)> normalizer) {
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
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
    total_size = ov::parallel_sum(num_elements, total_size, [&](size_t i) -> int {
        const std::string normalized = normalizer(std::string(chars + begins[i], chars + ends[i]));
        buffer[i] = normalized;
        return normalized.size();
    });

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
    if (m_compiled == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
        std::cerr << "PCRE2 compilation failed at offset " << erroroffset << ": " << buffer << std::endl;
        return;
    }
}

PCRE2Wrapper::~PCRE2Wrapper() {
    if (m_compiled != nullptr) {
        pcre2_code_free(m_compiled);
        m_compiled = nullptr;
    }
}

std::string PCRE2Wrapper::substitute(const std::string& orig_str, 
                                     const absl::string_view& replace_pattern,
                                     bool global_replace) {
    if (m_compiled == nullptr) {
        return orig_str;
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = orig_str.size();
    
    // Usually found pattern is replaced by shorter string, but set 3 times more space for safety.
    // Allocate dynamically since lenght depends dynamically on the lenght of input string.
    // Allocated memory will be freed at the exit from function.
    auto buffer = (PCRE2_UCHAR*) std::malloc(sizeof(PCRE2_UCHAR) * subject_length * 3);
    
    // Check if the string matches the pattern
    int match_result = pcre2_match(
        m_compiled,
        (PCRE2_SPTR) orig_str.c_str(), subject_length,
        0,
        0,
        match_data,
        NULL
    );
    if (match_result < 0 || match_result == PCRE2_ERROR_NOMATCH) {
        pcre2_match_data_free(match_data);
        return orig_str;
    }

    int rc = pcre2_substitute(
        m_compiled,
        (PCRE2_SPTR) orig_str.c_str(), orig_str.size(),
        0,
        global_replace ? PCRE2_SUBSTITUTE_GLOBAL : 0,
        match_data,
        NULL,
        (PCRE2_SPTR) replace_pattern.data(), replace_pattern.size(),
        buffer,
        &subject_length
    );

    if (rc < 0) {
        if (rc == PCRE2_ERROR_NOMEMORY) {
            std::cerr << "Buffer overflow" << std::endl;
        } else {
            std::cerr << "PCRE2 substitution failed with error code " << rc << std::endl;
        }
        pcre2_match_data_free(match_data);
        return orig_str;
    }
    auto res = std::string(reinterpret_cast<char*>(buffer), subject_length);
    std::free(buffer);
    pcre2_match_data_free(match_data); 
    return res;
}

std::pair<size_t, size_t> PCRE2Wrapper::match(const std::string& str, size_t curr_start) {
    if (m_compiled == nullptr) {
        return {SIZE_MAX, SIZE_MAX};
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(m_compiled, NULL);
    PCRE2_SIZE subject_length = str.length();

    int match_result = pcre2_match(
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
    // If we survived the previous IF the is at least one match,
    // not out of bound can happen here.
    PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    
    pcre2_match_data_free(match_data); 
    return {ovector[0], ovector[1]};
}
