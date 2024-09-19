# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from openvino import Model, Type
from openvino.preprocess import PrePostProcessor
from openvino.runtime import opset12 as opset
from openvino.runtime.op import Constant
from openvino.runtime.op.util import VariableInfo, Variable
from openvino.runtime.passes import ModelPass, Manager
from openvino.runtime import Node, Input, Shape, Output, PartialShape

from .constants import (
    LOGITS_OUTPUT_NAME,
    ORIGINAL_TOKENIZER_CLASS_NAME,
    SPACE_SYMBOLS,
    TOKEN_IDS_OUTPUT_NAME,
    rt_info_to_hf_attribute_map,
)


logger = logging.getLogger(__name__)


def connect_models(
    first: Model,
    second: Model,
    name_map: Optional[Union[Sequence[Tuple[str, str]], Dict[str, str]]] = None,
    by_indices: bool = False,
    keep_second_model_unaligned_inputs: bool = True,
    keep_remaining_first_model_outputs: bool = False,
) -> Model:
    if by_indices:
        min_len = min(len(first.outputs), len(second.inputs))
        aligned_first_outputs = first.outputs[:min_len]
        aligned_second_inputs = second.inputs[:min_len]
    elif name_map is None:
        aligned_first_outputs = first.outputs
        aligned_second_inputs = [second.input(model1_output.get_any_name()) for model1_output in aligned_first_outputs]
    else:
        if isinstance(name_map, dict):
            name_map = list(name_map.items())
        aligned_first_outputs = [first.output(name1) for name1, _ in name_map]
        aligned_second_inputs = [second.input(name2) for _, name2 in name_map]

    for second_input, first_output in zip(aligned_second_inputs, aligned_first_outputs):
        logger.debug(f"Connecting: {first_output.get_any_name()} -> {second_input.get_any_name()}")
        for target in second_input.get_target_inputs():
            target.replace_source_output(first_output.get_node().input_value(0))
            # target.replace_source_output(model1_output)  # TODO: Produces incorrect topology

    new_inputs = first.inputs
    remaining_inputs = [input_ for input_ in second.inputs if input_ not in aligned_second_inputs]
    if keep_second_model_unaligned_inputs:
        new_inputs.extend(remaining_inputs)
    elif remaining_inputs:
        logger.info(
            "Some inputs of the second model were left uncovered and not included in the connected model: "
            + ", ".join(input_.name for input_ in remaining_inputs)
            + ". To add them set `keep_unaligned_inputs` to `True`"
        )
    new_inputs = [input_.get_node() for input_ in new_inputs]

    new_outputs = second.outputs
    remaining_outputs = [output for output in first.outputs if output not in aligned_first_outputs]
    if keep_remaining_first_model_outputs:
        new_outputs.extend(remaining_outputs)
    elif remaining_outputs:
        logger.info(
            "Some outputs of the first model were left uncovered and not included in the connected model: "
            + ", ".join(output.name for output in remaining_outputs)
            + ". To add them set `keep_unaligned_outputs` to `True`"
        )

    connected_model = Model(new_outputs, new_inputs, f"{first.get_name()}_with_{second.get_name()}")
    # TODO: Cleanup model1 and mode2 to avoid using them, they are ill-formed after the reconnection
    connected_model.validate_nodes_and_infer_types()
    return connected_model


def greedy_decoder(input) -> Model:
    argmax = opset.topk(
        data=input,
        k=1,
        axis=-1,
        mode="max",
        sort="none",
        name="ArgMax",
    )
    token_ids = opset.squeeze(
        data=argmax.output(1),
        axes=-1,
    )
    return token_ids.output(0)


def add_greedy_decoding(
    text_generation_model: Model, logits_output: str = LOGITS_OUTPUT_NAME, output_type: Type = Type.i64
) -> Model:
    ppp = PrePostProcessor(text_generation_model)
    ppp.output(logits_output).postprocess().custom(greedy_decoder)
    ppp.output(logits_output).tensor().set_element_type(output_type)
    model = ppp.build()
    model.output(logits_output).tensor.set_names({TOKEN_IDS_OUTPUT_NAME})
    return model


def change_inputs_type(model: Model, input_type: Type) -> Model:
    ppp = PrePostProcessor(model)
    for idx, _ in enumerate(model.inputs):
        ppp.input(idx).tensor().set_element_type(input_type)
    return ppp.build()


def change_outputs_type(model: Model, output_type: Type) -> Model:
    ppp = PrePostProcessor(model)
    for idx, _ in enumerate(model.outputs):
        ppp.output(idx).tensor().set_element_type(output_type)
    return ppp.build()


def has_incompatible_re2_op(pattern: str) -> bool:
    return "(?=" in pattern or "(?!" in pattern or "(?<=" in pattern or "(?<!" in pattern


_subpattern_regex = re.compile(r"(?:[^()|]+|\([^)]*\))+")


def filter_re2_incompatible(pattern: str) -> str:
    not_filtered = []

    for subpattern in (match.group() for match in _subpattern_regex.finditer(pattern)):
        if has_incompatible_re2_op(subpattern):
            logging.warning(f"Subpattern `{subpattern}` is not supported by re2 and filtered out.")
            continue
        not_filtered.append(subpattern)

    return "|".join(not_filtered)


# from transformers.models.gpt2.tokenization_gpt2
@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = (chr(n) for n in cs)
    return dict(zip(bs, cs))


def apply_bytes_to_unicode(token: str) -> str:
    bytes_to_unicode_dict = bytes_to_unicode()
    return "".join(bytes_to_unicode_dict[byte] for char in token for byte in char.encode())


def get_hf_tokenizer_attribute(
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
    attributes: Tuple[str],
) -> Any:
    return next((value for attr in attributes if (value := getattr(hf_tokenizer, attr, None)) is not None), None)


def update_rt_info(
    ov_tokenizer: Model,
    hf_tokenizer: "PreTrainedTokenizerBase",  # noqa
) -> None:
    ov_tokenizer.set_rt_info(str(type(hf_tokenizer)), ORIGINAL_TOKENIZER_CLASS_NAME)

    for rt_field_name, hf_attributes in rt_info_to_hf_attribute_map.items():
        attribute = get_hf_tokenizer_attribute(hf_tokenizer, hf_attributes)
        if attribute is not None:
            ov_tokenizer.set_rt_info(attribute, rt_field_name)


def generate_tokens_with_space_symbols(token: str, depth: int = 1):
    for symbol_1 in SPACE_SYMBOLS:
        new_token = token + symbol_1
        yield new_token
        if depth > 1:
            yield from generate_tokens_with_space_symbols(new_token, depth - 1)


def make_combine_segments_stateful(model: Model, default_flag: bool = True) -> None:
    """
    Transforms the given ov.Model to make the "CombineSegments" operation stateful.

    It patches every 3rd input (ends input) of "CombineSegments" operations if it is a Constant:
    adds mul operation to make them depend on a runtime-updatable state instead of fixed constants.
    E.g. if there are 7 inputs, then inputs 1 and 4 will be patched.

    Args:
        model (Model): The OpenVINO model to modify.
        default_flag (bool, optional): A flag used to initialize the state variable. Defaults to True.

    Returns:
        None
    """

    def check_const(output: Output) -> bool:
        res = output.get_node().get_type_info().name == "Constant"
        res &= output.get_element_type() == Type.i32
        res &= output.get_partial_shape() == PartialShape([])
        return res
    
    class MakeCombineSegmentsStatefull(ModelPass):
        def __init__(self):
            super().__init__()

        def run_on_model(self, model):
            combine_segments = [node for node in model.get_ops() if node.get_type_info().name == "CombineSegments"]
            consts_to_patch = []
            for node in combine_segments:
                consts_to_patch.extend([node.input(idx) for idx in range(1, len(node.inputs()) - 1, 3) if check_const(node.input_value(idx))])

            var_info = VariableInfo()
            var_info.data_shape = PartialShape([])
            var_info.data_type = Type.i32
            from openvino_tokenizers.constants import SPECIAL_TOKENS_STATE_NAME
            var_info.variable_id = SPECIAL_TOKENS_STATE_NAME
            variable = Variable(var_info)
            
            default_val = Constant(Type.i32, Shape([]), [int(default_flag)])
            read_value_node = opset.read_value(default_val, variable)
            
            # Remove this when plugin will be ready to consume ReadVelue without Assign
            model.add_variables([variable])
            model.add_sinks([opset.assign(read_value_node, variable)])

            for input in consts_to_patch:
                mul_node = opset.multiply(read_value_node, input.get_source_output())
                input.replace_source_output(mul_node.output(0))

    manager = Manager()
    manager.register_pass(MakeCombineSegmentsStatefull())
    manager.run_passes(model)
