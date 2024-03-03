# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Dict, Optional, Sequence, Tuple, Union

from openvino import Model, Type
from openvino.preprocess import PrePostProcessor
from openvino.runtime import opset12 as opset

from .constants import LOGITS_OUTPUT_NAME, TOKEN_IDS_OUTPUT_NAME


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

    new_inputs = first.get_parameters()
    aligned_second_input_names = [ second_input.get_any_name() for second_input in aligned_second_inputs ]
    remaining_inputs = [ param for param in second.get_parameters() if param.get_friendly_name() not in aligned_second_input_names ]
    if keep_second_model_unaligned_inputs:
        new_inputs.extend(remaining_inputs)
    elif remaining_inputs:
        logger.info(
            "Some inputs of the second model were left uncovered and not included in the connected model: "
            + ", ".join(input_.name for input_ in remaining_inputs)
            + ". To add them set `keep_unaligned_inputs` to `True`"
        )

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
