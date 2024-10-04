# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Optional, Tuple, Union

from openvino.runtime import Model, Type
from openvino.runtime.exceptions import OVTypeError

from openvino_tokenizers.constants import UTF8ReplaceMode
from openvino_tokenizers.utils import (
    change_inputs_type,
    change_outputs_type,
    update_rt_info,
    TokenzierConversionParams,
)

logger = logging.getLogger(__name__)


def capture_arg(func):
    def wrapper(*argc, **kwargs):
        params = None
        if len(argc) > 1 and argc[1] != None:
            params = argc[1]
        if 'params' in kwargs:
            params = kwargs['params']
        
        if params is not None:
            for key in TokenzierConversionParams.__match_args__:
                if kwargs[key] is not None:
                    msg = "Cannot specify both 'params' and individual convert_tokenizer arguments simultaneously. " \
                          "Please pass all conversion params either individually, e.g. " \
                          "convert_tokenizer(tokenizer_object, with_detokenizr=True, add_special_tokens=True,...). " \
                          "Or within 'params' argument, e.g. " \
                          "convert_tokenzier(tokenizer_object, params={'with_detokenizr': True, 'add_special_tokens': True, ...})"
                    raise ValueError(msg)
        
        return func(*argc, **kwargs)
    
    # Embed convert_tokenizer docstring with TokenzierConversionParams docstring.
    pos = func.__doc__.find('    Returns:')
    wrapper.__doc__ = func.__doc__[:pos] + TokenzierConversionParams.__doc__ + '\n' + func.__doc__[pos:]
    return wrapper


@capture_arg
def convert_tokenizer(
    tokenizer_object: Any,
    params: Union[TokenzierConversionParams, dict] = None,
    *,
    with_detokenizer: bool = False,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: Optional[bool] = None,
    tokenizer_output_type: Type = Type.i64,
    detokenizer_input_type: Type = Type.i64,
    streaming_detokenizer: bool = False,
    use_max_padding: bool = False,
    handle_special_tokens_with_re: Optional[bool] = None,
    use_sentencepiece_backend: bool = False,
    utf8_replace_mode: Optional[UTF8ReplaceMode] = None,
) -> Union[Model, Tuple[Model, Model]]:
    """
    Converts a given tokenizer object into an OpenVINO-compatible model.

    If no `params` are provided, the function will construct a `TokenzierConversionParams` instance
    using the passed keyword arguments to control the behavior of the conversion.

    Parameters:
    -----------
    tokenizer_object : Any
        The tokenizer object to convert. This should be an instance of a supported tokenizer, such
        as Huggingface's `PreTrainedTokenizerBase` or `PreTrainedTokenizerFast`.

    params : TokenzierConversionParams, optional
        If provided, the `TokenzierConversionParams` object containing conversion parameters.
        If not provided, the parameters will be constructed from the other keyword arguments.
    Returns:
    --------
    Union[Model, Tuple[Model, Model]]
        The converted tokenizer model, or a tuple tokenizer and detokenizer depending on with_detokenizer value.
    """

    if params is None:
        params = TokenzierConversionParams(
            with_detokenizer=with_detokenizer,
            add_special_tokens=add_special_tokens,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            tokenizer_output_type=tokenizer_output_type,
            detokenizer_input_type=detokenizer_input_type,
            streaming_detokenizer=streaming_detokenizer,
            use_max_padding=use_max_padding,
            handle_special_tokens_with_re=handle_special_tokens_with_re,
            use_sentencepiece_backend=use_sentencepiece_backend,
            utf8_replace_mode=utf8_replace_mode,
        )

    if isinstance(params, dict):
        params = TokenzierConversionParams(**params)

    ov_tokenizers = None

    if "transformers" not in sys.modules:
        raise EnvironmentError(
            "No transformers library in the environment. Install required dependencies with one of two options:\n"
            "1. pip install openvino-tokenizers[transformers]\n"
            "2. pip install transformers[sentencepiece] tiktoken\n"
        )

    from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
    from .hf_parser import (
        convert_fast_tokenizer,
        convert_sentencepiece_model_tokenizer,
        convert_tiktoken_model_tokenizer,
        is_sentencepiece_bpe_model,
        is_sentencepiece_model,
        is_tiktoken_model,
    )

    can_use_sentencepiece = is_sentencepiece_model(tokenizer_object)
    is_unigram = can_use_sentencepiece and not is_sentencepiece_bpe_model(tokenizer_object)
    if isinstance(tokenizer_object, PreTrainedTokenizerBase):
        if can_use_sentencepiece and (is_unigram or not tokenizer_object.is_fast or use_sentencepiece_backend):
            logger.info("Convert tokenizer using SentencePiece .model file.")
            ov_tokenizers = convert_sentencepiece_model_tokenizer(tokenizer_object, params)
        elif is_tiktoken_model(tokenizer_object):
            logger.info("Convert tiktoken-based tokenizer")
            ov_tokenizers = convert_tiktoken_model_tokenizer(tokenizer_object, params)
        elif isinstance(tokenizer_object, PreTrainedTokenizerFast):
            logger.info("Convert Huggingface Fast tokenizer pipeline.")
            ov_tokenizers = convert_fast_tokenizer(tokenizer_object, params)
        else:
            raise OVTypeError(f"Huggingface tokenizer type is not supported: {type(tokenizer_object)}")

        if isinstance(ov_tokenizers, tuple):
            for ov_model in ov_tokenizers:
                update_rt_info(ov_model, tokenizer_object, params)
        else:
            update_rt_info(ov_tokenizers, tokenizer_object, params)

    if ov_tokenizers is None:
        raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")

    if isinstance(ov_tokenizers, tuple):
        return (
            change_outputs_type(ov_tokenizers[0], params.tokenizer_output_type),
            change_inputs_type(ov_tokenizers[1], params.detokenizer_input_type),
        )
    return change_outputs_type(ov_tokenizers, params.tokenizer_output_type)
