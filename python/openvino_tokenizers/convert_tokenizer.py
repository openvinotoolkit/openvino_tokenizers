# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Optional, Tuple, Union
from functools import wraps

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

    @wraps(func)
    def wrapper(*args, **kwargs):
        params = None
        if len(args) > 1 and args[1] != None:
            params = args[1]
        if 'params' in kwargs:
            params = kwargs['params']
        
        if params is not None:
            for key in TokenzierConversionParams.__match_args__:
                if kwargs.get(key) is not None:
                    msg = (
                        "Cannot specify both 'params' and individual convert_tokenizer arguments simultaneously. "
                        "Please pass all conversion params either individually, e.g. "
                        "convert_tokenizer(tokenizer_object, with_detokenizr=True, add_special_tokens=True,...). "
                        "Or within 'params' argument, e.g. "
                        "convert_tokenzier(tokenizer_object, params={'with_detokenizr': True, 'add_special_tokens': True, ...})"
                    )
                    raise ValueError(msg)
        
        if isinstance(params, dict):
            params = TokenzierConversionParams(**params)
        if params is None:
            params = TokenzierConversionParams(**kwargs)
        return func(args[0], params)
    
    # Embed convert_tokenizer docstring with TokenzierConversionParams docstring.
    wrapper.__doc__ = func.__doc__.replace('Returns:',  'Returns:\n'+ TokenzierConversionParams.__doc__ + '\n')
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
    using the passed keyword arguments to control the behavior of the conversion. Either params, 
    or keyword arguments should be passed, if both are specified an Error will be thrown.

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
        if can_use_sentencepiece and (is_unigram or not tokenizer_object.is_fast or params.use_sentencepiece_backend):
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
