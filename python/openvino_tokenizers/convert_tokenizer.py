# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Optional, Tuple, Union

from openvino.runtime import Model, Type
from openvino.runtime.exceptions import OVTypeError

from openvino_tokenizers.utils import change_inputs_type, change_outputs_type, update_rt_info, make_combine_segments_stateful
from openvino_tokenizers.constants import UTF8ReplaceMode

logger = logging.getLogger(__name__)


def convert_tokenizer(
    tokenizer_object: Any,
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
    ov_tokenizers = None

    if "transformers" in sys.modules:
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
                ov_tokenizers = convert_sentencepiece_model_tokenizer(
                    tokenizer_object,
                    add_attention_mask=True,
                    with_detokenizer=with_detokenizer,
                    streaming_detokenizer=streaming_detokenizer,
                    add_special_tokens=add_special_tokens,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    handle_special_tokens_with_re=handle_special_tokens_with_re,
                    utf8_replace_mode=utf8_replace_mode,
                )
            elif is_tiktoken_model(tokenizer_object):
                logger.info("Convert tiktoken-based tokenizer")
                ov_tokenizers = convert_tiktoken_model_tokenizer(
                    tokenizer_object,
                    with_detokenizer=with_detokenizer,
                    add_special_tokens=add_special_tokens,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    use_max_padding=use_max_padding,
                    utf8_replace_mode=utf8_replace_mode,
                )
            elif isinstance(tokenizer_object, PreTrainedTokenizerFast):
                logger.info("Convert Huggingface Fast tokenizer pipeline.")
                ov_tokenizers = convert_fast_tokenizer(
                    tokenizer_object,
                    number_of_inputs=1,
                    with_detokenizer=with_detokenizer,
                    add_special_tokens=add_special_tokens,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    use_max_padding=use_max_padding,
                    utf8_replace_mode=utf8_replace_mode,
                )
            else:
                raise OVTypeError(f"Huggingface tokenizer type is not supported: {type(tokenizer_object)}")

            if isinstance(ov_tokenizers, tuple):
                for ov_model in ov_tokenizers:
                    update_rt_info(ov_model, tokenizer_object)
            else:
                update_rt_info(ov_tokenizers, tokenizer_object)
    else:
        raise EnvironmentError(
            "No transformers library in the environment. Install required dependencies with one of two options:\n"
            "1. pip install openvino-tokenizers[transformers]\n"
            "2. pip install transformers[sentencepiece] tiktoken\n"
        )

    if ov_tokenizers is None:
        raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")
    
    ov_tokenizer = ov_tokenizers[0] if isinstance(ov_tokenizers, tuple) else ov_tokenizers
    make_combine_segments_stateful(ov_tokenizer, add_special_tokens)

    if isinstance(ov_tokenizers, tuple):
        return (
            change_outputs_type(ov_tokenizers[0], tokenizer_output_type),
            change_inputs_type(ov_tokenizers[1], detokenizer_input_type),
        )
    else:
        return change_outputs_type(ov_tokenizers, tokenizer_output_type)
